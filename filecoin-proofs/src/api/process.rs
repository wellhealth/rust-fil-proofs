use anyhow::{bail, Context, Result};
use log::info;
use scopeguard::defer;
use std::collections::VecDeque;
use std::io::Read;
use std::{
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
    process,
    sync::Mutex,
};
use storage_proofs_core::{merkle::MerkleTreeTrait, sector::SectorId, settings};
use uuid::Uuid;

use crate::{
    api::cores::get_l3_index, api::cores::serialize, PieceInfo, PoRepConfig, ProverId,
    SealPreCommitOutput, SealPreCommitPhase1Output, Ticket,
};
use serde::{Deserialize, Serialize};
pub const SHENSUANYUN_GPU_INDEX: &str = "SHENSUANYUN_GPU_INDEX";

#[derive(Serialize, Deserialize)]
struct P2Param<Tree: 'static + MerkleTreeTrait> {
    porep_config: PoRepConfig,
    #[serde(bound(
        serialize = "SealPreCommitPhase1Output<Tree>: Serialize",
        deserialize = "SealPreCommitPhase1Output<Tree>: Deserialize<'de>"
    ))]
    phase1_output: SealPreCommitPhase1Output<Tree>,
    cache_path: PathBuf,
    replica_path: PathBuf,
}

#[derive(Serialize, Deserialize)]
struct P1Param {
    porep_config: PoRepConfig,
    cache_path: PathBuf,
    in_path: PathBuf,
    prover_id: ProverId,
    sector_id: SectorId,
    ticket: Ticket,
    piece_infos: Vec<PieceInfo>,
}

pub fn p1<R, S, Tree: 'static + MerkleTreeTrait>(
    porep_config: PoRepConfig,
    cache_path: R,
    in_path: S,
    prover_id: ProverId,
    sector_id: SectorId,
    ticket: Ticket,
    piece_infos: &[PieceInfo],
) -> Result<SealPreCommitPhase1Output<Tree>>
where
    R: AsRef<Path>,
    S: AsRef<Path>,
{
    let l3_index = get_l3_index();
    let l3_params = l3_index
        .as_ref()
        .map(|x| &x.0[..])
        .map(serialize)
        .unwrap_or_default();

    info!("cores: {}", l3_params);
    let data = P1Param {
        porep_config,
        cache_path: cache_path.as_ref().to_owned(),
        in_path: in_path.as_ref().to_owned(),
        prover_id,
        sector_id,
        ticket,
        piece_infos: piece_infos.to_owned(),
    };
    let uuid = get_uuid();

    let param_folder = get_param_folder();
    std::fs::create_dir_all(&param_folder)
        .with_context(|| format!("cannot create dir: {:?}", param_folder))?;
    let in_path = Path::new(&param_folder).join(&uuid);
    let out_path = Path::new(&param_folder).join(&uuid);
    let program_folder = &settings::SETTINGS.program_folder;
    let program_path = Path::new(program_folder).join(&settings::SETTINGS.p1_program_name);

    let infile = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(&in_path)
        .with_context(|| format!("{:?}: cannot open file to pass in c2 parameter", sector_id))?;

    defer!({
        let _ = std::fs::remove_file(&out_path);
    });

    info!("{:?}: writing parameter to file: {:?}", sector_id, in_path);
    serde_json::to_writer(infile, &data)
        .with_context(|| format!("{:?}: cannot sealize to infile", sector_id))?;

    info!("{:?}: start p1 with program: {:?}", sector_id, program_path);
    let mut child = process::Command::new(&program_path)
        .arg(&uuid)
        .arg(u64::from(porep_config.sector_size).to_string())
        .arg(u64::from(sector_id).to_string())
        .arg(l3_params)
        .spawn()
        .with_context(|| format!("{:?}, cannot start program {:?} ", sector_id, program_path))?;

    let status = child.wait().expect("c2 is not running");

    match status.code() {
        Some(0) => {
            info!("{:?} p1 finished", sector_id);
        }
        Some(n) => {
            info!("{:?} p1 failed with exit number: {}", sector_id, n);
            bail!("{:?} p1 failed with exit number: {}", sector_id, n);
        }
        None => {
            info!("{:?} p1 crashed", sector_id);
            bail!("{:?} p1 crashed", sector_id);
        }
    }
    let out = File::open(&out_path).with_context(|| {
        format!(
            "{:?}: cannot open uuid file: {:?} for p1 output",
            sector_id, out_path
        )
    })?;

    serde_json::from_reader(out)
        .with_context(|| format!("{:?}: cannot serialzie output for p1 result", sector_id))
}

pub fn p2<R, S, Tree: 'static + MerkleTreeTrait>(
    porep_config: PoRepConfig,
    phase1_output: SealPreCommitPhase1Output<Tree>,
    cache_path: S,
    replica_path: R,
) -> Result<SealPreCommitOutput>
where
    R: AsRef<Path>,
    S: AsRef<Path>,
{
    let gpu_index = select_gpu_device()
        .with_context(|| format!("{:?}: cannot select gpu device", replica_path.as_ref()))?;
    defer! {

        info!("release gpu index: {}", gpu_index);
        release_gpu_device(gpu_index);
    };
    info!(
        "{:?}: selected gpu device: {}",
        replica_path.as_ref(),
        gpu_index
    );
    std::env::set_var(SHENSUANYUN_GPU_INDEX, &gpu_index.to_string());

    info!("{:?}: set_var finished", replica_path.as_ref());
    let cache_path = cache_path.as_ref().to_owned();
    let replica_path = replica_path.as_ref().to_owned();
    let param_folder = get_param_folder();
    info!("{:?}: get_param_folder finished", replica_path);
    std::fs::create_dir_all(&param_folder)
        .with_context(|| format!("cannot create dir: {:?}", param_folder))?;
    let program_folder = &settings::SETTINGS.program_folder;

    let data = P2Param {
        porep_config,
        phase1_output,
        cache_path,
        replica_path: replica_path.clone(),
    };
    info!("{:?}: data collected", replica_path);
    let uuid = get_uuid();

    let in_path = Path::new(&param_folder).join(&uuid);
    let p2 = Path::new(program_folder).join(&settings::SETTINGS.p2_program_name);
    let out_path = Path::new(&param_folder).join(&uuid);

    let infile = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(&in_path)
        .with_context(|| "cannot open file to pass p2 parameter".to_string())?;
    defer!({
        let _ = std::fs::remove_file(&out_path);
    });

    info!(
        "{:?}: writing parameter to file: {:?}",
        replica_path, in_path
    );
    serde_json::to_writer(infile, &data).with_context(|| "cannot sealize to infile".to_string())?;

    info!("start p2 with program: {:?}", p2);
    let mut p2_process = process::Command::new(&p2)
        .arg(&uuid)
        .arg(u64::from(porep_config.sector_size).to_string())
        .arg(gpu_index.to_string())
        .spawn()
        .with_context(|| format!("{:?}, cannot start {:?} ", replica_path, p2))?;

    let status = p2_process.wait().expect("p2 is not running");
    match status.code() {
        Some(0) => {
            info!("{:?} p2 finished", replica_path);
        }
        Some(n) => {
            info!("{:?} p2 failed with exit number: {}", replica_path, n);
            bail!("{:?} p2 failed with exit number: {}", replica_path, n);
        }
        None => {
            info!("{:?} p2 crashed", replica_path);
            bail!("{:?} p2 crashed", replica_path);
        }
    }

    let mut comm_r = [0u8; 32];
    let mut comm_d = [0u8; 32];
    let mut output = File::open(&out_path)
        .with_context(|| format!("{:?}: cannot open file to fetch output", replica_path))?;

    output
        .read_exact(&mut comm_r)
        .with_context(|| format!("{:?}, cannot read file to get comm_r", replica_path))?;

    output
        .read_exact(&mut comm_d)
        .with_context(|| format!("{:?}, cannot read file to get comm_d", replica_path))?;

    Ok(SealPreCommitOutput { comm_r, comm_d })
}

fn get_uuid() -> String {
    let mut buffer = Uuid::encode_buffer();

    let uuid = Uuid::new_v4().to_hyphenated().encode_upper(&mut buffer);
    uuid.to_owned()
}
fn get_param_folder() -> PathBuf {
    Path::new(&std::env::var("WORKER_PATH").unwrap_or_else(|_| ".".to_string())).join("param")
}

use rust_gpu_tools::opencl;
lazy_static::lazy_static! {
    pub static ref GPU_NVIDIA_DEVICES_QUEUE:  Mutex<VecDeque<usize>> = Mutex::new((0..opencl::Device::all().len()).collect());
}

pub fn select_gpu_device() -> Option<usize> {
    if opencl::Device::all().len() == 0 {
        Some(0)
    } else {
        GPU_NVIDIA_DEVICES_QUEUE.lock().unwrap().pop_front()
    }
}
pub fn release_gpu_device(gpu_index: usize) {
    if opencl::Device::all().len() > 0 {
        GPU_NVIDIA_DEVICES_QUEUE
            .lock()
            .unwrap()
            .push_back(gpu_index)
    }
}
