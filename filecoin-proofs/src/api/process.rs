use crate::{select_gpu_device, ChallengeSeed, PoStConfig, PrivateReplicaInfo};
use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use log::info;
use scopeguard::defer;
use std::fs::OpenOptions;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;
use std::process;
use std::{collections::BTreeMap, fs::File};
use storage_proofs::sector::SectorId;
use storage_proofs::settings;
use uuid::Uuid;

use crate::types::PoRepConfig;
use crate::ProverId;
use crate::SealCommitOutput;
use crate::SealCommitPhase1Output;
use crate::SealPreCommitOutput;
use crate::SealPreCommitPhase1Output;
use serde::{Deserialize, Serialize};
use storage_proofs::merkle::MerkleTreeTrait;

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
struct C2Param<Tree: 'static + MerkleTreeTrait> {
    porep_config: PoRepConfig,
    #[serde(bound(
        serialize = "SealCommitPhase1Output<Tree>: Serialize",
        deserialize = "SealCommitPhase1Output<Tree>: Deserialize<'de>"
    ))]
    phase1_output: SealCommitPhase1Output<Tree>,
    prover_id: ProverId,
    sector_id: SectorId,
}

#[derive(Serialize, Deserialize)]
struct WindowPostParam<Tree: 'static + MerkleTreeTrait> {
    post_config: PoStConfig,
    randomness: ChallengeSeed,
    #[serde(bound(
        serialize = "SealPreCommitPhase1Output<Tree>: Serialize",
        deserialize = "SealPreCommitPhase1Output<Tree>: Deserialize<'de>"
    ))]
    replicas: BTreeMap<SectorId, PrivateReplicaInfo<Tree>>,
    prover_id: ProverId,
    gpu_index: usize,
}

fn get_uuid() -> String {
    let mut buffer = Uuid::encode_buffer();
    let uuid = Uuid::new_v4().to_hyphenated().encode_upper(&mut buffer);
    uuid.to_owned()
}

fn get_param_folder() -> Option<PathBuf> {
    Some(Path::new(&std::env::var("WORKER_PATH").ok()?).join("param"))
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
        super::release_gpu_device(gpu_index);
    };
    info!(
        "{:?}: selected gpu device: {}",
        replica_path.as_ref(),
        gpu_index
    );
    std::env::set_var(
        crate::api::seal::SHENSUANYUN_GPU_INDEX,
        &gpu_index.to_string(),
    );

    info!("{:?}: set_var finished", replica_path.as_ref());
    let cache_path = cache_path.as_ref().to_owned();
    let replica_path = replica_path.as_ref().to_owned();
    let param_folder = get_param_folder().context("cannot get param folder")?;
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
    defer!({
        let _ = std::fs::remove_file(&out_path);
    });
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

pub fn c2<Tree: 'static + MerkleTreeTrait>(
    porep_config: PoRepConfig,
    phase1_output: SealCommitPhase1Output<Tree>,
    prover_id: ProverId,
    sector_id: SectorId,
) -> Result<SealCommitOutput> {
    let data = C2Param {
        porep_config,
        phase1_output,
        prover_id,
        sector_id,
    };
    let uuid = get_uuid();

    let param_folder = get_param_folder().context("cannot get param folder")?;
    std::fs::create_dir_all(&param_folder)
        .with_context(|| format!("cannot create dir: {:?}", param_folder))?;
    let in_path = Path::new(&param_folder).join(&uuid);
    let out_path = Path::new(&param_folder).join(&uuid);

    let program_folder = &settings::SETTINGS.program_folder;
    let c2_program_path = Path::new(program_folder).join(&settings::SETTINGS.c2_program_name);

    let infile = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(&in_path)
        .with_context(|| format!("{:?}: cannot open file to pass in c2 parameter", sector_id))?;

    info!("{:?}: writing parameter to file: {:?}", sector_id, in_path);
    serde_json::to_writer(infile, &data)
        .with_context(|| format!("{:?}: cannot sealize to infile", sector_id))?;

    let gpu_index = select_gpu_device()
        .with_context(|| format!("{:?}: cannot select gpu device", sector_id))?;
    defer! {
        info!("release gpu index: {}", gpu_index);
        super::release_gpu_device(gpu_index);
    };
    info!(
        "{:?}: start c2 with program: {:?}",
        sector_id, c2_program_path
    );
    let mut c2_process = process::Command::new(&c2_program_path)
        .arg(&uuid)
        .arg(u64::from(porep_config.sector_size).to_string())
        .arg(gpu_index.to_string())
        .arg(sector_id.0.to_string())
        .spawn()
        .with_context(|| {
            format!(
                "{:?}, cannot start program {:?} ",
                sector_id, c2_program_path
            )
        })?;

    let status = c2_process.wait().expect("c2 is not running");

    defer!({
        let _ = std::fs::remove_file(&out_path);
    });

    match status.code() {
        Some(0) => {
            info!("{:?} c2 finished", sector_id);
        }
        Some(n) => {
            info!("{:?} c2 failed with exit number: {}", sector_id, n);
            bail!("{:?} c2 failed with exit number: {}", sector_id, n);
        }
        None => {
            info!("{:?} c2 crashed", sector_id);
            bail!("{:?} c2 crashed", sector_id);
        }
    }

    let proof = std::fs::read(&out_path)
        .with_context(|| format!("{:?}, cannot open c2 output file for reuslt", sector_id))?;

    Ok(SealCommitOutput { proof })
}

pub fn window_post<Tree: 'static + MerkleTreeTrait>(
    post_config: &PoStConfig,
    randomness: &ChallengeSeed,
    replicas: &BTreeMap<SectorId, PrivateReplicaInfo<Tree>>,
    prover_id: ProverId,
    gpu_index: usize,
) -> Result<SealCommitOutput> {
    let data = {
        let post_config = post_config.clone();
        let randomness = *randomness;
        let replicas = replicas.clone();
        WindowPostParam {
            post_config,
            randomness,
            replicas,
            prover_id,
            gpu_index,
        }
    };
    let uuid = get_uuid();

    let param_folder = get_param_folder().context("cannot get param folder")?;
    std::fs::create_dir_all(&param_folder)
        .with_context(|| format!("cannot create dir: {:?}", param_folder))?;
    let in_path = Path::new(&param_folder).join(&uuid);
    let out_path = Path::new(&param_folder).join(&uuid);

    let program_folder = &settings::SETTINGS.program_folder;
    let program_path = Path::new(program_folder).join(&settings::SETTINGS.window_post_program_name);

    let infile = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(&in_path)
        .context("cannot open file to pass in window post parameter")?;

    info!("writing parameter to file: {:?}", in_path);
    serde_json::to_writer(infile, &data).context("cannot sealize to infile")?;

    defer! {
        info!("release gpu index: {}", gpu_index);
        super::release_gpu_device(gpu_index);
    };
    info!("start c2 with program: {:?}", program_path);
    let mut process = process::Command::new(&program_path)
        .arg(&uuid)
        .arg(u64::from(post_config.sector_size).to_string())
        .arg(gpu_index.to_string())
        .spawn()
        .with_context(|| format!("cannot start program {:?} ", program_path))?;

    let status = process.wait().expect("c2 is not running");

    defer!({
        let _ = std::fs::remove_file(&out_path);
    });

    match status.code() {
        Some(0) => {
            info!(" window post finished");
        }
        Some(n) => {
            info!("window post failed with exit number: {}", n);
            bail!("window post failed with exit number: {}", n);
        }
        None => {
            info!("window post crashed");
            bail!("window post crashed");
        }
    }

    let proof = std::fs::read(&out_path)
        .with_context(|| format!("cannot open c2 output file for reuslt"))?;

    Ok(SealCommitOutput { proof })
}
