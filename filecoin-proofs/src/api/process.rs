use crate::constants::*;
use crate::select_gpu_device;
use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use log::info;
use scopeguard::defer;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Read;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::process;
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
    std::env::set_var(
        crate::api::seal::SHENSUANYUN_GPU_INDEX,
        &gpu_index.to_string(),
    );

    let cache_path = cache_path.as_ref().to_owned();
    let replica_path = replica_path.as_ref().to_owned();
    let param_folder = get_param_folder().context("cannot get param folder")?;
    std::fs::create_dir_all(&param_folder)
        .with_context(|| format!("cannot create dir: {:?}", param_folder))?;
    let program_folder = &settings::SETTINGS.program_folder;

    let data = P2Param {
        porep_config,
        phase1_output,
        cache_path,
        replica_path: replica_path.clone(),
    };
    let uuid = get_uuid();

    let in_path = Path::new(&param_folder).join(&uuid);
    let p2 = Path::new(program_folder).join("lotus-p2");
    let out_path = Path::new(&param_folder).join(&uuid);

    let infile = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(&in_path)
        .with_context(|| format!("cannot open file to pass p2 parameter"))?;

    info!(
        "{:?}: writing parameter to file: {:?}",
        replica_path, in_path
    );
    serde_json::to_writer(infile, &data).with_context(|| format!("cannot sealize to infile"))?;

    info!("start p2 with program: {:?}", p2);
    let mut p2_process = process::Command::new(&p2)
        .arg(&uuid)
        .arg(u64::from(porep_config.sector_size).to_string())
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

    drop(&output);
    let _ = std::fs::remove_file(out_path);
    Ok(SealPreCommitOutput { comm_r, comm_d })
}

pub fn c2_sub_launcher() -> Result<()> {
    let mut args = std::env::args().skip(1).take(2);
    let uuid = args.next().context("cannot get uuid parameter")?;
    let shape = args
        .next()
        .context("cannot get shape parameter")?
        .parse()
        .context("cannot parse shape")?;

    match shape {
        SECTOR_SIZE_2_KIB => c2_sub::<SectorShape2KiB>(&uuid),
        SECTOR_SIZE_4_KIB => c2_sub::<SectorShape4KiB>(&uuid),
        SECTOR_SIZE_16_KIB => c2_sub::<SectorShape16KiB>(&uuid),
        SECTOR_SIZE_32_KIB => c2_sub::<SectorShape32KiB>(&uuid),
        SECTOR_SIZE_8_MIB => c2_sub::<SectorShape8MiB>(&uuid),
        SECTOR_SIZE_16_MIB => c2_sub::<SectorShape16MiB>(&uuid),
        SECTOR_SIZE_512_MIB => c2_sub::<SectorShape512MiB>(&uuid),
        SECTOR_SIZE_1_GIB => c2_sub::<SectorShape1GiB>(&uuid),
        SECTOR_SIZE_32_GIB => c2_sub::<SectorShape32GiB>(&uuid),
        SECTOR_SIZE_64_GIB => c2_sub::<SectorShape64GiB>(&uuid),
        _ => bail!("shape not recognized"),
    }
}

pub fn p2_sub_launcher() -> Result<()> {
    let mut args = std::env::args().skip(1).take(2);
    let uuid = args.next().context("cannot get uuid parameter")?;
    let shape = args
        .next()
        .context("cannot get shape parameter")?
        .parse()
        .context("cannot parse shape")?;

    match shape {
        SECTOR_SIZE_2_KIB => p2_sub::<SectorShape2KiB>(&uuid),
        SECTOR_SIZE_4_KIB => p2_sub::<SectorShape4KiB>(&uuid),
        SECTOR_SIZE_16_KIB => p2_sub::<SectorShape16KiB>(&uuid),
        SECTOR_SIZE_32_KIB => p2_sub::<SectorShape32KiB>(&uuid),
        SECTOR_SIZE_8_MIB => p2_sub::<SectorShape8MiB>(&uuid),
        SECTOR_SIZE_16_MIB => p2_sub::<SectorShape16MiB>(&uuid),
        SECTOR_SIZE_512_MIB => p2_sub::<SectorShape512MiB>(&uuid),
        SECTOR_SIZE_1_GIB => p2_sub::<SectorShape1GiB>(&uuid),
        SECTOR_SIZE_32_GIB => p2_sub::<SectorShape32GiB>(&uuid),
        SECTOR_SIZE_64_GIB => p2_sub::<SectorShape64GiB>(&uuid),
        _ => bail!("shape not recognized"),
    }
}

pub fn p2_sub<Tree: 'static + MerkleTreeTrait>(uuid: &str) -> Result<()> {
    let param_folder = get_param_folder().context("cannot get param folder")?;
    let in_path = Path::new(&param_folder).join(uuid);
    let out_path = Path::new(&param_folder).join(uuid);

    let infile = File::open(&in_path).with_context(|| format!("cannot open file {:?}", in_path))?;

    let data = serde_json::from_reader::<_, P2Param<Tree>>(infile)
        .context("failed to deserialize p2 params")?;

    let P2Param {
        porep_config,
        phase1_output,
        cache_path,
        replica_path,
    } = data;

    let out = super::official_p2(
        porep_config,
        phase1_output,
        cache_path,
        replica_path.clone(),
    )?;

    let mut out_file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(&out_path)
        .with_context(|| {
            format!(
                "{:?}: cannot open file: {:?} for output",
                replica_path, out_path
            )
        })?;

    out_file.write_all(&out.comm_r).with_context(|| {
        format!(
            "{:?} cannot write comm_r to file: {:?}",
            replica_path, out_path
        )
    })?;

    out_file.write_all(&out.comm_d).with_context(|| {
        format!(
            "{:?} cannot write comm_d to file: {:?}",
            replica_path, out_path
        )
    })?;
    Ok(())
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
    let gpu_index = select_gpu_device()
        .with_context(|| format!("{:?}: cannot select gpu device", sector_id))?;
    defer! {
        info!("release gpu index: {}", gpu_index);
        super::release_gpu_device(gpu_index);
    };
    std::env::set_var(
        crate::api::seal::SHENSUANYUN_GPU_INDEX,
        &gpu_index.to_string(),
    );
    let uuid = get_uuid();

    let param_folder = get_param_folder().context("cannot get param folder")?;
    std::fs::create_dir_all(&param_folder)
        .with_context(|| format!("cannot create dir: {:?}", param_folder))?;
    let in_path = Path::new(&param_folder).join(&uuid);
    let out_path = Path::new(&param_folder).join(&uuid);

    let program_folder = &settings::SETTINGS.program_folder;
    let c2_program_path = Path::new(program_folder).join("lotus-c2");

    let infile = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(&in_path)
        .with_context(|| format!("{:?}: cannot open file to pass in c2 parameter", sector_id))?;

    info!("{:?}: writing parameter to file: {:?}", sector_id, in_path);
    serde_json::to_writer(infile, &data)
        .with_context(|| format!("{:?}: cannot sealize to infile", sector_id))?;

    info!("start c2 with program: {:?}", c2_program_path);
    let mut c2_process = process::Command::new(&c2_program_path)
        .arg(&uuid)
        .arg(u64::from(porep_config.sector_size).to_string())
        .spawn()
        .with_context(|| {
            format!(
                "{:?}, cannot start program {:?} ",
                sector_id, c2_program_path
            )
        })?;

    let status = c2_process.wait().expect("c2 is not running");
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

    let _ = std::fs::remove_file(out_path);
    Ok(SealCommitOutput { proof })
}

pub fn c2_sub<Tree: 'static + MerkleTreeTrait>(uuid: &str) -> Result<()> {
    let param_folder = get_param_folder().context("cannot get param_folder")?;
    let in_path = Path::new(&param_folder).join(&uuid);
    let out_path = Path::new(&param_folder).join(&uuid);

    let infile = File::open(&in_path).with_context(|| format!("cannot open file {:?}", in_path))?;

    let data = serde_json::from_reader::<_, C2Param<Tree>>(infile)
        .context("failed to deserialize p2 params")?;

    let C2Param {
        porep_config,
        phase1_output,
        prover_id,
        sector_id,
    } = data;

    let out = super::custom::c2::whole(porep_config, phase1_output, prover_id, sector_id)?;

    std::fs::write(out_path, &out.proof)
        .with_context(|| format!("{:?}: cannot write result to file", sector_id))?;

    Ok(())
}
