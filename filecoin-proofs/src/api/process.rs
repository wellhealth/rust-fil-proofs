use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use log::info;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Read;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::process;
use storage_proofs::sector::SectorId;
use storage_proofs::settings;

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
    let cache_path = cache_path.as_ref().to_owned();
    let replica_path = replica_path.as_ref().to_owned();
    let param_folder = &settings::SETTINGS.param_folder;
    let program_folder = &settings::SETTINGS.program_folder;

    let data = P2Param {
        porep_config,
        phase1_output,
        cache_path,
        replica_path: replica_path.clone(),
    };

    let p2_param = Path::new(param_folder).join("p2-param");
    let p2 = Path::new(program_folder).join("p2");

    let infile = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(p2_param)
        .with_context(|| format!("cannot open file to pass p2 parameter"))?;

    serde_json::to_writer(infile, &data).with_context(|| format!("cannot sealize to infile"))?;

    let mut p2_process = process::Command::new(&p2)
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
    let p2_output = Path::new(param_folder).join("p2-output");

    let mut comm_r = [0u8; 32];
    let mut comm_d = [0u8; 32];
    let mut output = File::open(p2_output)
        .with_context(|| format!("{:?}: cannot open file to fetch output", replica_path))?;

    output
        .read_exact(&mut comm_r)
        .with_context(|| format!("{:?}, cannot read file to get comm_r", replica_path))?;

    output
        .read_exact(&mut comm_d)
        .with_context(|| format!("{:?}, cannot read file to get comm_d", replica_path))?;

    Ok(SealPreCommitOutput { comm_r, comm_d })
}

pub fn p2_sub<Tree: 'static + MerkleTreeTrait>() -> Result<()> {
    let param_folder = &settings::SETTINGS.param_folder;

    let p2_param = Path::new(param_folder).join("p2-param");

    let infile =
        File::open(&p2_param).with_context(|| format!("cannot open file {:?}", p2_param))?;

    let data = serde_json::from_reader::<_, P2Param<Tree>>(infile)
        .context("failed to deserialize p2 params")?;

    let P2Param {
        porep_config,
        phase1_output,
        cache_path,
        replica_path,
    } = data;
    let out = super::seal_pre_commit_phase2(
        porep_config,
        phase1_output,
        cache_path,
        replica_path.clone(),
    )?;

    let p2_output = Path::new(param_folder).join("p2-output");

    let mut out_file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(&p2_output)
        .with_context(|| {
            format!(
                "{:?}: cannot open file: {:?} for output",
                replica_path, p2_output
            )
        })?;

    out_file.write_all(&out.comm_r).with_context(|| {
        format!(
            "{:?} cannot write comm_r to file: {:?}",
            replica_path, p2_output
        )
    })?;
    out_file.write_all(&out.comm_d).with_context(|| {
        format!(
            "{:?} cannot write comm_d to file: {:?}",
            replica_path, p2_output
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
    let param_folder = &settings::SETTINGS.param_folder;
    let c2_param = Path::new(param_folder).join("c2-param");
    let program_folder = &settings::SETTINGS.program_folder;
    let c2_program_path = Path::new(program_folder).join("c2");

    let infile = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(c2_param)
        .with_context(|| format!("cannot open file to pass c2 parameter"))?;
    serde_json::to_writer(infile, &data).with_context(|| format!("cannot sealize to infile"))?;

    let mut c2_process = process::Command::new(&c2_program_path)
        .spawn()
        .with_context(|| format!("{:?}, cannot start {:?} ", sector_id, c2_program_path))?;

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

    let c2_output = Path::new(param_folder).join("c2-output");
    let mut proof = vec![];
    let mut outfile = File::open(c2_output)
        .with_context(|| format!("{:?}, cannot open c2 output file for reuslt", sector_id))?;
    outfile
        .read_to_end(&mut proof)
        .with_context(|| format!("{:?}, cannot read from c2 output file", sector_id))?;
    Ok(SealCommitOutput { proof })
}
