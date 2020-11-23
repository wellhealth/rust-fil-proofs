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
use storage_proofs::settings;

use crate::types::PoRepConfig;
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
