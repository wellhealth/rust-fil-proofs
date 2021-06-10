use anyhow::{bail, Context, Result};
use log::info;
use std::{
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
    process,
};
use uuid::Uuid;


use scopeguard::defer;
use storage_proofs_core::{merkle::MerkleTreeTrait, sector::SectorId, settings};

use crate::{
    api::cores::get_l3_index, api::cores::serialize, PieceInfo, PoRepConfig, ProverId,
    SealPreCommitPhase1Output, Ticket,
};
use serde::{Deserialize, Serialize};

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

fn get_uuid() -> String {
    let mut buffer = Uuid::encode_buffer();
	
    let uuid = Uuid::new_v4().to_hyphenated().encode_upper(&mut buffer);
    uuid.to_owned()
}
fn get_param_folder() -> PathBuf {
    Path::new(&std::env::var("WORKER_PATH").unwrap_or_else(|_| ".".to_string())).join("param")
}
