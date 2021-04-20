use anyhow::Context;
use anyhow::Result;
use bellperson::bls::Fr;
use ff::Field;
use ff::PrimeField;
use ff::PrimeFieldRepr;
use filecoin_hashers::PoseidonArity;
use log::info;
use neptune::batch_hasher::Batcher;
use neptune::{
    batch_hasher::{BatcherType::CustomGPU, GPUSelector},
    proteus::gpu::CLBatchHasher,
};
use rayon::prelude::*;
use std::fs::OpenOptions;
use std::io::Read;
use std::io::Write;
use std::path::Path;
use std::{fs::File, path::PathBuf};
use storage_proofs_core::settings::SETTINGS;

#[derive(Debug, Clone)]
pub struct TreeRConfig {
    pub node_count: usize,
    pub rows_to_discard: usize,
    pub paths: Vec<PathBuf>,
}

pub fn run<TreeArity>(
    config: &TreeRConfig,
    last_layer: &Path,
    unsealed: &Path,
    replica_path: &Path,
    gpu_index: usize,
) -> Result<()>
where
    TreeArity: PoseidonArity,
{
    info!(
        "{:?}: tree-r started with config: {:?}",
        replica_path, config
    );

    let mut batcher: CLBatchHasher<TreeArity> = match Batcher::<TreeArity>::new(
        &CustomGPU(GPUSelector::Index(gpu_index)),
        config.node_count,
    )
    .with_context(|| format!("failed to create tree_c batcher {}", gpu_index))?
    {
        Batcher::OpenCL(x) => x,
        Batcher::CPU(_) => panic!("neptune bug, batcher should be OPENCL"),
    };
    info!("{:?}: batcher created", replica_path,);

    let batch_size = SETTINGS.max_gpu_column_batch_size as usize;

    let unsealed_file = File::open(unsealed)
        .with_context(|| format!("cannot open unsealed file: {:?}", unsealed))?;
    let last_layer_file = File::open(last_layer)
        .with_context(|| format!("cannot open last layer file for tree-r {:?}", last_layer))?;

    let mut disk_data = [unsealed_file, last_layer_file];
    let mut sealed_file = OpenOptions::new()
        .truncate(true)
        .write(true)
        .create(true)
        .open(replica_path)
        .with_context(|| format!("cannot open sealed file: {:?}", replica_path))?;

    for (index, path) in config.paths.iter().enumerate() {
        info!(
            "{:?}: start collecting tree-r-last:{}",
            replica_path,
            index + 1
        );
        let mut tree_r_base = Vec::with_capacity(config.node_count);

        for node_index in (0..config.node_count).step_by(batch_size) {
            let rest_count = config.node_count - node_index;
            let chunked_node_count = std::cmp::min(rest_count, batch_size);
            let mut v = prepare_batch(&mut disk_data, chunked_node_count)?;
            tree_r_base.append(&mut v);
        }
        persist_sealed(&tree_r_base, &mut sealed_file)
            .with_context(|| format!("cannot save to sealed file: {:?}", sealed_file))?;
        info!(
            "{:?} sealed file done for tree-r-{}",
            replica_path,
            index + 1
        );

        let tree_r = super::build_tree(&mut batcher, &tree_r_base, config.rows_to_discard)?;
        persist_tree_r(index, path, &tree_r, replica_path)?;
    }

    Ok(())
}

fn prepare_batch(disk_data: &mut [File; 2], node_count: usize) -> Result<Vec<Fr>> {
    const FR_SIZE: usize = std::mem::size_of::<Fr>();
    let byte_count = node_count * FR_SIZE;

    let mut buf_bytes = [vec![0u8; byte_count], vec![0u8; byte_count]];
    let data = disk_data
        .par_iter_mut()
        .zip(buf_bytes.par_iter_mut())
        .map(|(f, b)| {
            f.read_exact(b)
                .with_context(|| format!("cannot read file for {:?}", f))?;

            b.chunks(FR_SIZE)
                .map(super::bytes_into_fr)
                .collect::<Result<Vec<_>>>()
                .with_context(|| format!("cannot convert bytes to Fr for file {:?}", f))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut it = data.into_iter();
    let mut first = it.next().expect("array length problem");
    let second = it.next().expect("array length problem");

    first
        .par_iter_mut()
        .zip(second.par_iter())
        .for_each(|(fr1, fr2)| {
            fr1.add_assign(fr2);
        });
    Ok(first)
}

fn persist_sealed(data: &[Fr], sealed: &mut File) -> Result<()> {
    use std::io::Cursor;
    let mut cursor = Cursor::new(Vec::<u8>::with_capacity(
        data.len() * std::mem::size_of::<Fr>(),
    ));

    for fr in data.iter().map(|x| x.into_repr()) {
        fr.write_le(&mut cursor)
            .with_context(|| format!("cannot write to cursor {:?}", sealed))?;
    }

    sealed
        .write_all(&cursor.into_inner())
        .with_context(|| format!("cannot write {:?} to file", sealed))?;

    Ok(())
}

fn persist_tree_r(index: usize, path: &Path, tree: &[Fr], replica_path: &Path) -> Result<()> {
    use std::io::Cursor;
    let mut cursor = Cursor::new(Vec::<u8>::with_capacity(
        tree.len() * std::mem::size_of::<Fr>(),
    ));
    info!(
        "{:?}.persisting, done: cursor created for tree-c {}",
        replica_path,
        index + 1
    );

    for fr in tree.iter().map(|x| x.into_repr()) {
        fr.write_le(&mut cursor)
            .with_context(|| format!("cannot write to cursor {:?}", path))?;
    }
    info!(
        "{:?}.persisting, done: put data into cursor for tree-c {}",
        replica_path,
        index + 1
    );

    std::fs::write(&path, cursor.into_inner())
        .with_context(|| format!("cannot write {:?} to file", path))?;

    info!(
        "{:?}.persisting, done: file written for tree-r {}",
        replica_path,
        index + 1
    );

    Ok(())
}
