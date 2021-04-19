#![allow(dead_code)]
use anyhow::Context;
use anyhow::Result;
use bellperson::bls::Fr;
use ff::PrimeField;
use ff::PrimeFieldRepr;
use filecoin_hashers::PoseidonArity;
use log::info;
use std::path::Path;

use neptune::{
    batch_hasher::{BatcherType, GPUSelector},
    tree_builder::{TreeBuilder, TreeBuilderTrait},
};

fn persist_tree_r(index: usize, path: &Path, tree: &[Fr], replica_path: &Path) -> Result<()> {
    use std::io::Cursor;
    let mut cursor = Cursor::new(Vec::<u8>::with_capacity(
        (tree.len()) * std::mem::size_of::<Fr>(),
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

fn create_tree<TreeArity>(
    data: &[Fr],
    tree_batch_size: usize,
    node_count: usize,
    replica_path: &Path,
    gpu_index: usize,
) -> Result<Vec<Fr>>
where
    TreeArity: PoseidonArity,
{
    let mut builder = TreeBuilder::<TreeArity>::new(
        Some(BatcherType::CustomGPU(GPUSelector::Index(gpu_index))),
        node_count,
        tree_batch_size,
        0,
    )
    .with_context(|| format!("{:?}: cannot create tree_builder", replica_path))?;

    Ok(builder.add_final_leaves(&data)?.1)
}
