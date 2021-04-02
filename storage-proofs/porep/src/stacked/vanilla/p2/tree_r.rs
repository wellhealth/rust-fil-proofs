use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::Path,
};

use anyhow::{Context, Result};
use merkletree::merkle::Element;
use storage_proofs_core::hasher::PoseidonArity;
use storage_proofs_core::merkle::MerkleTreeTrait;
use storage_proofs_core::{hasher::Hasher, util::NODE_SIZE};

pub fn tree_r_last_cpu<TreeArity, Tree>(
    tree_count: usize,
    last_label: &Path,
    label_size: usize,
) -> Result<()>
where
    TreeArity: PoseidonArity,
    Tree: MerkleTreeTrait,
{
    let single_size = label_size / tree_count;
    let mut layer_last =
        File::open(last_label).with_context(|| format!("cannot open file: {:?}", last_label))?;
    let parent = last_label
        .parent()
        .with_context(|| format!("last_label: [{:?}] doesn't have parent", last_label))?;

    for it in 0..tree_count {
        let mut buf = vec![0u8; single_size * NODE_SIZE];
        layer_last.read_exact(&mut buf)?;
        let buf = buf
            .chunks(NODE_SIZE)
            .map(<<<Tree as MerkleTreeTrait>::Hasher as Hasher>::Domain>::from_slice)
            .collect::<Vec<_>>();

        let tree = make_merkle(
            buf,
            TreeArity::to_usize(),
            &parent.join(format!("sc-02-data-{}", it)),
        );

        std::fs::write(parent.join(format!("tree-r-{}", it)), &tree)
            .with_context(|| format!("cannot write tree-r-{}", it))?;
    }

    Ok(())
}

fn append<P, Tree, E>(path: P, buf: &[E]) -> std::io::Result<()>
where
    Tree: MerkleTreeTrait,
    P: AsRef<Path>,
    E: Element,
{
    let bytes = buf.iter().fold(vec![], |mut v, x| {
        v.extend_from_slice(x.as_ref());
        v
    });

    OpenOptions::new()
        .append(true)
        .write(true)
        .open(path)?
        .write_all(&bytes)
}

fn make_merkle<E>(buf: Vec<E>, branch: usize, path: &Path) -> Vec<u8>
where
    E: Element,
{
    todo!()
}
