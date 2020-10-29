use ff::{PrimeField, PrimeFieldRepr};
use neptune::batch_hasher::Batcher;
use neptune::batch_hasher::BatcherType;
use std::fs::OpenOptions;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use generic_array::GenericArray;
use neptune::gpu::GPUBatchHasher;
use neptune::Arity;
use neptune::BatchHasher;
use paired::bls12_381::Fr;
use storage_proofs_core::hasher::PoseidonArity;

use super::hash::hash_single_column;

// Batcher<ColumnArity>

pub fn receive_and_generate_tree_c<ColumnArity, TreeArity, P>(
    nodes_count: usize,
    gpu_index: usize,
) -> Result<()>
where
    ColumnArity: 'static + PoseidonArity,
    TreeArity: PoseidonArity,
    P: AsRef<Path>,
{
    let batcher = match Batcher::<ColumnArity>::new(&BatcherType::GPU, nodes_count, gpu_index)
        .with_context(|| format!("failed to create tree_c batcher {}", gpu_index))?
    {
        Batcher::GPU(x) => x,
        _ => panic!(),
    };

    Ok(())
}

pub fn gpu_build_column<A: PoseidonArity>(
    batcher: &mut GPUBatchHasher<A>,
    data: &[GenericArray<Fr, A>],
) -> Result<Vec<Fr>> {
    batcher.hash(data).context("batcher.hash failed")
}

pub fn cpu_build_column<A: PoseidonArity>(data: &[GenericArray<Fr, A>]) -> Vec<Fr> {
    data.iter().map(|x| hash_single_column(x)).collect()
}

pub fn build_tree_gpu<TreeArity>(
    mut tree_data: Vec<Fr>,
    batcher: &mut GPUBatchHasher<TreeArity>,
) -> Result<(Vec<Fr>, Vec<Fr>)>
where
    TreeArity: PoseidonArity,
{
    let leaf_count = tree_data.len();
    let final_tree_size = tree_size::<TreeArity>(leaf_count);
    let intermediate_tree_size = final_tree_size + leaf_count;
    let arity = TreeArity::to_usize();
    let max_batch_size = batcher.max_batch_size();

    let (mut row_start, mut row_end) = (0, leaf_count);
    while row_end < intermediate_tree_size {
        let row_size = row_end - row_start;
        assert_eq!(0, row_size % arity);
        let new_row_size = row_size / arity;
        let (new_row_start, new_row_end) = (row_end, row_end + new_row_size);

        let mut total_hashed = 0;
        let mut batch_start = row_start;
        while total_hashed < new_row_size {
            let batch_end = usize::min(batch_start + (max_batch_size * arity), row_end);
            let batch_size = (batch_end - batch_start) / arity;
            let preimages = as_generic_arrays::<TreeArity>(&tree_data[batch_start..batch_end]);
            let hashed = batcher
                .hash(&preimages)
                .context("cannot hash to generate merkle tree")?;

            #[allow(clippy::drop_ref)]
            drop(preimages); // make sure we don't reference tree_data anymore
            tree_data[new_row_start + total_hashed..new_row_start + total_hashed + hashed.len()]
                .copy_from_slice(&hashed);
            total_hashed += batch_size;
            batch_start = batch_end;
        }

        row_start = new_row_start;
        row_end = new_row_end;
    }
    let base_row = tree_data[..leaf_count].to_vec();
    let tree_to_keep = tree_data[tree_data.len() - final_tree_size..].to_vec();
    Ok((base_row, tree_to_keep))
}

pub fn persist_tree_c<P>(path: P, base: &[Fr], tree: &[Fr]) -> Result<()>
where
    P: AsRef<Path>,
{
    let mut tree_c = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(&path)
        .with_context(|| format!("trying to open file {:?}, but failed", path.as_ref()))?;

    for fr in base.iter().chain(tree.iter()) {
        fr.into_repr()
            .write_le(&mut tree_c)
            .with_context(|| format!("trying to write to file {:?}, but failed", path.as_ref()))?;
    }
    Ok(())
}

pub fn tree_size<TreeArity>(leaf_count: usize) -> usize
where
    TreeArity: PoseidonArity,
{
    let arity = TreeArity::to_usize();

    let mut tree_size = 0;
    let mut current_row_size = leaf_count;

    // Exclude the base row, along with the rows to be discarded.
    let mut remaining_rows_to_exclude = 1;

    while current_row_size >= 1 {
        if remaining_rows_to_exclude > 0 {
            remaining_rows_to_exclude -= 1;
        } else {
            tree_size += current_row_size;
        }
        if current_row_size != 1 {
            assert_eq!(
                0,
                current_row_size % arity,
                "Tree leaf count {} is not a power of arity {}.",
                leaf_count,
                arity
            )
        }
        current_row_size /= arity;
    }

    tree_size
}

fn as_generic_arrays<'a, A: Arity<Fr>>(vec: &'a [Fr]) -> &'a [GenericArray<Fr, A>] {
    // It is a programmer error to call `as_generic_arrays` on a vector whose underlying data cannot be divided
    // into an even number of `GenericArray<Fr, Arity>`.
    assert_eq!(
        0,
        (vec.len() * std::mem::size_of::<Fr>()) % std::mem::size_of::<GenericArray<Fr, A>>()
    );

    // This block does not affect the underlying `Fr`s. It just groups them into `GenericArray`s of length `Arity`.
    // We know by the assertion above that `vec` can be evenly divided into these units.
    unsafe {
        std::slice::from_raw_parts(
            vec.as_ptr() as *const () as *const GenericArray<Fr, A>,
            vec.len() / A::to_usize(),
        )
    }
}
