pub mod tree_c;
pub mod tree_r;
use anyhow::Context;
use anyhow::Result;
use filecoin_hashers::PoseidonArity;
use generic_array::GenericArray;
use lazy_static::lazy_static;
use log::error;
use neptune::{Arity, BatchHasher};
use rayon::prelude::*;
use std::fmt::Debug;
use std::{borrow::Cow, io::Write};

use bellperson::bls::Fr;
use ff::PrimeFieldRepr;
use ff::{Field, PrimeField};

lazy_static! {
    pub static ref GPU_INDEX: usize = select_gpu_device();
    pub static ref POOL: rayon::ThreadPool = {
        rayon::ThreadPoolBuilder::new()
            .num_threads(16)
            .build()
            .expect("failed to build rayon ThreadPool")
    };
}

pub fn select_gpu_device() -> usize {
    std::env::args()
        .nth(3)
        .and_then(|x| x.parse().ok())
        .unwrap_or(0)
}

const fn calculate_tree_size(leaf_count: usize, arity: usize, rows_to_be_discard: usize) -> usize
where
{
    let mut tree_size = 0;
    let mut current_row_size = leaf_count;

    // Exclude the base row, along with the rows to be discarded.
    let mut remaining_rows_to_exclude = 1 + rows_to_be_discard;

    while current_row_size >= 1 {
        if remaining_rows_to_exclude > 0 {
            remaining_rows_to_exclude -= 1;
        } else {
            tree_size += current_row_size;
        }
        current_row_size /= arity;
    }

    tree_size
}

fn persist_frs<W>(data: &[Fr], file: &mut W) -> Result<()>
where
    W: Write + Debug,
{
    const FR_SIZE: usize = std::mem::size_of::<Fr>();
    let mut fr_bytes = vec![0u8; data.len() * FR_SIZE];

    fr_bytes
        .par_chunks_mut(FR_SIZE)
        .zip(data.par_iter())
        .for_each(|(bytes, fr)| {
            fr.into_repr()
                .write_le(bytes)
                .expect("cannot write_le ,persist_frs");
        });

    file.write_all(&fr_bytes)
        .with_context(|| format!("cannot write to file: {:?}", file))?;

    Ok(())
}

fn hash_batches<B, Arity>(
    batcher: &mut B,
    batch_size: usize,
    preimages: &[GenericArray<Fr, Arity>],
    result: &mut [Fr],
) where
    B: BatchHasher<Arity>,
    Arity: PoseidonArity,
{
    for (p_chunk, r_chunk) in preimages
        .chunks(batch_size)
        .zip(result.chunks_mut(batch_size))
    {
        while let Err(e) = batcher.hash_into_slice(r_chunk, p_chunk) {
            error!("opencl error, {:?}", e);
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }
}

pub fn build_tree<TreeArity, B>(
    batcher: &mut B,
    data: &[Fr],
    rows_to_discard: usize,
) -> Result<Vec<Fr>>
where
    B: BatchHasher<TreeArity>,
    TreeArity: PoseidonArity,
{
    let tree_size = calculate_tree_size(data.len(), TreeArity::to_usize(), rows_to_discard);
    let arity = TreeArity::to_usize();
    let batch_size = {
        let tmp = data.len() / arity / arity;
        if tmp == 0 {
            1
        } else {
            tmp
        }
    };

    // calculate discarded rows
    let hashed = (0..rows_to_discard).fold(Cow::Borrowed(data), |x, _| {
        let preimages = as_generic_arrays::<TreeArity>(&x);
        let mut hashed = vec![Fr::zero(); preimages.len()];

        hash_batches(batcher, batch_size, preimages, &mut hashed);
        Cow::Owned(hashed)
    });

    let mut tree: Vec<Fr> = vec![Fr::zero(); tree_size];
    let mut rest = &mut tree[..];
    let mut origin = &hashed[..];
    while !rest.is_empty() {
        let layer_len = origin.len() / arity;
        let preimages = as_generic_arrays::<TreeArity>(&origin);
        let (to_be_hashed, unhashed) = rest.split_at_mut(layer_len);

        hash_batches(batcher, batch_size, preimages, to_be_hashed);

        origin = to_be_hashed;
        rest = unhashed;
    }

    Ok(tree)
}

fn as_generic_arrays<A: Arity<Fr>>(vec: &[Fr]) -> &[GenericArray<Fr, A>] {
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

fn bytes_into_fr(bytes: &[u8]) -> Result<Fr> {
    let mut fr_repr = <<Fr as PrimeField>::Repr as Default>::default();
    fr_repr
        .read_le(bytes)
        .context("cannot convert bytes to Fr")?;

    Fr::from_repr(fr_repr).context("cannot convert fr_repr to fr")
}
