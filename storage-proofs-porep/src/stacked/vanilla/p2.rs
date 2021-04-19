pub mod tree_c;
pub mod tree_r;
use anyhow::Result;
use filecoin_hashers::PoseidonArity;
use generic_array::GenericArray;
use lazy_static::lazy_static;
use neptune::{Arity, BatchHasher};
use std::borrow::Cow;

use bellperson::bls::Fr;
use ff::Field;

lazy_static! {
    pub static ref GPU_INDEX: usize = select_gpu_device();
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

    // calculate discarded rows
    let hashed = (0..rows_to_discard).fold(Cow::Borrowed(data), |x, _| {
        let preimages = as_generic_arrays::<TreeArity>(&x);
        let hashed = batcher
            .hash(&preimages)
            .expect("cannot hash preimages to build tree");
        Cow::Owned(hashed)
    });

    let mut tree: Vec<Fr> = vec![Fr::zero(); tree_size];
    let mut rest: &mut [_] = &mut tree;
    let mut origin = &hashed[..];
    while !rest.is_empty() {
        let layer_len = origin.len() / arity;
        let preimages = as_generic_arrays::<TreeArity>(&origin);
        let (to_be_hashed, unhashed) = rest.split_at_mut(layer_len);

        batcher.hash_into_slice(to_be_hashed, preimages)?;

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
