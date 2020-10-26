use bellperson::gadgets::{boolean::Boolean, num};
use bellperson::{ConstraintSystem, SynthesisError};
use fil_sapling_crypto::circuit::pedersen_hash;
use paired::bls12_381::Bls12;

use crate::crypto::pedersen::{JJ_PARAMS, PEDERSEN_BLOCK_SIZE};

/// Pedersen hashing for inputs with length multiple of the block size. Based on a Merkle-Damgard construction.
pub fn pedersen_md_no_padding<CS>(
    mut cs: CS,
    data: &[Boolean],
) -> Result<num::AllocatedNum<Bls12>, SynthesisError>
where
    CS: ConstraintSystem<Bls12>,
{
    assert!(
        data.len() >= 2 * PEDERSEN_BLOCK_SIZE,
        "must be at least 2 block sizes long ({})",
        data.len()
    );

    assert_eq!(
        data.len() % PEDERSEN_BLOCK_SIZE,
        0,
        "data must be a multiple of the block size ({})",
        data.len()
    );

    let mut chunks = data.chunks(PEDERSEN_BLOCK_SIZE);
    let mut cur: Vec<Boolean> = chunks.next().expect("chunks.next failure").to_vec();
    let chunks_len = chunks.len();

    for (i, block) in chunks.enumerate() {
        let mut cs = cs.namespace(|| format!("block {}", i));
        for b in block {
            // TODO: no cloning
            cur.push(b.clone());
        }
        if i == chunks_len - 1 {
            // last round, skip
        } else {
            cur = pedersen_compression(cs.namespace(|| "hash"), &cur)?;
        }
    }

    // hash and return a num at the end
    pedersen_compression_num(cs.namespace(|| "last hash"), &cur)
}

pub fn pedersen_compression_num<CS: ConstraintSystem<Bls12>>(
    mut cs: CS,
    bits: &[Boolean],
) -> Result<num::AllocatedNum<Bls12>, SynthesisError> {
    Ok(pedersen_hash::pedersen_hash(
        cs.namespace(|| "inner hash"),
        pedersen_hash::Personalization::None,
        &bits,
        &*JJ_PARAMS,
    )?
    .get_x()
    .clone())
}

pub fn pedersen_compression<CS: ConstraintSystem<Bls12>>(
    mut cs: CS,
    bits: &[Boolean],
) -> Result<Vec<Boolean>, SynthesisError> {
    let h = pedersen_compression_num(cs.namespace(|| "compression"), bits)?;
    let mut out = h.to_bits_le(cs.namespace(|| "h into bits"))?;

    // needs padding, because x does not always translate to exactly 256 bits
    while out.len() < PEDERSEN_BLOCK_SIZE {
        out.push(Boolean::Constant(false));
    }

    Ok(out)
}

