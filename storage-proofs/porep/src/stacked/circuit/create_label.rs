use bellperson::gadgets::{boolean::Boolean, num, sha256::sha256 as sha256_circuit, uint32};
use bellperson::{ConstraintSystem, SynthesisError};
use ff::PrimeField;
use fil_sapling_crypto::jubjub::JubjubEngine;
use storage_proofs_core::{gadgets::multipack, gadgets::uint64, util::reverse_bit_numbering};

use crate::stacked::vanilla::TOTAL_PARENTS;

/// Compute a single label.
pub fn create_label_circuit<E, CS>(
    mut cs: CS,
    replica_id: &[Boolean],
    parents: Vec<Vec<Boolean>>,
    layer_index: uint32::UInt32,
    node: uint64::UInt64,
) -> Result<num::AllocatedNum<E>, SynthesisError>
where
    E: JubjubEngine,
    CS: ConstraintSystem<E>,
{
    assert!(replica_id.len() >= 32, "replica id is too small");
    assert!(replica_id.len() <= 256, "replica id is too large");
    assert_eq!(parents.len(), TOTAL_PARENTS, "invalid sized parents");

    // ciphertexts will become a buffer of the layout
    // id | node | parent_node_0 | parent_node_1 | ...

    let mut ciphertexts = replica_id.to_vec();

    // pad to 32 bytes
    while ciphertexts.len() < 256 {
        ciphertexts.push(Boolean::constant(false));
    }

    ciphertexts.extend_from_slice(&layer_index.into_bits_be());
    ciphertexts.extend_from_slice(&node.to_bits_be());
    // pad to 64 bytes
    while ciphertexts.len() < 512 {
        ciphertexts.push(Boolean::constant(false));
    }

    for parent in parents.iter() {
        ciphertexts.extend_from_slice(parent);

        // pad such that each parents take 32 bytes
        while ciphertexts.len() % 256 != 0 {
            ciphertexts.push(Boolean::constant(false));
        }
    }

    // 32b replica id
    // 32b layer_index + node
    // 37 * 32b  = 1184b parents
    assert_eq!(ciphertexts.len(), (1 + 1 + TOTAL_PARENTS) * 32 * 8);

    // Compute Sha256
    let alloc_bits = sha256_circuit(cs.namespace(|| "hash"), &ciphertexts[..])?;

    // Convert the hash result into a single Fr.
    let bits = reverse_bit_numbering(alloc_bits);
    multipack::pack_bits(
        cs.namespace(|| "result_num"),
        &bits[0..(E::Fr::CAPACITY as usize)],
    )
}
