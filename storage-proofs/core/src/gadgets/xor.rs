use bellperson::gadgets::boolean::Boolean;
use bellperson::{ConstraintSystem, SynthesisError};
use fil_sapling_crypto::jubjub::JubjubEngine;

pub fn xor<E, CS>(
    cs: &mut CS,
    key: &[Boolean],
    input: &[Boolean],
) -> Result<Vec<Boolean>, SynthesisError>
where
    E: JubjubEngine,
    CS: ConstraintSystem<E>,
{
    let key_len = key.len();
    assert_eq!(key_len, 32 * 8);

    input
        .iter()
        .enumerate()
        .map(|(i, byte)| {
            Boolean::xor(
                cs.namespace(|| format!("xor bit: {}", i)),
                byte,
                &key[i % key_len],
            )
        })
        .collect::<Result<Vec<_>, SynthesisError>>()
}