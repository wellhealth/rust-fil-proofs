use std::hash::Hasher as StdHasher;

use crate::crypto::sloth;
use crate::error::{Error, Result};
use crate::hasher::types::{
    PoseidonArity, PoseidonMDArity, POSEIDON_CONSTANTS_16, POSEIDON_CONSTANTS_2,
    POSEIDON_CONSTANTS_4, POSEIDON_CONSTANTS_8, POSEIDON_MD_CONSTANTS,
};
use crate::hasher::{Domain, HashFunction, Hasher};
use anyhow::ensure;
use bellperson::gadgets::{boolean, num};
use bellperson::{ConstraintSystem, SynthesisError};
use ff::{Field, PrimeField, PrimeFieldRepr, ScalarEngine};
use generic_array::typenum;
use generic_array::typenum::marker_traits::Unsigned;
use merkletree::hash::{Algorithm as LightAlgorithm, Hashable};
use merkletree::merkle::Element;
use neptune::circuit::poseidon_hash;
use neptune::poseidon::Poseidon;
use paired::bls12_381::{Bls12, Fr, FrRepr};
use serde::{Deserialize, Serialize};

#[derive(Default, Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoseidonHasher {}

impl Hasher for PoseidonHasher {
    type Domain = PoseidonDomain;
    type Function = PoseidonFunction;

    fn name() -> String {
        "poseidon_hasher".into()
    }

    #[inline]
    fn sloth_encode(key: &Self::Domain, ciphertext: &Self::Domain) -> Result<Self::Domain> {
        // Unrapping here is safe; `Fr` elements and hash domain elements are the same byte length.
        let key = Fr::from_repr(key.0)?;
        let ciphertext = Fr::from_repr(ciphertext.0)?;
        Ok(sloth::encode(&key, &ciphertext).into())
    }

    #[inline]
    fn sloth_decode(key: &Self::Domain, ciphertext: &Self::Domain) -> Result<Self::Domain> {
        // Unrapping here is safe; `Fr` elements and hash domain elements are the same byte length.
        let key = Fr::from_repr(key.0)?;
        let ciphertext = Fr::from_repr(ciphertext.0)?;

        Ok(sloth::decode(&key, &ciphertext).into())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PoseidonFunction(Fr);

impl Default for PoseidonFunction {
    fn default() -> PoseidonFunction {
        PoseidonFunction(Fr::from_repr(FrRepr::default()).expect("failed default"))
    }
}

impl Hashable<PoseidonFunction> for Fr {
    fn hash(&self, state: &mut PoseidonFunction) {
        let mut bytes = Vec::with_capacity(32);
        self.into_repr()
            .write_le(&mut bytes)
            .expect("write_le failure");
        state.write(&bytes);
    }
}

impl Hashable<PoseidonFunction> for PoseidonDomain {
    fn hash(&self, state: &mut PoseidonFunction) {
        let mut bytes = Vec::with_capacity(32);
        self.0
            .write_le(&mut bytes)
            .expect("Failed to write `FrRepr`");
        state.write(&bytes);
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct PoseidonDomain(pub FrRepr);

impl AsRef<PoseidonDomain> for PoseidonDomain {
    fn as_ref(&self) -> &PoseidonDomain {
        self
    }
}

impl std::hash::Hash for PoseidonDomain {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let raw: &[u64] = self.0.as_ref();
        std::hash::Hash::hash(raw, state);
    }
}

impl PartialEq for PoseidonDomain {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ref() == other.0.as_ref()
    }
}

impl Eq for PoseidonDomain {}

impl Default for PoseidonDomain {
    fn default() -> PoseidonDomain {
        PoseidonDomain(FrRepr::default())
    }
}

impl Ord for PoseidonDomain {
    #[inline(always)]
    fn cmp(&self, other: &PoseidonDomain) -> ::std::cmp::Ordering {
        (self.0).cmp(&other.0)
    }
}

impl PartialOrd for PoseidonDomain {
    #[inline(always)]
    fn partial_cmp(&self, other: &PoseidonDomain) -> Option<::std::cmp::Ordering> {
        Some((self.0).cmp(&other.0))
    }
}

impl AsRef<[u8]> for PoseidonDomain {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        as_ref(&(self.0).0)
    }
}

// This is unsafe, and I wish it wasn't here, but I really need AsRef<[u8]> to work, without allocating.
// https://internals.rust-lang.org/t/safe-trasnsmute-for-slices-e-g-u64-u32-particularly-simd-types/2871
// https://github.com/briansmith/ring/blob/abb3fdfc08562f3f02e95fb551604a871fd4195e/src/polyfill.rs#L93-L110
#[inline(always)]
#[allow(clippy::needless_lifetimes)]
fn as_ref<'a>(src: &'a [u64; 4]) -> &'a [u8] {
    unsafe {
        std::slice::from_raw_parts(
            src.as_ptr() as *const u8,
            src.len() * std::mem::size_of::<u64>(),
        )
    }
}

impl Domain for PoseidonDomain {
    fn into_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(PoseidonDomain::byte_len());
        self.0.write_le(&mut out).expect("write_le failure");

        out
    }

    fn try_from_bytes(raw: &[u8]) -> Result<Self> {
        ensure!(raw.len() == PoseidonDomain::byte_len(), Error::BadFrBytes);
        let mut res: FrRepr = Default::default();
        res.read_le(raw)?;

        Ok(PoseidonDomain(res))
    }

    fn write_bytes(&self, dest: &mut [u8]) -> Result<()> {
        self.0.write_le(dest)?;
        Ok(())
    }

    fn random<R: rand::RngCore>(rng: &mut R) -> Self {
        // generating an Fr and converting it, to ensure we stay in the field
        Fr::random(rng).into()
    }
}

impl Element for PoseidonDomain {
    fn byte_len() -> usize {
        32
    }

    fn from_slice(bytes: &[u8]) -> Self {
        match PoseidonDomain::try_from_bytes(bytes) {
            Ok(res) => res,
            Err(err) => panic!(err),
        }
    }

    fn copy_to_slice(&self, bytes: &mut [u8]) {
        bytes.copy_from_slice(&self.into_bytes());
    }
}

impl StdHasher for PoseidonFunction {
    #[inline]
    fn write(&mut self, msg: &[u8]) {
        self.0 = Fr::from_repr(shared_hash(msg).0).expect("from_repr failure");
    }

    #[inline]
    fn finish(&self) -> u64 {
        unimplemented!()
    }
}

fn shared_hash(data: &[u8]) -> PoseidonDomain {
    // FIXME: We shouldn't unwrap here, but doing otherwise will require an interface change.
    // We could truncate so `bytes_into_frs` cannot fail, then ensure `data` is always `fr_safe`.
    let preimage = data
        .chunks(32)
        .map(|ref chunk| {
            <Bls12 as ff::ScalarEngine>::Fr::from_repr(PoseidonDomain::from_slice(chunk).0)
                .expect("from_repr failure")
        })
        .collect::<Vec<_>>();

    shared_hash_frs(&preimage).into()
}

fn shared_hash_frs(
    preimage: &[<Bls12 as ff::ScalarEngine>::Fr],
) -> <Bls12 as ff::ScalarEngine>::Fr {
    match preimage.len() {
        2 => {
            let mut p = Poseidon::new_with_preimage(&preimage, &POSEIDON_CONSTANTS_2);
            p.hash()
        }
        4 => {
            let mut p = Poseidon::new_with_preimage(&preimage, &POSEIDON_CONSTANTS_4);
            p.hash()
        }
        8 => {
            let mut p = Poseidon::new_with_preimage(&preimage, &POSEIDON_CONSTANTS_8);
            p.hash()
        }
        16 => {
            let mut p = Poseidon::new_with_preimage(&preimage, &POSEIDON_CONSTANTS_16);
            p.hash()
        }

        _ => panic!("Unsupported arity for Poseidon hasher: {}", preimage.len()),
    }
}

impl HashFunction<PoseidonDomain> for PoseidonFunction {
    fn hash(data: &[u8]) -> PoseidonDomain {
        shared_hash(data)
    }

    fn hash2(a: &PoseidonDomain, b: &PoseidonDomain) -> PoseidonDomain {
        let mut p =
            Poseidon::new_with_preimage(&[(*a).into(), (*b).into()][..], &*POSEIDON_CONSTANTS_2);
        let fr: <Bls12 as ScalarEngine>::Fr = p.hash();
        fr.into()
    }

    fn hash_md(input: &[PoseidonDomain]) -> PoseidonDomain {
        assert!(input.len() > 1, "hash_md needs more than one element.");
        let arity = PoseidonMDArity::to_usize();

        let mut p = Poseidon::new(&*POSEIDON_MD_CONSTANTS);

        let fr_input = input
            .iter()
            .map(|x| <Bls12 as ScalarEngine>::Fr::from_repr(x.0).expect("from_repr failure"))
            .collect::<Vec<_>>();

        fr_input[1..]
            .chunks(arity - 1)
            .fold(fr_input[0], |acc, elts| {
                p.reset();
                p.input(acc).expect("input failure"); // These unwraps will panic iff arity is incorrect, but it was checked above.
                elts.iter().for_each(|elt| {
                    let _ = p.input(*elt).expect("input failure");
                });
                p.hash()
            })
            .into()
    }

    fn hash_leaf_circuit<CS: ConstraintSystem<Bls12>>(
        cs: CS,
        left: &num::AllocatedNum<Bls12>,
        right: &num::AllocatedNum<Bls12>,
        _height: usize,
    ) -> ::std::result::Result<num::AllocatedNum<Bls12>, SynthesisError> {
        let preimage = vec![left.clone(), right.clone()];

        poseidon_hash::<CS, Bls12, typenum::U2>(cs, preimage, typenum::U2::PARAMETERS())
    }

    fn hash_multi_leaf_circuit<Arity: 'static + PoseidonArity, CS: ConstraintSystem<Bls12>>(
        cs: CS,
        leaves: &[num::AllocatedNum<Bls12>],
        _height: usize,
    ) -> ::std::result::Result<num::AllocatedNum<Bls12>, SynthesisError> {
        let params = Arity::PARAMETERS();
        poseidon_hash::<CS, Bls12, Arity>(cs, leaves.to_vec(), params)
    }

    fn hash_md_circuit<CS: ConstraintSystem<Bls12>>(
        cs: &mut CS,
        elements: &[num::AllocatedNum<Bls12>],
    ) -> ::std::result::Result<num::AllocatedNum<Bls12>, SynthesisError> {
        let params = PoseidonMDArity::PARAMETERS();
        let arity = PoseidonMDArity::to_usize();

        let mut hash = elements[0].clone();
        let mut preimage = vec![hash.clone(); arity]; // Allocate. This will be overwritten.
        for (hash_num, elts) in elements[1..].chunks(arity - 1).enumerate() {
            preimage[0] = hash;
            for (i, elt) in elts.iter().enumerate() {
                preimage[i + 1] = elt.clone();
            }
            // any terminal padding
            #[allow(clippy::needless_range_loop)]
            for i in (elts.len() + 1)..arity {
                preimage[i] =
                    num::AllocatedNum::alloc(cs.namespace(|| format!("padding {}", i)), || {
                        Ok(Fr::zero())
                    })
                    .expect("alloc failure");
            }
            let cs = cs.namespace(|| format!("hash md {}", hash_num));
            hash =
                poseidon_hash::<_, Bls12, PoseidonMDArity>(cs, preimage.clone(), params)?.clone();
        }

        Ok(hash)
    }

    fn hash_circuit<CS: ConstraintSystem<Bls12>>(
        _cs: CS,
        _bits: &[boolean::Boolean],
    ) -> std::result::Result<num::AllocatedNum<Bls12>, SynthesisError> {
        unimplemented!();
    }

    fn hash2_circuit<CS>(
        cs: CS,
        a: &num::AllocatedNum<Bls12>,
        b: &num::AllocatedNum<Bls12>,
    ) -> std::result::Result<num::AllocatedNum<Bls12>, SynthesisError>
    where
        CS: ConstraintSystem<Bls12>,
    {
        let preimage = vec![a.clone(), b.clone()];
        poseidon_hash::<CS, Bls12, typenum::U2>(cs, preimage, typenum::U2::PARAMETERS())
    }
}

impl LightAlgorithm<PoseidonDomain> for PoseidonFunction {
    #[inline]
    fn hash(&mut self) -> PoseidonDomain {
        self.0.into()
    }

    #[inline]
    fn reset(&mut self) {
        self.0 = Fr::from_repr(FrRepr::from(0)).expect("failed 0");
    }

    fn leaf(&mut self, leaf: PoseidonDomain) -> PoseidonDomain {
        leaf
    }

    fn node(
        &mut self,
        left: PoseidonDomain,
        right: PoseidonDomain,
        _height: usize,
    ) -> PoseidonDomain {
        shared_hash_frs(&[
            <Bls12 as ff::ScalarEngine>::Fr::from_repr(left.0).expect("from_repr failure"),
            <Bls12 as ff::ScalarEngine>::Fr::from_repr(right.0).expect("from_repr failure"),
        ])
        .into()
    }

    fn multi_node(&mut self, parts: &[PoseidonDomain], _height: usize) -> PoseidonDomain {
        match parts.len() {
            1 | 2 | 4 | 8 | 16 => shared_hash_frs(
                &parts
                    .iter()
                    .map(|x| {
                        <Bls12 as ff::ScalarEngine>::Fr::from_repr(x.0).expect("from_repr failure")
                    })
                    .collect::<Vec<_>>(),
            )
            .into(),
            arity => panic!("unsupported arity {}", arity),
        }
    }
}

impl From<Fr> for PoseidonDomain {
    #[inline]
    fn from(val: Fr) -> Self {
        PoseidonDomain(val.into_repr())
    }
}

impl From<FrRepr> for PoseidonDomain {
    #[inline]
    fn from(val: FrRepr) -> Self {
        PoseidonDomain(val)
    }
}

impl From<PoseidonDomain> for Fr {
    #[inline]
    fn from(val: PoseidonDomain) -> Self {
        Fr::from_repr(val.0).expect("from_repr failure")
    }
}
