use num_bigint::BigUint;
use num_traits::cast::ToPrimitive;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use storage_proofs_core::hasher::Domain;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerChallenges {
    /// How many layers we are generating challenges for.
    layers: usize,
    /// The maximum count of challenges
    max_count: usize,
}

impl LayerChallenges {
    pub const fn new(layers: usize, max_count: usize) -> Self {
        LayerChallenges { layers, max_count }
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    pub fn challenges_count_all(&self) -> usize {
        self.max_count
    }

    /// Derive all challenges.
    pub fn derive<D: Domain>(
        &self,
        leaves: usize,
        replica_id: &D,
        seed: &[u8; 32],
        k: u8,
    ) -> Vec<usize> {
        self.derive_internal(self.challenges_count_all(), leaves, replica_id, seed, k)
    }

    pub fn derive_internal<D: Domain>(
        &self,
        challenges_count: usize,
        leaves: usize,
        replica_id: &D,
        seed: &[u8; 32],
        k: u8,
    ) -> Vec<usize> {
        assert!(leaves > 2, "Too few leaves: {}", leaves);

        (0..challenges_count)
            .map(|i| {
                let j: u32 = ((challenges_count * k as usize) + i) as u32;

                let hash = Sha256::new()
                    .chain(replica_id.into_bytes())
                    .chain(seed)
                    .chain(&j.to_le_bytes())
                    .finalize();

                let big_challenge = BigUint::from_bytes_le(hash.as_ref());

                // We cannot try to prove the first node, so make sure the challenge
                // can never be 0.
                let big_mod_challenge = big_challenge % (leaves - 1);
                let big_mod_challenge = big_mod_challenge
                    .to_usize()
                    .expect("`big_mod_challenge` exceeds size of `usize`");
                big_mod_challenge + 1
            })
            .collect()
    }
}

#[derive(Debug, Default)]
pub struct ChallengeRequirements {
    pub minimum_challenges: usize,
}

