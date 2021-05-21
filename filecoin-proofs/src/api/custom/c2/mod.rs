use crate::caches::get_stacked_params;
use crate::parameters::setup_params;
use crate::util::as_safe_commitment;
use crate::verify_seal;
use crate::Commitment;
use crate::DefaultPieceDomain;
use crate::DefaultPieceHasher;
use crate::PaddedBytesAmount;
use crate::PoRepConfig;
use crate::PoRepProofPartitions;
use crate::ProverId;
use crate::SealCommitOutput;
use crate::SealCommitPhase1Output;
use crate::Ticket;
use crate::SINGLE_PARTITION_PROOF_LEN;
use anyhow::{ensure, Context, Result};
use bellperson::domain::Scalar;
use bellperson::Circuit;
use bellperson::{
    bls::{Bls12, Fr, FrRepr},
    groth16::prover::ProvingAssignment,
    groth16::{self, MappedParameters},
    multiexp::QueryDensity,
    ConstraintSystem, Index, SynthesisError, Variable,
};
use ff::Field;
use lazy_static::lazy_static;
use log::info;
use rand::rngs::OsRng;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::sync::Arc;
use storage_proofs::{
    compound_proof::{self, CompoundProof},
    hasher::Domain,
    merkle::MerkleTreeTrait,
    multi_proof::MultiProof,
    porep::stacked::{self, StackedCircuit, StackedCompound, StackedDrg},
    sector::SectorId,
};

const GIT_VERSION: &str =
    git_version::git_version!(args = ["--abbrev=40", "--always", "--dirty=-modified"]);
// mod stage1;
mod stage2;

pub mod fft;

lazy_static! {
    pub static ref SECTOR_ID: SectorId = SectorId::from(
        std::env::args()
            .nth(4)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    );
    pub static ref C2_RANDOM_FACTOR: (Vec<Fr>, Vec<Fr>) =
        (|| { serde_json::from_str(&std::env::var("C2_RANDOM_FACTOR").ok()?).ok() })()
            .unwrap_or_default();
}

pub struct C2PreparationData<'a, Tree>
where
    Tree: MerkleTreeTrait + 'static,
{
    ticket: Ticket,
    params: Arc<MappedParameters<Bls12>>,
    circuits: Vec<StackedCircuit<'a, Tree, DefaultPieceHasher>>,
    comm_r: Commitment,
    comm_d: Commitment,
    seed: Ticket,
}

pub fn whole<Tree: 'static + MerkleTreeTrait>(
    porep_config: PoRepConfig,
    phase1_output: SealCommitPhase1Output<Tree>,
    prover_id: ProverId,
    sector_id: SectorId,
) -> Result<SealCommitOutput> {
    info!(
        "{:?}: process params: {:?}",
        sector_id,
        std::env::args().collect::<Vec<_>>()
    );

    info!("{:?}: c2 procedure started", sector_id);
    info!("{:?}: c2 git-version:{}", sector_id, GIT_VERSION);

    let gpu_index = super::get_gpu_index().unwrap_or(0);
    info!("{:?}: gpu index: {}", sector_id, gpu_index);

    let mut rng = OsRng;
    let C2PreparationData {
        ticket,
        params,
        circuits,
        comm_r,
        comm_d,
        seed,
    } = init(porep_config, phase1_output, sector_id)?;
    info!("{:?}: c2 initialized", sector_id);

    let (r_s, s_s) = if C2_RANDOM_FACTOR.0.len() == circuits.len()
        && C2_RANDOM_FACTOR.1.len() == circuits.len()
    {
        info!("random factor from parent process");
        info!("{:?}", *C2_RANDOM_FACTOR);
        C2_RANDOM_FACTOR.clone()
    } else {
        info!("self generated random factor");
        (0..circuits.len())
            .map(|_| (Fr::random(&mut rng), Fr::random(&mut rng)))
            .unzip()
    };

    let fft_handler = fft::create_fft_handler::<Bls12, Scalar<Bls12>>(bellperson::gpu::gpu_count());
    let mut provers: Vec<ProvingAssignment<Bls12>> = c2_stage1(circuits)
        .with_context(|| format!("{:?}: c2 cpu computation failed", sector_id))?;

    info!("{:?}: c2 stage1 finished", sector_id);
    let proofs = stage2::run(&mut provers, &params, r_s, s_s, fft_handler, gpu_index)?;
    info!("{:?}: c2 stage2 finished", sector_id);

    let groth_proofs = proofs
        .into_iter()
        .map(|groth_proof| {
            let mut proof_vec = vec![];
            groth_proof.write(&mut proof_vec)?;
            let gp = groth16::Proof::<Bls12>::read(&proof_vec[..])?;
            Ok(gp)
        })
        .collect::<Result<Vec<_>>>()?;
    info!("{:?}: c2 groth proof generated", sector_id);

    let proof = MultiProof::new(groth_proofs, &params.pvk);

    let mut buf = Vec::with_capacity(
        SINGLE_PARTITION_PROOF_LEN * usize::from(PoRepProofPartitions::from(porep_config)),
    );

    proof.write(&mut buf)?;
    info!("{:?}: c2 proof serialized", sector_id);

    // Verification is cheap when parameters are cached,
    // and it is never correct to return a proof which does not verify.
    verify_seal::<Tree>(
        porep_config,
        comm_r,
        comm_d,
        prover_id,
        sector_id,
        ticket,
        seed,
        &buf,
    )
    .context("post-seal verification sanity check failed")?;

    info!("seal_commit_phase2:finish: {:?}", sector_id);

    let out = SealCommitOutput { proof: buf };

    Ok(out)
}

pub fn c2_stage1<Tree: 'static + MerkleTreeTrait>(
    circuits: Vec<StackedCircuit<'static, Tree, DefaultPieceHasher>>,
) -> Result<Vec<ProvingAssignment<Bls12>>, SynthesisError> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(circuits.len())
        .build()
        .expect("cannot initialize thread-pool")
        .install(|| {
            circuits
                .into_par_iter()
                .enumerate()
                .map(|(index, circuit)| -> Result<_, SynthesisError> {
                    let mut prover = ProvingAssignment::new();

                    prover.alloc_input(|| "", || Ok(Fr::one()))?;

                    circuit.synthesize(&mut prover)?;
                    // stage1::circuit_synthesize(circuit, &mut prover)?;

                    for i in 0..prover.input_assignment.len() {
                        prover.enforce(
                            || "",
                            |lc| lc + Variable(Index::Input(i)),
                            |lc| lc,
                            |lc| lc,
                        );
                    }

                    info!("{:?}: done prover: {}", *SECTOR_ID, index + 1);
                    Ok(prover)
                })
                .collect::<Result<Vec<_>, SynthesisError>>()
        })
}

pub fn init<Tree: 'static + MerkleTreeTrait>(
    porep_config: PoRepConfig,
    phase1_output: SealCommitPhase1Output<Tree>,
    sector_id: SectorId,
) -> Result<C2PreparationData<'static, Tree>> {
    info!("seal_commit_phase2:start: {:?}", sector_id);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //select gpu index

    let SealCommitPhase1Output {
        vanilla_proofs,
        comm_r,
        comm_d,
        replica_id,
        seed,
        ticket,
    } = phase1_output;

    ensure!(comm_d != [0; 32], "Invalid all zero commitment (comm_d)");
    ensure!(comm_r != [0; 32], "Invalid all zero commitment (comm_r)");

    let comm_r_safe = as_safe_commitment(&comm_r, "comm_r")?;
    let comm_d_safe = DefaultPieceDomain::try_from_bytes(&comm_d)?;

    let pub_in = stacked::PublicInputs {
        replica_id,
        seed,
        tau: Some(stacked::Tau {
            comm_d: comm_d_safe,
            comm_r: comm_r_safe,
        }),
        k: None,
    };

    let params: Arc<MappedParameters<Bls12>> = get_stacked_params::<Tree>(porep_config)?;

    info!(
        "got groth params ({}) while sealing",
        u64::from(PaddedBytesAmount::from(porep_config))
    );

    let compound_setup_params = compound_proof::SetupParams {
        vanilla_params: setup_params(
            PaddedBytesAmount::from(porep_config),
            usize::from(PoRepProofPartitions::from(porep_config)),
            porep_config.porep_id,
        )?,
        partitions: Some(usize::from(PoRepProofPartitions::from(porep_config))),
        priority: false,
    };

    let compound_public_params = <StackedCompound<Tree, DefaultPieceHasher> as CompoundProof<
        StackedDrg<Tree, DefaultPieceHasher>,
        _,
    >>::setup(&compound_setup_params)?;
    let pub_params = &compound_public_params.vanilla_params;

    ensure!(
        !vanilla_proofs.is_empty(),
        "cannot create a circuit proof over missing vanilla proofs"
    );

    let circuits: Vec<StackedCircuit<Tree, DefaultPieceHasher>> = vanilla_proofs
        .into_par_iter()
        .enumerate()
        .map(|(k, vanilla_proof)| {
            StackedCompound::<Tree, DefaultPieceHasher>::circuit(
                &pub_in,
                (),
                &vanilla_proof,
                &pub_params,
                Some(k),
            )
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(C2PreparationData {
        ticket,
        params,
        circuits,
        comm_r,
        comm_d,
        seed,
    })
}

pub fn cpu_compute_exp<Q, D>(exponents: &[FrRepr], density_map: D) -> Vec<FrRepr>
where
    for<'a> &'a Q: QueryDensity,
    D: Send + Sync + 'static + Clone + AsRef<Q>,
{
    exponents
        .iter()
        .zip(density_map.as_ref().iter())
        .filter(|(_, d)| *d)
        .map(|(&e, _)| e)
        .collect()
}
