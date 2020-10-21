use bellperson::domain::EvaluationDomain;
use bellperson::gpu::LockedFFTKernel;
use bellperson::groth16::ParameterSource;
use bellperson::multicore::Worker;
use bellperson::SynthesisError;
use rayon::iter::IntoParallelRefMutIterator;
use serde::{Deserialize, Serialize};
use std::fs::{self, metadata, File, OpenOptions};
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use bincode::{deserialize, serialize};
use log::{info, trace};
use memmap::MmapOptions;
use merkletree::store::{DiskStore, Store, StoreConfig};
use paired::bls12_381::Fr;
use std::fs::{self, metadata, File, OpenOptions};
use std::io::prelude::*;
use std::path::{Path, PathBuf};

use storage_proofs::cache_key::CacheKey;
use storage_proofs::compound_proof::{self, CompoundProof};
use storage_proofs::drgraph::Graph;
use storage_proofs::hasher::{Domain, Hasher};
use storage_proofs::measurements::{measure_op, Operation::CommD};
use storage_proofs::merkle::{create_base_merkle_tree, BinaryMerkleTree, MerkleTreeTrait};
use storage_proofs::multi_proof::MultiProof;
use storage_proofs::porep::stacked::{
    self, generate_replica_id, ChallengeRequirements, StackedCompound, StackedDrg, Tau,
    TemporaryAux, TemporaryAuxCache,
};
use storage_proofs::proof::ProofScheme;
use storage_proofs::sector::SectorId;
use storage_proofs::settings;
use storage_proofs::util::default_rows_to_discard;

use crate::api::util::{
    as_safe_commitment, commitment_from_fr, get_base_tree_leafs, get_base_tree_size,
};
use crate::caches::get_stacked_verifying_key;
use crate::constants::{
    DefaultBinaryTree, DefaultPieceDomain, DefaultPieceHasher, POREP_MINIMUM_CHALLENGES,
};
use crate::parameters::setup_params;
pub use crate::pieces;
pub use crate::pieces::verify_pieces;
use crate::types::{
    Commitment, PaddedBytesAmount, PieceInfo, PoRepConfig, PoRepProofPartitions, ProverId,
    SealCommitOutput, SealCommitPhase1Output, SealPreCommitOutput, SealPreCommitPhase1Output,
    SectorSize, Ticket, BINARY_ARITY,
};
use crate::Labels;
use std::marker::PhantomData;

#[allow(clippy::too_many_arguments)]
pub fn seal_pre_commit_phase1<R, S, T, Tree: 'static + MerkleTreeTrait>(
    porep_config: PoRepConfig,
    cache_path: R,
    in_path: S,
    out_path: T,
    prover_id: ProverId,
    sector_id: SectorId,
    ticket: Ticket,
    piece_infos: &[PieceInfo],
) -> Result<SealPreCommitPhase1Output<Tree>>
where
    R: AsRef<Path>,
    S: AsRef<Path>,
    T: AsRef<Path>,
{
    info!("seal_pre_commit_phase1:start: {:?}", sector_id);

    // Sanity check all input path types.
    ensure!(
        metadata(in_path.as_ref())?.is_file(),
        "in_path must be a file"
    );
    ensure!(
        metadata(out_path.as_ref())?.is_file(),
        "out_path must be a file"
    );
    ensure!(
        metadata(cache_path.as_ref())?.is_dir(),
        "cache_path must be a directory"
    );

    let sector_bytes = usize::from(PaddedBytesAmount::from(porep_config));
    fs::metadata(&in_path)
        .with_context(|| format!("could not read in_path={:?})", in_path.as_ref().display()))?;

    fs::metadata(&out_path)
        .with_context(|| format!("could not read out_path={:?}", out_path.as_ref().display()))?;

    // Copy unsealed data to output location, where it will be sealed in place.
    fs::copy(&in_path, &out_path).with_context(|| {
        format!(
            "could not copy in_path={:?} to out_path={:?}",
            in_path.as_ref().display(),
            out_path.as_ref().display()
        )
    })?;

    let f_data = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&out_path)
        .with_context(|| format!("could not open out_path={:?}", out_path.as_ref().display()))?;

    // Zero-pad the data to the requested size by extending the underlying file if needed.
    f_data.set_len(sector_bytes as u64)?;

    let data = unsafe {
        MmapOptions::new()
            .map_mut(&f_data)
            .with_context(|| format!("could not mmap out_path={:?}", out_path.as_ref().display()))?
    };

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

    info!("building merkle tree for the original data");
    let (config, comm_d) = measure_op(CommD, || -> Result<_> {
        let base_tree_size = get_base_tree_size::<DefaultBinaryTree>(porep_config.sector_size)?;
        let base_tree_leafs = get_base_tree_leafs::<DefaultBinaryTree>(base_tree_size)?;
        ensure!(
            compound_public_params.vanilla_params.graph.size() == base_tree_leafs,
            "graph size and leaf size don't match"
        );

        trace!(
            "seal phase 1: sector_size {}, base tree size {}, base tree leafs {}",
            u64::from(porep_config.sector_size),
            base_tree_size,
            base_tree_leafs,
        );

        // MT for original data is always named tree-d, and it will be
        // referenced later in the process as such.
        let mut config = StoreConfig::new(
            cache_path.as_ref(),
            CacheKey::CommDTree.to_string(),
            default_rows_to_discard(base_tree_leafs, BINARY_ARITY),
        );
        let data_tree = create_base_merkle_tree::<BinaryMerkleTree<DefaultPieceHasher>>(
            Some(config.clone()),
            base_tree_leafs,
            &data,
        )?;
        drop(data);

        config.size = Some(data_tree.len());
        let comm_d_root: Fr = data_tree.root().into();
        let comm_d = commitment_from_fr(comm_d_root);

        drop(data_tree);

        Ok((config, comm_d))
    })?;

    info!("verifying pieces");

    ensure!(
        verify_pieces(&comm_d, piece_infos, porep_config.into())?,
        "pieces and comm_d do not match"
    );

    let replica_id = generate_replica_id::<Tree::Hasher, _>(
        &prover_id,
        sector_id.into(),
        &ticket,
        comm_d,
        &porep_config.porep_id,
    );

    let labels = StackedDrg::<Tree, DefaultPieceHasher>::replicate_phase1(
        &compound_public_params.vanilla_params,
        &replica_id,
        config.clone(),
    )?;

    let out = SealPreCommitPhase1Output {
        labels,
        config,
        comm_d,
    };

    info!("seal_pre_commit_phase1:finish: {:?}", sector_id);
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn seal_pre_commit_phase2<R, S, Tree: 'static + MerkleTreeTrait>(
    porep_config: PoRepConfig,
    phase1_output: SealPreCommitPhase1Output<Tree>,
    cache_path: S,
    replica_path: R,
) -> Result<SealPreCommitOutput>
where
    R: AsRef<Path>,
    S: AsRef<Path>,
{
    info!("seal_pre_commit_phase2:start");

    // Sanity check all input path types.
    ensure!(
        metadata(cache_path.as_ref())?.is_dir(),
        "cache_path must be a directory"
    );
    ensure!(
        metadata(replica_path.as_ref())?.is_file(),
        "replica_path must be a file"
    );

    let SealPreCommitPhase1Output {
        mut labels,
        mut config,
        comm_d,
        ..
    } = phase1_output;

    labels.update_root(cache_path.as_ref());
    config.path = cache_path.as_ref().into();

    let f_data = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&replica_path)
        .with_context(|| {
            format!(
                "could not open replica_path={:?}",
                replica_path.as_ref().display()
            )
        })?;
    let data = unsafe {
        MmapOptions::new().map_mut(&f_data).with_context(|| {
            format!(
                "could not mmap replica_path={:?}",
                replica_path.as_ref().display()
            )
        })?
    };
    let data: storage_proofs::Data<'_> = (data, PathBuf::from(replica_path.as_ref())).into();

    // Load data tree from disk
    let data_tree = {
        let base_tree_size = get_base_tree_size::<DefaultBinaryTree>(porep_config.sector_size)?;
        let base_tree_leafs = get_base_tree_leafs::<DefaultBinaryTree>(base_tree_size)?;

        trace!(
            "seal phase 2: base tree size {}, base tree leafs {}, rows to discard {}",
            base_tree_size,
            base_tree_leafs,
            default_rows_to_discard(base_tree_leafs, BINARY_ARITY)
        );
        ensure!(
            config.rows_to_discard == default_rows_to_discard(base_tree_leafs, BINARY_ARITY),
            "Invalid cache size specified"
        );

        let store: DiskStore<DefaultPieceDomain> =
            DiskStore::new_from_disk(base_tree_size, BINARY_ARITY, &config)?;
        BinaryMerkleTree::<DefaultPieceHasher>::from_data_store(store, base_tree_leafs)?
    };

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

    let (tau, (p_aux, t_aux)) = StackedDrg::<Tree, DefaultPieceHasher>::replicate_phase2(
        &compound_public_params.vanilla_params,
        labels,
        data,
        data_tree,
        config,
        replica_path.as_ref().to_path_buf(),
    )?;

    let comm_r = commitment_from_fr(tau.comm_r.into());

    // Persist p_aux and t_aux here
    let p_aux_path = cache_path.as_ref().join(CacheKey::PAux.to_string());
    let mut f_p_aux = File::create(&p_aux_path)
        .with_context(|| format!("could not create file p_aux={:?}", p_aux_path))?;
    let p_aux_bytes = serialize(&p_aux)?;
    f_p_aux
        .write_all(&p_aux_bytes)
        .with_context(|| format!("could not write to file p_aux={:?}", p_aux_path))?;

    let t_aux_path = cache_path.as_ref().join(CacheKey::TAux.to_string());
    let mut f_t_aux = File::create(&t_aux_path)
        .with_context(|| format!("could not create file t_aux={:?}", t_aux_path))?;
    let t_aux_bytes = serialize(&t_aux)?;
    f_t_aux
        .write_all(&t_aux_bytes)
        .with_context(|| format!("could not write to file t_aux={:?}", t_aux_path))?;

    let out = SealPreCommitOutput { comm_r, comm_d };

    info!("seal_pre_commit_phase2:finish");
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn seal_commit_phase1<T: AsRef<Path>, Tree: 'static + MerkleTreeTrait>(
    porep_config: PoRepConfig,
    cache_path: T,
    replica_path: T,
    prover_id: ProverId,
    sector_id: SectorId,
    ticket: Ticket,
    seed: Ticket,
    pre_commit: SealPreCommitOutput,
    piece_infos: &[PieceInfo],
) -> Result<SealCommitPhase1Output<Tree>> {
    info!("seal_commit_phase1:start: {:?}", sector_id);

    // Sanity check all input path types.
    ensure!(
        metadata(cache_path.as_ref())?.is_dir(),
        "cache_path must be a directory"
    );
    ensure!(
        metadata(replica_path.as_ref())?.is_file(),
        "replica_path must be a file"
    );

    let SealPreCommitOutput { comm_d, comm_r } = pre_commit;

    ensure!(comm_d != [0; 32], "Invalid all zero commitment (comm_d)");
    ensure!(comm_r != [0; 32], "Invalid all zero commitment (comm_r)");
    ensure!(
        verify_pieces(&comm_d, piece_infos, porep_config.into())?,
        "pieces and comm_d do not match"
    );

    let p_aux = {
        let p_aux_path = cache_path.as_ref().join(CacheKey::PAux.to_string());
        let p_aux_bytes = std::fs::read(&p_aux_path)
            .with_context(|| format!("could not read file p_aux={:?}", p_aux_path))?;

        deserialize(&p_aux_bytes)
    }?;

    let t_aux = {
        let t_aux_path = cache_path.as_ref().join(CacheKey::TAux.to_string());
        let t_aux_bytes = std::fs::read(&t_aux_path)
            .with_context(|| format!("could not read file t_aux={:?}", t_aux_path))?;

        let mut res: TemporaryAux<_, _> = deserialize(&t_aux_bytes)?;

        // Switch t_aux to the passed in cache_path
        res.set_cache_path(cache_path);
        res
    };

    // Convert TemporaryAux to TemporaryAuxCache, which instantiates all
    // elements based on the configs stored in TemporaryAux.
    let t_aux_cache: TemporaryAuxCache<Tree, DefaultPieceHasher> =
        TemporaryAuxCache::new(&t_aux, replica_path.as_ref().to_path_buf())
            .context("failed to restore contents of t_aux")?;

    let comm_r_safe = as_safe_commitment(&comm_r, "comm_r")?;
    let comm_d_safe = DefaultPieceDomain::try_from_bytes(&comm_d)?;

    let replica_id = generate_replica_id::<Tree::Hasher, _>(
        &prover_id,
        sector_id.into(),
        &ticket,
        comm_d_safe,
        &porep_config.porep_id,
    );

    let public_inputs = stacked::PublicInputs {
        replica_id,
        tau: Some(stacked::Tau {
            comm_d: comm_d_safe,
            comm_r: comm_r_safe,
        }),
        k: None,
        seed,
    };

    let private_inputs = stacked::PrivateInputs::<Tree, DefaultPieceHasher> {
        p_aux,
        t_aux: t_aux_cache,
    };

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

    let vanilla_proofs = StackedDrg::prove_all_partitions(
        &compound_public_params.vanilla_params,
        &public_inputs,
        &private_inputs,
        StackedCompound::partition_count(&compound_public_params),
    )?;

    let sanity_check = StackedDrg::<Tree, DefaultPieceHasher>::verify_all_partitions(
        &compound_public_params.vanilla_params,
        &public_inputs,
        &vanilla_proofs,
    )?;
    ensure!(sanity_check, "Invalid vanilla proof generated");

    let out = SealCommitPhase1Output {
        vanilla_proofs,
        comm_r,
        comm_d,
        replica_id,
        seed,
        ticket,
    };

    info!("seal_commit_phase1:finish: {:?}", sector_id);
    Ok(out)
}

pub struct C2DataStage1<Tree>
where
    Tree: 'static + MerkleTreeTrait,
{
    pub porep_config: PoRepConfig,
    pub ticket: Ticket,
    pub circuits: Vec<StackedCircuit<'static, Tree, DefaultPieceHasher>>,
    pub groth_params: Arc<MappedParameters<Bls12>>,
    pub prover_id: ProverId,
}

pub struct C2DataStage2 {
    pub porep_config: PoRepConfig,
    pub ticket: Ticket,
    pub groth_params: Arc<MappedParameters<Bls12>>,
    pub prover_id: ProverId,
    pub fft: Vec<Arc<Vec<Fr>>>,
}

pub fn stage1_2<Tree: 'static + MerkleTreeTrait>(
    porep_config: PoRepConfig,
    phase1_output: SealCommitPhase1Output<Tree>,
    prover_id: ProverId,
    sector_id: SectorId,
) -> Result<()> {
    let stage1_data = prepare_c2::<Tree>(porep_config, phase1_output, prover_id, sector_id)?;
    let C2DataStage1 {
        porep_config,
        ticket,
        circuits,
        groth_params,
        prover_id,
    } = stage1_data;

    use ff::Field;

    let mut rng = OsRng;
    let r_s = (0..circuits.len()).map(|_| Fr::random(&mut rng)).collect();
    let s_s = (0..circuits.len()).map(|_| Fr::random(&mut rng)).collect();

    let provers = c2_stage1::<Tree>(circuits)?;
    let _ = c2_stage2_fft(provers, &groth_params, r_s, s_s);
    Ok(())
}

pub fn prepare_c2<Tree: 'static + MerkleTreeTrait>(
    porep_config: PoRepConfig,
    phase1_output: SealCommitPhase1Output<Tree>,
    prover_id: ProverId,
    sector_id: SectorId,
) -> Result<C2DataStage1<Tree>> {
    info!("seal_commit_phase2:start: {:?}", sector_id);
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

    let public_inputs = stacked::PublicInputs {
        replica_id,
        tau: Some(stacked::Tau {
            comm_d: comm_d_safe,
            comm_r: comm_r_safe,
        }),
        k: None,
        seed,
    };

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

    let params = <StackedCompound<Tree, DefaultPieceHasher> as CompoundProof<
        StackedDrg<Tree, DefaultPieceHasher>,
        _,
    >>::setup(&compound_setup_params)?
    .vanilla_params;

    info!("snark_proof:start");

    ensure!(
        !vanilla_proofs.is_empty(),
        "cannot create a circuit proof over missing vanilla proofs"
    );

    let circuits: Vec<StackedCircuit<Tree, DefaultPieceHasher>> = vanilla_proofs
        .into_par_iter()
        .enumerate()
        .map(|(k, vanilla_proof)| {
            StackedCompound::<Tree, DefaultPieceHasher>::circuit(
                &public_inputs,
                <StackedCircuit::<Tree, DefaultPieceHasher> as CircuitComponent>::ComponentPrivateInputs::default(),
                &vanilla_proof,
                &params,
                Some(k),
            )
        })
        .collect::<Result<Vec<_>>>()?;

    let groth_params = get_stacked_params::<Tree>(porep_config)?;
    Ok(C2DataStage1 {
        porep_config,
        ticket,
        circuits,
        groth_params,
        prover_id,
    })
}

fn true_custom_c2<Tree: 'static + MerkleTreeTrait>(
    porep_config: PoRepConfig,
    phase1_output: SealCommitPhase1Output<Tree>,
    prover_id: ProverId,
    sector_id: SectorId,
) -> Result<()> {
	Ok(())
}

#[derive(Serialize, Deserialize)]
pub struct ConcreteProvingAssignment {
    // Density of queries
    pub a_aux_density: DensityTracker,
    pub b_input_density: DensityTracker,
    pub b_aux_density: DensityTracker,

    // Evaluations of A, B, C polynomials
    pub a: Vec<Scalar<Bls12>>,
    pub b: Vec<Scalar<Bls12>>,
    pub c: Vec<Scalar<Bls12>>,

    // Assignments of variables
    pub input_assignment: Vec<Fr>,
    pub aux_assignment: Vec<Fr>,
}

impl From<ProvingAssignment<Bls12>> for ConcreteProvingAssignment {
    fn from(input: ProvingAssignment<Bls12>) -> Self {
        ConcreteProvingAssignment {
            a_aux_density: input.a_aux_density,
            b_input_density: input.b_input_density,
            b_aux_density: input.b_aux_density,
            a: input.a,
            b: input.b,
            c: input.c,
            input_assignment: input.input_assignment,
            aux_assignment: input.aux_assignment,
        }
    }
}

pub fn c2_stage1<Tree: 'static + MerkleTreeTrait>(
    circuits: Vec<StackedCircuit<Tree, DefaultPieceHasher>>,
) -> Result<Vec<ConcreteProvingAssignment>> {
    circuits
        .into_par_iter()
        .map(|circuit| {
            let mut prover = ProvingAssignment::new();

            use ff::Field;
            prover.alloc_input(|| "", || Ok(Fr::one()))?;
            circuit
                .synthesize(&mut prover)
                .map_err(|e| anyhow::Error::from(e))?;

            for i in 0..prover.input_assignment.len() {
                prover.enforce(|| "", |lc| lc + Variable(Index::Input(i)), |lc| lc, |lc| lc);
            }

            Ok(prover.into())
        })
        .collect::<Result<Vec<ConcreteProvingAssignment>, _>>()
}

pub fn c2_stage2_fft(
    provers: Vec<ConcreteProvingAssignment>,
    params: &MappedParameters<paired::bls12_381::Bls12>,
    r_s: Vec<Fr>,
    s_s: Vec<Fr>,
) -> Result<Vec<Arc<Vec<Fr>>>> {
    let worker = Worker::new();
    let input_len = provers[0].input_assignment.len();
    let vk = params.get_vk(input_len)?;
    let n = provers[0].a.len();

    // Make sure all circuits have the same input len.
    for prover in &provers {
        assert_eq!(
            prover.a.len(),
            n,
            "only equaly sized circuits are supported"
        );
    }

    let mut log_d = 0;
    while (1 << log_d) < n {
        log_d += 1;
    }

    let mut provers = provers;

    let mut fft_kern = Some(LockedFFTKernel::<Bls12>::new(log_d, false));

    let a_s = provers
        .iter_mut()
        .map(|prover| {
            let mut a =
                EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.a, Vec::new()))?;
            let mut b =
                EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.b, Vec::new()))?;
            let mut c =
                EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.c, Vec::new()))?;

            b.ifft(&worker, &mut fft_kern).unwrap();
            b.coset_fft(&worker, &mut fft_kern).unwrap();
            a.ifft(&worker, &mut fft_kern).unwrap();
            a.coset_fft(&worker, &mut fft_kern).unwrap();
            c.ifft(&worker, &mut fft_kern).unwrap();
            c.coset_fft(&worker, &mut fft_kern).unwrap();

            a.mul_assign(&worker, &b);
            drop(b);
            a.sub_assign(&worker, &c);
            drop(c);
            a.divide_by_z_on_coset(&worker);
            a.icoset_fft(&worker, &mut fft_kern)?;
            let mut a = a.into_coeffs();
            let a_len = a.len() - 1;
            a.truncate(a_len);

            Ok({ Arc::new(a.into_iter().map(|s| s.0.into()).collect::<Vec<Fr>>()) })
        })
        .collect::<Result<Vec<Arc<Vec<Fr>>>, SynthesisError>>()?;

    Ok(a_s)
}

pub fn custom_c2<Tree: 'static + MerkleTreeTrait>(
    porep_config: PoRepConfig,
    phase1_output: SealCommitPhase1Output<Tree>,
    prover_id: ProverId,
    sector_id: SectorId,
) -> Result<SealCommitOutput> {
    info!("seal_commit_phase2:start: {:?}", sector_id);
    let SealCommitPhase1Output {
        vanilla_proofs,
        comm_d,
        comm_r,
        replica_id,
        seed,
        ticket,
    } = phase1_output;
    ensure!(comm_d != [0; 32], "Invalid all zero commitment (comm_d)");
    ensure!(comm_r != [0; 32], "Invalid all zero commitment (comm_r)");
    let comm_r_safe = as_safe_commitment(&comm_r, "comm_r")?;
    let comm_d_safe = DefaultPieceDomain::try_from_bytes(&comm_d)?;

    let public_inputs = stacked::PublicInputs {
        replica_id,
        tau: Some(stacked::Tau {
            comm_d: comm_d_safe,
            comm_r: comm_r_safe,
        }),
        k: None,
        seed,
    };

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

    let params = <StackedCompound<Tree, DefaultPieceHasher> as CompoundProof<
        StackedDrg<Tree, DefaultPieceHasher>,
        _,
    >>::setup(&compound_setup_params)?
    .vanilla_params;

    info!("snark_proof:start");

    ensure!(
        !vanilla_proofs.is_empty(),
        "cannot create a circuit proof over missing vanilla proofs"
    );

    let circuits = vanilla_proofs
        .into_par_iter()
        .enumerate()
        .map(|(k, vanilla_proof)| {
            StackedCompound::<Tree, DefaultPieceHasher>::circuit(
                &public_inputs,
                <StackedCircuit::<Tree, DefaultPieceHasher> as CircuitComponent>::ComponentPrivateInputs::default(),
                &vanilla_proof,
                &params,
                Some(k),
            )
        })
        .collect::<Result<Vec<_>>>()?;

    let groth_params = get_stacked_params::<Tree>(porep_config)?;
    let p: &MappedParameters<paired::bls12_381::Bls12> = &groth_params;
    let mut rng = OsRng;
    let groth_proofs: Vec<groth16::Proof<Bls12>> =
        groth16::create_random_proof_batch(circuits, p, &mut rng)?;

    let groth_proofs = groth_proofs
        .into_iter()
        .map(|groth_proof| {
            let mut proof_vec = vec![];
            groth_proof.write(&mut proof_vec)?;
            groth16::Proof::<Bls12>::read(&proof_vec[..]).map_err(|e| anyhow::Error::from(e))
        })
        .collect::<Result<Vec<_>>>()?;
    info!("snark_proof:finish");

    let proof = MultiProof::new(groth_proofs, &groth_params.vk);

    let mut buf = Vec::with_capacity(
        SINGLE_PARTITION_PROOF_LEN * usize::from(PoRepProofPartitions::from(porep_config)),
    );

    proof.write(&mut buf)?;

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

    let out = SealCommitOutput { proof: buf };

    info!("seal_commit_phase2:finish: {:?}", sector_id);
    Ok(out)
}

pub fn seal_commit_phase2<Tree: 'static + MerkleTreeTrait>(
    porep_config: PoRepConfig,
    phase1_output: SealCommitPhase1Output<Tree>,
    prover_id: ProverId,
    sector_id: SectorId,
) -> Result<SealCommitOutput> {
    crate::api::my::custom_c2::<Tree>(porep_config, phase1_output, prover_id, sector_id)
}

/// Computes a sectors's `comm_d` given its pieces.
///
/// # Arguments
///
/// * `porep_config` - this sector's porep config that contains the number of bytes in the sector.
/// * `piece_infos` - the piece info (commitment and byte length) for each piece in this sector.
pub fn compute_comm_d(sector_size: SectorSize, piece_infos: &[PieceInfo]) -> Result<Commitment> {
    info!("compute_comm_d:start");

    let result = pieces::compute_comm_d(sector_size, piece_infos);

    info!("compute_comm_d:finish");
    result
}

/// Verifies the output of some previously-run seal operation.
///
/// # Arguments
///
/// * `porep_config` - this sector's porep config that contains the number of bytes in this sector.
/// * `comm_r_in` - commitment to the sector's replica (`comm_r`).
/// * `comm_d_in` - commitment to the sector's data (`comm_d`).
/// * `prover_id` - the prover-id that sealed this sector.
/// * `sector_id` - this sector's sector-id.
/// * `ticket` - the ticket that was used to generate this sector's replica-id.
/// * `seed` - the seed used to derive the porep challenges.
/// * `proof_vec` - the porep circuit proof serialized into a vector of bytes.
#[allow(clippy::too_many_arguments)]
pub fn verify_seal<Tree: 'static + MerkleTreeTrait>(
    porep_config: PoRepConfig,
    comm_r_in: Commitment,
    comm_d_in: Commitment,
    prover_id: ProverId,
    sector_id: SectorId,
    ticket: Ticket,
    seed: Ticket,
    proof_vec: &[u8],
) -> Result<bool> {
    info!("verify_seal:start: {:?}", sector_id);
    ensure!(comm_d_in != [0; 32], "Invalid all zero commitment (comm_d)");
    ensure!(comm_r_in != [0; 32], "Invalid all zero commitment (comm_r)");

    let comm_r: <Tree::Hasher as Hasher>::Domain = as_safe_commitment(&comm_r_in, "comm_r")?;
    let comm_d: DefaultPieceDomain = as_safe_commitment(&comm_d_in, "comm_d")?;

    let replica_id = generate_replica_id::<Tree::Hasher, _>(
        &prover_id,
        sector_id.into(),
        &ticket,
        comm_d,
        &porep_config.porep_id,
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

    let compound_public_params: compound_proof::PublicParams<
        '_,
        StackedDrg<'_, Tree, DefaultPieceHasher>,
    > = StackedCompound::setup(&compound_setup_params)?;

    let public_inputs =
        stacked::PublicInputs::<<Tree::Hasher as Hasher>::Domain, DefaultPieceDomain> {
            replica_id,
            tau: Some(Tau { comm_r, comm_d }),
            seed,
            k: None,
        };

    let use_fil_blst = settings::SETTINGS
        .lock()
        .expect("use_fil_blst settings lock failure")
        .use_fil_blst;

    let result = if use_fil_blst {
        info!("verify_seal: use_fil_blst=true");
        let verifying_key_path = porep_config.get_cache_verifying_key_path::<Tree>()?;

        StackedCompound::verify_blst(
            &compound_public_params,
            &public_inputs,
            &proof_vec,
            proof_vec.len() / 192,
            &ChallengeRequirements {
                minimum_challenges: *POREP_MINIMUM_CHALLENGES
                    .read()
                    .expect("POREP_MINIMUM_CHALLENGES poisoned")
                    .get(&u64::from(SectorSize::from(porep_config)))
                    .expect("unknown sector size") as usize,
            },
            &verifying_key_path,
        )
    } else {
        let sector_bytes = PaddedBytesAmount::from(porep_config);
        let verifying_key = get_stacked_verifying_key::<Tree>(porep_config)?;

        info!(
            "got verifying key ({}) while verifying seal",
            u64::from(sector_bytes)
        );

        let proof = MultiProof::new_from_reader(
            Some(usize::from(PoRepProofPartitions::from(porep_config))),
            proof_vec,
            &verifying_key,
        )?;

        StackedCompound::verify(
            &compound_public_params,
            &public_inputs,
            &proof,
            &ChallengeRequirements {
                minimum_challenges: *POREP_MINIMUM_CHALLENGES
                    .read()
                    .expect("POREP_MINIMUM_CHALLENGES poisoned")
                    .get(&u64::from(SectorSize::from(porep_config)))
                    .expect("unknown sector size") as usize,
            },
        )
    };

    info!("verify_seal:finish: {:?}", sector_id);
    result
}

/// Verifies a batch of outputs of some previously-run seal operations.
///
/// # Arguments
///
/// * `porep_config` - this sector's porep config that contains the number of bytes in this sector.
/// * `[comm_r_ins]` - list of commitments to the sector's replica (`comm_r`).
/// * `[comm_d_ins]` - list of commitments to the sector's data (`comm_d`).
/// * `[prover_ids]` - list of prover-ids that sealed this sector.
/// * `[sector_ids]` - list of the sector's sector-id.
/// * `[tickets]` - list of tickets that was used to generate this sector's replica-id.
/// * `[seeds]` - list of seeds used to derive the porep challenges.
/// * `[proof_vecs]` - list of porep circuit proofs serialized into a vector of bytes.
#[allow(clippy::too_many_arguments)]
pub fn verify_batch_seal<Tree: 'static + MerkleTreeTrait>(
    porep_config: PoRepConfig,
    comm_r_ins: &[Commitment],
    comm_d_ins: &[Commitment],
    prover_ids: &[ProverId],
    sector_ids: &[SectorId],
    tickets: &[Ticket],
    seeds: &[Ticket],
    proof_vecs: &[&[u8]],
) -> Result<bool> {
    info!("verify_batch_seal:start");
    ensure!(!comm_r_ins.is_empty(), "Cannot prove empty batch");
    let l = comm_r_ins.len();
    ensure!(l == comm_d_ins.len(), "Inconsistent inputs");
    ensure!(l == prover_ids.len(), "Inconsistent inputs");
    ensure!(l == prover_ids.len(), "Inconsistent inputs");
    ensure!(l == sector_ids.len(), "Inconsistent inputs");
    ensure!(l == tickets.len(), "Inconsistent inputs");
    ensure!(l == seeds.len(), "Inconsistent inputs");
    ensure!(l == proof_vecs.len(), "Inconsistent inputs");

    for comm_d_in in comm_d_ins {
        ensure!(
            comm_d_in != &[0; 32],
            "Invalid all zero commitment (comm_d)"
        );
    }
    for comm_r_in in comm_r_ins {
        ensure!(
            comm_r_in != &[0; 32],
            "Invalid all zero commitment (comm_r)"
        );
    }

    let sector_bytes = PaddedBytesAmount::from(porep_config);

    let verifying_key = get_stacked_verifying_key::<Tree>(porep_config)?;
    info!(
        "got verifying key ({}) while verifying seal",
        u64::from(sector_bytes)
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

    let compound_public_params: compound_proof::PublicParams<
        '_,
        StackedDrg<'_, Tree, DefaultPieceHasher>,
    > = StackedCompound::setup(&compound_setup_params)?;

    let mut public_inputs = Vec::with_capacity(l);
    let mut proofs = Vec::with_capacity(l);

    for i in 0..l {
        let comm_r = as_safe_commitment(&comm_r_ins[i], "comm_r")?;
        let comm_d = as_safe_commitment(&comm_d_ins[i], "comm_d")?;

        let replica_id = generate_replica_id::<Tree::Hasher, _>(
            &prover_ids[i],
            sector_ids[i].into(),
            &tickets[i],
            comm_d,
            &porep_config.porep_id,
        );

        public_inputs.push(stacked::PublicInputs::<
            <Tree::Hasher as Hasher>::Domain,
            DefaultPieceDomain,
        > {
            replica_id,
            tau: Some(Tau { comm_r, comm_d }),
            seed: seeds[i],
            k: None,
        });
        proofs.push(MultiProof::new_from_reader(
            Some(usize::from(PoRepProofPartitions::from(porep_config))),
            proof_vecs[i],
            &verifying_key,
        )?);
    }

    let result = StackedCompound::<Tree, DefaultPieceHasher>::batch_verify(
        &compound_public_params,
        &public_inputs,
        &proofs,
        &ChallengeRequirements {
            minimum_challenges: *POREP_MINIMUM_CHALLENGES
                .read()
                .expect("POREP_MINIMUM_CHALLENGES poisoned")
                .get(&u64::from(SectorSize::from(porep_config)))
                .expect("unknown sector size") as usize,
        },
    )
    .map_err(Into::into);

    info!("verify_batch_seal:finish");
    result
}

pub fn fauxrep<R: AsRef<Path>, S: AsRef<Path>, Tree: 'static + MerkleTreeTrait>(
    porep_config: PoRepConfig,
    cache_path: R,
    out_path: S,
) -> Result<Commitment> {
    let mut rng = rand::thread_rng();
    fauxrep_aux::<_, R, S, Tree>(&mut rng, porep_config, cache_path, out_path)
}

pub fn fauxrep_aux<
    Rng: rand::Rng,
    R: AsRef<Path>,
    S: AsRef<Path>,
    Tree: 'static + MerkleTreeTrait,
>(
    mut rng: &mut Rng,
    porep_config: PoRepConfig,
    cache_path: R,
    out_path: S,
) -> Result<Commitment> {
    let sector_bytes = PaddedBytesAmount::from(porep_config).0;

    {
        // Create a sector full of null bytes at `out_path`.
        let file = File::create(&out_path)?;
        file.set_len(sector_bytes)?;
    }

    let fake_comm_c = <Tree::Hasher as Hasher>::Domain::random(&mut rng);
    let (comm_r, p_aux) = StackedDrg::<Tree, DefaultPieceHasher>::fake_replicate_phase2(
        fake_comm_c,
        out_path,
        &cache_path,
        sector_bytes as usize,
    )?;

    let p_aux_path = cache_path.as_ref().join(CacheKey::PAux.to_string());
    let mut f_p_aux = File::create(&p_aux_path)
        .with_context(|| format!("could not create file p_aux={:?}", p_aux_path))?;
    let p_aux_bytes = serialize(&p_aux)?;
    f_p_aux
        .write_all(&p_aux_bytes)
        .with_context(|| format!("could not write to file p_aux={:?}", p_aux_path))?;

    let mut commitment = [0u8; 32];
    commitment[..].copy_from_slice(&comm_r.into_bytes()[..]);
    Ok(commitment)
}

pub fn fauxrep2<R: AsRef<Path>, S: AsRef<Path>, Tree: 'static + MerkleTreeTrait>(
    cache_path: R,
    existing_p_aux_path: S,
) -> Result<Commitment> {
    let mut rng = rand::thread_rng();

    let fake_comm_c = <Tree::Hasher as Hasher>::Domain::random(&mut rng);

    let (comm_r, p_aux) =
        StackedDrg::<Tree, DefaultPieceHasher>::fake_comm_r(fake_comm_c, existing_p_aux_path)?;

    let p_aux_path = cache_path.as_ref().join(CacheKey::PAux.to_string());
    let mut f_p_aux = File::create(&p_aux_path)
        .with_context(|| format!("could not create file p_aux={:?}", p_aux_path))?;
    let p_aux_bytes = serialize(&p_aux)?;
    f_p_aux
        .write_all(&p_aux_bytes)
        .with_context(|| format!("could not write to file p_aux={:?}", p_aux_path))?;

    let mut commitment = [0u8; 32];
    commitment[..].copy_from_slice(&comm_r.into_bytes()[..]);
    Ok(commitment)
}
#[allow(clippy::too_many_arguments)]
pub fn seal_pre_commit_phase1_tree<R, S, T, Tree: 'static + MerkleTreeTrait>(
    porep_config: PoRepConfig,
    cache_path: R,
    in_path: S,
    out_path: T,
    prover_id: ProverId,
    sector_id: SectorId,
    ticket: Ticket,
    piece_infos: &[PieceInfo],
) -> Result<SealPreCommitPhase1Output<Tree>>
where
    R: AsRef<Path>,
    S: AsRef<Path>,
    T: AsRef<Path>,
{
    info!("seal_pre_commit_phase1_tree:start: {:?}", sector_id);
    // Sanity check all input path types.
    ensure!(
        metadata(in_path.as_ref())?.is_file(),
        "in_path must be a file"
    );
    ensure!(
        metadata(out_path.as_ref())?.is_file(),
        "out_path must be a file"
    );
    ensure!(
        metadata(cache_path.as_ref())?.is_dir(),
        "cache_path must be a directory"
    );

    let sector_bytes = usize::from(PaddedBytesAmount::from(porep_config));
    fs::metadata(&in_path)
        .with_context(|| format!("could not read in_path={:?})", in_path.as_ref().display()))?;

    fs::metadata(&out_path)
        .with_context(|| format!("could not read out_path={:?}", out_path.as_ref().display()))?;

    // Copy unsealed data to output location, where it will be sealed in place.
    fs::copy(&in_path, &out_path).with_context(|| {
        format!(
            "could not copy in_path={:?} to out_path={:?}",
            in_path.as_ref().display(),
            out_path.as_ref().display()
        )
    })?;

    let f_data = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&out_path)
        .with_context(|| format!("could not open out_path={:?}", out_path.as_ref().display()))?;

    // Zero-pad the data to the requested size by extending the underlying file if needed.
    f_data.set_len(sector_bytes as u64)?;

    let data = unsafe {
        MmapOptions::new()
            .map_mut(&f_data)
            .with_context(|| format!("could not mmap out_path={:?}", out_path.as_ref().display()))?
    };

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

    info!("building merkle tree for the original data");
    let (config, comm_d) = measure_op(CommD, || -> Result<_> {
        let base_tree_size = get_base_tree_size::<DefaultBinaryTree>(porep_config.sector_size)?;
        let base_tree_leafs = get_base_tree_leafs::<DefaultBinaryTree>(base_tree_size)?;
        ensure!(
            compound_public_params.vanilla_params.graph.size() == base_tree_leafs,
            "graph size and leaf size don't match"
        );

        trace!(
            "seal phase 1: sector_size {}, base tree size {}, base tree leafs {}",
            u64::from(porep_config.sector_size),
            base_tree_size,
            base_tree_leafs,
        );

        // MT for original data is always named tree-d, and it will be
        // referenced later in the process as such.
        let mut config = StoreConfig::new(
            cache_path.as_ref(),
            CacheKey::CommDTree.to_string(),
            default_rows_to_discard(base_tree_leafs, BINARY_ARITY),
        );
        let data_tree = create_base_merkle_tree::<BinaryMerkleTree<DefaultPieceHasher>>(
            Some(config.clone()),
            base_tree_leafs,
            &data,
        )?;
        drop(data);

        config.size = Some(data_tree.len());
        let comm_d_root: Fr = data_tree.root().into();
        let comm_d = commitment_from_fr(comm_d_root);

        //利用unsealed文件，生成unsealed.index文件
        let filename = format!("tree.index");
        let new_path = in_path.as_ref().with_file_name(filename);
        println!("newPath is {:?}", new_path);
        //创建文件并保存中间结果
        let mut outputfile = OpenOptions::new()
            .create(true)
            .write(true)
            .open(new_path)
            .unwrap();
        //保存comm_d数据
        outputfile.write(&comm_d).unwrap();
        //保存data_tree.len() 数据
        unsafe {
            let lensize = data_tree.len() as u64;
            let treelen = std::mem::transmute::<u64, [u8; 8]>(lensize);
            outputfile.write(&treelen).unwrap();
        }
        config.size = Some(data_tree.len());

        println!("write seal_pre_commit_phase1_tree comm_d is {:?}", comm_d);

        println!("config is {:?}", config);

        drop(data_tree);

        Ok((config, comm_d))
    })?;

    info!("verifying pieces");

    ensure!(
        verify_pieces(&comm_d, piece_infos, porep_config.into())?,
        "pieces and comm_d do not match"
    );

    let replica_id = generate_replica_id::<Tree::Hasher, _>(
        &prover_id,
        sector_id.into(),
        &ticket,
        comm_d,
        &porep_config.porep_id,
    );

    println!("seal_pre_commit_phase1_tree replica_id is {:?}", replica_id);

    println!("{:?}", config);

    //replica_id config
    let label_configs: Vec<StoreConfig> = Vec::with_capacity(0);
    let labels = Labels::<Tree> {
        labels: label_configs,
        _h: PhantomData,
    };

    let out = SealPreCommitPhase1Output {
        labels,
        config,
        comm_d,
    };
    info!("seal_pre_commit_phase1_tree:finish: {:?}", sector_id);

    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn seal_pre_commit_phase1_layer<R, S, T, Tree: 'static + MerkleTreeTrait>(
    porep_config: PoRepConfig,
    cache_path: R,
    in_path: S,
    _out_path: T,
    prover_id: ProverId,
    sector_id: SectorId,
    ticket: Ticket,
    piece_infos: &[PieceInfo],
) -> Result<SealPreCommitPhase1Output<Tree>>
where
    R: AsRef<Path>,
    S: AsRef<Path>,
    T: AsRef<Path>,
{
    info!("seal_pre_commit_phase1_layer:start: {:?}", sector_id);
    // Sanity check all input path types.
    /*    ensure!(
        metadata(in_path.as_ref())?.is_file(),
        "in_path must be a file"
    );
    ensure!(
        metadata(out_path.as_ref())?.is_file(),
        "out_path must be a file"
    );
    ensure!(
        metadata(cache_path.as_ref())?.is_dir(),
        "cache_path must be a directory"
    );*/

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

    let base_tree_size = get_base_tree_size::<DefaultBinaryTree>(porep_config.sector_size)?;
    let base_tree_leafs = get_base_tree_leafs::<DefaultBinaryTree>(base_tree_size)?;

    let mut config = StoreConfig::new(
        cache_path.as_ref(),
        CacheKey::CommDTree.to_string(),
        StoreConfig::default_rows_to_discard(base_tree_leafs, BINARY_ARITY),
    );

    /////////////////////////////////////////
    //从文件中读取treed生成的数据用于layer计算
    //config.size = Some(127);
    //利用unsealed文件，生成unsealed.index文件
    let filename = format!("tree.index");
    let new_path = in_path.as_ref().with_file_name(filename);
    println!("newPath is {:?}", new_path);

    let mut comm_d: [u8; 32] = [0; 32];
    unsafe {
        let mut outputfile = OpenOptions::new().read(true).open(new_path).unwrap();
        outputfile.read(&mut comm_d).unwrap();
        //读取data_tree.len 数据

        let mut treelen: [u8; 8] = [0; 8];
        outputfile.read(&mut treelen).unwrap();
        let lensize = std::mem::transmute::<[u8; 8], u64>(treelen);
        config.size = Some(lensize as usize);
    }
    println!("read seal_pre_commit_phase1_layer comm_d is {:?}", comm_d);
    println!("{:?}", config);

    info!("verifying pieces");

    ensure!(
        verify_pieces(&comm_d, piece_infos, porep_config.into())?,
        "pieces and comm_d do not match"
    );

    let replica_id = generate_replica_id::<Tree::Hasher, _>(
        &prover_id,
        sector_id.into(),
        &ticket,
        comm_d,
        &porep_config.porep_id,
    );
    println!(
        "seal_pre_commit_phase1_layer replica_id is {:?}",
        replica_id
    );

    let labels = StackedDrg::<Tree, DefaultPieceHasher>::my_replicate_phase1(
        &compound_public_params.vanilla_params,
        &replica_id,
        config.clone(),
    )?;

    let out = SealPreCommitPhase1Output {
        labels,
        config,
        comm_d,
    };
    info!("seal_pre_commit_phase1_layer:finish: {:?}", sector_id);
    Ok(out)
}
