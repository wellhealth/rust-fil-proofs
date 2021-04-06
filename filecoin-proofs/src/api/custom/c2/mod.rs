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
use bellperson::groth16::Proof;
use bellperson::{
    bls::{Bls12, Fr, FrRepr},
    domain::{EvaluationDomain, Scalar},
    gpu::{LockedFFTKernel, LockedMultiexpKernel},
    groth16::prover::ProvingAssignment,
    groth16::{self, MappedParameters},
    multicore::Worker,
    multiexp::{multiexp, multiexp_full, multiexp_precompute, FullDensity, QueryDensity},
    ConstraintSystem, Index, SynthesisError, Variable,
};
use bellperson::{groth16::ParameterSource, Circuit};
use ff::{Field, PrimeField};
use groupy::CurveAffine;
use groupy::CurveProjective;
use lazy_static::lazy_static;
use log::info;
use rand::rngs::OsRng;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use std::sync::{
    mpsc::{channel, sync_channel},
    Arc,
};
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
        (0..circuits.len())
            .map(|_| (Fr::random(&mut rng), Fr::random(&mut rng)))
            .unzip()
    };

    let mut provers: Vec<ProvingAssignment<Bls12>> = c2_stage1(circuits)
        .with_context(|| format!("{:?}: c2 cpu computation failed", sector_id))?;

    info!("{:?}: c2 stage1 finished", sector_id);
    let proofs = stage2::run(&mut provers, &params, r_s, s_s, gpu_index)?;
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
    circuits
        .into_par_iter()
        .enumerate()
        .map(|(index, circuit)| -> Result<_, SynthesisError> {
            let mut prover = ProvingAssignment::new();

            prover.alloc_input(|| "", || Ok(Fr::one()))?;

            circuit.synthesize(&mut prover)?;
            // stage1::circuit_synthesize(circuit, &mut prover)?;

            for i in 0..prover.input_assignment.len() {
                prover.enforce(|| "", |lc| lc + Variable(Index::Input(i)), |lc| lc, |lc| lc);
            }

            info!("{:?}: done prover: {}", *SECTOR_ID, index + 1);
            Ok(prover)
        })
        .collect::<Result<Vec<_>, SynthesisError>>()
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
                Default::default(),
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

#[allow(dead_code)]
fn c2_stage2(
    mut provers: Vec<ProvingAssignment<Bls12>>,
    params: &MappedParameters<Bls12>,
    r_s: Vec<Fr>,
    s_s: Vec<Fr>,
    gpu_index: usize,
    sector_id: SectorId,
) -> Result<Vec<Proof<Bls12>>> {
    let worker = Worker::new();
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

    let mut fft_kern = Some(LockedFFTKernel::<Bls12>::new(log_d, false, gpu_index));

    let (tx_1, rx_1) =
        sync_channel::<(EvaluationDomain<Bls12, Scalar<Bls12>>, _, _)>(provers.len());
    let (tx_2, rx_2) = sync_channel(provers.len());
    let (tx_3, rx_3) = sync_channel::<EvaluationDomain<Bls12, Scalar<Bls12>>>(provers.len());
    let (tx_hl, rx_hl) = channel();
    let (tx_ab, rx_ab) = channel();
    let (tx_g2, rx_g2) = channel();

    let mut a_s: Vec<Arc<Vec<FrRepr>>> = vec![];
    crossbeam::scope({
        let a_s = &mut a_s;
        let provers_len = provers.len();
        let provers = &mut provers;
        let worker = &worker;
        let fft_kern = &mut fft_kern;
        let tx_hl = tx_hl.clone();

        move |s| {
            s.spawn(move |_| {
                for (index, (mut a, b, c)) in rx_1.iter().enumerate() {
                    info!("{:?}: merge a, b, c. index: {}", sector_id, index);
                    a.mul_assign(&worker, &b);
                    drop(b);
                    a.sub_assign(&worker, &c);
                    drop(c);
                    a.divide_by_z_on_coset(&worker);
                    tx_2.send((index, a)).unwrap();
                }
            });

            s.spawn(move |_| {
                *a_s = rx_3
                    .iter()
                    .enumerate()
                    .map(|(index, a)| {
                        let mut a = a.into_coeffs();
                        let a_len = a.len() - 1;
                        a.truncate(a_len);
                        (index, a)
                    })
                    .map(|(_index, a)| {
                        Arc::new(a.into_iter().map(|s| s.0.into()).collect::<Vec<FrRepr>>())
                    })
                    .collect::<Vec<_>>();
                assert_eq!(a_s.len(), provers_len);
            });

            let len = provers.len();
            let mut spawn_index = if len <= 2 { 0 } else { len - 2 };
            for (index, prover) in provers.iter_mut().enumerate() {
                if index >= spawn_index {
                    spawn_index = len + 1;
                    let tx_hl = tx_hl.clone();
                    s.spawn(move |_| tx_hl.send(params.get_h(0)).unwrap());
                }
                let mut a =
                    EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.a, Vec::new()))
                        .unwrap();
                let mut b =
                    EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.b, Vec::new()))
                        .unwrap();
                let mut c =
                    EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.c, Vec::new()))
                        .unwrap();

                b.ifft(&worker, fft_kern).unwrap();
                b.coset_fft(&worker, fft_kern).unwrap();

                a.ifft(&worker, fft_kern).unwrap();
                a.coset_fft(&worker, fft_kern).unwrap();

                c.ifft(&worker, fft_kern).unwrap();
                c.coset_fft(&worker, fft_kern).unwrap();
                tx_1.send((a, b, c)).unwrap();
                for (index, mut a) in rx_2.try_iter() {
                    info!("{:?}, doing icoset_fft for a, index: {}", sector_id, index);
                    a.icoset_fft(&worker, fft_kern).unwrap();
                    tx_3.send(a).unwrap();
                }
            }
            drop(tx_1);
            for (index, mut a) in rx_2.iter() {
                info!("{:?}, doing icoset_fft for a, index: {}", sector_id, index);
                a.icoset_fft(&worker, fft_kern).unwrap();
                tx_3.send(a).unwrap();
            }
        }
    })
    .unwrap();
    info!("{:?}, done FFT", sector_id);

    let param_h = rx_hl.recv().unwrap()?;

    drop(fft_kern);

    let mut multiexp_kern = Some(LockedMultiexpKernel::<Bls12>::new(log_d, false, gpu_index));

    let h_s = crossbeam::scope({
        let worker = &worker;
        let multiexp_kern = &mut multiexp_kern;
        move |s| {
            s.spawn(move |_| {
                tx_hl.send(params.get_l(0)).unwrap();
            });
            a_s.into_iter()
                .map(|a| multiexp_full(worker, param_h.clone(), FullDensity, a, multiexp_kern))
                .collect::<Vec<_>>()
        }
    })
    .unwrap();

    info!("{:?}, done h_s", sector_id);

    let input_assignments = provers
        .par_iter_mut()
        .map(|prover| {
            let input_assignment = std::mem::replace(&mut prover.input_assignment, Vec::new());
            Arc::new(
                input_assignment
                    .into_iter()
                    .map(|s| s.into_repr())
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    info!("{:?}, done input_assignments", sector_id);

    let aux_assignments = provers
        .par_iter_mut()
        .map(|prover| {
            let aux_assignment = std::mem::replace(&mut prover.aux_assignment, Vec::new());
            Arc::new(
                aux_assignment
                    .into_iter()
                    .map(|s| s.into_repr())
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();

    info!("{:?}, done aux_assignments", sector_id);

    let mut l_s = Ok(vec![]);
    let param_l = rx_hl.recv().unwrap()?;
    drop(rx_hl);
    crossbeam::scope({
        let worker = &worker;
        let multiexp_kern = &mut multiexp_kern;
        let l_s = &mut l_s;
        let aux_assignments = &aux_assignments;
        move |s| {
            s.spawn(move |_| {
                let a = params.get_a(0, 0);
                tx_ab.send(a).unwrap();
                let g1 = params.get_b_g1(0, 0);
                tx_ab.send(g1).unwrap();
                let g2 = params.get_b_g2(0, 0);
                tx_g2.send(g2).unwrap();
            });

            *l_s = aux_assignments
                .iter()
                .map({
                    let param_l = param_l.clone();
                    move |aux_assignment| {
                        Ok(multiexp_full(
                            worker,
                            param_l.clone(),
                            FullDensity,
                            aux_assignment.clone(),
                            multiexp_kern,
                        ))
                    }
                })
                .collect::<Result<Vec<_>, SynthesisError>>();
        }
    })
    .unwrap();
    let l_s = l_s?;

    info!("{:?}, done l_s", sector_id);

    let param_a = rx_ab.recv().unwrap()?;
    let param_bg1 = rx_ab.recv().unwrap()?;
    let param_bg2 = rx_g2.recv().unwrap()?;
    let inputs = provers
        .into_iter()
        .zip(input_assignments.iter())
        .zip(aux_assignments.iter())
        .map(|((prover, input_assignment), aux_assignment)| {
            let a_inputs_source = param_a.0.clone();
            let a_aux_source = ((param_a.1).0.clone(), input_assignment.len());
            let b_input_density = Arc::new(prover.b_input_density);

            let a_aux_density = Arc::new(prover.a_aux_density);
            let b_aux_density = Arc::new(prover.b_aux_density);
            let (aux_tx, aux_rx) = channel();
            crossbeam::scope(|s| {
                s.spawn({
                    let aux_assignment = &aux_assignment;
                    let a_aux_density = a_aux_density.clone();
                    let b_aux_density = b_aux_density.clone();
                    let tx = aux_tx.clone();
                    move |_| {
                        tx.send(cpu_compute_exp(
                            aux_assignment.as_slice(),
                            a_aux_density.clone(),
                        ))
                        .unwrap();

                        tx.send(cpu_compute_exp(
                            aux_assignment.as_slice(),
                            b_aux_density.clone(),
                        ))
                        .unwrap();
                    }
                });

                let a_inputs = multiexp_full(
                    &worker,
                    a_inputs_source,
                    FullDensity,
                    input_assignment.clone(),
                    &mut multiexp_kern,
                );

                let a_aux = multiexp_precompute(
                    &worker,
                    a_aux_source,
                    a_aux_density,
                    aux_assignment.clone(),
                    &mut multiexp_kern,
                    Arc::new(aux_rx.recv().unwrap()),
                );

                let b_input_density_total = b_input_density.get_total_density();
                let b_g1_inputs_source = param_bg1.0.clone();
                let b_g1_aux_source = ((param_bg1.1).0.clone(), b_input_density_total);

                let b_g1_inputs = multiexp(
                    &worker,
                    b_g1_inputs_source,
                    b_input_density.clone(),
                    input_assignment.clone(),
                    &mut multiexp_kern,
                );

                let b_exps = Arc::new(aux_rx.recv().unwrap());

                let b_g1_aux = multiexp_precompute(
                    &worker,
                    b_g1_aux_source,
                    b_aux_density.clone(),
                    aux_assignment.clone(),
                    &mut multiexp_kern,
                    b_exps.clone(),
                );

                let b_g2_inputs_source = param_bg2.0.clone();
                let b_g2_aux_source = ((param_bg2.1).0.clone(), b_input_density_total);

                let b_g2_inputs = multiexp(
                    &worker,
                    b_g2_inputs_source,
                    b_input_density,
                    input_assignment.clone(),
                    &mut multiexp_kern,
                );
                let b_g2_aux = multiexp_precompute(
                    &worker,
                    b_g2_aux_source,
                    b_aux_density,
                    aux_assignment.clone(),
                    &mut multiexp_kern,
                    b_exps,
                );

                Ok((
                    a_inputs,
                    a_aux,
                    b_g1_inputs,
                    b_g1_aux,
                    b_g2_inputs,
                    b_g2_aux,
                ))
            })
            .unwrap()
        })
        .collect::<Result<Vec<_>, SynthesisError>>()?;

    drop(param_a);
    drop(param_bg1);
    drop(param_bg2);

    info!("{:?}, done inputs", sector_id);
    drop(multiexp_kern);

    let vk = &params.vk;
    let proofs = h_s
        .into_par_iter()
        .zip(l_s.into_par_iter())
        .zip(inputs.into_par_iter())
        .zip(r_s.into_par_iter())
        .zip(s_s.into_par_iter())
        .map(
            |(
                (((h, l), (a_inputs, a_aux, b_g1_inputs, b_g1_aux, b_g2_inputs, b_g2_aux)), r),
                s,
            )| {
                if vk.delta_g1.is_zero() || vk.delta_g2.is_zero() {
                    // If this element is zero, someone is trying to perform a
                    // subversion-CRS attack.
                    return Err(SynthesisError::UnexpectedIdentity);
                }

                let mut g_a = vk.delta_g1.mul(r);
                g_a.add_assign_mixed(&vk.alpha_g1);
                let mut g_b = vk.delta_g2.mul(s);
                g_b.add_assign_mixed(&vk.beta_g2);
                let mut g_c;
                {
                    let mut rs = r;
                    rs.mul_assign(&s);

                    g_c = vk.delta_g1.mul(rs);
                    g_c.add_assign(&vk.alpha_g1.mul(s));
                    g_c.add_assign(&vk.beta_g1.mul(r));
                }
                let mut a_answer = a_inputs.wait()?;
                a_answer.add_assign(&a_aux.wait()?);
                g_a.add_assign(&a_answer);
                a_answer.mul_assign(s);
                g_c.add_assign(&a_answer);

                let mut b1_answer = b_g1_inputs.wait()?;
                b1_answer.add_assign(&b_g1_aux.wait()?);
                let mut b2_answer = b_g2_inputs.wait()?;
                b2_answer.add_assign(&b_g2_aux.wait()?);

                g_b.add_assign(&b2_answer);
                b1_answer.mul_assign(r);
                g_c.add_assign(&b1_answer);
                g_c.add_assign(&h.wait()?);
                g_c.add_assign(&l.wait()?);

                Ok(Proof {
                    a: g_a.into_affine(),
                    b: g_b.into_affine(),
                    c: g_c.into_affine(),
                })
            },
        )
        .collect::<Result<Vec<_>, SynthesisError>>()?;

    Ok(proofs)
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
