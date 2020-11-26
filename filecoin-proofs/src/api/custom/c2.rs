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
use crate::SHENSUANYUN_GPU_INDEX;
use crate::SINGLE_PARTITION_PROOF_LEN;
use anyhow::ensure;
use anyhow::Context;
use anyhow::Result;
use bellperson::bls::Bls12;
use bellperson::bls::Fr;
use bellperson::domain::EvaluationDomain;
use bellperson::gpu::LockedFFTKernel;
use bellperson::gpu::LockedMultiexpKernel;
use bellperson::groth16;
use bellperson::groth16::prover::ProvingAssignment;
use bellperson::groth16::MappedParameters;
use bellperson::groth16::ParameterSource;
use bellperson::groth16::Proof;
use bellperson::multicore::Worker;
use bellperson::multiexp::multiexp;
use bellperson::multiexp::FullDensity;
use bellperson::Circuit;
use bellperson::ConstraintSystem;
use bellperson::Index;
use bellperson::SynthesisError;
use bellperson::Variable;
use ff::Field;
use ff::PrimeField;
use groupy::CurveAffine;
use groupy::CurveProjective;
use log::info;
use rand::rngs::OsRng;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use scopeguard::defer;
use std::sync::Arc;
use storage_proofs::compound_proof;
use storage_proofs::compound_proof::CompoundProof;
use storage_proofs::hasher::Domain;
use storage_proofs::merkle::MerkleTreeTrait;
use storage_proofs::multi_proof::MultiProof;
use storage_proofs::porep::stacked;
use storage_proofs::porep::stacked::StackedCircuit;
use storage_proofs::porep::stacked::StackedCompound;
use storage_proofs::porep::stacked::StackedDrg;
use storage_proofs::sector::SectorId;

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
    let gpu_index =
        super::get_gpu_index().with_context(|| format!("{:?}: cannot get gpu index", sector_id))?;

    let mut rng = OsRng;
    let C2PreparationData {
        ticket,
        params,
        circuits,
        comm_r,
        comm_d,
        seed,
    } = init(porep_config, phase1_output, sector_id)?;

    let r_s = (0..circuits.len()).map(|_| Fr::random(&mut rng)).collect();
    let s_s = (0..circuits.len()).map(|_| Fr::random(&mut rng)).collect();

    let provers: Vec<ProvingAssignment<Bls12>> =
        stage1(circuits).with_context(|| format!("{:?}: c2 cpu computation failed", sector_id))?;

    let proofs = stage2(provers, &params, r_s, s_s, gpu_index)?;

    let groth_proofs = proofs
        .into_iter()
        .map(|groth_proof| {
            let mut proof_vec = vec![];
            groth_proof.write(&mut proof_vec)?;
            let gp = groth16::Proof::<Bls12>::read(&proof_vec[..])?;
            Ok(gp)
        })
        .collect::<Result<Vec<_>>>()?;

    let proof = MultiProof::new(groth_proofs, &params.pvk);

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

    info!("seal_commit_phase2:finish: {:?}", sector_id);

    let out = SealCommitOutput { proof: buf };

    Ok(out)
}

pub fn stage1<Tree: 'static + MerkleTreeTrait>(
    circuits: Vec<StackedCircuit<'static, Tree, DefaultPieceHasher>>,
) -> Result<Vec<ProvingAssignment<Bls12>>, SynthesisError> {
    circuits
        .into_par_iter()
        .map(|circuit| -> Result<_, SynthesisError> {
            let mut prover = ProvingAssignment::new();

            prover.alloc_input(|| "", || Ok(Fr::one()))?;

            circuit.synthesize(&mut prover)?;

            for i in 0..prover.input_assignment.len() {
                prover.enforce(|| "", |lc| lc + Variable(Index::Input(i)), |lc| lc, |lc| lc);
            }

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
    let gpu_index = std::env::var(SHENSUANYUN_GPU_INDEX)
        .with_context(|| format!("{:?}: cannot get gpu index from env", sector_id))?
        .parse()
        .with_context(|| format!("{:?}: wrong gpu index", sector_id))?;

    log::info!("select gpu index: {}", gpu_index);

    defer! {
        log::info!("release gpu index: {}", gpu_index);
        crate::release_gpu_device(gpu_index);
    }

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

fn stage2(
    mut provers: Vec<ProvingAssignment<Bls12>>,
    params: &MappedParameters<Bls12>,
    r_s: Vec<Fr>,
    s_s: Vec<Fr>,
    gpu_index: usize,
) -> Result<Vec<Proof<Bls12>>> {
    let worker = Worker::new();
    let input_len = provers[0].input_assignment.len();
    let vk = params.get_vk(input_len)?;
    let n = provers[0].a.len();
    let mut log_d = 0;
    while (1 << log_d) < n {
        log_d += 1;
    }

    let mut fft_kern = Some(LockedFFTKernel::<Bls12>::new(log_d, false, gpu_index));

    let a_s = provers
        .iter_mut()
        .map(|prover| {
            let mut a =
                EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.a, Vec::new()))?;
            let mut b =
                EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.b, Vec::new()))?;
            let mut c =
                EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.c, Vec::new()))?;

            a.ifft(&worker, &mut fft_kern)?;
            a.coset_fft(&worker, &mut fft_kern)?;
            b.ifft(&worker, &mut fft_kern)?;
            b.coset_fft(&worker, &mut fft_kern)?;
            c.ifft(&worker, &mut fft_kern)?;
            c.coset_fft(&worker, &mut fft_kern)?;

            a.mul_assign(&worker, &b);
            drop(b);
            a.sub_assign(&worker, &c);
            drop(c);
            a.divide_by_z_on_coset(&worker);
            a.icoset_fft(&worker, &mut fft_kern)?;
            let mut a = a.into_coeffs();
            let a_len = a.len() - 1;
            a.truncate(a_len);

            Ok(Arc::new(
                a.into_iter().map(|s| s.0.into_repr()).collect::<Vec<_>>(),
            ))
        })
        .collect::<Result<Vec<_>, SynthesisError>>()?;

    drop(fft_kern);
    let mut multiexp_kern = Some(LockedMultiexpKernel::<Bls12>::new(log_d, false, gpu_index));

    let h_s = a_s
        .into_iter()
        .map(|a| {
            let h = multiexp(
                &worker,
                params.get_h(a.len())?,
                FullDensity,
                a,
                &mut multiexp_kern,
            );
            Ok(h)
        })
        .collect::<Result<Vec<_>, SynthesisError>>()?;

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

    let l_s = aux_assignments
        .iter()
        .map(|aux_assignment| {
            let l = multiexp(
                &worker,
                params.get_l(aux_assignment.len())?,
                FullDensity,
                aux_assignment.clone(),
                &mut multiexp_kern,
            );
            Ok(l)
        })
        .collect::<Result<Vec<_>, SynthesisError>>()?;

    let inputs = provers
        .into_iter()
        .zip(input_assignments.iter())
        .zip(aux_assignments.iter())
        .map(|((prover, input_assignment), aux_assignment)| {
            let a_aux_density_total = prover.a_aux_density.get_total_density();

            let (a_inputs_source, a_aux_source) =
                params.get_a(input_assignment.len(), a_aux_density_total)?;

            let a_inputs = multiexp(
                &worker,
                a_inputs_source,
                FullDensity,
                input_assignment.clone(),
                &mut multiexp_kern,
            );

            let a_aux = multiexp(
                &worker,
                a_aux_source,
                Arc::new(prover.a_aux_density),
                aux_assignment.clone(),
                &mut multiexp_kern,
            );

            let b_input_density = Arc::new(prover.b_input_density);
            let b_input_density_total = b_input_density.get_total_density();
            let b_aux_density = Arc::new(prover.b_aux_density);
            let b_aux_density_total = b_aux_density.get_total_density();

            let (b_g1_inputs_source, b_g1_aux_source) =
                params.get_b_g1(b_input_density_total, b_aux_density_total)?;

            let b_g1_inputs = multiexp(
                &worker,
                b_g1_inputs_source,
                b_input_density.clone(),
                input_assignment.clone(),
                &mut multiexp_kern,
            );

            let b_g1_aux = multiexp(
                &worker,
                b_g1_aux_source,
                b_aux_density.clone(),
                aux_assignment.clone(),
                &mut multiexp_kern,
            );

            let (b_g2_inputs_source, b_g2_aux_source) =
                params.get_b_g2(b_input_density_total, b_aux_density_total)?;

            let b_g2_inputs = multiexp(
                &worker,
                b_g2_inputs_source,
                b_input_density,
                input_assignment.clone(),
                &mut multiexp_kern,
            );
            let b_g2_aux = multiexp(
                &worker,
                b_g2_aux_source,
                b_aux_density,
                aux_assignment.clone(),
                &mut multiexp_kern,
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
        .collect::<Result<Vec<_>, SynthesisError>>()?;

    drop(multiexp_kern);

    let proofs = h_s
        .into_iter()
        .zip(l_s.into_iter())
        .zip(inputs.into_iter())
        .zip(r_s.into_iter())
        .zip(s_s.into_iter())
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
