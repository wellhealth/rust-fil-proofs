#![allow(dead_code)]
#![allow(unused_variables)]
use crate::custom::c2::SECTOR_ID;
use anyhow::{ensure, Context, Result};
use bellperson::{bls::G2Projective, multiexp::multiexp};
use bellperson::{
    bls::{Bls12, Fr, FrRepr, G1Affine, G1Projective, G2Affine},
    domain::{EvaluationDomain, Scalar},
    gpu::{LockedFFTKernel, LockedMultiexpKernel},
    groth16::{MappedParameters, ParameterSource, ProvingAssignment},
    multicore::{Waiter, Worker},
    multiexp::{multiexp_full, multiexp_precompute, FullDensity},
    SynthesisError,
};
use ff::PrimeField;
use log::info;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use std::sync::{
    mpsc::{channel, sync_channel, Receiver, SyncSender},
    Arc,
};
use storage_proofs::sector::SectorId;

use super::cpu_compute_exp;

struct InputParams {
    a: ((Arc<Vec<G1Affine>>, usize), (Arc<Vec<G1Affine>>, usize)),
    g1: ((Arc<Vec<G1Affine>>, usize), (Arc<Vec<G1Affine>>, usize)),
    g2: ((Arc<Vec<G2Affine>>, usize), (Arc<Vec<G2Affine>>, usize)),
}

pub fn calculate_h_cpu(
    worker: &Worker,
    param_h: (Arc<Vec<bellperson::bls::G1Affine>>, usize),
    a: Arc<Vec<FrRepr>>,
) -> bellperson::multicore::Waiter<
    std::result::Result<bellperson::bls::G1Projective, bellperson::SynthesisError>,
> {
    multiexp_full(worker, param_h.clone(), FullDensity, a, &mut None)
}

pub fn run(
    provers: &mut [ProvingAssignment<Bls12>],
    params: &MappedParameters<Bls12>,
    r_s: Vec<Fr>,
    s_s: Vec<Fr>,
    gpu_index: usize,
    sector_id: SectorId,
) -> Result<()> {
    ensure!(!provers.is_empty(), "provers cannot be empty");
    let n = provers.first().unwrap().a.len();

    ensure!(
        provers.iter().all(|x| x.a.len() == n),
        "only equaly sized circuits are supported"
    );
    let len = provers.len();
    let log_d = compute_log_d(n);
    let (tx_param_h, rx_param_h) = sync_channel(1);
    let (tx_fft, rx_fft) = sync_channel(len);
    let (tx_h, rx_h) = sync_channel(len);

    std::thread::spawn(move || {
        hs(rx_fft, rx_param_h, tx_h);
    });
    // EvaluationDomain<Bls12, Scalar<Bls12>>

    fft(provers, params, log_d, gpu_index, tx_fft, tx_param_h, 1)?;
    let input_assignments = collect_input_assignments(provers);
    let aux_assignments: Vec<Arc<Vec<FrRepr>>> = collect_aux_assignments(provers);
    let param_l: (Arc<Vec<G1Affine>>, usize) = params.get_l(0)?;

    let mut multiexp_kern: Option<LockedMultiexpKernel<Bls12>> =
        Some(LockedMultiexpKernel::<Bls12>::new(log_d, false, gpu_index));

    let (input_tx, input_rx) = sync_channel(1);
    let ls = crossbeam::scope({
        let aux_assignments = &aux_assignments;
        let kern = &mut multiexp_kern;

        move |s| {
            s.spawn(move |_| {
                let _ = input_tx.send(calculate_input_param(params));
            });
            ls(aux_assignments, param_l, kern)
        }
    })
    .unwrap();

    let input_param = input_rx
        .recv()
        .unwrap()
        .with_context(|| format!("{:?}: cannot get input params", *SECTOR_ID))?;

    let inputs = get_inputs(
        provers,
        &input_assignments,
        &aux_assignments,
        input_param,
        &mut multiexp_kern,
    )?;
    Ok(())
}

fn calculate_input_param(params: &MappedParameters<Bls12>) -> Result<InputParams, SynthesisError> {
    Ok(InputParams {
        a: params.get_a(0, 0)?,
        g1: params.get_b_g1(0, 0)?,
        g2: params.get_b_g2(0, 0)?,
    })
}

fn ls(
    aux_assignments: &[Arc<Vec<FrRepr>>],
    param_l: (Arc<Vec<G1Affine>>, usize),
    kern: &mut Option<LockedMultiexpKernel<Bls12>>,
) -> Vec<Waiter<Result<G1Projective, SynthesisError>>> {
    aux_assignments
        .iter()
        .map({
            |aux_assignment| {
                multiexp_full(
                    &Worker::new(),
                    param_l.clone(),
                    FullDensity,
                    aux_assignment.clone(),
                    kern,
                )
            }
        })
        .collect()
}

fn hs(
    rx_start: Receiver<(Arc<Vec<G1Affine>>, usize)>,
    rx_input: Receiver<Arc<Vec<FrRepr>>>,
    tx_output: SyncSender<Waiter<Result<G1Projective, SynthesisError>>>,
) {
    let param_h = rx_start.recv().expect("rx_start fails to recv");

    for (index, a) in rx_input.iter().enumerate() {
        let param_h = param_h.clone();
        let tx_output = tx_output.clone();
        std::thread::spawn(move || {
            let _ = tx_output.send(multiexp_full(
                &Worker::new(),
                param_h.clone(),
                FullDensity,
                a,
                &mut None,
            ));
            info!("h:{} finished", index + 1);
        });
    }
}

fn fft(
    provers: &mut [ProvingAssignment<Bls12>],
    params: &MappedParameters<Bls12>,
    log_d: usize,
    gpu_index: usize,
    h_start: SyncSender<(Arc<Vec<G1Affine>>, usize)>,
    h_data: SyncSender<Arc<Vec<FrRepr>>>,
    h_index: usize,
) -> Result<()> {
    let mut fft_kern = Some(LockedFFTKernel::<Bls12>::new(log_d, false, gpu_index));

    let (tx1, rx1) = channel::<(EvaluationDomain<Bls12, Scalar<Bls12>>, _, _)>();
    let (tx2, rx2) = channel();
    let (tx3, rx3) = sync_channel::<EvaluationDomain<Bls12, Scalar<Bls12>>>(provers.len());

    std::thread::spawn(move || {
        let worker = Worker::new();
        for (index, (mut a, b, c)) in rx1.iter().enumerate() {
            info!("{:?}: merge a, b, c. index: {}", *SECTOR_ID, index);
            a.mul_assign(&worker, &b);
            drop(b);
            a.sub_assign(&worker, &c);
            drop(c);
            a.divide_by_z_on_coset(&worker);
            let _ = tx2.send((index, a));
        }
    });

    std::thread::spawn(move || {
        rx3.iter()
            .enumerate()
            .map(|(index, a)| {
                let mut a = a.into_coeffs();
                let a_len = a.len() - 1;
                a.truncate(a_len);
                (index, a)
            })
            .map(|(_index, a)| Arc::new(a.into_iter().map(|s| s.0.into()).collect::<Vec<FrRepr>>()))
            .for_each(|x| h_data.send(x).unwrap_or_default());
    });

    for (index, prover) in provers.iter_mut().enumerate() {
        if index == h_index {
            h_start
                .send(params.get_h(0)?)
                .with_context(|| format!("{:?}: cannot send to h_start", *SECTOR_ID))?;
        }
        let mut a =
            EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.a, Vec::new())).unwrap();
        let mut b =
            EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.b, Vec::new())).unwrap();
        let mut c =
            EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.c, Vec::new())).unwrap();
        let worker = Worker::new();
        b.ifft(&worker, &mut fft_kern).unwrap();
        b.coset_fft(&worker, &mut fft_kern).unwrap();

        a.ifft(&worker, &mut fft_kern).unwrap();
        a.coset_fft(&worker, &mut fft_kern).unwrap();

        c.ifft(&worker, &mut fft_kern).unwrap();
        c.coset_fft(&worker, &mut fft_kern).unwrap();

        tx1.send((a, b, c))
            .with_context(|| format!("{:?}: cannot send abc to tx1", *SECTOR_ID))?;

        for (index, a) in rx2.try_iter() {
            fft_icoset(index, a, &mut fft_kern, &tx3);
        }
    }
    drop(tx1);

    for (index, a) in rx2.try_iter() {
        fft_icoset(index, a, &mut fft_kern, &tx3);
    }

    Ok(())
}

fn fft_icoset(
    index: usize,
    mut a: EvaluationDomain<Bls12, Scalar<Bls12>>,
    kern: &mut Option<LockedFFTKernel<Bls12>>,
    tx: &SyncSender<EvaluationDomain<Bls12, Scalar<Bls12>>>,
) {
    info!("{:?}, doing icoset_fft for a, index: {}", *SECTOR_ID, index);
    a.icoset_fft(&Worker::new(), kern).unwrap();
    tx.send(a).unwrap();
}

fn compute_log_d(n: usize) -> usize {
    let mut log_d = 0;
    while (1 << log_d) < n {
        log_d += 1;
    }
    log_d
}

fn collect_input_assignments(provers: &mut [ProvingAssignment<Bls12>]) -> Vec<Arc<Vec<FrRepr>>> {
    provers
        .par_iter_mut()
        .map(|prover| {
            let assignments = std::mem::replace(&mut prover.input_assignment, Vec::new());
            Arc::new(assignments.into_iter().map(|s| s.into_repr()).collect())
        })
        .collect()
}
fn collect_aux_assignments(provers: &mut [ProvingAssignment<Bls12>]) -> Vec<Arc<Vec<FrRepr>>> {
    provers
        .par_iter_mut()
        .map(|prover| {
            let assignments = std::mem::replace(&mut prover.aux_assignment, Vec::new());
            Arc::new(assignments.into_iter().map(|s| s.into_repr()).collect())
        })
        .collect()
}

fn get_inputs(
    provers: &mut [ProvingAssignment<Bls12>],
    input_assignments: &[Arc<Vec<FrRepr>>],
    aux_assignments: &[Arc<Vec<FrRepr>>],
    x: InputParams,
    kern: &mut Option<LockedMultiexpKernel<Bls12>>,
) -> std::result::Result<
    std::vec::Vec<(
        Waiter<Result<G1Projective, SynthesisError>>,
        Waiter<Result<G1Projective, SynthesisError>>,
        Waiter<Result<G1Projective, SynthesisError>>,
        Waiter<Result<G1Projective, SynthesisError>>,
        Waiter<Result<G2Projective, SynthesisError>>,
        Waiter<Result<G2Projective, SynthesisError>>,
    )>,
    SynthesisError,
> {
    let worker = Worker::new();
    let InputParams {
        a: param_a,
        g1: param_bg1,
        g2: param_bg2,
    } = x;

    provers
        .into_iter()
        .zip(input_assignments.iter())
        .zip(aux_assignments.iter())
        .map(|((prover, input_assignment), aux_assignment)| {
            let a_inputs_source = param_a.0.clone();
            let a_aux_source = ((param_a.1).0.clone(), input_assignment.len());
            let b_input_density = Arc::new(std::mem::replace(
                &mut prover.b_input_density,
                Default::default(),
            ));

            let a_aux_density = Arc::new(std::mem::replace(
                &mut prover.a_aux_density,
                Default::default(),
            ));

            let b_aux_density = Arc::new(std::mem::replace(
                &mut prover.b_aux_density,
                Default::default(),
            ));
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
                    kern,
                );

                let a_aux = multiexp_precompute(
                    &worker,
                    a_aux_source,
                    a_aux_density,
                    aux_assignment.clone(),
                    kern,
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
                    kern,
                );

                let b_exps = Arc::new(aux_rx.recv().unwrap());

                let b_g1_aux = multiexp_precompute(
                    &worker,
                    b_g1_aux_source,
                    b_aux_density.clone(),
                    aux_assignment.clone(),
                    kern,
                    b_exps.clone(),
                );

                let b_g2_inputs_source = param_bg2.0.clone();
                let b_g2_aux_source = ((param_bg2.1).0.clone(), b_input_density_total);

                let b_g2_inputs = multiexp(
                    &worker,
                    b_g2_inputs_source,
                    b_input_density,
                    input_assignment.clone(),
                    kern,
                );
                let b_g2_aux = multiexp_precompute(
                    &worker,
                    b_g2_aux_source,
                    b_aux_density,
                    aux_assignment.clone(),
                    kern,
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
        .collect::<Result<Vec<_>, SynthesisError>>()
}
