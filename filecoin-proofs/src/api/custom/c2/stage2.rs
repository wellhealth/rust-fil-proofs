#![allow(dead_code)]
use crate::custom::c2::SECTOR_ID;
use anyhow::{ensure, Context, Result};
use bellperson::{
    bls::{Bls12, Fr, FrRepr},
    domain::{EvaluationDomain, Scalar},
    gpu::LockedFFTKernel,
    groth16::{MappedParameters, ProvingAssignment},
    multicore::Worker,
    multiexp::{multiexp_full, FullDensity},
};
use log::info;
use std::sync::{
    mpsc::{channel, sync_channel, SyncSender},
    Arc,
};
use storage_proofs::sector::SectorId;

pub fn calculate_h_cpu(
    worker: &Worker,
    param_h: (Arc<Vec<bellperson::bls::G1Affine>>, usize),
    a: Arc<Vec<FrRepr>>,
) -> bellperson::multicore::Waiter<
    std::result::Result<bellperson::bls::G1Projective, bellperson::SynthesisError>,
> {
    multiexp_full(worker, param_h.clone(), FullDensity, a, &mut None)
}

fn multiexp_full_cpu(
    pool: &Worker,
    param: (Arc<Vec<bellperson::bls::G1Affine>>, usize),
    exponents: Arc<Vec<FrRepr>>,
) {
    let c = if exponents.len() < 32 {
        3u32
    } else {
        (f64::from(exponents.len() as u32)).ln().ceil() as u32
    };
}

pub fn whole(
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
    let log_d = compute_log_d(n);
    let (tx, rx) = sync_channel(1);
    let (tx_data, rx_data) = sync_channel(provers.len());
    fft(provers, log_d, gpu_index, tx, tx_data, 2);

    Ok(())
}

fn fft(
    provers: &mut [ProvingAssignment<Bls12>],
    log_d: usize,
    gpu_index: usize,
    h_start: SyncSender<()>,
    h_data: SyncSender<std::sync::Arc<std::vec::Vec<bellperson::bls::FrRepr>>>,
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
            tx2.send((index, a)).unwrap();
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
            .for_each(|x| h_data.send(x).unwrap());
    });

    for (index, prover) in provers.iter_mut().enumerate() {
        if index == h_index {
            h_start
                .send(())
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

        for (index, mut a) in rx2.try_iter() {
            fft_icoset(index, a, &mut fft_kern, &tx3);
        }
    }
    drop(tx1);

    for (index, mut a) in rx2.try_iter() {
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
