use std::io::Write;

use anyhow::{ensure, Result};
use bellperson::{
    bls::{Bls12, Fr},
    domain::Scalar,
};
use ff::{Field, PrimeField};
use gpu::GpuBuffer;
use log::info;
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

pub mod constants;
pub mod geforce_3080;
pub mod geforce_3090;
pub mod gpu;
pub type Program = rust_gpu_tools::opencl::Program;
#[derive(Default, Debug)]
pub struct SectorId(u64);
impl From<u64> for SectorId {
    fn from(n: u64) -> Self {
        Self(n)
    }
}

fn bytes_from_fr(src: &Fr) -> [u8; 32] {
    let node = src.into_repr();
    let mut buf = [0u8; 32];
    let mut cursor = std::io::Cursor::new(&mut buf[..]);
    cursor.write_all(&node.0[0].to_be_bytes()).unwrap();
    cursor.write_all(&node.0[1].to_be_bytes()).unwrap();
    cursor.write_all(&node.0[2].to_be_bytes()).unwrap();
    cursor.write_all(&node.0[3].to_be_bytes()).unwrap();

    buf
}

fn bytes_from_scalar(src: &Scalar<Bls12>) -> [u8; 32] {
    bytes_from_fr(&src.0)
}

pub fn hash_gpu_buf(data: &GpuBuffer) -> String {
    let mut mem = vec![Scalar(Fr::zero()); data.length()];

    let mut v = vec![0u8; mem.len() * 32];

    data.read_into(0, &mut mem).unwrap();

    mem.par_iter()
        .zip(v.par_chunks_mut(32))
        .for_each(|(node, bytes)| {
            bytes.copy_from_slice(&bytes_from_scalar(node));
        });

    sha256::digest_bytes(&v)
}

pub fn hash_scalar(data: &[Scalar<Bls12>]) -> String {
    let mut v = vec![0u8; data.len() * 32];
    data.par_iter()
        .zip(v.par_chunks_mut(32))
        .for_each(|(node, bytes)| {
            bytes.copy_from_slice(&bytes_from_scalar(node));
        });

    sha256::digest_bytes(&v)
}

pub fn fft_3090(
    fft: &Program,
    mut a: Vec<Scalar<Bls12>>,
    mut b: Vec<Scalar<Bls12>>,
    mut c: Vec<Scalar<Bls12>>,
    sector_id: SectorId,
) -> Result<Vec<Scalar<Bls12>>> {
    let len = a.len();
    ensure!(
        [a.len(), b.len(), c.len()].iter().all(|&x| x == len),
        "a, b, c length mismatch for gpu_fft"
    );
    let len = a.len();
    info!("{:?}: data retrieved from file", sector_id);
    let params = gpu::generate_params(len)?;
    let fft_n = 2usize.pow(params.log_n);

    info!("{:?}: params generated", sector_id);
    a.resize_with(fft_n, || Scalar(Fr::zero()));
    b.resize_with(fft_n, || Scalar(Fr::zero()));
    c.resize_with(fft_n, || Scalar(Fr::zero()));

    info!("{:?}: resized", sector_id);
    let omega = (params.log_n..Fr::S).fold(Fr::root_of_unity(), |mut x, _| {
        x.square();
        x
    });

    geforce_3090::fft(&fft, params, &mut a, b, c, omega, sector_id)?;
    Ok(a)
}
