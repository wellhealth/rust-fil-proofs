use std::convert::TryInto;

use bellperson::{
    bls::{Bls12, Fr},
    domain::{Group, Scalar},
};
use ff::{Field, PrimeField};
use rust_gpu_tools::{
    call_kernel,
    opencl::{Parameter, Program},
};

use crate::constants::{MAX_LOG2_LOCAL_WORK_SIZE, MAX_LOG2_RADIX};
pub type GpuBuffer = rust_gpu_tools::opencl::Buffer<Scalar<Bls12>>;
use anyhow::{bail, Context, Result};

pub const FFT: &str = include_str!("fft.c");
pub struct FftParams {
    pub log_n: u32,
    pub minv: Fr,
}

/// # Purpose of Existence
/// This struct has the layout identical to [`Fr`].
///
/// The reason why this struct to exist is that
/// we cannot just implement [`Parameter`] for [`Fr`] itself,
/// so a transparent wrapper is required for data to be passwd into the GPU.
#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct FrGpu(Fr);

impl Default for FrGpu {
    fn default() -> Self {
        FrGpu(Fr::zero())
    }
}

unsafe impl Parameter for FrGpu {}

pub fn create_fft_program(gpu_index: usize) -> Result<Program> {
    let devices = rust_gpu_tools::opencl::Device::all();
    if devices.is_empty() {
        bail!("gpu掉了");
    }

    // Select the specified device for FFT
    let device = devices[gpu_index].clone();

    rust_gpu_tools::opencl::Program::from_opencl(device, FFT)
        .with_context(|| format!("cannot generate program with gpu: {}", gpu_index))
}

pub fn multiply_by_field(fft: &Program, src: &GpuBuffer, scalar: Fr) -> Result<()> {
    let kernel = fft.create_kernel("mul_by_field", src.length(), None);
    call_kernel!(kernel, src, FrGpu(scalar)).context("cannot call kernel for mul_by_field")?;
    Ok(())
}

pub fn distribute_powers(fft: &Program, src: &GpuBuffer, g: Fr) -> Result<()> {
    let kernel = fft.create_kernel("distribute_powers", src.length(), None);

    call_kernel!(kernel, src, FrGpu(g)).context("cannot call kernel for distribute_powers")?;
    Ok(())
}

pub fn merge_mul(fft: &Program, buf_mut: &GpuBuffer, buf: &GpuBuffer) -> Result<()> {
    let kernel = fft.create_kernel("merge_mul", buf_mut.length(), None);
    call_kernel!(kernel, buf_mut, buf).context("cannot call kernel for distribute_powers")?;

    Ok(())
}

pub fn merge_sub(fft: &Program, buf_mut: &GpuBuffer, buf: &GpuBuffer) -> Result<()> {
    let kernel = fft.create_kernel("merge_sub", buf_mut.length(), None);
    call_kernel!(kernel, buf_mut, buf).context("cannot call kernel for distribute_powers")?;

    Ok(())
}

pub fn coset_fft(
    fft: &Program,
    src: &mut GpuBuffer,
    tmp: &mut GpuBuffer,
    params: &FftParams,
    omega: Fr,
) -> Result<()> {
    distribute_powers(fft, src, Fr::multiplicative_generator())?;
    gpu_fft(fft, params, src, tmp, omega)?;
    Ok(())
}

pub fn icoset_fft(
    fft: &Program,
    src: &mut GpuBuffer,
    tmp: &mut GpuBuffer,
    params: &FftParams,
    omega: Fr,
) -> Result<()> {
    ifft(fft, src, tmp, params, omega)?;
    distribute_powers(fft, src, Fr::multiplicative_generator().inverse().unwrap())?;
    Ok(())
}

pub fn ifft(
    fft: &Program,
    src: &mut GpuBuffer,
    tmp: &mut GpuBuffer,
    params: &FftParams,
    omega: Fr,
) -> Result<()> {
    gpu_fft(fft, params, src, tmp, omega.inverse().unwrap())?;
    multiply_by_field(fft, src, params.minv)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn radix_fft_round(
    fft: &Program,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    pq_buffer: &GpuBuffer,
    omegas_buffer: &GpuBuffer,
    log_n: u32,
    log_p: u32,
    deg: u32,
    max_deg: u32,
) -> Result<()> {
    let n = 1u32 << log_n;
    let local_work_size = 1 << std::cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
    let global_work_size = (n >> deg) * local_work_size;
    let kernel = fft.create_kernel(
        "radix_fft",
        global_work_size as usize,
        Some(local_work_size as usize),
    );

    call_kernel!(
        kernel,
        src,
        dst,
        pq_buffer,
        omegas_buffer,
        rust_gpu_tools::opencl::LocalBuffer::<Fr>::new(1 << deg),
        n,
        log_p,
        deg,
        max_deg
    )?;

    Ok(())
}

fn gpu_fft(
    fft: &Program,
    params: &FftParams,
    buf: &mut GpuBuffer,
    tmp: &mut GpuBuffer,
    omega: Fr,
) -> Result<()> {
    let max_deg = std::cmp::min(MAX_LOG2_RADIX, params.log_n);

    let pq_buffer = setup_pq(fft, &omega, params.log_n).context("cannot setup pq_buffer")?;
    let omegas_buffer = setup_omega(fft, &omega).context("cannot setup omegas_buffer")?;

    let mut log_p = 0u32;
    while log_p < params.log_n {
        let deg = std::cmp::min(max_deg, params.log_n - log_p);
        radix_fft_round(
            fft,
            buf,
            tmp,
            &pq_buffer,
            &omegas_buffer,
            params.log_n,
            log_p,
            deg,
            max_deg,
        )?;
        log_p += deg;
        std::mem::swap(buf, tmp);
    }

    Ok(())
}

pub fn divide_by_z_on_coset(fft: &Program, coeffs: &GpuBuffer) -> Result<()> {
    let i = get_z(
        coeffs.length().try_into().unwrap(),
        Fr::multiplicative_generator(),
    )
    .inverse()
    .context("cannot inverse after get_z")?;

    multiply_by_field(fft, coeffs, i)?;

    Ok(())
}

pub fn get_z(len: u64, tau: Fr) -> Fr {
    let mut tmp = tau.pow(&[len]);
    tmp.sub_assign(&Fr::one());

    tmp
}

pub fn setup_pq(fft: &Program, omega: &Fr, log_n: u32) -> Result<GpuBuffer> {
    let n = 2usize.pow(log_n);

    let max_deg = std::cmp::min(MAX_LOG2_RADIX, log_n);
    let mut pq_buffer = fft
        .create_buffer(1 << MAX_LOG2_RADIX >> 1)
        .context("cannot create pq buffer")?;
    let mut pq = vec![Scalar(Fr::zero()); 1 << max_deg >> 1];
    let twiddle = Scalar(omega.pow([(n >> max_deg) as u64]));
    pq[0] = Scalar(Fr::one());
    if max_deg > 1 {
        pq[1] = twiddle;
        for i in 2..(1 << max_deg >> 1) {
            pq[i] = pq[i - 1];
            pq[i].group_mul_assign(&twiddle.0);
        }
    }
    pq_buffer.write_from(0, &pq)?;
    Ok(pq_buffer)
}

pub fn setup_omega(fft: &Program, omega: &Fr) -> Result<GpuBuffer> {
    let mut omegas = [Scalar(Fr::zero()); 32];
    omegas[0] = Scalar(*omega);
    for i in 1..32 {
        let m: Fr = omegas[i - 1].0;
        let m = m.pow(&[2u64]);
        omegas[i] = Scalar(m);
    }

    let mut omegas_buffer = fft
        .create_buffer(omegas.len())
        .context("cannot create omegas buffer")?;
    omegas_buffer
        .write_from(0, &omegas)
        .context("cannot write omegas to GPU")?;

    Ok(omegas_buffer)
}

pub fn generate_params(length: usize) -> Result<FftParams> {
    let mut m = 1;
    let mut log_n = 0;
    while m < length {
        m *= 2;
        log_n += 1;

        if log_n >= Fr::S {
            bail!("Polynomial degree too large");
        }
    }

    let m = m;
    let log_n = log_n;
    let params = FftParams {
        log_n,
        minv: Fr::from_str(&format!("{}", m)).unwrap().inverse().unwrap(),
    };

    Ok(params)
}
