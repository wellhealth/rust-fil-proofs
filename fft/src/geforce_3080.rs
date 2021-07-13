use anyhow::{Context, Result};
use bellperson::{
    bls::{Bls12, Fr},
    domain::Scalar,
};
use log::info;
use rust_gpu_tools::opencl::Program;

use crate::{
    gpu::{FftParams, GpuBuffer},
    SectorId,
};

/// Allocate 2 buffers used for FFT computation for GPUs with lower memory
///
/// `fft`: the opencl program used for fft
///
/// `len`: buffer size (must be 2 ^ n, or there will be consequences)
fn allocate_gpu_buffer(fft: &Program, len: usize) -> Result<(GpuBuffer, GpuBuffer)> {
    let b1 = fft
        .create_buffer::<Scalar<Bls12>>(len)
        .context("cannot create tmp GPU buffer")?;

    let b2 = fft
        .create_buffer::<Scalar<Bls12>>(len)
        .context("cannot create tmp GPU buffer")?;
    Ok((b1, b2))
}

pub fn fft(
    fft: &Program,
    params: FftParams,
    buffer_a: &mut [Scalar<Bls12>],
    mut buffer_b: Vec<Scalar<Bls12>>,
    mut buffer_c: Vec<Scalar<Bls12>>,
    omega: Fr,
    sector_id: SectorId,
) -> Result<()> {
    let (mut buf_src, mut buf_tmp) = allocate_gpu_buffer(fft, buffer_a.len())?;

    let t = std::time::Instant::now();
    buf_src
        .write_from(0, &buffer_b)
        .context("cannot write buf_src from buffer_a")?;
    info!("{:?}: GPU buffer write: {:?}", sector_id, t.elapsed());

    crate::gpu::ifft(fft, &mut buf_src, &mut buf_tmp, &params, omega)?;
    crate::gpu::coset_fft(fft, &mut buf_src, &mut buf_tmp, &params, omega)?;

    buf_src
        .read_into(0, &mut buffer_b)
        .context("cannot read into buffer_b")?;

    let t = std::time::Instant::now();
    buf_src
        .write_from(0, &buffer_c)
        .context("cannot write buf_src from buffer_a")?;
    info!("{:?}: GPU buffer write: {:?}", sector_id, t.elapsed());

    crate::gpu::ifft(fft, &mut buf_src, &mut buf_tmp, &params, omega)?;
    crate::gpu::coset_fft(fft, &mut buf_src, &mut buf_tmp, &params, omega)?;

    let t = std::time::Instant::now();
    buf_src
        .read_into(0, &mut buffer_c)
        .context("cannot read into buffer_b")?;
    info!("{:?}: GPU buffer write: {:?}", sector_id, t.elapsed());

    let t = std::time::Instant::now();
    buf_src
        .write_from(0, buffer_a)
        .context("cannot write buf_src from buffer_a")?;
    info!("{:?}: GPU buffer write: {:?}", sector_id, t.elapsed());

    crate::gpu::ifft(fft, &mut buf_src, &mut buf_tmp, &params, omega)?;
    crate::gpu::coset_fft(fft, &mut buf_src, &mut buf_tmp, &params, omega)?;

    let t = std::time::Instant::now();
    buf_tmp
        .write_from(0, &buffer_b)
        .context("cannot write buffer_b to buf_tmp")?;
    info!("{:?}: GPU buffer write: {:?}", sector_id, t.elapsed());

    crate::gpu::merge_mul(fft, &buf_src, &buf_tmp)?;
    drop(buffer_b);

    let t = std::time::Instant::now();
    buf_tmp
        .write_from(0, &buffer_c)
        .context("cannot write buffer_b to buf_tmp")?;
    info!("{:?}: GPU buffer write: {:?}", sector_id, t.elapsed());

    crate::gpu::merge_sub(fft, &buf_src, &buf_tmp)?;
    drop(buffer_c);

    crate::gpu::divide_by_z_on_coset(fft, &buf_src)?;
    crate::gpu::icoset_fft(fft, &mut buf_src, &mut buf_tmp, &params, omega)?;

    buf_src
        .read_into(0, buffer_a)
        .context::<_>("cannot read into buffer_a from GPU memory")?;

    Ok(())
}
