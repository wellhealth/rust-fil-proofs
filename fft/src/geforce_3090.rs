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

pub fn fft(
    fft: &Program,
    params: FftParams,
    buffer_a: &mut [Scalar<Bls12>],
    buffer_b: Vec<Scalar<Bls12>>,
    buffer_c: Vec<Scalar<Bls12>>,
    omega: Fr,
    sector_id: SectorId,
) -> Result<()> {
    info!("{:?}: enter fft", sector_id);
    let (mut a, mut b, mut c) = move_buffers_to_gpu_memory(fft, buffer_a, buffer_b, buffer_c)?;
    info!("{:?}: buffer moved", sector_id);

    let mut tmp = fft
        .create_buffer::<Scalar<Bls12>>(buffer_a.len())
        .context("cannot create tmp GPU buffer")?;
    info!("{:?}: created tmp buffer", sector_id);

    crate::gpu::ifft(fft, &mut a, &mut tmp, &params, omega)?;
    crate::gpu::coset_fft(fft, &mut a, &mut tmp, &params, omega)?;
    info!("{:?}: finished fft for a", sector_id);

    crate::gpu::ifft(fft, &mut b, &mut tmp, &params, omega)?;
    crate::gpu::coset_fft(fft, &mut b, &mut tmp, &params, omega)?;
    info!("{:?}: finished fft for b", sector_id);

    crate::gpu::ifft(fft, &mut c, &mut tmp, &params, omega)?;
    crate::gpu::coset_fft(fft, &mut c, &mut tmp, &params, omega)?;
    info!("{:?}: finished fft for c", sector_id);

    crate::gpu::merge_mul(fft, &a, &b)?;
    crate::gpu::merge_sub(fft, &a, &c)?;
    info!("{:?}: merge done", sector_id);
    drop(b);
    drop(c);

    crate::gpu::divide_by_z_on_coset(fft, &a)?;
    crate::gpu::icoset_fft(fft, &mut a, &mut tmp, &params, omega)?;

    a.read_into(0, buffer_a)
        .context("cannot read into buffer_a from GPU memory")?;

    Ok(())
}

fn move_buffers_to_gpu_memory(
    fft: &Program,
    buffer_a: &[Scalar<Bls12>],
    buffer_b: Vec<Scalar<Bls12>>,
    buffer_c: Vec<Scalar<Bls12>>,
) -> Result<(GpuBuffer, GpuBuffer, GpuBuffer)> {
    let mut a = fft
        .create_buffer(buffer_a.len())
        .context("cannot create gpu_buf_a")?;
    a.write_from(0, buffer_a)
        .context("cannot write from buffer_a into gpu")?;
    let mut b = fft
        .create_buffer(buffer_b.len())
        .context("cannot create gpu_buf_b")?;

    b.write_from(0, &buffer_b)
        .context("cannot write from buffer_b into gpu")?;
    drop(buffer_b);

    let mut c = fft
        .create_buffer(buffer_c.len())
        .context("cannot create gpu_buf_c")?;

    c.write_from(0, &buffer_c)
        .context("cannot write from buffer_b into gpu")?;
    drop(buffer_c);
    Ok((a, b, c))
}
