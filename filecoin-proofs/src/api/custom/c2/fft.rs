use bellperson::{
    bls::Engine,
    domain::{Group, Scalar},
    gpu::{GPUError, GPUResult, LockedFFTKernel},
    SynthesisError,
};
use ff::SqrtField;
use ff::{Field, PrimeField, ScalarEngine};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};
use std::convert::{TryFrom, TryInto};

pub struct MultiGpuDomain<E, G>
where
    E: ScalarEngine,
    G: Group<E>,
{
    coeffs: Vec<G>,
    exp: u32,
    omega: E::Fr,
    omegainv: E::Fr,
    geninv: E::Fr,
    minv: E::Fr,
}

struct GpuFftIn<T, E>
where
    E: Engine,
    T: Group<E>,
{
    a: Vec<T>,
    omega: E::Fr,
    log_n: u32,
    tx_res: std::sync::mpsc::Sender<(Vec<T>, GPUResult<()>)>,
}

impl<E, G> TryFrom<Vec<G>> for MultiGpuDomain<E, G>
where
    E: Engine,
    G: Group<E>,
{
    type Error = SynthesisError;
    fn try_from(mut coeffs: Vec<G>) -> Result<Self, Self::Error> {
        // Compute the size of our evaluation domain
        let mut m = 1;
        let mut exp = 0;
        while m < coeffs.len() {
            m *= 2;
            exp += 1;

            // The pairing-friendly curve may not be able to support
            // large enough (radix2) evaluation domains.
            if exp >= E::Fr::S {
                return Err(SynthesisError::PolynomialDegreeTooLarge);
            }
        }
        // Compute omega, the 2^exp primitive root of unity
        let mut omega = E::Fr::root_of_unity();
        for _ in exp..E::Fr::S {
            omega.square();
        }

        // Extend the coeffs vector with zeroes if necessary
        coeffs.resize(m, G::group_zero());

        Ok(MultiGpuDomain {
            coeffs,
            exp,
            omega,
            omegainv: omega.inverse().unwrap(),
            geninv: E::Fr::multiplicative_generator().inverse().unwrap(),
            minv: E::Fr::from_str(&format!("{}", m))
                .unwrap()
                .inverse()
                .unwrap(),
        })
    }
}

impl<E, G> MultiGpuDomain<E, G>
where
    E: Engine,
    G: Group<E>,
{
    pub fn into_coeffs(self) -> Vec<G> {
        self.coeffs
    }

    pub fn icoset_fft(&mut self, kern: &mut [LockedFFTKernel<E>]) -> GPUResult<()> {
        let geninv = self.geninv;
        self.ifft(kern)?;
        distribute_powers(&mut self.coeffs, geninv);
        Ok(())
    }

    pub fn ifft(&mut self, kern: &mut [LockedFFTKernel<E>]) -> GPUResult<()> {
        multi_gpu_fft(kern, &mut self.coeffs, &self.omegainv, self.exp)?;
        let minv = &self.minv;

        let chunk_size = self.coeffs.len() / *bellperson::multicore::NUM_CPUS;
        self.coeffs.par_chunks_mut(chunk_size).for_each(|chunk| {
            for it in chunk {
                it.group_mul_assign(minv)
            }
        });

        Ok(())
    }

    pub fn coset_fft(&mut self, kern: &mut [LockedFFTKernel<E>]) -> GPUResult<()> {
        distribute_powers(&mut self.coeffs, E::Fr::multiplicative_generator());
        multi_gpu_fft(kern, &mut self.coeffs, &self.omega, self.exp)?;
        Ok(())
    }

    pub fn mul_assign(&mut self, other: &MultiGpuDomain<E, Scalar<E>>) {
        let chunk_size = self.coeffs.len() / *bellperson::multicore::NUM_CPUS;
        self.coeffs
            .par_chunks_mut(chunk_size)
            .zip(other.coeffs.par_chunks(chunk_size))
            .for_each(|(a, b)| {
                for (a, b) in a.iter_mut().zip(b.iter()) {
                    a.group_mul_assign(&b.0);
                }
            });
    }
    pub fn sub_assign(&mut self, other: &Self) {
        let chunk_size = self.coeffs.len() / *bellperson::multicore::NUM_CPUS;
        self.coeffs
            .par_chunks_mut(chunk_size)
            .zip(other.coeffs.par_chunks(chunk_size))
            .for_each(|(a, b)| {
                for (a, b) in a.iter_mut().zip(b.iter()) {
                    a.group_sub_assign(&b);
                }
            });
    }

    pub fn divide_by_z_on_coset(&mut self) {
        let i = get_z(
            self.coeffs.len().try_into().unwrap(),
            &E::Fr::multiplicative_generator(),
        )
        .inverse()
        .unwrap();

        let chunk_size = self.coeffs.len() / *bellperson::multicore::NUM_CPUS;
        self.coeffs.par_chunks_mut(chunk_size).for_each(|chunk| {
            for it in chunk {
                it.group_mul_assign(&i);
            }
        });
    }
}

pub fn multi_gpu_fft<E, T>(
    kern: &mut [LockedFFTKernel<E>],
    a: &mut [T],
    omega: &E::Fr,
    log_n: u32,
) -> GPUResult<()>
where
    E: Engine,
    T: Group<E>,
{
    let num_gpus = kern.len();

    let log_gpus = match num_gpus {
        1 => None,
        2 => Some(1),
        4 => Some(2),
        8 => Some(3),
        _ => return Err(GPUError::GPUCountMismatch),
    };

    let log_gpus = match log_gpus {
        Some(x) => x,
        None => {
            return kern
                .first_mut()
                .unwrap()
                .with(|k| bellperson::domain::gpu_fft(k, a, omega, log_n))
        }
    };

    let log_new_n = log_n - log_gpus;
    let mut buffer2d = vec![vec![T::group_zero(); 1 << log_new_n]; num_gpus];
    let new_omega = omega.pow(&[num_gpus as u64]);

    buffer2d
        .par_iter_mut()
        .zip(kern.par_iter_mut())
        .enumerate()
        .for_each(|(gpu_index, (single_gpu_buffer, k))| {
            // Shuffle into a sub-FFT
            let omega_gpu = omega.pow(&[gpu_index as u64]);
            let omega_step = omega.pow(&[(gpu_index as u64) << log_new_n]);

            let chunk_size = single_gpu_buffer.len() / *bellperson::multicore::NUM_CPUS;
            single_gpu_buffer
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(index, chunk)| {
                    let elt_gpu = omega_gpu.pow(&[(index * chunk_size) as u64]);
                    let elt_step = omega_step.pow(&[(num_gpus * index * chunk_size) as u64]);

                    let mut elt = E::Fr::one();
                    elt.mul_assign(&elt_gpu);
                    elt.mul_assign(&elt_step);

                    for (chunk_index, sub_tmp) in chunk.iter_mut().enumerate() {
                        let i = chunk_size * index + chunk_index;
                        for s in 0..num_gpus {
                            let idx = (i + (s << log_new_n)) % (1 << log_n);
                            let mut t = a[idx];
                            t.group_mul_assign(&elt);
                            sub_tmp.group_add_assign(&t);
                            elt.mul_assign(&omega_step);
                        }
                        elt.mul_assign(&omega_gpu);
                    }
                });

            k.with(|k| bellperson::domain::gpu_fft(k, single_gpu_buffer, &new_omega, log_new_n))
                .unwrap();
        });

    let chunk_size = a.len() / *bellperson::multicore::NUM_CPUS;
    a.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(idx, a)| {
            let mut idx = idx * chunk_size;
            let mask = (1 << log_gpus) - 1;
            for a in a {
                *a = buffer2d[idx & mask][idx >> log_gpus];
                idx += 1;
            }
        });

    Ok(())
}

pub fn distribute_powers<E, T>(coeffs: &mut [T], g: E::Fr)
where
    E: Engine,
    T: Group<E>,
{
    let chunk_size = coeffs.len() / *bellperson::multicore::NUM_CPUS;
    coeffs
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(index, chunk)| {
            let mut u = g.pow(&[(index * chunk_size) as u64]);
            for it in chunk.iter_mut() {
                it.group_mul_assign(&u);
                u.mul_assign(&g);
            }
        });
}

pub fn get_z<Fr>(len: u64, tau: &Fr) -> Fr
where
    Fr: PrimeField + SqrtField,
{
    let mut tmp = tau.pow(&[len]);
    tmp.sub_assign(&Fr::one());

    tmp
}

fn gpu_service<E, T>(mut kern: LockedFFTKernel<E>, rx: crossbeam::channel::Receiver<GpuFftIn<T, E>>)
where
    E: Engine,
    T: Group<E>,
{
    while let Ok(GpuFftIn::<T, E> {
        mut a,
        omega,
        log_n,
        tx_res,
    }) = rx.recv()
    {
        let e = kern.with(|k| bellperson::domain::gpu_fft(k, &mut a, &omega, log_n));

        tx_res.send((a, e)).expect("cannot send fft result back");
    }
}

fn create_fft_handler<E, T>(gpu_count: u32) -> crossbeam::channel::Sender<GpuFftIn<T, E>>
where
    E: Engine,
    T: Group<E> + 'static,
{
    let (tx, rx) = crossbeam::channel::unbounded();
    for kern in (0..bellperson::gpu::gpu_count())
        .into_iter()
        .map(|index| LockedFFTKernel::<E>::new(0 /*unused*/, false, index))
    {
        let rx = rx.clone();
        std::thread::spawn(move || gpu_service(kern, rx));
    }
    tx
}

fn gpu_fft_queued<E, T>(
    tx: crossbeam::channel::Sender<GpuFftIn<T, E>>,
    buffer: &mut Vec<T>,
    omega: E::Fr,
    log_n: u32,
) -> GPUResult<()>
where
    E: Engine,
    T: Group<E>,
{
    let (tx_res, rx_res) = std::sync::mpsc::channel();
    let tmp_buf = std::mem::take(buffer);

    tx.send(GpuFftIn {
        a: tmp_buf,
        omega,
        log_n,
        tx_res,
    })
    .expect("cannot send fft input to gpu service");

    let (mut tmp_buf, res) = rx_res.recv().expect("cannot receive fft output");
    std::mem::swap(&mut tmp_buf, buffer);
    res
}
