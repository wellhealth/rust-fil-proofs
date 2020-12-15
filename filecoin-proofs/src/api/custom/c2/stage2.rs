#![allow(dead_code)]
use std::sync::Arc;

use bellperson::{
    bls::FrRepr,
    multicore::Worker,
    multiexp::{multiexp_full, FullDensity},
};

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


pub fn whole() {
    
}
