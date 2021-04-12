use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use log::*;
use std::sync::Mutex;

use lazy_static::lazy_static;

lazy_static! {
    static ref L3_TOPOLOGY: Mutex<Vec<Vec<u32>>> = Mutex::new(get_l3_topo());
}
#[derive(Debug, Default)]
pub struct L3Index(Vec<u32>);

impl L3Index {
    pub fn get_main(&self) -> u32 {
        self.0[0]
    }
    pub fn get_rest(&self) -> &[u32] {
        &self.0[1..]
    }
}

impl Drop for L3Index {
    fn drop(&mut self) {
        let mut v = Default::default();
        std::mem::swap(&mut v, &mut self.0);
        L3_TOPOLOGY.lock().unwrap().push(v);
    }
}
pub fn get_l3_index() -> Option<L3Index> {
    let mut topo = L3_TOPOLOGY.lock().unwrap();
    if topo.is_empty() {
        None
    } else {
        Some(L3Index(topo.remove(0)))
    }
}
pub fn get_l3_topo() -> Vec<Vec<u32>> {
    let cache_count: u32 = std::env::var("L3_CACHE_COUNT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap();

    let unit_count: u32 = std::env::var("PU_COUNT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap();

    info!("Cache Count: {}", cache_count);
    info!("Unit Count: {}", unit_count);
    let l3_core_count = unit_count / cache_count;
    let res = (0..unit_count)
        .step_by(l3_core_count as usize)
        .map(|x| (x..x + l3_core_count).collect())
        .collect();

    info!("L3 array: {:?}", res);
    res
}

pub fn bind_core(index: u32) -> Result<()> {
    let status = std::process::Command::new("hwloc-bind")
        .arg("--tid")
        .arg(gettid::gettid().to_string())
        .arg(format!("pu:{}", index))
        .status()
        .context("cannot execute program hwloc-bind")?
        .code()
        .context("hwloc-bind crashed")?;

    if status != 0 {
        bail!("hwloc-bind returned {}", status);
    }
    Ok(())
}
