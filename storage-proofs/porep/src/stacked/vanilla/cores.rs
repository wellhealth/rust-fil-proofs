use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use lazy_static::lazy_static;
use log::*;
use std::sync::Mutex;
use storage_proofs_core::settings::SETTINGS;

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
    let hwloc_info = std::process::Command::new("hwloc-info")
        .output()
        .ok()
        .and_then(|x| String::from_utf8(x.stdout).ok())
        .unwrap_or_else(|| {
            error!("cannot get info from hwloc-info");
            std::process::exit(255)
        });

    let l3_cache_regex = regex::Regex::new(r#"([[:digit:]]*)[[:space:]]+L3Cache"#).unwrap();
    let pu_regex = regex::Regex::new(r#"([[:digit:]]*)[[:space:]]+PU"#).unwrap();

    let l3_cache_match = l3_cache_regex
        .captures_iter(&hwloc_info)
        .next()
        .unwrap_or_else(|| {
            error!("invalid hwloc-info");
            std::process::exit(255)
        });

    info!("l3_cache_match[0] -> {}", &l3_cache_match[1]);

    let pu_match = pu_regex
        .captures_iter(&hwloc_info)
        .next()
        .unwrap_or_else(|| {
            error!("invalid hwloc-info");
            std::process::exit(255)
        });
    info!("pu_match[0] -> {}", &pu_match[1]);

    let cache_count: u32 = l3_cache_match[1].parse().unwrap_or_else(|_| {
        error!("l3_cache_match cannot be parsed to u32");
        std::process::exit(255)
    });

    let unit_count: u32 = pu_match[1].parse().unwrap_or_else(|_| {
        error!("pu_match cannot be parsed to u32");
        std::process::exit(255)
    });

    info!("Cache Count: {}", cache_count);
    info!("Unit Count: {}", unit_count);

    let l3_core_count = unit_count / cache_count;
    let num_cores = SETTINGS.multicore_sdr_producers + 1;
    let mut task_cores = vec![];

    for x in (0..unit_count).step_by(l3_core_count as usize) {
        let sub_res: Vec<_> = (x..x + l3_core_count).collect();
        let core_groups: Vec<_> = sub_res
            .chunks(num_cores)
            .filter(|x| x.len() == num_cores)
            .map(ToOwned::to_owned)
            .collect();
        task_cores.push(core_groups);
    }
    let task_cores = task_cores;

    if task_cores.is_empty() || task_cores[0].is_empty() {
        return Default::default();
    }

    let mut res = vec![];

    for index in 0..task_cores[0].len() {
        for sub in &task_cores {
            res.push(sub[index].clone());
        }
    }

    info!("L3 array: {:?}", res);
    res
}

pub fn unbind_core() -> Result<()> {
    let status = std::process::Command::new("hwloc-bind")
        .arg("--tid")
        .arg(gettid::gettid().to_string())
        .arg("all")
        .status()
        .context("cannot execute program hwloc-bind")?
        .code()
        .context("hwloc-bind crashed")?;

    if status != 0 {
        bail!("hwloc-bind returned {}", status);
    }
    Ok(())
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
