use anyhow::bail;
use anyhow::Context;
use anyhow::Result;

pub fn get_l3_index() -> Option<Vec<u32>> {
    Some(serde_json::from_str(&std::env::args().nth(3)?).ok()?)
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
