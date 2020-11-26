pub mod c2;


use crate::SHENSUANYUN_GPU_INDEX;

pub fn get_gpu_index() -> Option<usize> {
    std::env::var(SHENSUANYUN_GPU_INDEX).ok()?
        .parse().ok()
}
