
pub mod tree_c;
pub mod tree_r;
use lazy_static::lazy_static;

lazy_static! {
	pub static ref GPU_INDEX: usize = select_gpu_device();
}

pub fn select_gpu_device() -> usize {
    std::env::args()
        .nth(3)
        .and_then(|x|x.parse().ok())
        .unwrap_or(0)
}
