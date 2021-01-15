pub mod c2;


pub fn get_gpu_index() -> Option<usize> {
    std::env::var("SHENSUANYUN_GPU_INDEX").ok()?
        .parse().ok()
}
