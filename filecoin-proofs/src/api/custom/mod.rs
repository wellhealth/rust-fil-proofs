pub mod c2;

pub fn get_gpu_index() -> Option<usize> {
    std::env::args().nth(3).unwrap_or_default().parse().ok()
}
