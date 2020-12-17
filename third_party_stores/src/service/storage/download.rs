#[derive(Debug)]
pub enum RangeReader {
    QiNiu(crate::qiniu::service::storage::download::RangeReader),
    Ali(crate::sds::service::storage::download::RangeReader),
}

impl RangeReader {
    pub fn read_last_bytes(&self, length: usize) -> std::io::Result<(u64, Vec<u8>)> {
        match self {
            RangeReader::Ali(x) => {x.read_last_bytes(length)}
            RangeReader::QiNiu(x) =>  {x.read_last_bytes(length)}
        }
    }

    pub fn download_bytes(&self) -> std::io::Result<Vec<u8>>  {
        match self {
            RangeReader::Ali(x) => {x.download_bytes()}
            RangeReader::QiNiu(x) =>  {x.download_bytes()}
        }
    }
 }

pub fn qiniu_is_enable() -> bool {
    crate::qiniu::service::storage::download::qiniu_is_enable()
}

pub fn sds_is_enable() -> bool {
    crate::sds::service::storage::download::sds_is_enable()
}

pub fn reader_from_env(path: &str) -> Option<RangeReader> {
    if !qiniu_is_enable() &&  !sds_is_enable(){
        return None;
    }
   if qiniu_is_enable(){
        crate::qiniu::service::storage::download::reader_from_env(path).map(|x|RangeReader::QiNiu(x))
    }
    else{
        crate::sds::service::storage::download::reader_from_env(path).map(|x|RangeReader::Ali(x))
    }
}