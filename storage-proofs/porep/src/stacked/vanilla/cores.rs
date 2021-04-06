use log::*;
use std::sync::Mutex;

use hwloc2::{CpuBindFlags, ObjectType, Topology, TopologyObject};
use lazy_static::lazy_static;

lazy_static! {
    static ref TOPOLOGY: Mutex<Topology> = Mutex::new(Topology::new().unwrap());
    static ref L3_TOPOLOGY: Mutex<Vec<Vec<u32>>> = Mutex::new(split_by_task());
    static ref TOPOLOGY_INDEX: std::sync::atomic::AtomicUsize = Default::default();
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
    L3_TOPOLOGY.lock().unwrap().pop().map(L3Index)
}

pub fn split_by_task() -> Vec<Vec<u32>> {
    let topo = TOPOLOGY.lock().expect("cannot lock TOPOLOGY");

    let l3_array = topo
        .objects_with_type(&ObjectType::L3Cache)
        .expect("cannot get L3 type");
    info!("L3 array {:?}", l3_array);
    let v = l3_array
        .into_iter()
        .map(get_pu_from_l3)
        .filter(|x| !x.is_empty())
        .collect::<Vec<_>>();

    if v[0].len() >= 8 {
        let mut res = vec![];
        for it in &v {
            let mut sub = vec![];
            sub.extend_from_slice(&it[..4]);
            res.push(sub);
        }
        for it in &v {
            let mut sub = vec![];
            sub.extend_from_slice(&it[4..]);
            res.push(sub);
        }
        res
    } else {
        v
    }
}

pub fn get_pu_from_l3(obj: &TopologyObject) -> Vec<u32> {
    let mut ret = vec![];
    obj.children().into_iter().for_each(|it| {
        it.children().into_iter().for_each(|it| {
            it.children().into_iter().for_each(|it| {
                it.children()
                    .into_iter()
                    .for_each(|it| ret.push(it.logical_index()))
            })
        })
    });
    ret
}

pub fn bind_core(index: u32) {
    let mut topology = TOPOLOGY.lock().expect("cannot lock TOPOLOGY");
    let mut v = topology
        .objects_with_type(&ObjectType::PU)
        .expect("cannot get PU objects");
    v.sort_by_key(|x| x.logical_index());
    let cpuset = v[index as usize].cpuset().unwrap();

    topology
        .set_cpubind_for_thread(get_thread_id(), cpuset, CpuBindFlags::CPUBIND_THREAD)
        .unwrap();
}

#[cfg(not(target_os = "windows"))]
pub type ThreadId = libc::pthread_t;

#[cfg(target_os = "windows")]
pub type ThreadId = winapi::winnt::HANDLE;

/// Helper method to get the thread id through libc, with current rust stable (1.5.0) its not
/// possible otherwise I think.
#[cfg(not(target_os = "windows"))]
fn get_thread_id() -> ThreadId {
    unsafe { libc::pthread_self() }
}

#[cfg(target_os = "windows")]
fn get_thread_id() -> ThreadId {
    unsafe { kernel32::GetCurrentThread() }
}
