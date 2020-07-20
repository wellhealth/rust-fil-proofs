#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use sha2raw::Sha256;
use storage_proofs_core::{
    error::Result,
    hasher::Hasher,
    util::{data_at_node_offset, NODE_SIZE},
};

use super::{cache::ParentCache, graph::StackedBucketGraph};

/*pub fn my_create_label<H: Hasher>(
    graph: &StackedBucketGraph<H>,
    store: DiskStore<<Tree::Hasher as Hasher>::Domain>,
    mut cache: Option<&mut ParentCache>,
    replica_id: &H::Domain,
    layer_labels: &mut [u8],
    layer_index: usize,
    node_start: usize,
    cache_node_size: usize,
) -> Result<()> {


    //////////////////////////////////////////////////////////////////
    //从DiskStore读取文件到内存
    let mut tmpLayer = vec![0u8; node_start * NODE_SIZE];
    let tmpBuf = &mut tmpLayer[..];
    store.store_read_into(0,has_calc_node,tmpBuf);
    //从tmpLayer临时内存中读取cache_node_size 长度的数据到 labels_buffer
    if(has_calc_node > 0)
    {
        graph.get_base_label_cache(cache,tmpBuf,labels_buffer,has_calc_node*NODE_SIZE,cache_node_size);
    }
    drop(tmpLayer);
    /////////////////////////////////////////

    for node in 0..cache_node_size {

        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 32];

        buffer[..4].copy_from_slice(&(layer_index as u32).to_be_bytes());
        buffer[4..12].copy_from_slice(&(node as u64).to_be_bytes());
        hasher.input(&[AsRef::<[u8]>::as_ref(replica_id), &buffer[..]][..]);
        // hash parents for all non 0 nodes
        let hash = if (node_start+node) > 0 { //第一个节点数据特殊处理
            // prefetch previous node, which is always a parent
            let prev = &layer_labels[(node - 1) * NODE_SIZE..node * NODE_SIZE];
            unsafe {
                _mm_prefetch(prev.as_ptr() as *const i8, _MM_HINT_T0);
            }

            graph.copy_parents_data(node as u32, &*layer_labels, hasher, cache)?


        } else {
            hasher.finish()
        };


        // store the newly generated key
        let start = data_at_node_offset(node);
        let end = start + NODE_SIZE;
        layer_labels[start..end].copy_from_slice(&hash[..]);

        // strip last two bits, to ensure result is in Fr.
        layer_labels[end - 1] &= 0b0011_1111;

    }

    Ok(())
}*/

pub fn create_label<H: Hasher>(
    graph: &StackedBucketGraph<H>,
    cache: Option<&mut ParentCache>,
    replica_id: &H::Domain,
    layer_labels: &mut [u8],
    layer_index: usize,
    node: usize,
) -> Result<()> {
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 32];

    buffer[..4].copy_from_slice(&(layer_index as u32).to_be_bytes());
    buffer[4..12].copy_from_slice(&(node as u64).to_be_bytes());
    hasher.input(&[AsRef::<[u8]>::as_ref(replica_id), &buffer[..]][..]);

    // hash parents for all non 0 nodes
    let hash = if node > 0 {
        // prefetch previous node, which is always a parent
        let prev = &layer_labels[(node - 1) * NODE_SIZE..node * NODE_SIZE];
        unsafe {
            _mm_prefetch(prev.as_ptr() as *const i8, _MM_HINT_T0);
        }

        graph.copy_parents_data(node as u32, &*layer_labels, hasher, cache)?
    } else {
        hasher.finish()
    };

    // store the newly generated key
    let start = data_at_node_offset(node);
    let end = start + NODE_SIZE;
    layer_labels[start..end].copy_from_slice(&hash[..]);

    // strip last two bits, to ensure result is in Fr.
    layer_labels[end - 1] &= 0b0011_1111;

    Ok(())
}

pub fn create_label_exp<H: Hasher>(
    graph: &StackedBucketGraph<H>,
    cache: Option<&mut ParentCache>,
    replica_id: &H::Domain,
    exp_parents_data: &[u8],
    layer_labels: &mut [u8],
    layer_index: usize,
    node: usize,
) -> Result<()> {
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 32];

    buffer[0..4].copy_from_slice(&(layer_index as u32).to_be_bytes());
    buffer[4..12].copy_from_slice(&(node as u64).to_be_bytes());
    hasher.input(&[AsRef::<[u8]>::as_ref(replica_id), &buffer[..]][..]);

    // hash parents for all non 0 nodes
    let hash = if node > 0 {
        // prefetch previous node, which is always a parent
        let prev = &layer_labels[(node - 1) * NODE_SIZE..node * NODE_SIZE];
        unsafe {
            _mm_prefetch(prev.as_ptr() as *const i8, _MM_HINT_T0);
        }

        graph.copy_parents_data_exp(node as u32, &*layer_labels, exp_parents_data, hasher, cache)?
    } else {
        hasher.finish()
    };

    // store the newly generated key
    let start = data_at_node_offset(node);
    let end = start + NODE_SIZE;
    layer_labels[start..end].copy_from_slice(&hash[..]);

    // strip last two bits, to ensure result is in Fr.
    layer_labels[end - 1] &= 0b0011_1111;

    Ok(())
}
