use anyhow::anyhow;
use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use bellperson::bls::Fr;
use ff::Field;
use ff::PrimeField;
use ff::PrimeFieldRepr;
use filecoin_hashers::PoseidonArity;
use generic_array::GenericArray;
use lazy_static::lazy_static;
use log::{error, info};
use merkletree::store::StoreConfig;
use neptune::{batch_hasher::Batcher, BatchHasher};
use rayon::prelude::*;
use scopeguard::defer;
use std::fs::OpenOptions;
use std::io::Read;
use std::{
    fs::File,
    path::{Path, PathBuf},
};
use storage_proofs_core::{
    merkle::{create_disk_tree, DiskTree, MerkleTreeTrait},
    settings,
};

use crate::stacked::hash::hash_single_column;

use neptune::{
    batch_hasher::{BatcherType::CustomGPU, GPUSelector},
    proteus::gpu::CLBatchHasher,
};

struct Channels {
    txs: Vec<std::sync::mpsc::Sender<(usize, Vec<Fr>)>>,
    rxs: Vec<std::sync::mpsc::Receiver<(usize, Vec<Fr>)>>,
}

pub struct ColumnData<ColumnArity>
where
    ColumnArity: 'static + PoseidonArity,
{
    columns: Vec<GenericArray<Fr, ColumnArity>>,
    node_index: usize,
    config_index: usize,
}

lazy_static! {
    static ref CHANNEL_CAPACITY: usize = 16;
}

pub fn run<Tree, ColumnArity, TreeArity>(
    nodes_count: usize,
    configs: &[StoreConfig],
    labels: &[(PathBuf, String)],
    replica_path: &Path,
) -> Result<DiskTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>>
where
    Tree: MerkleTreeTrait,
    ColumnArity: PoseidonArity + 'static,
    TreeArity: PoseidonArity,
{
    let gpu_index = *super::GPU_INDEX;
    let (file_data_tx, file_data_rx) = crossbeam::channel::bounded(*CHANNEL_CAPACITY);
    let (column_tx, column_rx) = crossbeam::channel::bounded(*CHANNEL_CAPACITY);
    let mut files = open_column_data_file(labels)?;
    let Channels { txs, rxs } = channels(configs.len());
    let rxs = rxs;
    let txs = txs;

    let scope_result = crossbeam::scope(move |s| {
        s.spawn(move |_| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(11)
                .build()
                .expect("failed to build thread pool for p2 file reading")
                .install(move || {
                    if let Err(e) = read_data_from_file(
                        &mut files,
                        configs.len(),
                        nodes_count,
                        settings::SETTINGS.max_gpu_column_batch_size as usize,
                        file_data_tx,
                        replica_path,
                    ) {
                        error!("{:?}: p2 file reading error: {:?}", replica_path, e);
                    }
                })
        });

        s.spawn(move |_| {
            generate_columns::<ColumnArity>(file_data_rx.clone(), column_tx, replica_path);
            info!("{:?}: debuglog: generate column finished", replica_path);
        });

        s.spawn({
            let column_rx = column_rx.clone();
            let txs = txs.clone();
            move |_| {
                if let Err(e) = gpu_build::<ColumnArity, TreeArity>(
                    nodes_count,
                    gpu_index,
                    column_rx,
                    &txs,
                    replica_path,
                ) {
                    info!("generate_tree_c_gpu error: {:?}", e);
                }
            }
        });

        s.spawn({
            let column_rx = column_rx.clone();
            let txs = txs.clone();
            move |_| {
                if let Err(e) = gpu_build::<ColumnArity, TreeArity>(
                    nodes_count,
                    gpu_index,
                    column_rx,
                    &txs,
                    replica_path,
                ) {
                    info!("generate_tree_c_gpu error: {:?}", e);
                }
            }
        });

        s.spawn(move |_| {
            if let Err(e) = gpu_build::<ColumnArity, TreeArity>(
                nodes_count,
                gpu_index,
                column_rx,
                &txs,
                replica_path,
            ) {
                info!("generate_tree_c_gpu error: {:?}", e);
            }
        });

        collect_and_persist_tree_c::<TreeArity>(
            &rxs,
            &configs
                .iter()
                .map(|x| StoreConfig::data_path(&x.path, &x.id))
                .collect::<Vec<_>>(),
            settings::SETTINGS.max_gpu_column_batch_size as usize,
            nodes_count,
            replica_path,
            gpu_index,
        )?;

        Ok(())
    });

    match scope_result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => {
            info!("{:?} tree-c error: {:?}", replica_path, e);
            Err(e)
        }
        Err(e) => {
            info!("{:?} tree-c panic error: {:?}", replica_path, e);
            Err(anyhow::anyhow!(format!("tree-c panic with error: {:?}", e)))
        }
    }?;

    create_disk_tree::<DiskTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>>(
        configs[0].size.expect("config size failure"),
        &configs,
    )
}

fn read_data_from_file(
    files: &mut [File],
    config_count: usize,
    node_count: usize,
    batch_size: usize,
    chan: crossbeam::channel::Sender<(usize, usize, Vec<Vec<Fr>>)>,
    replica_path: &Path,
) -> Result<()> {
    for config_index in 0..config_count {
        for node_index in (0..node_count).step_by(batch_size) {
            let rest_count = node_count - node_index;
            let chunked_node_count = std::cmp::min(rest_count, batch_size);

            let t = std::time::Instant::now();
            let data = read_single_batch(files, chunked_node_count)?;
            info!(
                "{:?}: file read: tree-c:{}, node:{} ({:?})",
                replica_path,
                config_index + 1,
                node_index,
                t.elapsed()
            );
            chan.send((config_index, node_index, data))
                .with_context(|| format!("{:?}: cannot send file data", replica_path))?;
        }
    }
    info!("{:?}: debuglog: read_data_from_file finished", replica_path);

    Ok(())
}

pub fn generate_tree_c_cpu<ColumnArity, TreeArity>(
    column_rx: crossbeam::channel::Receiver<ColumnData<ColumnArity>>,
    hashed_tx: &[std::sync::mpsc::Sender<(usize, Vec<Fr>)>],
    replica_path: &Path,
    start_condition: Option<std::sync::mpsc::Receiver<()>>,
) where
    ColumnArity: 'static + PoseidonArity,
    TreeArity: PoseidonArity,
{
    if let Some(x) = start_condition {
        x.recv().expect("cannot receive from start_condition");
    }
    info!("{:?}: start using CPU to hash columns", replica_path);

    for ColumnData {
        columns,
        node_index,
        config_index,
    } in column_rx.iter()
    {
        let result = cpu_build_column(&columns);
        info!(
            "{:?}: cpu built: tree-c:{}, node:{}",
            replica_path,
            config_index + 1,
            node_index
        );

        hashed_tx[config_index]
            .send((node_index, result))
            .expect("cannot send to hashed_tx");
    }

    info!("{:?}: debuglog: cpu tree-c finished", replica_path);
}

fn read_single_batch(files: &mut [File], chunked_node_count: usize) -> Result<Vec<Vec<Fr>>> {
    const FR_SIZE: usize = std::mem::size_of::<Fr>();
    let byte_count = chunked_node_count * FR_SIZE;

    let bytes_into_fr = |bytes: &[u8]| -> Result<Fr> {
        let mut fr_repr = <<Fr as PrimeField>::Repr as Default>::default();
        fr_repr
            .read_le(bytes)
            .context("cannot convert bytes to Fr")?;

        Fr::from_repr(fr_repr).context("cannot convert fr_repr to fr")
    };

    files
        .par_iter_mut()
        .map(|x| {
            let mut buf_bytes = vec![0u8; byte_count];
            x.read_exact(&mut buf_bytes)
                .with_context(|| format!("error occurred when reading file [{:?}]", x))?;

            buf_bytes
                .chunks(std::mem::size_of::<Fr>())
                .map(bytes_into_fr)
                .collect::<Result<Vec<_>>>()
                .with_context(|| format!("cannot convert bytes to Fr for file: {:?})", x))
        })
        .collect()
}

fn generate_columns<ColumnArity>(
    rx: crossbeam::channel::Receiver<(usize, usize, Vec<Vec<Fr>>)>,
    tx: crossbeam::channel::Sender<ColumnData<ColumnArity>>,
    replica_path: &Path,
) where
    ColumnArity: PoseidonArity,
{
    use generic_array::sequence::GenericSequence;
    for (config_index, node_index, data) in rx.iter() {
        let mut columns: Vec<GenericArray<Fr, ColumnArity>> =
            vec![GenericArray::<Fr, ColumnArity>::generate(|_i: usize| Fr::zero()); data[0].len()];

        columns.iter_mut().enumerate().for_each(|(index, column)| {
            for layer_index in 0..ColumnArity::to_usize() {
                column[layer_index] = data[layer_index][index];
            }
        });

        info!(
            "{:?}:column generated: tree-c:{}, node:{}",
            replica_path,
            config_index + 1,
            node_index
        );
        tx.send(ColumnData {
            columns,
            node_index,
            config_index,
        })
        .expect("failed to send column data");
    }
}

fn open_column_data_file(paths: &[(PathBuf, String)]) -> Result<Vec<File>> {
    paths
        .iter()
        .map(|x| StoreConfig::data_path(&x.0, &x.1))
        .map(|x| {
            File::open(&x).with_context(|| format!("cannot open layer file [{:?}] for tree-c", x))
        })
        .collect()
}

fn channels(config_count: usize) -> Channels {
    let mut txs = Vec::with_capacity(config_count);
    let mut rxs = Vec::with_capacity(config_count);
    for _ in 0..config_count {
        let (tx, rx) = std::sync::mpsc::channel();
        txs.push(tx);
        rxs.push(rx);
    }
    Channels { txs, rxs }
}

pub fn cpu_build_column<A: PoseidonArity>(data: &[GenericArray<Fr, A>]) -> Vec<Fr> {
    data.par_iter().map(|x| hash_single_column(x)).collect()
}

pub fn gpu_build_column<A, B>(batcher: &mut B, data: &[GenericArray<Fr, A>]) -> Result<Vec<Fr>>
where
    A: PoseidonArity,
    B: BatchHasher<A>,
{
    batcher.hash(data).context("batcher.hash failed")
}

pub fn gpu_build<ColumnArity, TreeArity>(
    nodes_count: usize,
    gpu_index: usize,
    column_rx: crossbeam::channel::Receiver<ColumnData<ColumnArity>>,
    hashed_tx: &[std::sync::mpsc::Sender<(usize, Vec<Fr>)>],
    replica_path: &Path,
) -> Result<()>
where
    ColumnArity: 'static + PoseidonArity,
    TreeArity: PoseidonArity,
{
    let mut batcher: CLBatchHasher<ColumnArity> =
        match Batcher::<ColumnArity>::new(&CustomGPU(GPUSelector::Index(gpu_index)), nodes_count)
            .with_context(|| format!("failed to create tree_c batcher {}", gpu_index))?
        {
            Batcher::OpenCL(x) => x,
            Batcher::CPU(_) => panic!("neptune bug, batcher should be GPU"),
        };

    for ColumnData {
        columns,
        node_index,
        config_index,
    } in column_rx.iter()
    {
        let t = std::time::Instant::now();
        let result = gpu_build_column(&mut batcher, &columns).unwrap_or_else(|e| {
            info!(
                "{:?}: p2 column hash GPU failed, falling back to CPU: {:?}",
                replica_path, e
            );
            cpu_build_column(&columns)
        });

        info!(
            "{:?}: built: tree-c:{}, node:{} ({:?})",
            replica_path,
            config_index + 1,
            node_index,
            t.elapsed(),
        );

        hashed_tx[config_index]
            .send((node_index, result))
            .with_context(|| {
                format!(
                    "cannot send hashed_tx[{}], node:{}",
                    config_index, node_index
                )
            })?;
    }
    info!("{:?}: debuglog: gpu tree-c finished", replica_path);

    Ok(())
}

pub fn collect_and_persist_tree_c<TreeArity>(
    rxs: &[std::sync::mpsc::Receiver<(usize, Vec<Fr>)>],
    paths: &[PathBuf],
    batch_size: usize,
    node_count: usize,
    replica_path: &Path,
    gpu_index: usize,
) -> Result<()>
where
    TreeArity: PoseidonArity,
{
    defer! { info!("collect_and_persist_tree_c exit");}
    let (build_tx, build_rx) = std::sync::mpsc::channel();

    let res = crossbeam::scope(move |s| -> Result<()> {
        let (tx, rx) = std::sync::mpsc::channel();

        s.spawn(move |_| {
            defer! { info!("build_tree_and_persist exit");}
            let data =
                build_tree_and_persist::<TreeArity>(rx, paths, node_count, replica_path, gpu_index);
            info!("tree-c persisted");
            build_tx
                .send(data)
                .expect("cannot send persisted result for error handling");
        });

        for (index, rx) in rxs.iter().enumerate() {
            let tree_c_column = collect_column_for_config(rx, batch_size, node_count, replica_path)
                .with_context(|| {
                    format!(
                        "{:?} cannot collect column for tree_c {}",
                        replica_path,
                        index + 1
                    )
                })?;
            info!(
                "{:?}: tree-c {} has been collected",
                replica_path,
                index + 1
            );
            tx.send((index, tree_c_column)).with_context(|| {
                format!("{:?}: cannot send column for tree building", replica_path)
            })?;
        }
        drop(tx);
        Ok(())
    });

    build_rx.recv().expect("cannot receive data for build_rx")?;

    match res {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(e)) => Err(e),
        Err(e) => Err(anyhow!(
            "{:?} collect_and_persist_tree_c panic: {:?}",
            replica_path,
            e
        )),
    }
}

fn build_tree_and_persist<TreeArity>(
    rx: std::sync::mpsc::Receiver<(usize, Vec<Fr>)>,
    paths: &[PathBuf],
    node_count: usize,
    replica_path: &Path,
    gpu_index: usize,
) -> Result<()>
where
    TreeArity: PoseidonArity,
{
    info!("{:?}: start creating builder", replica_path);

    let mut batcher: CLBatchHasher<TreeArity> =
        match Batcher::<TreeArity>::new(&CustomGPU(GPUSelector::Index(gpu_index)), node_count)
            .with_context(|| format!("failed to create tree_c batcher {}", gpu_index))?
        {
            Batcher::OpenCL(x) => x,
            Batcher::CPU(_) => panic!("neptune bug, batcher should be GPU"),
        };

    info!("{:?}: done creating builder", replica_path);

    for (index, column) in rx.iter() {
        let mut tree_c_file = OpenOptions::new()
            .truncate(true)
            .create(true)
            .write(true)
            .open(&paths[index])
            .with_context(|| format!("cannot open file {:?}", paths[index]))?;

        let (tx_err, rx_err) = std::sync::mpsc::channel();
        info!("{:?}: done creating channel", replica_path);
        let res = crossbeam::scope({
            let column = &column;
            let batcher = &mut batcher;
            let tree_c_file = &mut tree_c_file;
            move |s| {
                info!(
                    "{:?}: tree-c {} start building final leaves",
                    replica_path,
                    index + 1
                );

                s.spawn(move |_| {
                    let res = super::persist_frs(column, tree_c_file)
                        .with_context(|| format!("cannot persist_frs for {:?}", tree_c_file));
                    info!(
                        "{:?}: persisted tree-base for tree-c-{}",
                        replica_path,
                        index + 1
                    );
                    tx_err.send(res).expect("cannot send res persist_frs");
                });

                let tree = super::build_tree(batcher, &column, 0).with_context(|| {
                    format!(
                        "{:?}: cannot build tree for tree-c:{}",
                        replica_path,
                        index + 1
                    )
                })?;

                Ok(tree)
            }
        });
        rx_err.recv().expect("cannot recv rx_err")?;

        let tree = match res {
            Ok(Ok(tree)) => tree,
            Ok(Err(e)) => return Err(e),
            Err(e) => return Err(anyhow::anyhow!("building tree-c panic, error: {:?}", e)),
        };
        super::persist_frs(&tree, &mut tree_c_file)
            .with_context(|| format!("cannot persist tree-c-{}'s tree", index + 1))?;
    }
    Ok(())
}

// fn persist_tree_c(
//     index: usize,
//     path: &Path,
//     base: &[Fr],
//     tree: &[Fr],
//     replica_path: &Path,
// ) -> Result<()> {
//     use std::io::Cursor;
//     let mut cursor = Cursor::new(Vec::<u8>::with_capacity(
//         (base.len() + tree.len()) * std::mem::size_of::<Fr>(),
//     ));
//     info!(
//         "{:?}.persisting, done: cursor created for tree-c {}",
//         replica_path,
//         index + 1
//     );
//
//     for fr in base.iter().chain(tree.iter()).map(|x| x.into_repr()) {
//         fr.write_le(&mut cursor)
//             .with_context(|| format!("cannot write to cursor {:?}", path))?;
//     }
//     info!(
//         "{:?}.persisting, done: put data into cursor for tree-c {}",
//         replica_path,
//         index + 1
//     );
//
//     let tree_c_bytes = cursor.into_inner();
//     std::fs::write(&path, tree_c_bytes).with_context(|| format!("cannot open file: {:?}", path))?;
//
//     info!(
//         "{:?}.persisting, done: file written for tree-c {}",
//         replica_path,
//         index + 1
//     );
//
//     Ok(())
// }

pub fn collect_column_for_config(
    rx: &std::sync::mpsc::Receiver<(usize, Vec<Fr>)>,
    batch_size: usize,
    node_count: usize,
    replica_path: &Path,
) -> Result<Vec<Fr>> {
    let last_batch = if node_count % batch_size == 0 { 0 } else { 1 };
    let recv_count = node_count / batch_size + last_batch;
    let mut final_data = vec![Fr::zero(); node_count];

    for _ in 0..recv_count {
        let (index, data) = rx
            .recv()
            .with_context(|| format!("{:?}: cannot recv column", replica_path))?;

        final_data.as_mut_slice()[index..data.len() + index].copy_from_slice(&data);
    }
    Ok(final_data)
}
