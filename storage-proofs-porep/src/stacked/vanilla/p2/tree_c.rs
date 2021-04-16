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
// use neptune::{triton::gpu::GPUBatchHasher, BatchHasher};
use rayon::prelude::*;
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
use neptune::batch_hasher::BatcherType;
use neptune::batch_hasher::BatcherType::CustomGPU;
use neptune::batch_hasher::GPUSelector;
use neptune::proteus::gpu::CLBatchHasher;
use neptune::tree_builder::TreeBuilder;
use neptune::tree_builder::TreeBuilderTrait;

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
    let (max_gpu_column_batch_size, max_gpu_tree_batch_size) = {
        (
            settings::SETTINGS.max_gpu_column_batch_size as usize,
            settings::SETTINGS.max_gpu_tree_batch_size as usize,
        )
    };

    let (file_data_tx, file_data_rx) = crossbeam::channel::bounded(*CHANNEL_CAPACITY);
    let (column_tx, column_rx) = crossbeam::channel::bounded(*CHANNEL_CAPACITY);
    let mut files = open_column_data_file(labels)?;
    let Channels { txs, rxs } = channels(configs.len());
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
                        max_gpu_column_batch_size,
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

        s.spawn(move |_| {
            if let Err(e) = generate_tree_c_gpu::<ColumnArity, TreeArity>(
                nodes_count,
                gpu_index,
                column_rx,
                &txs,
                replica_path,
            ) {
                info!("generate_tree_c_gpu error: {:?}", e);
            }
        });

        // s.spawn(move |_| {
        //     P2_POOL.install(move || {
        //         generate_tree_c_cpu::<ColumnArity, TreeArity>(
        //             column_rx,
        //             &txs,
        //             replica_path,
        //             cpu_start_condition,
        //         )
        //     })
        // });

        collect_and_persist_tree_c::<TreeArity>(
            &rxs,
            &configs
                .iter()
                .map(|x| StoreConfig::data_path(&x.path, &x.id))
                .collect::<Vec<_>>(),
            max_gpu_column_batch_size,
            nodes_count,
            max_gpu_tree_batch_size,
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

pub fn generate_tree_c_gpu<ColumnArity, TreeArity>(
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
            .expect("cannot send gpu data to hashed_tx");
    }
    info!("{:?}: debuglog: gpu tree-c finished", replica_path);

    Ok(())
}

pub fn collect_and_persist_tree_c<TreeArity>(
    rxs: &[std::sync::mpsc::Receiver<(usize, Vec<Fr>)>],
    paths: &[PathBuf],
    batch_size: usize,
    node_count: usize,
    tree_batch_size: usize,
    replica_path: &Path,
    gpu_index: usize,
) -> Result<()>
where
    TreeArity: PoseidonArity,
{
    let (build_tx, build_rx) = std::sync::mpsc::channel();
    let res = crossbeam::scope(move |s| -> Result<()> {
        let (tx, rx) = std::sync::mpsc::channel();

        s.spawn(move |_| {
            let data = build_tree_and_persist::<TreeArity>(
                rx,
                paths,
                node_count,
                tree_batch_size,
                replica_path,
                gpu_index,
            );
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
        Err(_) => bail!("{:?} collect_and_persist_tree_c panic", replica_path),
    }
}

fn build_tree_and_persist<TreeArity>(
    rx: std::sync::mpsc::Receiver<(usize, Vec<Fr>)>,
    paths: &[PathBuf],
    node_count: usize,
    tree_batch_size: usize,
    replica_path: &Path,
    gpu_index: usize,
) -> Result<()>
where
    TreeArity: PoseidonArity,
{
    info!("start creating builder");
    let mut builder = TreeBuilder::<TreeArity>::new(
        Some(BatcherType::CustomGPU(GPUSelector::Index(gpu_index))),
        node_count,
        tree_batch_size,
        0,
    )
    .with_context(|| format!("{:?}: cannot create tree_builder", replica_path))?;
    info!("{:?}: done creating builder", replica_path);

    let (tx_err, rx_err) = std::sync::mpsc::channel();
    info!("{:?}: done creating channel", replica_path);

    let res = crossbeam::scope(|s| -> Result<()> {
        for (index, column) in rx.iter() {
            info!(
                "{:?}: tree-c {} start building final leaves",
                replica_path,
                index + 1
            );
            let (base, tree) = builder
                .add_final_leaves(&column)
                .with_context(|| format!("{:?} cannot add final leaves", replica_path))?;

            info!(
                "{:?}: tree-c {} has been built from column",
                replica_path,
                index + 1
            );

            let tx = tx_err.clone();
            s.spawn(move |_| {
                if let Err(e) = persist_tree_c(index, &paths[index], &base, &tree, replica_path) {
                    error!("cannot persisit tree-c {}, error: {:?}", index + 1, e);
                    tx.send((index + 1, e))
                        .expect("cannot send persisting error to tx");
                }
            });
        }
        Ok(())
    });
    drop(tx_err);
    if let Ok((index, e)) = rx_err.try_recv() {
        bail!("cannot persisit tree-c {}, error: {:?}", index, e);
    }

    match res {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(e)) => Err(e),
        Err(e) => Err(anyhow::anyhow!("building tree-c panic, error: {:?}", e)),
    }
}

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

fn persist_tree_c(
    index: usize,
    path: &Path,
    base: &[Fr],
    tree: &[Fr],
    replica_path: &Path,
) -> Result<()> {
    use std::fs::OpenOptions;
    use std::io::Cursor;
    use std::io::Write;
    let mut cursor = Cursor::new(Vec::<u8>::with_capacity(
        (base.len() + tree.len()) * std::mem::size_of::<Fr>(),
    ));
    info!(
        "{:?}.persisting, done: cursor created for tree-c {}",
        replica_path,
        index + 1
    );

    for fr in base.iter().chain(tree.iter()).map(|x| x.into_repr()) {
        fr.write_le(&mut cursor)
            .with_context(|| format!("cannot write to cursor {:?}", path))?;
    }
    info!(
        "{:?}.persisting, done: put data into cursor for tree-c {}",
        replica_path,
        index + 1
    );

    let mut tree_c = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(&path)
        .with_context(|| format!("cannot open file: {:?}", path))?;

    info!(
        "{:?}.persisting, done: file opened for {}",
        replica_path,
        index + 1
    );
    tree_c
        .write_all(&cursor.into_inner())
        .with_context(|| format!("cannot write to file: {:?}", path))?;
    info!(
        "{:?}.persisting, done: file written for tree-c {}",
        replica_path,
        index + 1
    );

    Ok(())
}
