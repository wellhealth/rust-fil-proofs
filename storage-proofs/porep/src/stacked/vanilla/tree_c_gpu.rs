use anyhow::anyhow;
use anyhow::ensure;
use anyhow::Context;
use anyhow::Result;
use bellperson::bls::Fr;
use ff::Field;
use generic_array::{sequence::GenericSequence, GenericArray};
use log::info;
use merkletree::store::StoreConfig;
use neptune::batch_hasher::BatcherType;
use neptune::column_tree_builder::ColumnTreeBuilder;
use neptune::column_tree_builder::ColumnTreeBuilderTrait;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Cursor;
use std::io::Read;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::mpsc::channel;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::SyncSender;
use storage_proofs_core::hasher::Hasher;
use storage_proofs_core::hasher::PoseidonArity;
use storage_proofs_core::merkle::create_disk_tree;
use storage_proofs_core::merkle::DiskTree;
use storage_proofs_core::merkle::MerkleTreeTrait;
use storage_proofs_core::settings;

#[allow(unused_variables)]
pub fn custom_tree_c_gpu<ColumnArity, TreeArity, Tree>(
    layers: usize,
    nodes_count: usize,
    tree_count: usize,
    configs: Vec<StoreConfig>,
    labels: &[(PathBuf, String)],
    gpu_index: usize,
) -> Result<DiskTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>>
where
    ColumnArity: 'static + PoseidonArity,
    TreeArity: PoseidonArity,
    Tree: 'static + MerkleTreeTrait,
{
    let replica_path: &Path = &configs[0].path;
    info!("{:?}: Building column hashes", replica_path);

    // NOTE: The max number of columns we recommend sending to the GPU at once is
    // 400000 for columns and 700000 for trees (conservative soft-limits discussed).
    //
    // 'column_write_batch_size' is how many nodes to chunk the base layer of data
    // into when persisting to disk.
    //
    // Override these values with care using environment variables:
    // FIL_PROOFS_MAX_GPU_COLUMN_BATCH_SIZE, FIL_PROOFS_MAX_GPU_TREE_BATCH_SIZE, and
    // FIL_PROOFS_COLUMN_WRITE_BATCH_SIZE respectively.

    let (max_gpu_column_batch_size, max_gpu_tree_batch_size) = {
        let setting = &*settings::SETTINGS;

        (
            setting.max_gpu_column_batch_size as usize,
            setting.max_gpu_tree_batch_size as usize,
        )
    };

    // This channel will receive batches of columns and add them to the ColumnTreeBuilder.
    let (builder_tx, builder_rx) = mpsc::sync_channel(10);

    let configs = &configs;
    let replica_path = &replica_path;

    let (r1, r2) = {
        rayon::join(
            move || {
                create_batch_gpu::<ColumnArity, Tree>(
                    nodes_count,
                    configs,
                    labels,
                    builder_tx,
                    max_gpu_column_batch_size,
                    replica_path,
                )
            },
            move || {
                receive_and_generate_tree_c::<ColumnArity, TreeArity, _>(
                    nodes_count,
                    configs,
                    builder_rx,
                    max_gpu_tree_batch_size,
                    max_gpu_column_batch_size,
                    replica_path,
                    gpu_index,
                )
            },
        )
    };

    match (r1, r2) {
        (Ok(_), Ok(_)) => {}
        (Ok(_), Err(e)) => return Err(e),
        (Err(e), Ok(_)) => return Err(e),
        (Err(e1), Err(e2)) => {
            return Err(anyhow!("tree-c error: {:?}, {:?}", e1, e2));
        }
    }

    create_disk_tree::<DiskTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>>(
        configs[0].size.expect("config size failure"),
        &configs,
    )
}

fn fast_create_batch<ColumnArity, Tree>(
    nodes_count: usize,
    config_counts: usize,
    batch_size: usize,
    paths: &[(PathBuf, String)],
    mut builder_tx: SyncSender<(usize, Vec<GenericArray<Fr, ColumnArity>>, bool)>,
    replica_path: &Path,
) -> Result<()>
where
    ColumnArity: 'static + PoseidonArity,
    Tree: 'static + MerkleTreeTrait,
{
    let mut files = paths
        .iter()
        .map(|x| StoreConfig::data_path(&x.0, &x.1))
        .map(|x| {
            anyhow::Context::with_context(File::open(&x), || {
                format!("cannot open layer file [{:?}] for tree-c", x)
            })
        })
        .collect::<Result<Vec<_>>>()?;
    info!("{:?}, file opened for p2", replica_path);

    for _ in 0..config_counts {
        fast_create_batch_for_config::<ColumnArity, Tree>(
            nodes_count,
            batch_size,
            &mut builder_tx,
            &mut files,
            replica_path.as_ref(),
        )?;
    }
    Ok(())
}
fn fast_create_batch_for_config<ColumnArity, Tree>(
    nodes_count: usize,
    batch_size: usize,
    builder_tx: &mut SyncSender<(usize, Vec<GenericArray<Fr, ColumnArity>>, bool)>,
    files: &mut [File],
    replica_path: &Path,
) -> Result<()>
where
    ColumnArity: 'static + PoseidonArity,
    Tree: 'static + MerkleTreeTrait,
{
    let bytes_per_item = batch_size * std::mem::size_of::<Fr>() * 11;
    let sync_size = 8 * 1024 * 1024 * 1024 / bytes_per_item;

    let (tx, rx) = mpsc::sync_channel(sync_size);
    let (r1, r2) = rayon::join(
        || read_column_batch_from_file::<Tree>(files, nodes_count, batch_size, tx, replica_path),
        || {
            create_column_in_memory::<ColumnArity>(
                rx,
                nodes_count,
                batch_size,
                builder_tx,
                replica_path,
            )
        },
    );

    r1?;
    r2
}
fn read_column_batch_from_file<Tree>(
    files: &mut [File],
    nodes_count: usize,
    batch_size: usize,
    tx: SyncSender<(usize, Vec<Vec<Fr>>)>,
    replica_path: &Path,
) -> Result<()>
where
    Tree: 'static + MerkleTreeTrait,
{
    use merkletree::merkle::Element;

    for node_index in (0..nodes_count).step_by(batch_size) {
        let chunked_nodes_count = std::cmp::min(nodes_count - node_index, batch_size);
        let chunk_byte_count = chunked_nodes_count * std::mem::size_of::<Fr>();

        info!(
            "{:?} read from file for node: [{}]",
            replica_path, node_index
        );
        let data = files
            .par_iter_mut()
            .map(|x| {
                let mut buf_bytes = vec![0u8; chunk_byte_count];
                x.read_exact(&mut buf_bytes)
                    .with_context(|| format!("error occurred when reading file [{:?}]", x))?;
                let size = std::mem::size_of::<<Tree::Hasher as Hasher>::Domain>();
                Ok(buf_bytes
                    .chunks(size)
                    .map(|x| <<Tree::Hasher as Hasher>::Domain>::from_slice(x))
                    .map(Into::into)
                    .collect())
            })
            .collect::<Result<Vec<_>>>()?;

        info!("{:?} file data collected: [{}]", replica_path, node_index);
        tx.send((node_index, data)).unwrap();
        info!("{:?} file data sent: [{}]", replica_path, node_index);
    }
    Ok(())
}
fn create_column_in_memory<ColumnArity>(
    rx: Receiver<(usize, Vec<Vec<Fr>>)>,
    nodes_count: usize,
    batch_size: usize,
    builder_tx: &mut SyncSender<(usize, Vec<GenericArray<Fr, ColumnArity>>, bool)>,
    replica_path: &Path,
) -> Result<()>
where
    ColumnArity: 'static + PoseidonArity,
{
    let layers = ColumnArity::to_usize();
    for node_index in (0..nodes_count).step_by(batch_size) {
        let (node, data) = rx
            .recv()
            .map_err(|e| anyhow!("failed to receive column data, err: {:?}", e))?;

        ensure!(node == node_index, "node index mismatch for p2");
        info!(
            "{:?} start creating columns for node {}",
            replica_path, node
        );

        let mut columns: Vec<GenericArray<Fr, ColumnArity>> =
            vec![GenericArray::<Fr, ColumnArity>::generate(|_i: usize| Fr::zero()); data[0].len()];

        let is_final = node_index + batch_size >= nodes_count;
        columns.iter_mut().enumerate().for_each(|(index, column)| {
            for layer_index in 0..layers {
                column[layer_index] = data[layer_index][index];
            }
        });

        builder_tx.send((node, columns, is_final)).unwrap();
    }
    Ok(())
}
fn create_batch_gpu<ColumnArity, Tree>(
    nodes_count: usize,
    configs: &[StoreConfig],
    paths: &[(PathBuf, String)],
    builder_tx: SyncSender<(usize, Vec<GenericArray<Fr, ColumnArity>>, bool)>,
    batch_size: usize,
    replica_path: &Path,
) -> Result<()>
where
    ColumnArity: 'static + PoseidonArity,
    Tree: 'static + MerkleTreeTrait,
{
    fast_create_batch::<ColumnArity, Tree>(
        nodes_count,
        configs.len(),
        batch_size,
        paths,
        builder_tx,
        replica_path,
    )
}
fn receive_and_generate_tree_c<ColumnArity, TreeArity, P>(
    nodes_count: usize,
    configs: &[StoreConfig],
    builder_rx: Receiver<(usize, Vec<GenericArray<Fr, ColumnArity>>, bool)>,
    max_gpu_tree_batch_size: usize,
    max_gpu_column_batch_size: usize,
    replica_path: P,
    gpu_index: usize,
) -> Result<()>
where
    ColumnArity: 'static + PoseidonArity,
    TreeArity: PoseidonArity,
    P: AsRef<Path> + Send + Sync,
{
    let mut column_tree_builder = ColumnTreeBuilder::<ColumnArity, TreeArity>::new(
        Some(BatcherType::GPU),
        nodes_count,
        max_gpu_column_batch_size,
        max_gpu_tree_batch_size,
        gpu_index,
    )
    .expect("failed to create ColumnTreeBuilder");

    let mut i = 0;
    let mut config = &configs[i];

    // Loop until all trees for all configs have been built.
    let config_count = configs.len();

    let (persist_tx, persist_rx) = channel();
    while i < config_count {
        let t0 = std::time::Instant::now();
        let (node, columns, is_final) = builder_rx.recv().expect("failed to recv columns");
        let t1 = std::time::Instant::now();
        info!(
            "{:?} received node [{}] with {:?}",
            replica_path.as_ref(),
            node,
            t1 - t0
        );

        // Just add non-final column batches.
        if !is_final {
            column_tree_builder
                .add_columns(&columns)
                .expect("failed to add columns");
            info!("{:?} built node [{}]", replica_path.as_ref(), node);
            continue;
        };

        // If we get here, this is a final column: build a sub-tree.
        let (base_data, tree_data) = column_tree_builder
            .add_final_columns(&columns)
            .expect("failed to add final columns");

        drop(columns);

        assert_eq!(base_data.len(), nodes_count);
        info!(
            "{:?}: persist tree-c: {}/{}",
            replica_path.as_ref(),
            i + 1,
            configs.len()
        );

        let replica_path = PathBuf::from(replica_path.as_ref());
        rayon::spawn({
            let config = config.clone();
            let persist_tx = persist_tx.clone();
            move || {
                info!(
                    "{:?}, start persisting {}/{}",
                    replica_path,
                    i + 1,
                    config_count
                );

                let tree_len = base_data.len() + tree_data.len();
                assert_eq!(tree_len, config.size.expect("config size failure"));

                let path = StoreConfig::data_path(&config.path, &config.id);

                let r = persist_tree_c(i + 1, path, &base_data, &tree_data, &replica_path);
                if let Err(e) = r.as_ref() {
                    info!("{:?} persist_tree_c failed, error: {:?}", replica_path, e);
                }
                persist_tx.send(r).unwrap();
                info!(
                    "{:?}, done persisting {}/{}",
                    replica_path,
                    i + 1,
                    config_count
                );
            }
        });

        // Move on to the next config.
        i += 1;
        if i == configs.len() {
            break;
        }
        config = &configs[i];
    }
    drop(persist_tx);
    match persist_rx.iter().filter_map(|r| r.err()).next() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}
fn persist_tree_c<P>(
    index: usize,
    path: P,
    base: &[Fr],
    tree: &[Fr],
    replica_path: &Path,
) -> Result<()>
where
    P: AsRef<Path>,
{
    use ff::{PrimeField, PrimeFieldRepr};
    let mut cursor = Cursor::new(Vec::<u8>::with_capacity(
        (base.len() + tree.len()) * std::mem::size_of::<Fr>(),
    ));
    info!(
        "{:?}.persisting, done: cursor created for tree-c {}",
        replica_path, index
    );

    for fr in base.iter().chain(tree.iter()).map(|x| x.into_repr()) {
        fr.write_le(&mut cursor)
            .with_context(|| format!("cannot write to cursor {:?}", path.as_ref()))?;
    }
    info!(
        "{:?}.persisting, done: put data into cursor for tree-c {}",
        replica_path, index
    );

    let mut tree_c = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(&path)
        .with_context(|| format!("cannot open file: {:?}", path.as_ref()))?;

    info!(
        "{:?}.persisting, done: file opened for {}",
        replica_path, index
    );
    tree_c
        .write_all(&cursor.into_inner())
        .with_context(|| format!("cannot write to file: {:?}", path.as_ref()))?;
    info!(
        "{:?}.persisting, done: file written for {}",
        replica_path, index
    );

    Ok(())
}
