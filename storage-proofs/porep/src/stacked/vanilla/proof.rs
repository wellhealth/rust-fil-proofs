use std::fs::File;
use std::fs::OpenOptions;
use std::io::Read;
use std::io::Write;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, RwLock};

use anyhow::Context;
use bincode::deserialize;
use generic_array::typenum::{self, Unsigned};
use log::{info, trace};
use merkletree::merkle::{
    get_merkle_tree_cache_size, get_merkle_tree_leafs, get_merkle_tree_len,
    is_merkle_tree_size_valid,
};
use merkletree::store::{DiskStore, StoreConfig};
use mpsc::{Receiver, SyncSender};
use paired::bls12_381::Fr;
use rayon::prelude::*;
use scopeguard::defer;
use storage_proofs_core::{
    cache_key::CacheKey,
    data::Data,
    drgraph::Graph,
    error::Result,
    hasher::{Domain, HashFunction, Hasher, PoseidonArity},
    measurements::{
        measure_op,
        Operation::{CommD, EncodeWindowTimeAll, GenerateTreeC, GenerateTreeRLast},
    },
    merkle::*,
    settings,
    util::{default_rows_to_discard, NODE_SIZE},
};
use typenum::{U11, U2, U8};

use super::{
    challenges::LayerChallenges,
    column::Column,
    create_label,
    graph::StackedBucketGraph,
    hash::hash_single_column,
    params::{
        get_node, Labels, LabelsCache, PersistentAux, Proof, PublicInputs, PublicParams,
        ReplicaColumnProof, Tau, TemporaryAux, TemporaryAuxCache, TransformedLayers, BINARY_ARITY,
    },
    EncodingProof, LabelingProof,
};

use ff::Field;
use generic_array::{sequence::GenericSequence, GenericArray};
use neptune::batch_hasher::BatcherType;
use neptune::column_tree_builder::{ColumnTreeBuilder, ColumnTreeBuilderTrait};
use neptune::tree_builder::{TreeBuilder, TreeBuilderTrait};
use storage_proofs_core::fr32::fr_into_bytes;

use crate::encode::{decode, encode};
use crate::PoRep;

pub const TOTAL_PARENTS: usize = 37;

// pub fn join3<O1, O2, O3, R1, R2, R3>(oper_1: O1, oper_2: O2, oper_3: O3) -> (R1, R2, R3)
// where
//     O1: FnOnce() -> R1 + Send,
//     O2: FnOnce() -> R2 + Send,
//     O3: FnOnce() -> R3 + Send,
//     R1: Send,
//     R2: Send,
//     R3: Send,
// {
//     let (r1, (r2, r3)) = rayon::join(oper_1, || (oper_2(), oper_3()));
//     (r1, r2, r3)
// }

#[derive(Debug)]
pub struct StackedDrg<'a, Tree: 'a + MerkleTreeTrait, G: 'a + Hasher> {
    _a: PhantomData<&'a Tree>,
    _b: PhantomData<&'a G>,
}

impl<'a, Tree: 'static + MerkleTreeTrait, G: 'static + Hasher> StackedDrg<'a, Tree, G> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prove_layers(
        graph: &StackedBucketGraph<Tree::Hasher>,
        pub_inputs: &PublicInputs<<Tree::Hasher as Hasher>::Domain, <G as Hasher>::Domain>,
        p_aux: &PersistentAux<<Tree::Hasher as Hasher>::Domain>,
        t_aux: &TemporaryAuxCache<Tree, G>,
        layer_challenges: &LayerChallenges,
        layers: usize,
        _total_layers: usize,
        partition_count: usize,
    ) -> Result<Vec<Vec<Proof<Tree, G>>>> {
        assert!(layers > 0);
        assert_eq!(t_aux.labels.len(), layers);

        let graph_size = graph.size();

        // Sanity checks on restored trees.
        assert!(pub_inputs.tau.is_some());
        assert_eq!(
            pub_inputs.tau.as_ref().expect("as_ref failure").comm_d,
            t_aux.tree_d.root()
        );

        let get_drg_parents_columns = |x: usize| -> Result<Vec<Column<Tree::Hasher>>> {
            let base_degree = graph.base_graph().degree();

            let mut columns = Vec::with_capacity(base_degree);

            let mut parents = vec![0; base_degree];
            graph.base_parents(x, &mut parents)?;

            for parent in &parents {
                columns.push(t_aux.column(*parent)?);
            }

            debug_assert!(columns.len() == base_degree);

            Ok(columns)
        };

        let get_exp_parents_columns = |x: usize| -> Result<Vec<Column<Tree::Hasher>>> {
            let mut parents = vec![0; graph.expansion_degree()];
            graph.expanded_parents(x, &mut parents)?;

            parents.iter().map(|parent| t_aux.column(*parent)).collect()
        };

        (0..partition_count)
            .map(|k| {
                trace!("proving partition {}/{}", k + 1, partition_count);

                // Derive the set of challenges we are proving over.
                let challenges = pub_inputs.challenges(layer_challenges, graph_size, Some(k));

                // Stacked commitment specifics
                challenges
                    .into_par_iter()
                    .enumerate()
                    .map(|(challenge_index, challenge)| {
                        trace!(" challenge {} ({})", challenge, challenge_index);
                        assert!(challenge < graph.size(), "Invalid challenge");
                        assert!(challenge > 0, "Invalid challenge");

                        // Initial data layer openings (c_X in Comm_D)
                        let comm_d_proof = t_aux.tree_d.gen_proof(challenge)?;
                        assert!(comm_d_proof.validate(challenge));

                        // Stacked replica column openings
                        let rcp = {
                            let (c_x, drg_parents, exp_parents) = {
                                assert_eq!(p_aux.comm_c, t_aux.tree_c.root());
                                let tree_c = &t_aux.tree_c;

                                // All labels in C_X
                                trace!("  c_x");
                                let c_x = t_aux.column(challenge as u32)?.into_proof(tree_c)?;

                                // All labels in the DRG parents.
                                trace!("  drg_parents");
                                let drg_parents = get_drg_parents_columns(challenge)?
                                    .into_iter()
                                    .map(|column| column.into_proof(tree_c))
                                    .collect::<Result<_>>()?;

                                // Labels for the expander parents
                                trace!("  exp_parents");
                                let exp_parents = get_exp_parents_columns(challenge)?
                                    .into_iter()
                                    .map(|column| column.into_proof(tree_c))
                                    .collect::<Result<_>>()?;

                                (c_x, drg_parents, exp_parents)
                            };

                            ReplicaColumnProof {
                                c_x,
                                drg_parents,
                                exp_parents,
                            }
                        };

                        // Final replica layer openings
                        trace!("final replica layer openings");
                        let comm_r_last_proof = t_aux.tree_r_last.gen_cached_proof(
                            challenge,
                            Some(t_aux.tree_r_last_config_rows_to_discard),
                        )?;

                        debug_assert!(comm_r_last_proof.validate(challenge));

                        // Labeling Proofs Layer 1..l
                        let mut labeling_proofs = Vec::with_capacity(layers);
                        let mut encoding_proof = None;

                        for layer in 1..=layers {
                            trace!("  encoding proof layer {}", layer,);
                            let parents_data: Vec<<Tree::Hasher as Hasher>::Domain> = if layer == 1
                            {
                                let mut parents = vec![0; graph.base_graph().degree()];
                                graph.base_parents(challenge, &mut parents)?;

                                parents
                                    .into_iter()
                                    .map(|parent| t_aux.domain_node_at_layer(layer, parent))
                                    .collect::<Result<_>>()?
                            } else {
                                let mut parents = vec![0; graph.degree()];
                                graph.parents(challenge, &mut parents)?;
                                let base_parents_count = graph.base_graph().degree();

                                parents
                                    .into_iter()
                                    .enumerate()
                                    .map(|(i, parent)| {
                                        if i < base_parents_count {
                                            // parents data for base parents is from the current layer
                                            t_aux.domain_node_at_layer(layer, parent)
                                        } else {
                                            // parents data for exp parents is from the previous layer
                                            t_aux.domain_node_at_layer(layer - 1, parent)
                                        }
                                    })
                                    .collect::<Result<_>>()?
                            };

                            // repeat parents
                            let mut parents_data_full = vec![Default::default(); TOTAL_PARENTS];
                            for chunk in parents_data_full.chunks_mut(parents_data.len()) {
                                chunk.copy_from_slice(&parents_data[..chunk.len()]);
                            }

                            let proof = LabelingProof::<Tree::Hasher>::new(
                                layer as u32,
                                challenge as u64,
                                parents_data_full.clone(),
                            );

                            {
                                let labeled_node = rcp.c_x.get_node_at_layer(layer)?;
                                assert!(
                                    proof.verify(&pub_inputs.replica_id, &labeled_node),
                                    format!("Invalid encoding proof generated at layer {}", layer)
                                );
                                trace!("Valid encoding proof generated at layer {}", layer);
                            }

                            labeling_proofs.push(proof);

                            if layer == layers {
                                encoding_proof = Some(EncodingProof::new(
                                    layer as u32,
                                    challenge as u64,
                                    parents_data_full,
                                ));
                            }
                        }

                        Ok(Proof {
                            comm_d_proofs: comm_d_proof,
                            replica_column_proofs: rcp,
                            comm_r_last_proof,
                            labeling_proofs,
                            encoding_proof: encoding_proof.expect("invalid tapering"),
                        })
                    })
                    .collect()
            })
            .collect()
    }

    pub(crate) fn extract_and_invert_transform_layers(
        graph: &StackedBucketGraph<Tree::Hasher>,
        layer_challenges: &LayerChallenges,
        replica_id: &<Tree::Hasher as Hasher>::Domain,
        data: &mut [u8],
        config: StoreConfig,
    ) -> Result<()> {
        trace!("extract_and_invert_transform_layers");

        let layers = layer_challenges.layers();
        assert!(layers > 0);

        // generate labels
        let (labels, _) = Self::generate_labels(graph, layer_challenges, replica_id, config)?;

        let last_layer_labels = labels.labels_for_last_layer()?;
        let size = merkletree::store::Store::len(last_layer_labels);

        for (key, encoded_node_bytes) in last_layer_labels
            .read_range(0..size)?
            .into_iter()
            .zip(data.chunks_mut(NODE_SIZE))
        {
            let encoded_node =
                <Tree::Hasher as Hasher>::Domain::try_from_bytes(encoded_node_bytes)?;
            let data_node = decode::<<Tree::Hasher as Hasher>::Domain>(key, encoded_node);

            // store result in the data
            encoded_node_bytes.copy_from_slice(AsRef::<[u8]>::as_ref(&data_node));
        }

        Ok(())
    }

    fn generate_labels(
        graph: &StackedBucketGraph<Tree::Hasher>,
        layer_challenges: &LayerChallenges,
        replica_id: &<Tree::Hasher as Hasher>::Domain,
        config: StoreConfig,
    ) -> Result<(LabelsCache<Tree>, Labels<Tree>)> {
        let mut parent_cache = graph.parent_cache()?;

        if settings::SETTINGS
            .lock()
            .expect("use_multicore_sdr settings lock failure")
            .use_multicore_sdr
        {
            info!("multi core replication");
            create_label::multi::create_labels(
                graph,
                &parent_cache,
                layer_challenges.layers(),
                replica_id,
                config,
            )
        } else {
            info!("single core replication");
            create_label::single::create_labels(
                graph,
                &mut parent_cache,
                layer_challenges.layers(),
                replica_id,
                config,
            )
        }
    }

    #[allow(clippy::type_complexity)]
    fn my_generate_labels(
        graph: &StackedBucketGraph<Tree::Hasher>,
        layer_challenges: &LayerChallenges,
        replica_id: &<Tree::Hasher as Hasher>::Domain,
        config: StoreConfig,
    ) -> Result<(LabelsCache<Tree>, Labels<Tree>)> {
        let mut parent_cache = graph.parent_cache()?;

        if settings::SETTINGS
            .lock()
            .expect("use_multicore_sdr settings lock failure")
            .use_multicore_sdr
        {
            info!("multi core replication");
            create_label::multi::my_create_labels(
                graph,
                &parent_cache,
                layer_challenges.layers(),
                replica_id,
                config,
            )
        } else {
            info!("single core replication");
            create_label::single::my_create_labels(
                graph,
                &mut parent_cache,
                layer_challenges.layers(),
                replica_id,
                config,
            )
        }
    }

    fn build_binary_tree<K: Hasher>(
        tree_data: &[u8],
        config: StoreConfig,
    ) -> Result<BinaryMerkleTree<K>> {
        trace!("building tree (size: {})", tree_data.len());

        let leafs = tree_data.len() / NODE_SIZE;
        assert_eq!(tree_data.len() % NODE_SIZE, 0);

        let tree = MerkleTree::from_par_iter_with_config(
            (0..leafs)
                .into_par_iter()
                // TODO: proper error handling instead of `unwrap()`
                .map(|i| get_node::<K>(tree_data, i).expect("get_node failure")),
            config,
        )?;
        Ok(tree)
    }

    fn generate_tree_c<ColumnArity, TreeArity, P>(
        nodes_count: usize,
        configs: Vec<StoreConfig>,
        tree_count: usize,
        labels: &[(PathBuf, String)],
        old_labels: &LabelsCache<Tree>,
        replica_path: P,
        gpu_index: usize,
    ) -> Result<DiskTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>>
    where
        ColumnArity: 'static + PoseidonArity,
        TreeArity: PoseidonArity,
        P: AsRef<Path> + Send + Sync,
    {
        if settings::SETTINGS
            .lock()
            .expect("use_gpu_column_builder settings lock failure")
            .use_gpu_column_builder
        {
            Self::generate_tree_c_gpu::<ColumnArity, TreeArity, _>(
                nodes_count,
                configs,
                labels,
                replica_path,
                gpu_index,
            )
        } else {
            Self::generate_tree_c_cpu::<ColumnArity, TreeArity>(
                ColumnArity::to_usize(),
                nodes_count,
                tree_count,
                configs,
                old_labels,
            )
        }
    }

    fn fast_create_batch<ColumnArity, P>(
        nodes_count: usize,
        config_counts: usize,
        batch_size: usize,
        paths: &[(PathBuf, String)],
        mut builder_tx: SyncSender<(Vec<GenericArray<Fr, ColumnArity>>, bool)>,
        replica_path: P,
    ) where
        ColumnArity: 'static + PoseidonArity,
        P: AsRef<Path> + Send + Sync,
    {
        let mut files = paths
            .iter()
            .map(|x| StoreConfig::data_path(&x.0, &x.1))
            .map(|x| File::open(x).context("cannot open layer file for tree-c"))
            .collect::<Result<Vec<_>>>()
            .unwrap();
        info!("{:?}, file opened for p2", replica_path.as_ref());

        for _ in 0..config_counts {
            Self::fast_create_batch_for_config(
                nodes_count,
                batch_size,
                &mut builder_tx,
                &mut files,
                &replica_path,
            );
        }
    }

    fn fast_create_batch_for_config<ColumnArity, P>(
        nodes_count: usize,
        batch_size: usize,
        builder_tx: &mut SyncSender<(Vec<GenericArray<Fr, ColumnArity>>, bool)>,
        files: &mut [File],
        replica_path: P,
    ) where
        ColumnArity: 'static + PoseidonArity,
        P: AsRef<Path> + Send + Sync,
    {
        let bytes_per_item = batch_size * std::mem::size_of::<Fr>() * 11;
        let sync_size = 32 * 1024 * 1024 * 1024 / bytes_per_item;
        let (tx, rx) = mpsc::sync_channel(sync_size);

        let replica_path = &replica_path;
        rayon::join(
            || Self::read_column_batch_from_file(files, nodes_count, batch_size, tx, &replica_path),
            || {
                Self::create_column_in_memory::<ColumnArity, _>(
                    rx,
                    nodes_count,
                    batch_size,
                    builder_tx,
                    replica_path,
                )
            },
        );
    }

    fn create_column_in_memory<ColumnArity, P>(
        rx: Receiver<Vec<Vec<Fr>>>,
        nodes_count: usize,
        batch_size: usize,
        builder_tx: &mut SyncSender<(Vec<GenericArray<Fr, ColumnArity>>, bool)>,
        replica_path: P,
    ) where
        ColumnArity: 'static + PoseidonArity,
        P: AsRef<Path>,
    {
        let layers = ColumnArity::to_usize();
        for node_index in (0..nodes_count).step_by(batch_size) {
            let data = rx.recv().unwrap();
            let t0 = std::time::Instant::now();
            let mut columns: Vec<GenericArray<Fr, ColumnArity>> = vec![
                    GenericArray::<Fr, ColumnArity>::generate(|_i: usize| Fr::zero());
                    data[0].len()
                ];

            let is_final = node_index + batch_size >= nodes_count;
            for layer_index in 0..layers {
                for (index, column) in columns.iter_mut().enumerate() {
                    column[layer_index] = data[layer_index][index];
                }
            }
            let t1 = std::time::Instant::now();
            info!(
                "{:?} column creation takes {:?}",
                replica_path.as_ref(),
                t1 - t0
            );
            builder_tx.send((columns, is_final)).unwrap();
        }
    }

    fn read_column_batch_from_file<P>(
        files: &mut [File],
        nodes_count: usize,
        batch_size: usize,
        tx: SyncSender<Vec<Vec<Fr>>>,
        replica_path: P,
    ) where
        P: AsRef<Path> + Send + Sync,
    {
        use merkletree::merkle::Element;

        for node_index in (0..nodes_count).step_by(batch_size) {
            let chunked_nodes_count = std::cmp::min(nodes_count - node_index, batch_size);
            let chunk_byte_count = chunked_nodes_count * std::mem::size_of::<Fr>();

            let data = files
                .par_iter_mut()
                .map(|x| {
                    let mut buf_bytes = vec![0u8; chunk_byte_count];
                    let t0 = std::time::Instant::now();
                    x.read_exact(&mut buf_bytes).unwrap();
                    let t1 = std::time::Instant::now();
                    let size = std::mem::size_of::<<Tree::Hasher as Hasher>::Domain>();
                    let res = buf_bytes
                        .chunks(size)
                        .map(|x| <<Tree::Hasher as Hasher>::Domain>::from_slice(x))
                        .map(Into::into)
                        .collect();
                    let t2 = std::time::Instant::now();
                    info!(
                        "{:?}: read label file [{:?}], collect info: [{:?}]",
                        replica_path.as_ref(),
                        t1 - t0,
                        t2 - t1
                    );
                    res
                })
                .collect::<Vec<Vec<Fr>>>();

            info!("{:?}: node index: {}", replica_path.as_ref(), node_index);

            tx.send(data).unwrap();
        }
    }

    fn create_batch_gpu<ColumnArity, P: AsRef<Path> + Send + Sync>(
        nodes_count: usize,
        configs: &[StoreConfig],
        paths: &[(PathBuf, String)],
        builder_tx: SyncSender<(Vec<GenericArray<Fr, ColumnArity>>, bool)>,
        batch_size: usize,
        replica_path: P,
    ) where
        ColumnArity: 'static + PoseidonArity,
    {
        Self::fast_create_batch::<ColumnArity, _>(
            nodes_count,
            configs.len(),
            batch_size,
            paths,
            builder_tx,
            replica_path,
        );
    }

    fn receive_and_generate_tree_c<ColumnArity, TreeArity, P>(
        nodes_count: usize,
        configs: &[StoreConfig],
        builder_rx: Receiver<(Vec<GenericArray<Fr, ColumnArity>>, bool)>,
        max_gpu_tree_batch_size: usize,
        max_gpu_column_batch_size: usize,
        column_write_batch_size: usize,
        replica_path: P,
        gpu_index: usize,
    ) where
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

        let mut chans = (0..config_count)
            .map(|_| mpsc::channel())
            .map(|(tx, rx)| (tx, Some(rx)))
            .collect::<Vec<_>>();

        while i < config_count {
            let (columns, is_final) = builder_rx.recv().expect("failed to recv columns");

            info!(
                "sector: {:?}: recv in config: {}",
                replica_path.as_ref(),
                i + 1
            );
            // Just add non-final column batches.
            if !is_final {
                column_tree_builder
                    .add_columns(&columns)
                    .expect("failed to add columns");
                continue;
            };

            // If we get here, this is a final column: build a sub-tree.
            let (base_data, tree_data) = column_tree_builder
                .add_final_columns(&columns)
                .expect("failed to add final columns");

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
                let sync_tx = chans[i].0.clone();
                let sync_rx = match i {
                    0 => None,
                    _ => chans[i - 1].1.take(),
                };
                move || {
                    if let Some(x) = sync_rx {
                        x.recv().unwrap()
                    }
                    defer!({
                        sync_tx.send(()).unwrap();
                    });
                    let tree_len = base_data.len() + tree_data.len();
                    assert_eq!(tree_len, config.size.expect("config size failure"));

                    let path = StoreConfig::data_path(&config.path, &config.id);
                    if path.exists() {
                        if let Err(e) = std::fs::remove_file(&path) {
                            info!(
                                "sector: {:?}: cannot delete file: {:?}, erro: {}",
                                replica_path, path, e
                            );
                        }
                    }

                    // Persist the base and tree data to disk based using the current store config.
                    let tree_c_store =
                        DiskStore::<<Tree::Hasher as Hasher>::Domain>::new_with_config(
                            tree_len,
                            Tree::Arity::to_usize(),
                            config.clone(),
                        )
                        .expect("failed to create DiskStore for base tree data");

                    let store = Arc::new(RwLock::new(tree_c_store));
                    let batch_size = std::cmp::min(base_data.len(), column_write_batch_size);
                    let flatten_and_write_store = |data: &Vec<Fr>, offset| {
                        data.into_par_iter()
                            .chunks(column_write_batch_size)
                            .enumerate()
                            .try_for_each(|(index, fr_elements)| {
                                let mut buf = Vec::with_capacity(batch_size * NODE_SIZE);

                                for fr in fr_elements {
                                    buf.extend(fr_into_bytes(&fr));
                                }
                                store
                                    .write()
                                    .expect("failed to access store for write")
                                    .copy_from_slice(&buf[..], offset + (batch_size * index))
                            })
                    };

                    flatten_and_write_store(&base_data, 0)
                        .expect("failed to flatten and write store");

                    let base_offset = base_data.len();
                    flatten_and_write_store(&tree_data, base_offset)
                        .expect("failed to flatten and write store");
                }
            });

            // Move on to the next config.
            i += 1;
            if i == configs.len() {
                break;
            }
            config = &configs[i];
        }
        chans.last().unwrap().1.as_ref().unwrap().recv().unwrap();
    }
    fn generate_tree_c_gpu_impl<ColumnArity, TreeArity, P>(
        nodes_count: usize,
        configs: Vec<StoreConfig>,
        labels: &[(PathBuf, String)],
        replica_path: P,
        gpu_index: usize,
    ) -> Result<DiskTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>>
    where
        ColumnArity: 'static + PoseidonArity,
        TreeArity: PoseidonArity,
        P: AsRef<Path> + Send + Sync,
    {
        info!("{:?}, Building column hashes", replica_path.as_ref());

        // NOTE: The max number of columns we recommend sending to the GPU at once is
        // 400000 for columns and 700000 for trees (conservative soft-limits discussed).
        //
        // 'column_write_batch_size' is how many nodes to chunk the base layer of data
        // into when persisting to disk.
        //
        // Override these values with care using environment variables:
        // FIL_PROOFS_MAX_GPU_COLUMN_BATCH_SIZE, FIL_PROOFS_MAX_GPU_TREE_BATCH_SIZE, and
        // FIL_PROOFS_COLUMN_WRITE_BATCH_SIZE respectively.

        let (max_gpu_column_batch_size, max_gpu_tree_batch_size, column_write_batch_size) = {
            let settings_lock = settings::SETTINGS
                .lock()
                .expect("max_gpu_column_batch_size settings lock failure");

            (
                settings_lock.max_gpu_column_batch_size as usize,
                settings_lock.max_gpu_tree_batch_size as usize,
                settings_lock.column_write_batch_size as usize,
            )
        };

        // This channel will receive batches of columns and add them to the ColumnTreeBuilder.
        let (builder_tx, builder_rx) = mpsc::sync_channel(0);

        let configs = &configs;
        let replica_path = &replica_path;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(64)
            .build()
            .unwrap();

        pool.install(|| {
            rayon::join(
                move || {
                    Self::create_batch_gpu::<ColumnArity, _>(
                        nodes_count,
                        configs,
                        labels,
                        builder_tx,
                        max_gpu_column_batch_size,
                        replica_path,
                    )
                },
                move || {
                    Self::receive_and_generate_tree_c::<ColumnArity, TreeArity, _>(
                        nodes_count,
                        configs,
                        builder_rx,
                        max_gpu_tree_batch_size,
                        max_gpu_column_batch_size,
                        column_write_batch_size,
                        replica_path,
                        gpu_index,
                    )
                },
            );
        });

        create_disk_tree::<
            DiskTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>,
        >(configs[0].size.expect("config size failure"), &configs)
    }

    #[allow(clippy::needless_range_loop)]
    fn generate_tree_c_gpu<ColumnArity, TreeArity, P>(
        nodes_count: usize,
        configs: Vec<StoreConfig>,
        labels: &[(PathBuf, String)],
        replica_path: P,
        gpu_index: usize,
    ) -> Result<DiskTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>>
    where
        ColumnArity: 'static + PoseidonArity,
        TreeArity: PoseidonArity,
        P: AsRef<Path> + Send + Sync,
    {
        info!(
            "{:?}, generating tree c using the GPU",
            replica_path.as_ref()
        );
        // Build the tree for CommC
        measure_op(GenerateTreeC, || {
            Self::generate_tree_c_gpu_impl::<ColumnArity, TreeArity, _>(
                nodes_count,
                configs,
                labels,
                replica_path,
                gpu_index,
            )
        })
    }

    #[allow(dead_code)]
    fn generate_tree_c_cpu<ColumnArity, TreeArity>(
        layers: usize,
        nodes_count: usize,
        tree_count: usize,
        configs: Vec<StoreConfig>,
        labels: &LabelsCache<Tree>,
    ) -> Result<DiskTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>>
    where
        ColumnArity: PoseidonArity,
        TreeArity: PoseidonArity,
    {
        info!("generating tree c using the CPU");
        measure_op(GenerateTreeC, || {
            info!("Building column hashes");

            let mut trees = Vec::with_capacity(tree_count);
            for (i, config) in configs.iter().enumerate() {
                let mut hashes: Vec<<Tree::Hasher as Hasher>::Domain> =
                    vec![<Tree::Hasher as Hasher>::Domain::default(); nodes_count];

                rayon::scope(|s| {
                    let n = num_cpus::get();

                    // only split if we have at least two elements per thread
                    let num_chunks = if n > nodes_count * 2 { 1 } else { n };

                    // chunk into n chunks
                    let chunk_size = (nodes_count as f64 / num_chunks as f64).ceil() as usize;

                    // calculate all n chunks in parallel
                    for (chunk, hashes_chunk) in hashes.chunks_mut(chunk_size).enumerate() {
                        let labels = &labels;

                        s.spawn(move |_| {
                            for (j, hash) in hashes_chunk.iter_mut().enumerate() {
                                let data: Vec<_> = (1..=layers)
                                    .map(|layer| {
                                        let store = labels.labels_for_layer(layer);
                                        let el: <Tree::Hasher as Hasher>::Domain = store
                                            .read_at((i * nodes_count) + j + chunk * chunk_size)
                                            .expect("store read_at failure");
                                        el.into()
                                    })
                                    .collect();

                                *hash = hash_single_column(&data).into();
                            }
                        });
                    }
                });

                info!("building base tree_c {}/{}", i + 1, tree_count);
                trees.push(DiskTree::<
                    Tree::Hasher,
                    Tree::Arity,
                    typenum::U0,
                    typenum::U0,
                >::from_par_iter_with_config(
                    hashes.into_par_iter(), config.clone()
                ));
            }

            assert_eq!(tree_count, trees.len());
            create_disk_tree::<
                DiskTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>,
            >(configs[0].size.expect("config size failure"), &configs)
        })
    }

    fn generate_tree_r_last<'b, TreeArity>(
        mut data: Data<'b>,
        nodes_count: usize,
        tree_count: usize,
        tree_r_last_config: StoreConfig,
        replica_path: PathBuf,
        labels: &'a LabelsCache<Tree>,
        gpu_index: usize,
    ) -> Result<(
        LCTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>,
        Data<'b>,
    )>
    where
        TreeArity: PoseidonArity,
    {
        let (configs, replica_config) = split_config_and_replica(
            tree_r_last_config.clone(),
            replica_path.clone(),
            nodes_count,
            tree_count,
        )?;

        data.ensure_data()?;
        let last_layer_labels = labels.labels_for_last_layer()?;

        if settings::SETTINGS
            .lock()
            .expect("use_gpu_tree_builder settings lock failure")
            .use_gpu_tree_builder
        {
            info!(
                "{:?}: generating tree r last using the GPU, replica",
                replica_path
            );
            let max_gpu_tree_batch_size = settings::SETTINGS
                .lock()
                .expect("max_gpu_tree_batch_size settings lock failure")
                .max_gpu_tree_batch_size as usize;

            // This channel will receive batches of leaf nodes and add them to the TreeBuilder.
            let (builder_tx, builder_rx) = mpsc::sync_channel::<(Vec<Fr>, bool)>(0);
            let config_count = configs.len(); // Don't move config into closure below.
            let configs = &configs;

            crossbeam::scope(|s| {
                let data = &mut data;
                s.spawn(move |_| {
                    for i in 0..config_count {
                        let mut node_index = 0;
                        while node_index != nodes_count {
                            let chunked_nodes_count =
                                std::cmp::min(nodes_count - node_index, max_gpu_tree_batch_size);
                            let start = (i * nodes_count) + node_index;
                            let end = start + chunked_nodes_count;
                            trace!(
                                "processing config {}/{} with leaf nodes {} [{}, {}, {}-{}]",
                                i + 1,
                                tree_count,
                                chunked_nodes_count,
                                node_index,
                                nodes_count,
                                start,
                                end,
                            );

                            let encoded_data = last_layer_labels
                                .read_range(start..end)
                                .expect("failed to read layer range")
                                .into_par_iter()
                                .zip(
                                    data.as_mut()[(start * NODE_SIZE)..(end * NODE_SIZE)]
                                        .par_chunks_mut(NODE_SIZE),
                                )
                                .map(|(key, data_node_bytes)| {
                                    let data_node =
                                        <Tree::Hasher as Hasher>::Domain::try_from_bytes(
                                            data_node_bytes,
                                        )
                                        .expect("try_from_bytes failed");
                                    let encoded_node =
                                        encode::<<Tree::Hasher as Hasher>::Domain>(key, data_node);
                                    data_node_bytes
                                        .copy_from_slice(AsRef::<[u8]>::as_ref(&encoded_node));

                                    encoded_node
                                });

                            node_index += chunked_nodes_count;
                            trace!(
                                "node index {}/{}/{}",
                                node_index,
                                chunked_nodes_count,
                                nodes_count,
                            );

                            let encoded: Vec<_> =
                                encoded_data.into_par_iter().map(|x| x.into()).collect();

                            let is_final = node_index == nodes_count;
                            builder_tx
                                .send((encoded, is_final))
                                .expect("failed to send encoded");
                        }
                    }
                });

                {
                    let tree_r_last_config = &tree_r_last_config;
                    s.spawn(move |_| {
                        let mut tree_builder = TreeBuilder::<Tree::Arity>::new(
                            Some(BatcherType::GPU),
                            nodes_count,
                            max_gpu_tree_batch_size,
                            tree_r_last_config.rows_to_discard,
                            gpu_index,
                        )
                        .expect("failed to create TreeBuilder");

                        let mut i = 0;
                        let mut config = &configs[i];

                        // Loop until all trees for all configs have been built.
                        while i < configs.len() {
                            let (encoded, is_final) =
                                builder_rx.recv().expect("failed to recv encoded data");

                            // Just add non-final leaf batches.
                            if !is_final {
                                tree_builder
                                    .add_leaves(&encoded)
                                    .expect("failed to add leaves");
                                continue;
                            };

                            // If we get here, this is a final leaf batch: build a sub-tree.
                            info!(
                                "{:?}: building base tree_r_last with GPU {}/{}, replica",
                                replica_path,
                                i + 1,
                                tree_count,
                            );
                            let (_, tree_data) = tree_builder
                                .add_final_leaves(&encoded)
                                .expect("failed to add final leaves");
                            let tree_data_len = tree_data.len();
                            let cache_size = get_merkle_tree_cache_size(
                                get_merkle_tree_leafs(
                                    config.size.expect("config size failure"),
                                    Tree::Arity::to_usize(),
                                )
                                .expect("failed to get merkle tree leaves"),
                                Tree::Arity::to_usize(),
                                config.rows_to_discard,
                            )
                            .expect("failed to get merkle tree cache size");
                            assert_eq!(tree_data_len, cache_size);

                            let flat_tree_data: Vec<_> = tree_data
                                .into_par_iter()
                                .flat_map(|el| fr_into_bytes(&el))
                                .collect();

                            // Persist the data to the store based on the current config.
                            let tree_r_last_path = StoreConfig::data_path(&config.path, &config.id);
                            trace!(
                                "persisting tree r of len {} with {} rows to discard at path {:?}",
                                tree_data_len,
                                config.rows_to_discard,
                                tree_r_last_path
                            );
                            let mut f = OpenOptions::new()
                                .create(true)
                                .write(true)
                                .open(&tree_r_last_path)
                                .expect("failed to open file for tree_r_last");

                            f.write_all(&flat_tree_data)
                                .expect("failed to wrote tree_r_last data");

                            // Move on to the next config.
                            i += 1;
                            if i == configs.len() {
                                break;
                            }
                            config = &configs[i];
                        }
                    });
                }
            }).unwrap();
        } else {
            info!(
                "{:?}: generating tree r last using the CPU, replica",
                replica_path
            );
            let size = Store::len(last_layer_labels);

            let mut start = 0;
            let mut end = size / tree_count;

            for (i, config) in configs.iter().enumerate() {
                let encoded_data = last_layer_labels
                    .read_range(start..end)?
                    .into_par_iter()
                    .zip(
                        data.as_mut()[(start * NODE_SIZE)..(end * NODE_SIZE)]
                            .par_chunks_mut(NODE_SIZE),
                    )
                    .map(|(key, data_node_bytes)| {
                        let data_node =
                            <Tree::Hasher as Hasher>::Domain::try_from_bytes(data_node_bytes)
                                .expect("try from bytes failed");
                        let encoded_node =
                            encode::<<Tree::Hasher as Hasher>::Domain>(key, data_node);
                        data_node_bytes.copy_from_slice(AsRef::<[u8]>::as_ref(&encoded_node));

                        encoded_node
                    });

                info!(
                    "{:?}: building base tree_r_last with CPU {}/{}, replica",
                    replica_path,
                    i + 1,
                    tree_count,
                );
                LCTree::<Tree::Hasher, Tree::Arity, typenum::U0, typenum::U0>::from_par_iter_with_config(encoded_data, config.clone())?;

                start = end;
                end += size / tree_count;
            }
        };

        create_lc_tree::<LCTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>>(
            tree_r_last_config.size.expect("config size failure"),
            &configs,
            &replica_config,
        )
        .map(|x| (x, data))
    }

    pub(crate) fn transform_and_replicate_layers(
        graph: &StackedBucketGraph<Tree::Hasher>,
        layer_challenges: &LayerChallenges,
        replica_id: &<Tree::Hasher as Hasher>::Domain,
        data: Data,
        data_tree: Option<BinaryMerkleTree<G>>,
        config: StoreConfig,
        replica_path: PathBuf,
        gpu_index: usize,
    ) -> Result<TransformedLayers<Tree, G>> {
        // Generate key layers.
        let (_, labels) = measure_op(EncodeWindowTimeAll, || {
            Self::generate_labels(graph, layer_challenges, replica_id, config.clone())
        })?;

        Self::transform_and_replicate_layers_inner(
            graph,
            layer_challenges,
            data,
            data_tree,
            config,
            replica_path,
            labels,
            gpu_index,
        )
    }

    #[allow(dead_code)]
    fn replicate_inner_tree_d<'b>(
        tree_d_config: &StoreConfig,
        data_tree: Option<BinaryMerkleTree<G>>,
        mut data: Data<'b>,
    ) -> Result<(BinaryMerkleTree<G>, Data<'b>)> {
        let tree_d = match data_tree {
            Some(t) => {
                trace!("using existing original data merkle tree");
                assert_eq!(t.len(), 2 * (data.len() / NODE_SIZE) - 1);

                t
            }
            None => {
                trace!("building merkle tree for the original data");
                data.ensure_data()?;
                measure_op(CommD, || {
                    Self::build_binary_tree::<G>(data.as_ref(), tree_d_config.clone())
                })?
            }
        };
        Ok((tree_d, data))
    }

    #[allow(dead_code)]
    fn replicate_inner_tree_r<'b>(
        tree_r_config: &StoreConfig,
        data: Data<'b>,
        nodes_count: usize,
        tree_count: usize,
        replica_path: PathBuf,
        labels: &LabelsCache<Tree>,
        gpu_index: usize,
    ) -> Result<LCTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>> {
        let tree_r_last = measure_op(GenerateTreeRLast, || {
            Self::generate_tree_r_last::<Tree::Arity>(
                data,
                nodes_count,
                tree_count,
                tree_r_config.clone(),
                replica_path.clone(),
                labels,
                gpu_index,
            )
        })?;

        Ok(tree_r_last.0)
    }

    pub(crate) fn transform_and_replicate_layers_inner(
        graph: &StackedBucketGraph<Tree::Hasher>,
        layer_challenges: &LayerChallenges,
        data: Data,
        data_tree: Option<BinaryMerkleTree<G>>,
        config: StoreConfig,
        replica_path: PathBuf,
        label_configs: Labels<Tree>,
        gpu_index: usize,
    ) -> Result<TransformedLayers<Tree, G>> {
        trace!("transform_and_replicate_layers");
        let nodes_count = graph.size();

        assert_eq!(data.len(), nodes_count * NODE_SIZE);
        trace!("nodes count {}, data len {}", nodes_count, data.len());

        let tree_count = get_base_tree_count::<Tree>();
        let nodes_count = graph.size() / tree_count;

        // Ensure that the node count will work for binary and oct arities.
        let binary_arity_valid = is_merkle_tree_size_valid(nodes_count, BINARY_ARITY);
        let other_arity_valid = is_merkle_tree_size_valid(nodes_count, Tree::Arity::to_usize());
        trace!(
            "is_merkle_tree_size_valid({}, BINARY_ARITY) = {}",
            nodes_count,
            binary_arity_valid
        );
        trace!(
            "is_merkle_tree_size_valid({}, {}) = {}",
            nodes_count,
            Tree::Arity::to_usize(),
            other_arity_valid
        );
        assert!(binary_arity_valid);
        assert!(other_arity_valid);

        let layers = layer_challenges.layers();
        assert!(layers > 0);

        // Generate all store configs that we need based on the
        // cache_path in the specified config.
        let tree_d_config = {
            let mut cfg = StoreConfig::from_config(
                &config,
                CacheKey::CommDTree.to_string(),
                Some(get_merkle_tree_len(nodes_count, BINARY_ARITY)?),
            );
            cfg.rows_to_discard = default_rows_to_discard(nodes_count, BINARY_ARITY);
            cfg
        };
        // A default 'rows_to_discard' value will be chosen for tree_r_last, unless the user overrides this value via the
        // environment setting (FIL_PROOFS_ROWS_TO_DISCARD).  If this value is specified, no checking is done on it and it may
        // result in a broken configuration.  Use with caution.  It must be noted that if/when this unchecked value is passed
        // through merkle_light, merkle_light now does a check that does not allow us to discard more rows than is possible
        // to discard.
        let tree_r_last_config = {
            let mut cfg = StoreConfig::from_config(
                &config,
                CacheKey::CommRLastTree.to_string(),
                Some(get_merkle_tree_len(nodes_count, Tree::Arity::to_usize())?),
            );
            cfg.rows_to_discard = default_rows_to_discard(nodes_count, Tree::Arity::to_usize());
            cfg
        };

        trace!(
            "tree_r_last using rows_to_discard={}",
            tree_r_last_config.rows_to_discard
        );

        let tree_c_config = {
            let mut cfg = StoreConfig::from_config(
                &config,
                CacheKey::CommCTree.to_string(),
                Some(get_merkle_tree_len(nodes_count, Tree::Arity::to_usize())?),
            );
            cfg.rows_to_discard = default_rows_to_discard(nodes_count, Tree::Arity::to_usize());
            cfg
        };

        let paths = label_configs
            .labels
            .iter()
            .map(|x| (x.path.clone(), x.id.clone()))
            .collect::<Vec<(PathBuf, String)>>();

        let labels = LabelsCache::<Tree>::new(&label_configs)?;
        let configs = split_config(tree_c_config.clone(), tree_count)?;

        let (c_tx, c_rx) = mpsc::sync_channel(5);
        let (r_tx, r_rx) = mpsc::sync_channel(5);

        crossbeam::scope(|s| {
            s.spawn(|_| {
                let tree_c = match layers {
                    2 => Self::generate_tree_c::<U2, Tree::Arity, _>(
                        nodes_count,
                        configs,
                        tree_count,
                        &paths,
                        &labels,
                        &replica_path,
                        gpu_index,
                    ),
                    8 => Self::generate_tree_c::<U8, Tree::Arity, _>(
                        nodes_count,
                        configs,
                        tree_count,
                        &paths,
                        &labels,
                        &replica_path,
                        gpu_index,
                    ),
                    11 => Self::generate_tree_c::<U11, Tree::Arity, _>(
                        nodes_count,
                        configs,
                        tree_count,
                        &paths,
                        &labels,
                        &replica_path,
                        gpu_index,
                    ),
                    _ => panic!("Unsupported column arity"),
                };
                info!("{:?}: tree_c done", &replica_path);
                c_tx.send(tree_c).unwrap();
            });
            // Build the MerkleTree over the original data (if needed).
            let (tree_d, data) = match data_tree {
                Some(t) => {
                    trace!("using existing original data merkle tree");
                    assert_eq!(t.len(), 2 * (data.len() / NODE_SIZE) - 1);

                    (t, data)
                }
                None => {
                    let mut data = data;
                    trace!("building merkle tree for the original data");
                    data.ensure_data().unwrap();
                    (
                        measure_op(CommD, || {
                            Self::build_binary_tree::<G>(data.as_ref(), tree_d_config.clone())
                        })
                        .unwrap(),
                        data,
                    )
                }
            };
            let tree_d_config = StoreConfig {
                size: Some(tree_d.len()),
                ..tree_d_config
            };
            // tree_d_config.size = Some(tree_d.len());
            assert_eq!(
                tree_d_config.size.expect("config size failure"),
                tree_d.len()
            );
            let tree_d_root = tree_d.root();
            drop(tree_d);

            // Encode original data into the last layer.
            info!("{:?}: building tree_r_last", &replica_path);
            let (tree_r_last, data) = measure_op(GenerateTreeRLast, || {
                Self::generate_tree_r_last::<Tree::Arity>(
                    data,
                    nodes_count,
                    tree_count,
                    tree_r_last_config.clone(),
                    replica_path.clone(),
                    &labels,
                    gpu_index,
                )
            })
            .unwrap();
            info!("tree_r_last done");

            r_tx.send((tree_r_last, tree_d_root, data, tree_d_config))
                .unwrap();
        }).unwrap();

        let tree_c = c_rx.recv().unwrap();
        let (tree_r_last, tree_d_root, data, tree_d_config) = r_rx.recv().unwrap();
        let tree_c_root = tree_c?.root();

        let tree_r_last_root = tree_r_last.root();
        drop(tree_r_last);

        drop(data);

        // comm_r = H(comm_c || comm_r_last)
        let comm_r: <Tree::Hasher as Hasher>::Domain =
            <Tree::Hasher as Hasher>::Function::hash2(&tree_c_root, &tree_r_last_root);

        Ok((
            Tau {
                comm_d: tree_d_root,
                comm_r,
            },
            PersistentAux {
                comm_c: tree_c_root,
                comm_r_last: tree_r_last_root,
            },
            TemporaryAux {
                labels: label_configs,
                tree_d_config,
                tree_r_last_config,
                tree_c_config,
                _g: PhantomData,
            },
        ))
    }

    /// Phase1 of replication.
    pub fn my_replicate_phase1(
        pp: &'a PublicParams<Tree>,
        replica_id: &<Tree::Hasher as Hasher>::Domain,
        config: StoreConfig,
    ) -> Result<Labels<Tree>> {
        info!("my_replicate_phase1");

        let (_, labels) = measure_op(EncodeWindowTimeAll, || {
            Self::my_generate_labels(&pp.graph, &pp.layer_challenges, replica_id, config)
        })?;

        Ok(labels)
    }

    /// Phase1 of replication.
    pub fn replicate_phase1(
        pp: &'a PublicParams<Tree>,
        replica_id: &<Tree::Hasher as Hasher>::Domain,
        config: StoreConfig,
    ) -> Result<Labels<Tree>> {
        info!("replicate_phase1");

        let (_, labels) = measure_op(EncodeWindowTimeAll, || {
            Self::generate_labels(&pp.graph, &pp.layer_challenges, replica_id, config)
        })?;

        Ok(labels)
    }

    #[allow(clippy::type_complexity)]
    /// Phase2 of replication.
    #[allow(clippy::type_complexity)]
    pub fn replicate_phase2(
        pp: &'a PublicParams<Tree>,
        labels: Labels<Tree>,
        data: Data<'a>,
        data_tree: BinaryMerkleTree<G>,
        config: StoreConfig,
        replica_path: PathBuf,
        gpu_index: usize,
    ) -> Result<(
        <Self as PoRep<'a, Tree::Hasher, G>>::Tau,
        <Self as PoRep<'a, Tree::Hasher, G>>::ProverAux,
    )> {
        info!("replicate_phase2");

        let (tau, paux, taux) = Self::transform_and_replicate_layers_inner(
            &pp.graph,
            &pp.layer_challenges,
            data,
            Some(data_tree),
            config,
            replica_path,
            labels,
            gpu_index,
        )?;

        Ok((tau, (paux, taux)))
    }

    // Assumes data is all zeros.
    // Replica path is used to create configs, but is not read.
    // Instead new zeros are provided (hence the need for replica to be all zeros).
    fn generate_fake_tree_r_last<TreeArity>(
        nodes_count: usize,
        tree_count: usize,
        tree_r_last_config: StoreConfig,
        replica_path: PathBuf,
        gpu_index: usize,
    ) -> Result<LCTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>>
    where
        TreeArity: PoseidonArity,
    {
        let (configs, replica_config) = split_config_and_replica(
            tree_r_last_config.clone(),
            replica_path,
            nodes_count,
            tree_count,
        )?;

        if settings::SETTINGS
            .lock()
            .expect("use_gpu_tree_builder settings lock failure")
            .use_gpu_tree_builder
        {
            info!("generating tree r last using the GPU");
            let max_gpu_tree_batch_size = settings::SETTINGS
                .lock()
                .expect("max_gpu_tree_batch_size settings lock failure")
                .max_gpu_tree_batch_size as usize;

            let mut tree_builder = TreeBuilder::<Tree::Arity>::new(
                Some(BatcherType::GPU),
                nodes_count,
                max_gpu_tree_batch_size,
                tree_r_last_config.rows_to_discard,
                gpu_index,
            )
            .expect("failed to create TreeBuilder");

            // Allocate zeros once and reuse.
            let zero_leaves: Vec<Fr> = vec![Fr::zero(); max_gpu_tree_batch_size];
            for (i, config) in configs.iter().enumerate() {
                let mut consumed = 0;
                while consumed < nodes_count {
                    let batch_size = usize::min(max_gpu_tree_batch_size, nodes_count - consumed);

                    consumed += batch_size;

                    if consumed != nodes_count {
                        tree_builder
                            .add_leaves(&zero_leaves[0..batch_size])
                            .expect("failed to add leaves");
                        continue;
                    };

                    // If we get here, this is a final leaf batch: build a sub-tree.
                    info!(
                        "building base tree_r_last with GPU {}/{}",
                        i + 1,
                        tree_count
                    );

                    let (_, tree_data) = tree_builder
                        .add_final_leaves(&zero_leaves[0..batch_size])
                        .expect("failed to add final leaves");
                    let tree_data_len = tree_data.len();
                    let cache_size = get_merkle_tree_cache_size(
                        get_merkle_tree_leafs(
                            config.size.expect("config size failure"),
                            Tree::Arity::to_usize(),
                        )
                        .expect("failed to get merkle tree leaves"),
                        Tree::Arity::to_usize(),
                        config.rows_to_discard,
                    )
                    .expect("failed to get merkle tree cache size");
                    assert_eq!(tree_data_len, cache_size);

                    let flat_tree_data: Vec<_> = tree_data
                        .into_par_iter()
                        .flat_map(|el| fr_into_bytes(&el))
                        .collect();

                    // Persist the data to the store based on the current config.
                    let tree_r_last_path = StoreConfig::data_path(&config.path, &config.id);
                    trace!(
                        "persisting tree r of len {} with {} rows to discard at path {:?}",
                        tree_data_len,
                        config.rows_to_discard,
                        tree_r_last_path
                    );
                    let mut f = OpenOptions::new()
                        .create(true)
                        .write(true)
                        .open(&tree_r_last_path)
                        .expect("failed to open file for tree_r_last");
                    f.write_all(&flat_tree_data)
                        .expect("failed to wrote tree_r_last data");
                }
            }
        } else {
            info!("generating tree r last using the CPU");
            for (i, config) in configs.iter().enumerate() {
                let encoded_data = vec![<Tree::Hasher as Hasher>::Domain::default(); nodes_count];

                info!(
                    "building base tree_r_last with CPU {}/{}",
                    i + 1,
                    tree_count
                );
                LCTree::<Tree::Hasher, Tree::Arity, typenum::U0, typenum::U0>::from_par_iter_with_config(encoded_data, config.clone())?;
            }
        };

        create_lc_tree::<LCTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>>(
            tree_r_last_config.size.expect("config size failure"),
            &configs,
            &replica_config,
        )
    }

    pub fn fake_replicate_phase2<R: AsRef<Path>, S: AsRef<Path>>(
        tree_c_root: <Tree::Hasher as Hasher>::Domain,
        replica_path: R,
        cache_path: S,
        sector_size: usize,
        gpu_index: usize,
    ) -> Result<(
        <Tree::Hasher as Hasher>::Domain,
        PersistentAux<<Tree::Hasher as Hasher>::Domain>,
    )> {
        let leaf_count = sector_size / NODE_SIZE;
        let replica_pathbuf = PathBuf::from(replica_path.as_ref());
        assert_eq!(0, sector_size % NODE_SIZE);
        let tree_count = get_base_tree_count::<Tree>();
        let nodes_count = leaf_count / tree_count;

        let config = StoreConfig::new(
            cache_path.as_ref(),
            CacheKey::CommRLastTree.to_string(),
            default_rows_to_discard(nodes_count, Tree::Arity::to_usize()),
        );
        let tree_r_last_config = StoreConfig::from_config(
            &config,
            CacheKey::CommRLastTree.to_string(),
            Some(get_merkle_tree_len(nodes_count, Tree::Arity::to_usize())?),
        );

        // Encode original data into the last layer.
        info!("building tree_r_last");
        let tree_r_last = Self::generate_fake_tree_r_last::<Tree::Arity>(
            nodes_count,
            tree_count,
            tree_r_last_config,
            replica_pathbuf,
            gpu_index,
        )?;
        info!("tree_r_last done");

        let tree_r_last_root = tree_r_last.root();
        drop(tree_r_last);

        // comm_r = H(comm_c || comm_r_last)
        let comm_r: <Tree::Hasher as Hasher>::Domain =
            <Tree::Hasher as Hasher>::Function::hash2(&tree_c_root, &tree_r_last_root);

        let p_aux = PersistentAux {
            comm_c: tree_c_root,
            comm_r_last: tree_r_last_root,
        };

        Ok((comm_r, p_aux))
    }

    pub fn fake_comm_r<R: AsRef<Path>>(
        tree_c_root: <Tree::Hasher as Hasher>::Domain,
        existing_p_aux_path: R,
    ) -> Result<(
        <Tree::Hasher as Hasher>::Domain,
        PersistentAux<<Tree::Hasher as Hasher>::Domain>,
    )> {
        let existing_p_aux: PersistentAux<<Tree::Hasher as Hasher>::Domain> = {
            let p_aux_bytes = std::fs::read(&existing_p_aux_path)?;

            deserialize(&p_aux_bytes)
        }?;

        let existing_comm_r_last = existing_p_aux.comm_r_last;

        // comm_r = H(comm_c || comm_r_last)
        let comm_r: <Tree::Hasher as Hasher>::Domain =
            <Tree::Hasher as Hasher>::Function::hash2(&tree_c_root, &existing_comm_r_last);

        let p_aux = PersistentAux {
            comm_c: tree_c_root,
            comm_r_last: existing_comm_r_last,
        };

        Ok((comm_r, p_aux))
    }
}
