use crate::types::stacked::circuit::params::enforce_inclusion;
use crate::DefaultPieceHasher;
use bellperson::bls::Bls12;
use bellperson::bls::Fr;
use bellperson::gadgets::boolean::Boolean;
use bellperson::gadgets::num;
use bellperson::gadgets::uint32;
use bellperson::gadgets::Assignment;
use bellperson::groth16::ProvingAssignment;
use bellperson::ConstraintSystem;
use bellperson::Index;
use bellperson::SynthesisError;
use bellperson::Variable;
use num::AllocatedNum;
use storage_proofs::gadgets::constraint;
use storage_proofs::gadgets::encode::encode;
use storage_proofs::gadgets::uint64;
use storage_proofs::hasher::HashFunction;
use storage_proofs::hasher::Hasher;
use storage_proofs::merkle::MerkleTreeTrait;
use storage_proofs::porep::stacked::circuit::hash::hash_single_column;
use storage_proofs::porep::stacked::create_label_circuit as create_label;
use storage_proofs::porep::stacked::params::Proof;
use storage_proofs::porep::stacked::StackedCircuit;
use storage_proofs::util::reverse_bit_numbering;

pub fn circuit_synthesize<Tree: 'static + MerkleTreeTrait>(
    circuit: StackedCircuit<'static, Tree, DefaultPieceHasher>,
    cs: &mut ProvingAssignment<Bls12>,
) -> Result<(), SynthesisError> {
    let StackedCircuit {
        public_params,
        replica_id,
        comm_d,
        comm_r,
        comm_r_last,
        comm_c,
        proofs,
        ..
    } = circuit;

    // Allocate replica_id
    let replica_id_num = num_alloc(cs, || {
        replica_id
            .map(Into::into)
            .ok_or(SynthesisError::AssignmentMissing)
    })?;

    // make replica_id a public input
    num_inputize(&replica_id_num, cs)?;

    let replica_id_bits =
        reverse_bit_numbering(replica_id_num.to_bits_le(cs.namespace(|| "replica_id_bits"))?);

    // Allocate comm_d as Fr
    let comm_d_num = num_alloc(cs, || {
        comm_d
            .map(Into::into)
            .ok_or(SynthesisError::AssignmentMissing)
    })?;

    // make comm_d a public input
    num_inputize(&comm_d_num, cs)?;

    // Allocate comm_r as Fr
    let comm_r_num = num_alloc(cs, || {
        comm_r
            .map(Into::into)
            .ok_or(SynthesisError::AssignmentMissing)
    })?;

    // make comm_r a public input
    num_inputize(&comm_r_num, cs)?;

    // Allocate comm_r_last as Fr
    let comm_r_last_num = num_alloc(cs, || {
        comm_r_last
            .map(Into::into)
            .ok_or(SynthesisError::AssignmentMissing)
    })?;

    // Allocate comm_c as Fr
    let comm_c_num = num_alloc(cs, || {
        comm_c
            .map(Into::into)
            .ok_or(SynthesisError::AssignmentMissing)
    })?;

    // Verify comm_r = H(comm_c || comm_r_last)
    {
        let hash_num = <Tree::Hasher as Hasher>::Function::hash2_circuit(
            cs.namespace(|| "H_comm_c_comm_r_last"),
            &comm_c_num,
            &comm_r_last_num,
        )?;

        // Check actual equality
        constraint::equal(
            cs,
            || "enforce comm_r = H(comm_c || comm_r_last)",
            &comm_r_num,
            &hash_num,
        );
    }

    for proof in proofs.into_iter() {
        proof_synthesize(
            proof,
            cs,
            public_params.layer_challenges.layers(),
            &comm_d_num,
            &comm_c_num,
            &comm_r_last_num,
            &replica_id_bits,
        )?;
    }

    Ok(())
}

pub fn num_alloc<F>(
    cs: &mut ProvingAssignment<Bls12>,
    value: F,
) -> Result<AllocatedNum<Bls12>, SynthesisError>
where
    F: FnOnce() -> Result<Fr, SynthesisError>,
{
    let mut new_value = None;
    let var = cs_alloc(
        cs,
        || "num",
        || {
            let tmp = value()?;

            new_value = Some(tmp);

            Ok(tmp)
        },
    )?;

    Ok(AllocatedNum {
        value: new_value,
        variable: var,
    })
}

fn cs_alloc<F, A, AR>(
    x: &mut ProvingAssignment<Bls12>,
    _: A,
    f: F,
) -> Result<Variable, SynthesisError>
where
    F: FnOnce() -> Result<Fr, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
{
    x.aux_assignment.push(f()?);
    x.a_aux_density.add_element();
    x.b_aux_density.add_element();

    Ok(Variable(Index::Aux(x.aux_assignment.len() - 1)))
}

pub fn num_inputize(
    num: &AllocatedNum<Bls12>,
    cs: &mut ProvingAssignment<Bls12>,
) -> Result<(), SynthesisError>
where
{
    let input = cs.alloc_input(|| "input variable", || Ok(*num.value.get()?))?;

    cs.enforce(
        || "enforce input is correct",
        |lc| lc + input,
        |lc| lc + <ProvingAssignment<Bls12> as ConstraintSystem<Bls12>>::one(),
        |lc| lc + num.variable,
    );

    Ok(())
}

pub fn proof_synthesize<Tree: 'static + MerkleTreeTrait>(
    proof: Proof<Tree, DefaultPieceHasher>,
    cs: &mut ProvingAssignment<Bls12>,
    layers: usize,
    comm_d: &num::AllocatedNum<Bls12>,
    comm_c: &num::AllocatedNum<Bls12>,
    comm_r_last: &num::AllocatedNum<Bls12>,
    replica_id: &[Boolean],
) -> Result<(), SynthesisError> {
    let Proof {
        comm_d_path,
        data_leaf,
        challenge,
        comm_r_last_path,
        comm_c_path,
        drg_parents_proofs,
        exp_parents_proofs,
        ..
    } = proof;

    assert!(!drg_parents_proofs.is_empty());
    assert!(!exp_parents_proofs.is_empty());

    // -- verify initial data layer

    // PrivateInput: data_leaf
    let data_leaf_num = num_alloc(cs, || data_leaf.ok_or(SynthesisError::AssignmentMissing))?;

    // enforce inclusion of the data leaf in the tree D
    enforce_inclusion(
        cs.namespace(|| "comm_d_inclusion"),
        comm_d_path,
        comm_d,
        &data_leaf_num,
    )?;

    // -- verify replica column openings

    // Private Inputs for the DRG parent nodes.
    let mut drg_parents = Vec::with_capacity(layers);

    for (i, parent) in drg_parents_proofs.into_iter().enumerate() {
        let (parent_col, inclusion_path) =
            parent.alloc(cs.namespace(|| format!("drg_parent_{}_num", i)))?;
        assert_eq!(layers, parent_col.len());

        // calculate column hash
        let val = parent_col.hash(cs.namespace(|| format!("drg_parent_{}_constraint", i)))?;
        // enforce inclusion of the column hash in the tree C
        enforce_inclusion(
            cs.namespace(|| format!("drg_parent_{}_inclusion", i)),
            inclusion_path,
            comm_c,
            &val,
        )?;
        drg_parents.push(parent_col);
    }

    // Private Inputs for the Expander parent nodes.
    let mut exp_parents = Vec::new();

    for (i, parent) in exp_parents_proofs.into_iter().enumerate() {
        let (parent_col, inclusion_path) =
            parent.alloc(cs.namespace(|| format!("exp_parent_{}_num", i)))?;
        assert_eq!(layers, parent_col.len());

        // calculate column hash
        let val = parent_col.hash(cs.namespace(|| format!("exp_parent_{}_constraint", i)))?;
        // enforce inclusion of the column hash in the tree C
        enforce_inclusion(
            cs.namespace(|| format!("exp_parent_{}_inclusion", i)),
            inclusion_path,
            comm_c,
            &val,
        )?;
        exp_parents.push(parent_col);
    }

    // -- Verify labeling and encoding

    // stores the labels of the challenged column
    let mut column_labels = Vec::new();

    // PublicInput: challenge index
    let challenge_num = uint64::UInt64::alloc(cs.namespace(|| "challenge"), challenge)?;
    challenge_num.pack_into_input(cs.namespace(|| "challenge input"))?;

    for layer in 1..=layers {
        let layer_num = uint32::UInt32::constant(layer as u32);

        let mut cs = cs.namespace(|| format!("labeling_{}", layer));

        // Collect the parents
        let mut parents = Vec::new();

        // all layers have drg parents
        for parent_col in &drg_parents {
            let parent_val_num = parent_col.get_value(layer);
            let parent_val_bits = reverse_bit_numbering(
                parent_val_num
                    .to_bits_le(cs.namespace(|| format!("drg_parent_{}_bits", parents.len())))?,
            );
            parents.push(parent_val_bits);
        }

        // the first layer does not contain expander parents
        if layer > 1 {
            for parent_col in &exp_parents {
                // subtract 1 from the layer index, as the exp parents, are shifted by one, as they
                // do not store a value for the first layer
                let parent_val_num = parent_col.get_value(layer - 1);
                let parent_val_bits =
                    reverse_bit_numbering(parent_val_num.to_bits_le(
                        cs.namespace(|| format!("exp_parent_{}_bits", parents.len())),
                    )?);
                parents.push(parent_val_bits);
            }
        }

        // Duplicate parents, according to the hashing algorithm.
        let mut expanded_parents = parents.clone();
        expanded_parents.extend_from_slice(&parents);
        if layer > 1 {
            expanded_parents.extend_from_slice(&parents[..9]); // 37
        } else {
            // layer 1 only has drg parents
            expanded_parents.extend_from_slice(&parents); // 18
            expanded_parents.extend_from_slice(&parents); // 24
            expanded_parents.extend_from_slice(&parents); // 30
            expanded_parents.extend_from_slice(&parents); // 36
            expanded_parents.push(parents[0].clone()); // 37
        };

        // Reconstruct the label
        let label = create_label(
            cs.namespace(|| "create_label"),
            replica_id,
            expanded_parents,
            layer_num,
            challenge_num.clone(),
        )?;
        column_labels.push(label);
    }

    // -- encoding node
    {
        // encode the node

        // key is the last label
        let key = &column_labels[column_labels.len() - 1];
        let encoded_node = encode(cs.namespace(|| "encode_node"), key, &data_leaf_num)?;

        // verify inclusion of the encoded node
        enforce_inclusion(
            cs.namespace(|| "comm_r_last_data_inclusion"),
            comm_r_last_path,
            comm_r_last,
            &encoded_node,
        )?;
    }

    // -- ensure the column hash of the labels is included
    {
        // calculate column_hash
        let column_hash = hash_single_column(cs.namespace(|| "c_x_column_hash"), &column_labels)?;

        // enforce inclusion of the column hash in the tree C
        enforce_inclusion(
            cs.namespace(|| "c_x_inclusion"),
            comm_c_path,
            comm_c,
            &column_hash,
        )?;
    }

    Ok(())
}
