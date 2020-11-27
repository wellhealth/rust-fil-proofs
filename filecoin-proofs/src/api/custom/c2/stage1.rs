use bellperson::bls::Bls12;
use bellperson::bls::Fr;
use bellperson::gadgets::num;
use bellperson::gadgets::Assignment;
use bellperson::groth16::ProvingAssignment;
use bellperson::ConstraintSystem;
use bellperson::Index;
use bellperson::SynthesisError;
use bellperson::Variable;
use num::AllocatedNum;
use storage_proofs::gadgets::constraint;
use storage_proofs::hasher::HashFunction;
use storage_proofs::hasher::Hasher;
use storage_proofs::merkle::MerkleTreeTrait;
use storage_proofs::porep::stacked::StackedCircuit;
use storage_proofs::util::reverse_bit_numbering;

use crate::DefaultPieceHasher;

pub fn circuit_synthesize<Tree: 'static + MerkleTreeTrait>(
    s: StackedCircuit<'static, Tree, DefaultPieceHasher>,
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
    } = s;

    // Allocate replica_id
    let replica_id_num = num_alloc(cs, || {
        replica_id
            .map(Into::into)
            .ok_or_else(|| SynthesisError::AssignmentMissing)
    })?;

    // make replica_id a public input
    replica_id_num.inputize(cs.namespace(|| "replica_id_input"))?;

    let replica_id_bits =
        reverse_bit_numbering(replica_id_num.to_bits_le(cs.namespace(|| "replica_id_bits"))?);

    // Allocate comm_d as Fr
    let comm_d_num = num_alloc(cs, || {
        comm_d
            .map(Into::into)
            .ok_or_else(|| SynthesisError::AssignmentMissing)
    })?;

    // make comm_d a public input
    comm_d_num.inputize(cs.namespace(|| "comm_d_input"))?;

    // Allocate comm_r as Fr
    let comm_r_num = num_alloc(cs, || {
        comm_r
            .map(Into::into)
            .ok_or_else(|| SynthesisError::AssignmentMissing)
    })?;

    // make comm_r a public input
    num_inputize(&comm_r_num, cs)?;

    // Allocate comm_r_last as Fr
    let comm_r_last_num = num_alloc(cs, || {
        comm_r_last
            .map(Into::into)
            .ok_or_else(|| SynthesisError::AssignmentMissing)
    })?;

    // Allocate comm_c as Fr
    let comm_c_num = num_alloc(cs, || {
        comm_c
            .map(Into::into)
            .ok_or_else(|| SynthesisError::AssignmentMissing)
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
        proof.synthesize(
            cs.namespace(|| ""),
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
