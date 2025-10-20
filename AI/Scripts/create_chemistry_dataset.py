# Symmer imports
from symmer.evolution.variational_optimization import ADAPT_VQE
from symmer.evolution import PauliwordOp_to_QuantumCircuit
from symmer import PauliwordOp, QubitTapering

# Openfermion imports
from openfermion.transforms import jordan_wigner, bravyi_kitaev
from openfermion import MolecularData, FermionOperator

# PySCF imports
from pyscf.cc.addons import spatial2spin
from pyscf import gto, scf, cc

# Qiskit imports
from qiskit import qasm2, transpile

# Standard library imports
from typing import Literal, Tuple, List
import numpy as np


def build_adapt_circuit(
        geometry: List[Tuple[str, Tuple[float, float, float]]],
        basis: str,
        mapper: Literal["JW", "BK"] = "JW",
        max_adapt_cycles: int = 8,
        seed: int = 0,
):
    # 1) PySCF HF + CCSD (gets t1,t2 and HF occupation)
    mol = gto.M(atom=geometry, basis=basis, unit="Angstrom", charge=0, spin=0)
    mf = scf.RHF(mol).run()
    mycc = cc.CCSD(mf).run()

    # HF bitstring (alpha then beta in PySCF MO order)
    nelec_a, nelec_b = mol.nelec
    nmo = mf.mo_coeff.shape[1]
    hf_occ = np.zeros(2 * nmo, dtype=int)
    hf_occ[:nelec_a] = 1
    hf_occ[nmo:nmo + nelec_b] = 1
    hf_bitstring = "".join(map(str, hf_occ.tolist()))

    # 2) OpenFermion molecular Hamiltonian (second-quantized) → qubit
    # one-/two-electron integrals from PySCF
    of_mol = MolecularData(geometry=geometry, basis=basis, charge=0, multiplicity=1)
    of_mol.n_orbitals = nmo
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, mf._eri, nmo) if hasattr(mf, "_eri") else mf._eri  # type: ignore
    of_mol.one_body_integrals = h1
    of_mol.two_body_integrals = eri
    from openfermion.chem import MolecularData as _  # ensures attributes exist

    from openfermion.hamiltonians import MolecularHamiltonian
    mol_ham = MolecularHamiltonian(of_mol.one_body_integrals, of_mol.two_body_integrals, of_mol.nuclear_repulsion)
    ferm_op = mol_ham.get_molecular_hamiltonian()

    qubit_map = {"JW": jordan_wigner, "BK": bravyi_kitaev}[mapper]
    H_q = qubit_map(ferm_op)
    H = PauliwordOp.from_openfermion(H_q)

    # 3) taper using HF sector
    QT = QubitTapering(H)
    H_taper = QT.taper_it(ref_state=hf_bitstring)
    hf_state = QT.tapered_ref_state

    # 4) CCSD-based excitation pool → PauliwordOp (anti-Hermitian generator)
    t1 = spatial2spin(mycc.t1, orbspin=None)
    t2 = spatial2spin(mycc.t2, orbspin=None)
    no, nv = t1.shape
    nso = no + nv
    occ_mask = np.zeros(nso, dtype=bool);
    occ_mask[:no] = True
    singles = [(i, j) for i in range(nso) for j in range(nso) if
               (not occ_mask[i]) and occ_mask[j] and abs(t1[j, i]) > 1e-10]
    doubles = [(a, i, b, j) for a in range(nso) for i in range(nso) for b in range(nso) for j in range(nso)
               if
               (not occ_mask[a]) and occ_mask[i] and (not occ_mask[b]) and occ_mask[j] and abs(t2[i, a, j, b]) > 1e-10]
    gen = FermionOperator()
    for (a, i) in singles:  gen += FermionOperator(f"{a}^ {i}", t1[i, a])
    for (a, i, b, j) in doubles: gen += FermionOperator(f"{a}^ {i} {b}^ {j}", 0.25 * t2[i, a, j, b])
    G = PauliwordOp.from_openfermion(qubit_map(gen)) - PauliwordOp.from_openfermion(
        qubit_map(gen.conjugate_transpose()))
    G = QT.taper_it(aux_operator=G).sort()  # align with tapered H

    # 5) tiny ADAPT run to pick a few ops (keeps generation fast)
    adapt = ADAPT_VQE(observable=H_taper, excitation_pool=G, ref_state=hf_state)
    adapt.topology_aware = False
    out = adapt.optimize(max_cycles=max_adapt_cycles, random_seed=seed)
    picked_ops = out["adapt_operator"]  # list of (coeff, 'P...') tuples

    # 6) build parameterized circuit (HF prep + selected exponentials), dump QASM2
    from qiskit import QuantumCircuit
    qc_sel = PauliwordOp_to_QuantumCircuit(PauliwordOp.from_list(picked_ops), bind_params=False, include_barriers=False)
    ansatz = QuantumCircuit(qc_sel.num_qubits)
    for k, bit in enumerate(hf_state.to_dictionary.keys()[0] if hasattr(hf_state, 'to_dictionary') else hf_bitstring):
        if bit == "1":
            ansatz.x(qc_sel.num_qubits - 1 - k)
    for inst in qc_sel.data:
        ansatz.append(inst)

    # random parameters and unroll to QASM2-safe basis
    rng = np.random.default_rng(seed)
    vals = rng.uniform(-2 * np.pi, 2 * np.pi, len(ansatz.parameters))
    qc = ansatz.assign_parameters(dict(zip(ansatz.parameters, vals)))
    qc = transpile(qc, basis_gates=['u', 'cx'], optimization_level=0)
    return qc, qasm2.dumps(qc)
