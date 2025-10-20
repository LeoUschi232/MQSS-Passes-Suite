# System configuration and specification
import os
import psutil

num_cpus = os.cpu_count()
total_memory = psutil.virtual_memory().total / (1024 * 1024)  # in MBs
print(f"OS: {os.name}")
print(f"Num_cpus: {num_cpus}")
print(f"Memory: {total_memory}")

# set env var for the num. of threads in numba, related to ray and symmer runtum
# otherwise, nbed will get error
os.environ["NUMBA_NUM_THREADS"] = "8"

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

from pyscf import scf, fci
from nbed.ham_builder import HamiltonianBuilder
from nbed.driver import NbedDriver

from symmer.evolution.variational_optimization import ADAPT_VQE
from symmer import PauliwordOp, QubitTapering
from symmer.utils import exact_gs_energy
from symmer.evolution import PauliwordOp_to_QuantumCircuit

# Prepare the geometry and nbed config for the molecules

nh3 = "./molecules/nh3.xyz"
h2o = "./molecules/h2o.xyz"
#acetonitrile = "./mols/acetonitrile.xyz"
#formamide = "./mols/formamide.xyz"

basis = "sto-3g"  # could also use 6-31G as an example of a larger basis set

nbed_config = {
    "basis": basis,
    "xc_functional": "vwnrpa",
    "projector": "mu",
    "localization": "spade",
    "convergence": 1e-6,
    "run_ccsd_emb": False,
    "run_fci_emb": False,
    "pyscf_print_level": 0,
    "run_virtual_localization": False,
    "max_ram_memory": 4000,  # could also change to some fraction of total available memory
    "symmetry": True,
}

# Define the embedding driver using Nbed with projection-based embedding
#nh3_driver = NbedDriver(geometry=nh3, n_active_atoms=1, charge=0, **nbed_config)
h2o_driver = NbedDriver(geometry=h2o, n_active_atoms=2, charge=0, **nbed_config)
# acetonitrile_driver = NbedDriver(geometry=acetonitrile, n_active_atoms=3, charge=0, **nbed_config)
# formamide_driver = NbedDriver(geometry=formamide, n_active_atoms=2, charge=0, **nbed_config)

mol_driver = h2o_driver
# From mol_driver to build mol in quantum domain
mol = mol_driver._build_mol()

# calculate restricted Hartree-Fock and full configuration interaction energies:
pyscf_rhf = scf.RHF(mol).run(verbose=0)
pyscf_fci = fci.FCI(pyscf_rhf).run(verbose=0)

print('RHF energy:', pyscf_rhf.e_tot)
print('FCI energy:', pyscf_fci.e_tot)

# Extract the engergy
# calculated by the built-in PBE function in Nbed
nbed_embed_scf = mol_driver.embedded_scf
nbed_embed_fci = mol_driver._run_emb_FCI(nbed_embed_scf)

# embedding correction
emb_corr = (
        mol_driver.e_env
        + mol_driver.two_e_cross
        - mol_driver._mu["correction"]
        - mol_driver._mu["beta_correction"]
)

# the FCI-in-DFT energy:
nbed_pbe_e = nbed_embed_fci.e_tot + emb_corr

print('FCI-in-DFT energy:', nbed_pbe_e)

# Create an Hamiltonian builder object with the wave functions following the FCI method, which requires
#  + embedded_scf
#  + constant_e_shift
ham_builder = HamiltonianBuilder(
    scf_method=nbed_embed_scf,
    constant_e_shift=mol_driver.classical_energy
)

# Build PauliwordOp - Pauli operators
H = PauliwordOp.from_openfermion(ham_builder.build(taper=False))

# Get the Hartree-fock state
hf_state = ham_builder.get_hartree_fock_state()

# Tapering qubits to optimize Hamiltonian
QT = QubitTapering(H)  # deprecated
H = QT.taper_it(ref_state=hf_state)
hf_state = QT.tapered_ref_state

# Assert the number of qubits in Hamiltonian and Hartree-fock state
assert H.n_qubits == hf_state.n_qubits

# Check the ground state energy
e0, psi0 = exact_gs_energy(H.to_sparse_matrix)

# Assert the emb_FCI
assert np.isclose(nbed_pbe_e, e0, atol=1e-7)

print('diagonalized Hamiltonian energy:', e0)

# Another method to build Hamiltonian and wave function
#  + CSSD from embed_scf
#  + Generate operators using openfermion
ccsd, error_ccsd = mol_driver._run_emb_CCSD(nbed_embed_scf)

from pyscf.cc.addons import spatial2spin
from openfermion import InteractionOperator, FermionOperator
from openfermion.transforms import jordan_wigner, bravyi_kitaev


def array_to_dict_nonzero_indices(arr, tol=1e-10):
    where_nonzero = np.where(~np.isclose(arr, 0, atol=tol))
    nonzero_indices = list(zip(*where_nonzero))
    return dict(zip(nonzero_indices, arr[where_nonzero]))


def fermion_to_qubit_operator(fermionic_operator: FermionOperator,
                              qubit_transformation: str = 'JW',
                              n_qubits: int = None):
    """
    Function to convert from fermion operators to qubit operators.
    Note see openfermion.transforms for different fermion to qubit mappings

    Args:
        Fermionic_operator(FermionOperator): any fermionic operator (openfermion)
        qubit_mapping_str (str): fermion to qubit mapping
        N_qubits (int): number of qubits (or spin orbitals)

    Returns:
        qubit_operator (PauliwordOp): qubit operator of fermonic operator (under certain mapping)
    """
    fermonic_to_qubit_map = {
        'JW': jordan_wigner, 'jordan_wigner': jordan_wigner,
        'BK': bravyi_kitaev, 'bravyi_kitaev': bravyi_kitaev,
    }

    if qubit_transformation not in fermonic_to_qubit_map.keys():
        print(f'valid qubit mappings : {list(fermonic_to_qubit_map.keys())}')
        raise ValueError(f'unknown qubit mapping: {qubit_transformation}')

    mapping = fermonic_to_qubit_map[qubit_transformation]
    qubit_operator = mapping(fermionic_operator)

    return PauliwordOp.from_openfermion(qubit_operator, n_qubits)


def get_coupled_cluster_operator(cc_obj=None, t1=None, t2=None, hf_array=None,
                                 operator_type='qubit', qubit_transformation='JW', orbspin=None):
    """
    """
    if cc_obj is not None:
        t1 = spatial2spin(cc_obj.t1, orbspin=orbspin)
        t2 = spatial2spin(cc_obj.t2, orbspin=orbspin)
    else:
        assert (
                t1 is not None and
                t2 is not None
        ), 'Must supply t1 and t2 matrices'

    no, nv = t1.shape
    nmo = no + nv

    if hf_array is None:
        warnings.warn('No Hartree-Fock state provided: assumed singlet configuration')
        occ_mask = np.zeros(nmo, dtype=bool)
        occ_mask[:no] = True
    else:
        occ_mask = hf_array.reshape(-1).astype(bool)

    indices = np.arange(0, nmo)
    single_mask = np.ix_(indices[~occ_mask], indices[occ_mask])
    double_mask = np.ix_(indices[~occ_mask], indices[occ_mask],
                         indices[~occ_mask], indices[occ_mask])

    # dictionary of single aplitudes of form {(i,j):t_ij}
    single_amplitudes = np.zeros((nmo, nmo))
    single_amplitudes[single_mask] = t1.T
    single_amp_dict = array_to_dict_nonzero_indices(single_amplitudes)

    # dictionary of double aplitudes of form {(i,j,k,l):t_ijkl}
    double_amplitudes = np.zeros((nmo, nmo, nmo, nmo))
    double_amplitudes[double_mask] = .25 * t2.transpose(2, 0, 3, 1)
    double_amp_dict = array_to_dict_nonzero_indices(double_amplitudes)

    generator = FermionOperator()
    for (i, j), t_ij in single_amp_dict.items():
        generator += FermionOperator(f'{i}^ {j}', t_ij)
    for (i, j, k, l), t_ijkl in double_amp_dict.items():
        generator += FermionOperator(f'{i}^ {j} {k}^ {l}', t_ijkl)

    if operator_type == 'fermion':
        return generator
    elif operator_type == 'qubit':
        return fermion_to_qubit_operator(generator, qubit_transformation=qubit_transformation, n_qubits=nmo)


# Generate the ccsd
CCSD_generator = get_coupled_cluster_operator(ccsd)
CCSD_generator -= CCSD_generator.dagger
CCSD_generator = QT.taper_it(aux_operator=CCSD_generator).sort()
CCSD_generator.sigfig = 8
assert CCSD_generator.n_qubits == H.n_qubits

print("List of possible Pauli operators:")
print(CCSD_generator)

from symmer.evolution.variational_optimization import ADAPT_VQE

adapt = ADAPT_VQE(observable=H, excitation_pool=CCSD_generator, ref_state=hf_state)
adapt.topology_aware = False
adapt_result = adapt.optimize(max_cycles=15)

from matplotlib import pyplot as plt

fig, axis = plt.subplots()
fig.tight_layout()
Y_adapt = [nbed_pbe_e]
X_adapt = [0]

# niter = len(out['interim_data']['history'])
niter = len(adapt_result['interim_data'].keys()) - 1
offset = 0
prev = 0
colors = []
for i in range(1, niter + 1):
    c = plt.cm.plasma_r(i / niter)
    colors.append(c)
    # i = str(i)
    # params   = out['interim_data'][i]['history']['params']
    energy = adapt_result['interim_data'][i]['history']['energy']
    gradient = adapt_result['interim_data'][i]['history']['gradient']
    X, Y = zip(*energy.items())
    Y = np.array(Y)
    X = offset + np.array([int(x) for x in X])
    axis.plot(X, abs(Y - nbed_pbe_e), lw=1.2, zorder=10, color=c)
    grad_norm = np.linalg.norm(list(gradient.values()), axis=1)
    offset += (len(X) - 1)
    X_adapt.append(offset)
    Y_adapt.append(adapt_result['interim_data'][i]['output']['fun'])

axis.plot(
    X_adapt, abs(np.array(Y_adapt) - nbed_pbe_e), ms=5,
    ls=':', color='black', marker='d', label='ADAPT energy after\neach VQE routine', zorder=10, lw=1
)

axis.set_ylabel('Error from target embedding energy [Ha]')
axis.set_xlabel('Optimization step')
axis.set_yscale('log')
axis.legend()
axis.grid()
plt.show()

print(adapt_result['adapt_operator'])

from symmer.evolution import PauliwordOp_to_QuantumCircuit
from qiskit import QuantumCircuit

qc = PauliwordOp_to_QuantumCircuit(
    PauliwordOp.from_list(adapt_result['adapt_operator']),
    bind_params=False,
    include_barriers=False
)

adapt_ansatz = QuantumCircuit(qc.num_qubits)

# Prepare HF state
hf_bitstring = list(hf_state.to_dictionary.keys())[0]
for bit_idx in range(qc.num_qubits):
    bit = hf_bitstring[bit_idx]
    if bit == "1":
        adapt_ansatz.x(qc.num_qubits - 1 - bit_idx)

# Add excitations found by ADAPT-VQE
for inst in qc.data:
    adapt_ansatz.append(inst)
adapt_ansatz.draw()

adapt_e = adapt_result["result"]["fun"]
adapt_params = adapt_result["result"]["x"]
print("Energy obtained from ADAPT:", adapt_e)
print("Parameters obtained from ADAPT:", adapt_params)

from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
import numpy as np
from scipy.optimize import minimize


def run_circuit(circ, shots):
    ### This function is setup to execute the circuit using qiskit's default simulators
    ### If running on real hardware you may need to adapt this function

    backend = Sampler()
    result = backend.run(circ, shots=shots).result()
    dist = result.quasi_dists[0]

    counts = {}
    for res, val in dist.items():
        res_bin = bin(res)[2:].zfill(circ.num_qubits)
        res_counts = int(val * shots)
        counts[res_bin] = res_counts

    return counts


def get_exp_val(measurement_results):
    ### Given a set of measurement results we calculate the corresponding expectation value
    def parity(bitstring):
        p = 1
        for b in bitstring:
            if b == "1":
                p *= -1
        return p

    total_shots = sum(list(measurement_results.values()))

    even_parity_shots = 0
    odd_parity_shots = 0

    for bitstring, counts in measurement_results.items():
        p = parity(bitstring)
        if p == 1:
            even_parity_shots += counts
        else:
            odd_parity_shots += counts

    exp_value = even_parity_shots / total_shots - odd_parity_shots / total_shots

    return exp_value


def calc_energy(qc, ham, num_shots):
    ham_dict = ham.to_dictionary
    num_terms = len(list(ham_dict.keys()))

    energy_estimate = 0

    for pauli_string, weight in ham_dict.items():

        if pauli_string == "I" * qc.num_qubits:
            energy_estimate += weight
            continue

        def pauli_weight(ps):
            w = 0
            for p in ps:
                if p != "I": w += 1
            return w

        # Create a temp circuit that is a copy of qc
        # N.B. we don't need to measure the qubits where the pauli_string has an "I" term
        temp_qc = QuantumCircuit(qc.num_qubits, pauli_weight(pauli_string))
        for inst in qc.data:
            temp_qc.append(inst)

        # Add the gates needed to measure the pauli_string
        current_classical_bit = 0
        for p_idx in range(qc.num_qubits):
            p = pauli_string[p_idx]
            if p == "X":
                temp_qc.h(p_idx)
                temp_qc.measure(p_idx, current_classical_bit)
                current_classical_bit += 1
            elif p == "Y":
                temp_qc.sdg(p_idx)
                temp_qc.h(p_idx)
                temp_qc.measure(p_idx, current_classical_bit)
                current_classical_bit += 1
            elif p == "Z":
                temp_qc.measure(p_idx, current_classical_bit)
                current_classical_bit += 1
            else:
                pass

        # Number of shots to use
        shots = int(num_shots / num_terms)
        measurement_results = run_circuit(temp_qc, shots=shots)
        exp_value = get_exp_val(measurement_results)

        energy_estimate += exp_value * weight

    return energy_estimate


def run_optimisation(ansatz_circuit, ham, maxiter=100, shots=10000):
    num_params = len(ansatz_circuit.parameters)

    # start with random parameters
    initial_params = np.random.random(num_params)

    # define the optimisation func
    def opt_func(x):
        param_dict = {}
        for param_idx in range(num_params):
            param = ansatz_circuit.parameters[param_idx]
            param_dict[param] = x[param_idx]

        qc = ansatz_circuit.assign_parameters(param_dict)
        energy = calc_energy(qc, ham, shots)

        return energy

    opt_res = minimize(opt_func, initial_params, method="COBYLA", options={'maxiter': maxiter})

    energy_estimate = opt_res.fun
    optimised_parameters = opt_res.x

    return energy_estimate, optimised_parameters

energy_estimate, optimised_parameters = run_optimisation(qc, H)
print(energy_estimate)
print(optimised_parameters)

# Copmaring to ADAPT:
energy_diff = np.abs(energy_estimate - adapt_e)
param_diffs = [np.abs(optimised_parameters[i] - adapt_params[i]) for i in range(len(adapt_params))]
assert np.isclose(energy_diff, 0.0)
for d in param_diffs:
    assert np.isclose(d, 0.0)

from qiskit.circuit.library import PauliTwoDesign

ansatz = PauliTwoDesign(num_qubits=qc.num_qubits, reps=5)
energy_estimate, _ = run_optimisation(ansatz, H)
print(energy_estimate)

from qiskit_algorithms import VQE
from qiskit.primitives import Estimator, Sampler
from qiskit_algorithms.optimizers import COBYLA

estimator = Estimator()
ansatz = adapt_ansatz
optimizer = COBYLA(maxiter=200)
initial_point = [4 * np.pi * (np.random.random() - 0.5) for _ in range(15)]

optimisation_dict = {"optimisation_number": [], "optimisation_parameters": [], "optimisation_value": []}


def callback(i, a, f, _):
    optimisation_dict["optimisation_number"].append(i)
    optimisation_dict["optimisation_parameters"].append(a)
    optimisation_dict["optimisation_value"].append(f)
    return


vqe = VQE(estimator, ansatz, optimizer, initial_point=initial_point, callback=callback)

observable = H.to_qiskit
result = vqe.compute_minimum_eigenvalue(observable)
energy_estimate = result.eigenvalue
print(energy_estimate)

plt.plot(optimisation_dict["optimisation_number"], optimisation_dict["optimisation_value"])
plt.show()

#Pennylane

''' The idea is to :
1. Convert the qiskit ansatz circuit to a pennylane quantum function
2. Use it in a QNode which is the format followed in pennylane to declare quantum circuits and tie a device to it for computation
3. Use the integration of pennylane of classical and quantum computations to compute gradients of the ansatz
4. Then optimise gradients with classical optimisers from different frameworks provided
5. Compare the resulting parameters with that of ADAPT_VQE '''

#IDEA 2: https://pennylane.ai/qml/demos/tutorial_vqe_qng
import pennylane as qml

#from qiskit import aer

pl_circ = qml.from_qiskit(qc)  #qc has the ansatz

#device = qml.device("qiskit.aer", backend="qasm_simulator", wires=8)
device = qml.device("default.qubit", wires=8)


@qml.qnode(device, interface="autograd")
def cost_fn(params):
    pl_circ(params)
    return qml.expval(H)

