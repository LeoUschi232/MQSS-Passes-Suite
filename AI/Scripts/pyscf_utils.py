from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, BravyiKitaevMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit import QuantumCircuit, transpile, qasm2
from numpy.random import default_rng
from numpy import pi


def get_standard_gates():
    # 'id', 'sx', 'x', 'cx', 'rz', 'r', 'c3sx', 'ccx', 'dcx', 'ch', 'cp', 'crx', 'cry', 'crz', 'cswap', 'csx', 'cu',
    # 'cu1', 'cu3', 'cy', 'cz', 'ccz', 'global_phase', 'h', 'p', 'rccx', 'rcccx', 'rx', 'rxx', 'ry', 'ryy', 'rzz',
    # 'rzx', 'xx_minus_yy', 'xx_plus_yy', 'ecr', 's', 'sdg', 'cs', 'csdg', 'swap', 'iswap', 'sxdg', 't', 'tdg', 'u',
    # 'u1', 'u2', 'u3', 'y', 'z', 'delay', 'reset', 'measure', ’barrier'
    return list(get_standard_gate_name_mapping().keys())


def get_pyscf_mainstream_bases():
    return ["sto3g", "sto6g", "ccpvdz", "631g", "321g"]


def fill_circuit_with_random_params(quantum_circuit: QuantumCircuit, seed: int = None):
    rng = default_rng(seed)
    params = list(quantum_circuit.parameters)
    theta = rng.uniform(-2 * pi, 2 * pi, len(params))
    return quantum_circuit.assign_parameters(dict(zip(params, theta)), inplace=False)


def flattened_qasm_circuit_v1(quantum_circuit: QuantumCircuit):
    return quantum_circuit.decompose()


def flattened_qasm_circuit_v2(quantum_circuit: QuantumCircuit):
    return transpile(quantum_circuit, basis_gates=get_standard_gates(), optimization_level=0)


def compare_qasm_flattening_functions(quantum_circuit: QuantumCircuit):
    qc1 = flattened_qasm_circuit_v1(quantum_circuit)
    qc2 = flattened_qasm_circuit_v2(quantum_circuit)
    for line1, line2 in zip(qasm2.dumps(qc1).splitlines(), qasm2.dumps(qc2).splitlines()):
        print(f"{line1:<30} | {line2:<30} | {line1 == line2}")


def make_ansatz_circuits_from_problem(problem: ElectronicStructureProblem) -> list[QuantumCircuit]:
    mappers = [JordanWignerMapper(), ParityMapper(), BravyiKitaevMapper()]
    circuits = []
    for mapper in mappers:
        try:
            circuits.append(UCCSD(
                problem.num_spatial_orbitals, problem.num_particles, mapper,
                initial_state=HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
            ).decompose())
        except Exception as error:
            print(f"Error: ", error)
    return circuits


def get_random_filled_flattened_circuits_from_problem(problem: ElectronicStructureProblem, seed: int = None):
    return [
        flattened_qasm_circuit_v2(fill_circuit_with_random_params(circuit, seed))
        for circuit in make_ansatz_circuits_from_problem(problem)
    ]

