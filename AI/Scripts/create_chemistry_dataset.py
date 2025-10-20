from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver

driver = PySCFDriver(
    atom="H 0 0 0; H 0 0 0.735",
    basis="sto3g",
    charge=0,
    spin=0,
    unit=DistanceUnit.ANGSTROM,
)

problem = driver.run()
print(problem)

# after `problem = driver.run()`
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

mapper = JordanWignerMapper()
ansatz = UCCSD(
    problem.num_spatial_orbitals,
    problem.num_particles,
    mapper,
    initial_state=HartreeFock(
        problem.num_spatial_orbitals, problem.num_particles, mapper
    ),
)

qc = ansatz.decompose()  # <-- QuantumCircuit you can benchmark

from qiskit import qasm2

import numpy as np
from qiskit import transpile
from qiskit import qasm2

# optional: from qiskit.qasm3 import dumps as qasm3_dumps

# 1) sample random angles in [-2π, 2π)
rng = np.random.default_rng(123)  # set/omit seed as you like
params = list(qc.parameters)
theta = rng.uniform(-2 * np.pi, 2 * np.pi, len(params))

# 2) bind them
qc_filled = qc.assign_parameters(dict(zip(params, theta)), inplace=False)

# 3) unroll/flatten to a defined basis (QASM2-friendly)
#    For strict OpenQASM2, use ['u','cx']; for IBM native, use ['rz','sx','x','cx','id']
all_basis_gates = set(
    ['u1', 'u2', 'u3', 'cx', 'id', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'rx', 'ry', 'rz', 'r', 'u',
     'cz', 'ccx', 'cy', 'ch', 'swap', 'cswap', 'crx', 'cry', 'crz', 'cu1', 'cu3', "cs", "csdg", "ct", "ctdg"])
qc_flat = transpile(qc_filled, basis_gates=all_basis_gates, optimization_level=0)

# 4) export
print(qasm2.dumps(qc_flat))
# print(qasm3_dumps(qc_flat))  # uncomment if you need QASM3
