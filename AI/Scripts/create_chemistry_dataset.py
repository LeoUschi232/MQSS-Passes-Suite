# Qiskit Nature imports
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevMapper, ParityMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC

# Qiskit imports
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit import qasm2, transpile

# Standard imports
from numpy import random, pi, min, max, any
from os import environ, makedirs, path
from itertools import product
from time import sleep
from tqdm import tqdm

# ------------ CONFIGURATION -------------
SPEED_TIER_CONFIG = dict(nr_orbitals=4, excitations="s")
OCCURED_EXCEPTIONS = []
REPS = 1

# Reduce oversubscription
environ.setdefault("OMP_NUM_THREADS", "1")
environ.setdefault("MKL_NUM_THREADS", "1")
environ.setdefault("OPENBLAS_NUM_THREADS", "1")
mappers = {"JW": JordanWignerMapper(), "BK": BravyiKitaevMapper(), "parity": ParityMapper()}


def choose_k_and_particles(n_so, n_alpha, n_beta, k_target):
    if n_so < 2:
        # Too tiny to do anything meaningful.
        return None

    # Start with your target but cannot exceed available orbitals.
    k = min([k_target, n_so])

    # If k is not strictly larger than the max spin count, try to bump k.
    max_spin = max([n_alpha, n_beta])
    if k <= max_spin:
        if max_spin + 1 <= n_so:
            k = max_spin + 1
        else:
            # Can't bump k —> must reduce electrons.
            pass

    # Clamp per spin to < k.
    na = int(min([n_alpha, k - 1]))
    nb = int(min([n_beta, k - 1]))

    # Cap total to 2*(k-1), strictly less than full double occupancy.
    max_total = 2 * (k - 1)
    if na + nb > max_total:
        overflow = na + nb - max_total
        # Trim the larger spin first.
        while overflow > 0 and (na > 0 or nb > 0):
            if na >= nb and na > 0:
                na -= 1
            elif nb > 0:
                nb -= 1
            overflow -= 1

    # If after all this k is still invalid, shrink k until valid.
    while not (k > na and k > nb) and k >= 2:
        k -= 1
        na = min([na, k - 1])
        nb = min([nb, k - 1])
    if k < 2 or not (k > na and k > nb):
        return None
    return k, (na, nb)


def fill_params(circuit, seed=None):
    if len(circuit.parameters):
        rng = random.default_rng(seed)
        return circuit.assign_parameters(
            parameters={param: rng.uniform(-2 * pi, 2 * pi) for param in circuit.parameters},
            inplace=False)
    return circuit


def build_circuit_fast(problem, mapper, k_target, excitations, reps=1):
    n_so = problem.num_spatial_orbitals
    n_alpha, n_beta = problem.num_particles
    pick = choose_k_and_particles(n_so, n_alpha, n_beta, k_target)
    if pick is None:
        raise RuntimeError(f"no feasible active space: n_so={n_so}, (na,nb)=({n_alpha},{n_beta}), k_target={k_target}")
    k, (na, nb) = pick
    circuit = UCC(
        num_spatial_orbitals=k,
        num_particles=(na, nb),
        qubit_mapper=mapper,
        excitations=excitations,
        reps=reps,
        initial_state=HartreeFock(k, (na, nb), mapper),
    )
    circuit.measure_all()
    return fill_params(transpile(
        circuits=circuit,
        basis_gates=list(get_standard_gate_name_mapping().keys()),
        optimization_level=0
    ))


def sanitize_basis(basis: str) -> str:
    return basis.replace("**", "dstar").replace("*", "star")


def build_problem(geometry, basis, charge, spin):
    global OCCURED_EXCEPTIONS
    problem = PySCFDriver(atom=geometry, basis=basis, unit=DistanceUnit.ANGSTROM, charge=charge, spin=spin).run()
    try:
        problem = FreezeCoreTransformer().transform(problem)
    except Exception as error:
        OCCURED_EXCEPTIONS.append(str(error))
        pass
    return problem


def build_and_save(name, geometry, basis, charge, spin, mapper_name, mapper, output_dir):
    circuit = build_circuit_fast(
        problem=build_problem(geometry, basis, charge, spin),
        mapper=mapper,
        k_target=SPEED_TIER_CONFIG["nr_orbitals"],
        excitations=SPEED_TIER_CONFIG["excitations"],
        reps=REPS
    )
    makedirs(output_dir, exist_ok=True)
    with open(path.join(output_dir, f"{name}_{sanitize_basis(basis)}_{mapper_name}.qasm"), "w") as f:
        f.write(qasm2.dumps(circuit))
    return name, basis, mapper_name, circuit.num_qubits, len(circuit.parameters)


# [Name, Geometry, Charge, Spin]
molecules = [
    # --- Diatomics ---
    ("H2-eq", "H 0 0 0; H 0 0 0.735", 0, 0),
    ("LiH-eq", "Li 0 0 0; H 0 0 1.60", 0, 0),
    ("F2-eq", "F 0 0 0; F 0 0 1.41", 0, 0),
    ("N2-eq", "N 0 0 0; N 0 0 1.10", 0, 0),
    ("O2-triplet", "O 0 0 0; O 0 0 1.21", 0, 2),
    ("CO-eq", "C 0 0 0; O 0 0 1.13", 0, 0),
    ("HF-eq", "H 0 0 0; F 0 0 0.92", 0, 0),
    ("HCl-eq", "H 0 0 0; Cl 0 0 1.27", 0, 0),
    ("HeH-plus", "He 0 0 0; H 0 0 0.77", 1, 0),
    ("Li2-eq", "Li 0 0 -1.335; Li 0 0 1.335", 0, 0),
    ("NaH-eq", "Na 0 0 0; H 0 0 1.89", 0, 0),
    ("LiF-eq", "Li 0 0 0; F 0 0 1.56", 0, 0),
    ("LiCl-eq", "Li 0 0 0; Cl 0 0 2.02", 0, 0),
    ("NaF-eq", "Na 0 0 0; F 0 0 1.93", 0, 0),
    ("NaCl-eq", "Na 0 0 0; Cl 0 0 2.36", 0, 0),
    ("Cl2-eq", "Cl 0 0 0; Cl 0 0 2.00", 0, 0),
    ("SiO-eq", "Si 0 0 0; O 0 0 1.51", 0, 0),
    ("CS-eq", "C 0 0 0; S 0 0 1.55", 0, 0),

    # --- Linear Triatomics ---
    ("CO2-lin", "O 0 0 -1.16; C 0 0 0; O 0 0 1.16", 0, 0),
    ("HCN-lin", "H 0 0 -1.06; C 0 0 0; N 0 0 1.153", 0, 0),
    ("HCO-plus-lin", "H 0 0 -1.10; C 0 0 0; O 0 0 1.12", 1, 0),
    ("N2O-lin", "N 0 0 -1.12; N 0 0 0; O 0 0 1.18", 0, 0),
    ("HNC-lin", "H 0 0 -1.10; N 0 0 0; C 0 0 1.16", 0, 0),
    ("HOF-lin", "H 0 0 -1.18; O 0 0 0; F 0 0 0.92", 0, 0),
    ("HOCl-lin", "H 0 0 -1.28; O 0 0 0; Cl 0 0 1.28", 0, 0),
    ("SO2-lin", "O 0 0 -1.19; S 0 0 0; O 0 0 1.19", 0, 0),

    # --- Nonlinear Triatomics ---
    ("H2O-bent", "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", 0, 0),
    ("H2S-bent", "S 0 0 0; H 0 -1.00 0.95; H 0 1.00 0.95", 0, 0),
    ("BeH2-lin", "Be 0 0 0; H 0 0 -1.32; H 0 0 1.32", 0, 0),
    ("HOF-bent", "H 0 -0.95 0.76; O 0 0 0; F 0 0.95 0.76", 0, 0),
    ("H2O2-approx", "O 0 0 0; O 1.47 0 0; H -0.36 0.94 0; H 1.83 -0.94 0", 0, 0),
    ("H2N-like", "N 0 0 0; H 0 0 1.02; H 0.94 0 -0.34", 0, 1),

    # --- Small Polyatomics ---
    ("NH3-pyramidal", "N 0 0 0; H 0 0.938 0.382; H 0.812 -0.469 0.382; H -0.812 -0.469 0.382", 0, 0),
    ("CH4-tetra", "C 0 0 0; H 0 0 1.089; H 1.026 0 -0.363; H -0.513 -0.889 -0.363; H -0.513 0.889 -0.363", 0, 0),
    ("CH2O-min", "C 0 0 0; O 0 0 1.21; H 0.94 0 -0.53; H -0.94 0 -0.53", 0, 0),
    ("CH3OH-compact", "C 0 0 0; O 0 0 1.43; H 0 -0.94 -0.53; H 0 0.94 -0.53", 0, 0),
    ("BH3-planar", "B 0 0 0; H 0 0 1.19; H 1.03 0 -0.40; H -1.03 0 -0.40", 0, 0),
    ("PH3-trigonal", "P 0 0 0; H 0 0 1.42; H 1.24 0 -0.47; H -1.24 0 -0.47", 0, 0),

    # --- Simple Hydrides / Pnictogens / Chalcogens ---
    ("SiH4-tetra", "Si 0 0 0; H 0 0 1.48; H 1.21 0 -0.49; H -0.61 -1.05 -0.49; H -0.61 1.05 -0.49", 0, 0),
    ("GeH4-tetra", "Ge 0 0 0; H 0 0 1.52; H 1.25 0 -0.50; H -0.62 -1.09 -0.50; H -0.62 1.09 -0.50", 0, 0),
    ("AlH3-planar", "Al 0 0 0; H 0 0 1.65; H 1.43 0 -0.55; H -1.43 0 -0.55", 0, 0),
    ("SiH3-approx", "Si 0 0 0; H 0 0 1.48; H 0.00 1.26 -0.49; H 1.09 -0.63 -0.49", 0, 1),
    ("AsH3-trigonal", "As 0 0 0; H 0 0 1.52; H 1.31 0 -0.50; H -1.31 0 -0.50", 0, 0),

    # --- Hydrocarbons ---
    ("C2H2-lin", "C 0 0 -0.60; C 0 0 0.60; H 0 0 -1.66; H 0 0 1.66", 0, 0),
    ("C2H4-planar", "C -0.66 0 0; C 0.66 0 0; H -1.23 0.93 0; H -1.23 -0.93 0; H 1.23 0.93 0; H 1.23 -0.93 0", 0, 0),
    ("C2H6-approx",
     "C -0.77 0 0; C 0.77 0 0; "
     "H -1.39 0.92 0; H -1.39 -0.92 0; H -1.39 0 0.92; "
     "H 1.39 0.92 0; H 1.39 -0.92 0; H 1.39 0 0.92", 0, 0),

    # --- Small Cations ---
    ("H3-plus-tri", "H 0.0 0.0 0.0; H 0.0 0.0 0.87; H 0.75 0.0 -0.435", 1, 0),
    ("NH4-plus", "N 0 0 0; H 0 0 1.03; H 0.89 0 -0.34; H -0.89 0 -0.34; H 0 0 -1.03", 1, 0),

    # --- Small Organics / Functional Groups ---
    ("CH3F", "C 0 0 0; F 0 0 -1.38; H 0 0 1.09; H 1.03 0 -0.36; H -0.52 0.90 -0.36", 0, 0),
    ("CH3Cl", "C 0 0 0; Cl 0 0 -1.77; H 0 0 1.09; H 1.03 0 -0.36; H -0.52 0.90 -0.36", 0, 0),
    ("H2CO-formaldehyde", "C 0 0 0; O 0 0 1.20; H 0.94 0 -0.53; H -0.94 0 -0.53", 0, 0),
    ("CH3OH-methanol", "C 0 0 0; O 0 0 1.36; H 0 -0.94 -0.53; H 0 0.94 -0.53", 0, 0),
    ("HCOOH-formic", "C 0 0 0; O 0 0 1.22; H 0 0 -1.10; H 0.94 0 2.15", 0, 0),
    ("CH3COOH-skeleton", "C 0 0 0; O 0 0 1.23; H 0 0 -1.09; O 0 0 2.36; H 0.94 0 3.20", 0, 0),
    ("CH3CN-acetonitrile", "C -0.66 0 0; C 0.66 0 0; N 1.83 0 0; H -1.23 0.93 0; H -1.23 -0.93 0; H -0.10 0 0", 0, 0),
    ("CH3NH2-methylamine", "C 0 0 0; H 0 0 1.09; H 1.03 0 -0.36; H -0.52 -0.90 -0.36; N 0 0 -1.16", 0, 0),
    ("CH3CHO-acetaldehyde",
     "C -0.60 0 0; C 0.60 0 0; O 1.80 0 0; "
     "H -1.20 0.93 0; H -1.20 -0.93 0; H -0.10 0 0; H 2.35 0 0", 0, 0),

    # --- C3 Hydrocarbons ---
    ("C3H4-allene-like",
     "C -1.26 0 0; C 0 0 0; C 1.26 0 0; H -1.90 0.92 0; H -1.90 -0.92 0; H 1.90 0.92 0; H 1.90 -0.92 0", 0, 0),
    ("C3H6-propene",
     "C -1.26 0 0; C 0 0 0; C 1.26 0 0; "
     "H -1.90 0.92 0; H -1.90 -0.92 0; "
     "H 0 0 1.09; H 0 0 -1.09; "
     "H 1.90 0.92 0; H 1.90 -0.92 0", 0, 0),
    ("C3H8-propane",
     "C -1.26 0 0; C 0 0 0; C 1.26 0 0; H -1.89 0.94 0; H -1.89 -0.94 0; H -0.37 0.94 0; H -0.37 -0.94 0; H 1.89 0.94 0; H 1.89 -0.94 0",
     0, 0),

    # --- Simple Inorganics ---
    ("COCl2-skeleton", "C 0 0 0; O 0 0 1.16; Cl 0 0 -1.75; Cl 0 0 2.80", 0, 0),
    ("SO3-planar", "S 0 0 0; O 0 0 1.43; O 1.24 0 -0.72; O -1.24 0 -0.72", 0, 0),
    ("CSO-chain", "C 0 0 0; S 0 0 1.56; O 0 0 2.72", 0, 0),
    ("CNO-chain", "C 0 0 0; N 0 0 1.17; O 0 0 2.29", 0, 1),
    ("BF3-planar", "B 0 0 0; F 0 0 1.31; F 1.14 0 -0.66; F -1.14 0 -0.66", 0, 0),
    ("CF3-like", "C 0 0 0; F 0 0 1.32; F 1.14 0 -0.66; F -1.14 0 -0.66", 0, 1)
]
bases = {"sto3g", "sto6g", "ccpvdz", "631g", "321g", "631g*", "631g**", "6311g"}

HEAVY = {"Ge", "As"}
ILLEGAL_BASES_FOR_HEAVY = {"321g", "631g", "631g*", "631g**", "6311g", "ccpvdz"}


def is_illegal_combination(geometry, basis):
    return any([substring in geometry for substring in HEAVY]) and basis.lower() in ILLEGAL_BASES_FOR_HEAVY


if __name__ == "__main__":
    # This script takes very long to run,
    # so better take a coffee break once you launch it.
    legal_combinations = []
    for (name, geometry, charge, spin), basis in product(molecules, bases):
        if is_illegal_combination(geometry, basis):
            continue
        legal_combinations.append((name, geometry, charge, spin, basis))
    dataset_folder = "../Datasets/Qasm/Chemistry"
    nr_circuits = len(legal_combinations) * len(mappers)
    print("Nr Molecules:", len(molecules), "\nNr circuits:", nr_circuits)
    sleep(0.1)
    pbar = tqdm(total=nr_circuits, desc="Circuits generated", unit="circuit")
    for name, geometry, charge, spin, basis in legal_combinations:
        pbar.set_postfix({"Molecule": name})
        for mapper_name, mapper in mappers.items():
            try:
                build_and_save(name, geometry, basis, charge, spin, mapper_name, mapper, dataset_folder)
            except Exception as error:
                OCCURED_EXCEPTIONS.append(f"Task failed: {error} on: {name} {basis} {mapper_name}")
            pbar.update(1)
    pbar.close()
    sleep(0.1)
    if OCCURED_EXCEPTIONS:
        print("Following exceptions occured:")
        for error in OCCURED_EXCEPTIONS:
            print(error)
    print("Generation of Chemistry Dataset finished.")
