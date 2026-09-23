import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye, csr_matrix
from scipy.sparse.linalg import expm_multiply
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Operator, Statevector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from tqdm import tqdm

import os
os.makedirs("figures", exist_ok=True)

# ==========================================
# 1. CONFIGURATION & PARAMETERS
# ==========================================
params = {
    "N_SITES": 9,
    "N_LEVELS": 4,
    "N_QUBITS_PER_SITE": 2,
    "A": 1.0,
    "MU": 0.5,
    "W0": 1.0,
    "T_MAX": 10.0,
    "DT": 0.05,
    "SHOTS": 2000,
    "STRING_MAG": 1.5
}

N_Q_TOTAL = params["N_SITES"] * params["N_QUBITS_PER_SITE"]
TIMES = np.arange(0, params["T_MAX"] + params["DT"], params["DT"])

# ==========================================
# 2. OPERATOR CONSTRUCTION
# ==========================================

class SchwingerOperators:
    def __init__(self, p):
        self.p = p
        self.dim = p["N_LEVELS"]
        
        # A. Local Matrix Operators
        diag_vals = np.sqrt(np.arange(1, self.dim))
        self.a_mat = np.diag(diag_vals, k=1)
        self.adag_mat = self.a_mat.T.conj()
        
        self.phi_mat = (1.0 / np.sqrt(2 * p["W0"])) * (self.a_mat + self.adag_mat)
        self.pi_mat = -1j * np.sqrt(p["W0"] / 2.0) * (self.a_mat - self.adag_mat)

        self.phi2_mat = self.phi_mat @ self.phi_mat
        self.pi2_mat = self.pi_mat @ self.pi_mat
        self.id_mat = np.eye(self.dim)

        # B. Sparse Global Hamiltonian
        print("Building sparse Hamiltonian...")
        self.H_sparse = self._build_sparse_hamiltonian()

        # C. Qiskit Operators
        print("Building Pauli operators...")
        self.phi_op = SparsePauliOp.from_operator(Operator(self.phi_mat))
        self.pi_op = SparsePauliOp.from_operator(Operator(self.pi_mat))
        self.phi2_op = SparsePauliOp.from_operator(Operator(self.phi2_mat))
        self.pi2_op = SparsePauliOp.from_operator(Operator(self.pi2_mat))
        self.int_op = self.phi_op.tensor(self.phi_op)

    def _build_sparse_hamiltonian(self):
        N = self.p["N_SITES"]
        dim_full = self.dim ** N
        H = csr_matrix((dim_full, dim_full))
        
        c_kin = 0.5 * self.p["A"]
        c_mass = 0.5 * self.p["A"] * self.p["MU"]**2
        c_grad = 0.5 / self.p["A"]

        for i in range(N):
            term_k = self._embed_operator(self.pi2_mat, i)
            H += c_kin * term_k
            term_m = self._embed_operator(self.phi2_mat, i)
            H += c_mass * term_m
            n_neighbors = 1 if (i == 0 or i == N-1) else 2
            H += c_grad * n_neighbors * term_m

        for i in range(N - 1):
            op_list = [self.id_mat] * N
            op_list[i] = self.phi_mat
            op_list[i+1] = self.phi_mat
            term_int = op_list[0]
            for k in range(1, N):
                term_int = kron(term_int, op_list[k], format='csr')
            H -= (1.0 / self.p["A"]) * term_int
        return H

    def _embed_operator(self, op, site_idx):
        ops = [self.id_mat] * self.p["N_SITES"]
        ops[site_idx] = op
        full_op = ops[0]
        for k in range(1, self.p["N_SITES"]):
            full_op = kron(full_op, ops[k], format='csr')
        return full_op

    def get_initial_state_vector(self, system_type):
        N = self.p["N_SITES"]
        vac = np.zeros(self.dim); vac[0] = 1.0
        alpha = self.p["STRING_MAG"]
        string_state = expm_multiply(alpha * self.adag_mat - np.conj(alpha) * self.a_mat, vac)
        full_psi = np.array([1.0])
        
        if system_type == 1:
            print("Initializing String between site 3 and 6")
            for i in range(N):
                if 3 <= i < 6: 
                    full_psi = np.kron(full_psi, string_state)
                else:
                    full_psi = np.kron(full_psi, vac)
        elif system_type == 2:
            print("Initializing Point Charge at site 4")
            for i in range(N):
                if i >= 4:
                    full_psi = np.kron(full_psi, string_state)
                else:
                    full_psi = np.kron(full_psi, vac)
        return full_psi

    def site_qubits(self, site_idx):
        """
        Return list of qubit indices corresponding to `site_idx` *consistent with
        build_trotter_circuit* mapping: site 0 -> highest-index qubits.
        """
        N = self.p["N_SITES"]
        nq = self.p["N_QUBITS_PER_SITE"]
        start = (N - 1 - site_idx) * nq
        return list(range(start, start + nq))

    def build_trotter_circuit(self, dt):
        N = self.p["N_SITES"]
        nq = self.p["N_QUBITS_PER_SITE"]
        qc = QuantumCircuit(N * nq)
        c_kin = 0.5 * self.p["A"]
        c_mass = 0.5 * self.p["A"] * self.p["MU"]**2
        c_grad = 0.5 / self.p["A"]
        c_int = -1.0 / self.p["A"]

        # Helper to map site index 'i' to Qiskit qubits (High to Low)
        def get_qubits(site_idx):
            # Site 0 is at the top (highest indices)
            # Site N-1 is at the bottom (lowest indices)
            start = (N - 1 - site_idx) * nq
            return list(range(start, start + nq))

        for i in range(N):
            q_indices = get_qubits(i)
            qc.append(PauliEvolutionGate(self.pi2_op, time=c_kin * dt), q_indices)
            
        for i in range(N):
            q_indices = get_qubits(i)
            n_bonds = 1 if (i==0 or i==N-1) else 2
            coeff = c_mass + n_bonds * c_grad
            qc.append(PauliEvolutionGate(self.phi2_op, time=coeff * dt), q_indices)
            
        for i in range(N - 1):
            # Interaction involves site i and i+1
            # q_indices must cover both sites. 
            # Since we reversed mapping, site i is 'higher' than site i+1.
            # We need to be careful with the order passed to PauliEvolutionGate 
            # if the operator is not symmetric, but phi-phi is symmetric.
            q_i = get_qubits(i)
            q_next = get_qubits(i+1)
            # Combined indices. Note: q_next (site i+1) are lower indices, q_i are higher.
            # Qiskit gates usually take list [q_low, ..., q_high] or specific map.
            # We strictly concatenate the lists.
            q_indices = q_i + q_next 
            qc.append(PauliEvolutionGate(self.int_op, time=c_int * dt), q_indices)
            
        return qc

# ==========================================
# 3. SIMULATION ENGINE
# ==========================================

def measure_pauli_string(qc, pauli_string, backend, shots):
    """
    Manually measures a specific Pauli string expectation value.
    pauli_string: e.g., "IIZII" (measure Z on qubit 2)
    """
    meas_qc = qc.copy()
    num_qubits = qc.num_qubits
    
    # 1. Basis rotation
    # We only measure non-identity terms
    measured_qubits = []
    for i, p_char in enumerate(reversed(pauli_string)): # Qiskit order is q0 at right
        if p_char == 'X':
            meas_qc.h(i)
            measured_qubits.append(i)
        elif p_char == 'Y':
            meas_qc.sdg(i)
            meas_qc.h(i)
            measured_qubits.append(i)
        elif p_char == 'Z':
            measured_qubits.append(i)
            
    # 2. Measurement
    if not measured_qubits:
        return 1.0 # Expectation of Identity is 1
        
    meas_qc.measure_all()
    
    # 3. Run
    job = backend.run(transpile(meas_qc, backend), shots=shots)
    counts = job.result().get_counts()
    
    # 4. Compute Expectation <P> = (Counts(even) - Counts(odd)) / Total
    expectation = 0
    for bitstring, count in counts.items():
        parity = 1
        # Qiskit bitstring is q(N-1)...q0
        # We only care about parity of the measured qubits
        for q_idx in measured_qubits:
            # bitstring is reversed relative to qubit index
            bit = bitstring[num_qubits - 1 - q_idx] 
            if bit == '1':
                parity *= -1
        expectation += parity * count
        
    return expectation / shots

def measure_sparse_pauli_op(qc, op, backend, shots):
    """
    Measures expectation value of a SparsePauliOp sum of terms.
    """
    total_exp = 0.0
    # Grouping would be more efficient, but simple loop is fine for this scale
    for pauli, coeff in zip(op.paulis, op.coeffs):
        # Convert Pauli object to string representation
        p_str = pauli.to_label()
        exp_val = measure_pauli_string(qc, p_str, backend, shots)
        total_exp += np.real(coeff * exp_val)
    return total_exp

def run_simulations(ops, system_type):
    psi_0 = ops.get_initial_state_vector(system_type)
    
    results = {
        "time": TIMES,
        "bench_field": [],
        "trot_field": [],
        "shot_field": [],
        "bench_current": [],
        "trot_current": [],
        "shot_current": [],
        "bench_energy": []
    }
    
    # --- A. Benchmark ---
    print(f"Running Benchmark (Exact) for System {system_type}...")
    psi_t = psi_0.copy()
    
    def get_local_exp(vec, op, site):
        dim = ops.dim
        n = ops.p["N_SITES"]
        shape = (dim**site, dim, dim**(n-1-site))
        v_reshaped = vec.reshape(shape)
        v_prime = np.tensordot(v_reshaped, op, axes=([1], [1]))
        v_prime = np.transpose(v_prime, (0, 2, 1))
        return np.vdot(vec.flatten(), v_prime.flatten()).real

    for t in tqdm(TIMES):
        fields = [get_local_exp(psi_t, ops.phi_mat, i) for i in range(ops.p["N_SITES"])]
        results["bench_field"].append(fields)
        currents = [- (fields[i+1] - fields[i]) for i in range(len(fields)-1)]
        results["bench_current"].append(currents)
        en = np.vdot(psi_t, ops.H_sparse @ psi_t).real
        results["bench_energy"].append(en)
        
        if t < TIMES[-1]:
            psi_t = expm_multiply(-1j * ops.H_sparse * ops.p["DT"], psi_t)

    # --- B. Trotter ---
    print(f"Running Trotter (Statevector) for System {system_type}...")
    qs_0 = Statevector(psi_0)
    step_circ = ops.build_trotter_circuit(ops.p["DT"])
    curr_qs = qs_0
    
    for t in tqdm(TIMES):
        fields = []
        for i in range(ops.p["N_SITES"]):
            q_indices = ops.site_qubits(i)   # <- use consistent mapping
            val = curr_qs.expectation_value(ops.phi_op, qargs=q_indices).real
            fields.append(val)
        results["trot_field"].append(fields)
        currents = [- (fields[i+1] - fields[i]) for i in range(len(fields)-1)]
        results["trot_current"].append(currents)
        
        if t < TIMES[-1]:
            curr_qs = curr_qs.evolve(step_circ)

# --- C. Quantum (Estimator V2) ---
    print(f"Running Quantum Simulation (Shots={ops.p['SHOTS']}) for System {system_type}...")
    
    # 1. Create Observables (Padded to full size)
    observables = []
    for i in range(ops.p["N_SITES"]):
        # Calculate starting qubit index for site i (Site 0 is high index)
        start = (ops.p["N_SITES"] - 1 - i) * ops.p["N_QUBITS_PER_SITE"]
        
        # Pad phi_op with identities: I_upper (x) phi (x) I_lower
        full_op = ops.phi_op
        
        # Add lower identities (if any)
        if start > 0:
            full_op = full_op.tensor(SparsePauliOp("I" * start))
            
        # Add upper identities (if any)
        remaining = (ops.p["N_SITES"] * ops.p["N_QUBITS_PER_SITE"]) - (start + ops.p["N_QUBITS_PER_SITE"])
        if remaining > 0:
            full_op = SparsePauliOp("I" * remaining).tensor(full_op)
            
        observables.append(full_op)

    # 2. Setup EstimatorV2
    sim_backend = AerSimulator()
    estimator = AerEstimator()
    # Fix for strict Options validation: Set shots via attribute
    estimator.options.default_shots = ops.p["SHOTS"]

    # 3. Initialize Circuit
    main_qc = QuantumCircuit(ops.p["N_SITES"] * ops.p["N_QUBITS_PER_SITE"])
    main_qc.initialize(psi_0, main_qc.qubits)
    
    for t in tqdm(TIMES, desc="EstimatorV2"):
        # Transpile to ISA circuit (required for V2)
        isa_circuit = transpile(main_qc, sim_backend)
        
        # Create PUB (Primitive Unified Bloc): (Circuit, Observables)
        # This batches all site measurements into a single execution
        pub = (isa_circuit, observables)
        
        # Run Estimator
        job = estimator.run([pub])
        result = job.result()
        
        # Extract expectation values
        fields = result[0].data.evs
            
        results["shot_field"].append(fields)
        currents = [- (fields[i+1] - fields[i]) for i in range(len(fields)-1)]
        results["shot_current"].append(currents)
        
        # Evolve
        if t < TIMES[-1]:
            main_qc.append(step_circ, range(main_qc.num_qubits))

    return results

# ==========================================
# 4. PLOTTING
# ==========================================

def plot_results(res1, res2, n_sites):
    systems = [(res1, "System 1 (String)"), (res2, "System 2 (Point Charge)")]
    
    # 1. Heatmaps
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for i, (res, title) in enumerate(systems):
        data = np.array(res["bench_field"])
        im = axs[i].imshow(data, aspect='auto', origin='lower', cmap='RdBu_r', 
                           extent=[0, n_sites-1, 0, res["time"][-1]])
        axs[i].set_title(f"{title}: Benchmark Field $\\langle \\phi \\rangle$")
        axs[i].set_xlabel("Site")
        axs[i].set_ylabel("Time")
        plt.colorbar(im, ax=axs[i])
    plt.tight_layout()
    plt.savefig("figures/benchmark_heatmaps.png", dpi=600, bbox_inches='tight')
    plt.close()

    # 2. Comparison Heatmaps
    for idx, (res, title) in enumerate(systems):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"{title}: Field Profile Comparison")
        vmin, vmax = np.min(res["bench_field"]), np.max(res["bench_field"])
        
        plots = [("Benchmark", res["bench_field"]), ("Trotter", res["trot_field"]), ("Quantum (Shots)", res["shot_field"])]
        for i, (name, data) in enumerate(plots):
            im = axs[i].imshow(np.array(data), aspect='auto', origin='lower', cmap='RdBu_r',
                               extent=[0, n_sites-1, 0, res["time"][-1]], vmin=vmin, vmax=vmax)
            axs[i].set_title(name)
            axs[i].set_xlabel("Site")
            if i==0: axs[i].set_ylabel("Time")
        plt.colorbar(im, ax=axs[2])
        plt.tight_layout()
        plt.savefig(f"figures/comparison_heatmaps_system{idx+1}.png", dpi=600, bbox_inches='tight')
        plt.close()

    # 3. Induced Current
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for i, (res, title) in enumerate(systems):
        data = np.array(res["bench_current"])
        times_to_plot = [0, 20, 40, 60, 80, 100]
        for ti in times_to_plot:
            if ti < len(data):
                axs[i].plot(data[ti], label=f"t={res['time'][ti]:.1f}")
        axs[i].set_title(f"{title}: Induced Current (Benchmark)")
        axs[i].set_xlabel("Bond Index")
        axs[i].legend(fontsize='small')
        axs[i].grid(True)
    plt.tight_layout()
    plt.savefig("figures/induced_current_comparison.png", dpi=600, bbox_inches='tight')
    plt.close()

    # 4. Mid-Bond Current
    mid_bond = (n_sites - 1) // 2
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for i, (res, title) in enumerate(systems):
        axs[i].plot(res["time"], np.array(res["bench_current"])[:, mid_bond], 'k-', label="Benchmark")
        axs[i].plot(res["time"], np.array(res["trot_current"])[:, mid_bond], 'b--', label="Trotter")
        axs[i].plot(res["time"], np.array(res["shot_current"])[:, mid_bond], 'r.', label="Quantum", alpha=0.3)
        axs[i].set_title(f"{title}: Current at Bond {mid_bond}")
        axs[i].set_xlabel("Time")
        axs[i].legend()
    plt.tight_layout()
    plt.savefig("figures/midbond_current_comparison.png", dpi=600, bbox_inches='tight')
    plt.close()
    
    # 5. Energy
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for i, (res, title) in enumerate(systems):
        axs[i].plot(res["time"], res["bench_energy"])
        axs[i].set_title(f"{title}: Total Energy")
        axs[i].set_xlabel("Time")
        axs[i].grid(True)
    plt.tight_layout()
    plt.savefig("figures/energy_comparison.png", dpi=600, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    ops = SchwingerOperators(params)
    print("\n--- Simulating System 1 ---")
    res1 = run_simulations(ops, 1)
    print("\n--- Simulating System 2 ---")
    res2 = run_simulations(ops, 2)
    plot_results(res1, res2, params["N_SITES"])