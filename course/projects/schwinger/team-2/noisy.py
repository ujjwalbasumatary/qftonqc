import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, kron, eye
from scipy.sparse.linalg import eigsh, expm_multiply
from scipy.linalg import expm
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Operator, Statevector, partial_trace, entropy
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from tqdm import tqdm
import os

os.makedirs("figures_new", exist_ok=True)

class SchwingerBoson:
    def __init__(self, n, dim, a=1.0, e=1.0, m=0.5, theta=0.0):
        self.n, self.dim, self.a, self.e, self.m, self.theta = n, dim, a, e, m, theta
        self.mu = e / np.sqrt(np.pi)
        # Operators
        vals = np.sqrt(np.arange(1, dim))
        a_op = np.diag(vals, k=1)
        self.phi = (a_op + a_op.T.conj()) / np.sqrt(2)
        self.pi = 1j * (a_op.T.conj() - a_op) / np.sqrt(2)
        self.phi2, self.pi2 = self.phi @ self.phi, self.pi @ self.pi
        self.id = np.eye(dim)
        self.n_op = a_op.T.conj() @ a_op
        self.adag = a_op.T.conj()
        
        # Qiskit Ops
        self.phi_pop = SparsePauliOp.from_operator(Operator(self.phi))
        self.pi2_pop = SparsePauliOp.from_operator(Operator(self.pi2))
        self.phi2_pop = SparsePauliOp.from_operator(Operator(self.phi2))
        self.n_pop = SparsePauliOp.from_operator(Operator(self.n_op))

    def embed(self, op, s, sparse=True):
        ops = [self.id if sparse else SparsePauliOp("I"*int(np.log2(self.dim)))] * self.n
        ops[s] = csr_matrix(op) if sparse else op
        full = ops[0]
        for k in range(1, self.n):
            full = kron(full, ops[k]) if sparse else full.tensor(ops[k])
        return full

    def get_H(self, ext_Q=None):
        H = csr_matrix((self.dim**self.n, self.dim**self.n), dtype=complex)
        ck, cg = 0.5/self.a, 0.5/self.a
        cm = 0.5 * self.a * self.mu**2
        ci = -0.2 * self.a * self.m * self.e
        
        for i in range(self.n):
            H += ck * self.embed(self.pi2, i) + cm * self.embed(self.phi2, i)
            th = self.theta - (2*np.pi*ext_Q[i] if ext_Q is not None else 0)
            arg = 2*np.sqrt(np.pi)*self.phi + th*self.id
            H += ci * self.embed(0.5*(expm(1j*arg)+expm(-1j*arg)), i)
            
        for i in range(self.n-1):
            H += cg*(self.embed(self.phi2,i) + self.embed(self.phi2,i+1))
            H -= 2*cg * (self.embed(self.phi, i) @ self.embed(self.phi, i+1))
        return H

    def get_trotter(self, dt, steps, ext_Q=None):
        qc = QuantumCircuit(self.n * int(np.log2(self.dim)))
        op_K, op_P, op_I = [SparsePauliOp(["I"*qc.num_qubits], coeffs=[0.0]) for _ in range(3)]
        
        ck, cg, cm = 0.5/self.a, 0.5/self.a, 0.5*self.a*self.mu**2
        ci = -0.2 * self.a * self.m * self.e

        for i in range(self.n):
            op_K += ck * self.embed(self.pi2_pop, i, False)
            th = self.theta - (2*np.pi*ext_Q[i] if ext_Q is not None else 0)
            arg = 2*np.sqrt(np.pi)*self.phi + th*self.id
            cos_op = SparsePauliOp.from_operator(Operator(0.5*(expm(1j*arg)+expm(-1j*arg))))
            n_b = 1 if i in [0, self.n-1] else 2
            op_P += (cm + n_b*cg) * self.embed(self.phi2_pop, i, False) + ci * self.embed(cos_op, i, False)

        for i in range(self.n-1):
            op_I += (-2*cg) * self.embed(self.phi_pop, i, False).compose(self.embed(self.phi_pop, i+1, False))

        step = QuantumCircuit(qc.num_qubits)
        for op in [op_K, op_P, op_I]: step.append(PauliEvolutionGate(op, time=dt), range(qc.num_qubits))
        for _ in range(steps): qc.append(step, range(qc.num_qubits))
        return qc

    def get_hamiltonian_op(self, ext_Q=None):
        """Constructs the full Hamiltonian as a SparsePauliOp for Estimator."""
        qc = QuantumCircuit(self.n * int(np.log2(self.dim)))
        H_op = SparsePauliOp(["I"*qc.num_qubits], coeffs=[0.0])
        
        ck, cg, cm = 0.5/self.a, 0.5/self.a, 0.5*self.a*self.mu**2
        ci = -0.2 * self.a * self.m * self.e

        for i in range(self.n):
            H_op += ck * self.embed(self.pi2_pop, i, False)
            H_op += (cm + (1 if i in [0, self.n-1] else 2)*cg) * self.embed(self.phi2_pop, i, False)
            
            th = self.theta - (2*np.pi*ext_Q[i] if ext_Q is not None else 0)
            arg = 2*np.sqrt(np.pi)*self.phi + th*self.id
            cos_op = SparsePauliOp.from_operator(Operator(0.5*(expm(1j*arg)+expm(-1j*arg))))
            H_op += ci * self.embed(cos_op, i, False)

        for i in range(self.n-1):
            H_op += (-2*cg) * self.embed(self.phi_pop, i, False).compose(self.embed(self.phi_pop, i+1, False))
            
        return H_op

# --- PLOTTING TASKS ---

def task_1_lightcone():
    # Quantum Measurement vs Exact Diagonalization
    print("Task 1: Lightcone (Quantum Measurement vs ED)")
    N, sim = 7, SchwingerBoson(7, 4, m=0.5, theta=np.pi) # dim=4 (2 qubits per site)
    
    # Classical ED for initial state
    psi0 = eigsh(SchwingerBoson(N, 4, m=0.5).get_H(), k=1, which='SA')[1][:,0]
    
    times = np.linspace(0, 6, 15)
    obs = [sim.embed(sim.phi_pop, i, False) for i in range(N)]
    est = AerEstimator()
    est.options.default_shots = 4000
    
    res_q = np.zeros((len(times), N))
    res_c = np.zeros((len(times), N))
    
    for t_idx, t in enumerate(tqdm(times)):
        # Classical
        psi_t = expm_multiply(-1j * sim.get_H() * t, psi0)
        res_c[t_idx] = [(psi_t.conj().T @ sim.embed(sim.phi, i) @ psi_t).real for i in range(N)]
        
        # Quantum
        if t == 0:
            res_q[t_idx] = res_c[t_idx]
            continue
            
        qc = QuantumCircuit(N * 2) 
        qc.initialize(psi0) # FIX: Using initialize instead of prepare_state
        qc.append(sim.get_trotter(t/max(1, int(t/0.1)), max(1, int(t/0.1))), range(qc.num_qubits))
        isa = transpile(qc, AerSimulator())
        res_q[t_idx] = est.run([(isa, [o.apply_layout(isa.layout) for o in obs])]).result()[0].data.evs

    # Calculate Charge Density Gradient
    def get_rho(res):
        rho = np.zeros_like(res)
        for t in range(len(times)):
            rho[t, 0] = res[t, 1] - res[t, 0]
            rho[t, 1:-1] = (res[t, 2:] - res[t, :-2]) / 2.0
            rho[t, -1] = res[t, -1] - res[t, -2]
        return rho

    rho_c, rho_q = get_rho(res_c), get_rho(res_q)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot Classical
    im1 = ax1.imshow(rho_c, aspect='auto', origin='lower', cmap='RdBu_r', extent=[0, N-1, 0, 6])
    ax1.set_title("Exact Diagonalization")
    ax1.set_xlabel("Lattice Site")
    ax1.set_ylabel("Time ($1/g$)")
    
    # Plot Quantum
    im2 = ax2.imshow(rho_q, aspect='auto', origin='lower', cmap='RdBu_r', extent=[0, N-1, 0, 6])
    ax2.set_title("Quantum Measurement")
    ax2.set_xlabel("Lattice Site")
    
    fig.colorbar(im2, ax=[ax1, ax2], label=r"Induced Charge Density $\langle j^0(x) \rangle$")
    plt.suptitle("Screening Light Cone Dynamics")
    plt.savefig("figures_new/plot1_lightcone.png")
    plt.close()

def task_2_potential():
    # Quantum Measurement (Energy Meas) vs Classical ED
    print("Task 2: Potential V(L) (Quantum Energy Meas vs ED)")
    N, Ls = 8, range(1, 6) 
    model = SchwingerBoson(N, 4, m=0.3) # dim=4
    
    # Vacuum Reference
    H_vac_op = model.get_hamiltonian_op(np.zeros(N))
    H_vac_mat = model.get_H(np.zeros(N))
    
    # Classical Vacuum Energy
    E_vac_c = eigsh(H_vac_mat, k=1, which='SA')[0][0]
    
    # Quantum Vacuum Energy
    est = AerEstimator()
    est.options.default_shots = 4000
    vac_vec = eigsh(H_vac_mat, k=1, which='SA')[1][:,0]
    qc_vac = QuantumCircuit(N * 2); qc_vac.initialize(vac_vec) # FIX
    qc_isa = transpile(qc_vac, AerSimulator())
    E_vac_q = est.run([(qc_isa, [H_vac_op.apply_layout(qc_isa.layout)])]).result()[0].data.evs[0]

    E_f_c, E_i_c = [], []
    E_f_q, E_i_q = [], []

    for L in tqdm(Ls):
        s, e = (N-L)//2, (N-L)//2 + L
        D_f, D_i = np.zeros(N), np.zeros(N)
        D_f[s:e], D_i[s:e] = 0.5, 1.0
        
        # Classical
        E_f_c.append(eigsh(model.get_H(D_f), k=1, which='SA')[0][0] - E_vac_c)
        E_i_c.append(eigsh(model.get_H(D_i), k=1, which='SA')[0][0] - E_vac_c)
        
        # Quantum
        vec_f = eigsh(model.get_H(D_f), k=1, which='SA')[1][:,0]
        vec_i = eigsh(model.get_H(D_i), k=1, which='SA')[1][:,0]
        
        qc_f, qc_i = QuantumCircuit(N * 2), QuantumCircuit(N * 2)
        qc_f.initialize(vec_f); qc_i.initialize(vec_i) # FIX
        
        H_f_op = model.get_hamiltonian_op(D_f)
        H_i_op = model.get_hamiltonian_op(D_i)
        
        isa_f = transpile(qc_f, AerSimulator())
        isa_i = transpile(qc_i, AerSimulator())
        
        res_f = est.run([(isa_f, [H_f_op.apply_layout(isa_f.layout)])]).result()[0].data.evs[0]
        res_i = est.run([(isa_i, [H_i_op.apply_layout(isa_i.layout)])]).result()[0].data.evs[0]
        
        E_f_q.append(res_f - E_vac_q)
        E_i_q.append(res_i - E_vac_q)

    plt.figure()
    plt.plot(Ls, E_f_c, 'r-', label="Q=0.5e (Confining) - Classical")
    plt.plot(Ls, E_i_c, 'b-', label="Q=1.0e (Screening) - Classical")
    plt.plot(Ls, E_f_q, 'r--', marker='o', label="Q=0.5e - Quantum Meas")
    plt.plot(Ls, E_i_q, 'b--', marker='s', label="Q=1.0e - Quantum Meas")
    
    plt.xlabel("Separation Distance $L$ (lattice sites)")
    plt.ylabel("Potential Energy $V(L) = - E_0(L) + E_{vac}$")
    plt.title("Static Potential between External Charges")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures_new/plot2_potential.png"); plt.close()

def task_3_entropy():
    # Classical Statevector vs Trotterized Statevector
    print("Task 3: Entropy (Classical vs Trotter)")
    N, model = 6, SchwingerBoson(6, 4, m=0.1) # dim=4
    vac = eigsh(model.get_H(), k=1, which='SA')[1][:,0]
    
    xL, xR = N//3, 2*N//3
    psi = vac.copy().reshape([4]*N)
    for i, x in enumerate([xL, xR]):
        op = [(model.adag if j==x else model.id) for j in range(N)]
        m_op = op[0]
        for k in range(1, N): m_op = np.kron(m_op, op[k])
        vac = m_op @ vac * np.exp(1j * (1.5 if i==0 else -1.5) * x)
        
    psi0 = vac / np.linalg.norm(vac)
    times = np.linspace(0, 8, 50)
    
    S_c, S_q = [], []
    
    # Classical Exact
    for t in times:
        psi_t = expm_multiply(-1j * model.get_H() * t, psi0)
        S_c.append(entropy(partial_trace(Statevector(psi_t), list(range(N//2 * 2))), base=np.e))

    # Trotterized Evolution
    sv = Statevector(psi0)
    gate = model.get_trotter(times[1], 5)
    for _ in times:
        S_q.append(entropy(partial_trace(sv, list(range(N//2 * 2))), base=np.e))
        sv = sv.evolve(gate)

    plt.figure()
    plt.plot(times, S_c, 'k-', label="Exact Dynamics")
    plt.plot(times, S_q, 'm--', marker='d', label="Trotterized Dynamics")
    plt.legend()
    plt.xlabel("Time ($1/g$)")
    plt.ylabel("One-third-Chain Entanglement Entropy $S_{vN}$")
    plt.title("Entanglement Growth during Scattering")
    plt.grid(True)
    plt.savefig("figures_new/plot3_entropy.png"); plt.close()

def task_4_scattering():
    # Quantum Measurement of Particle Number vs Classical ED
    print("Task 4: Scattering Prob (Quantum Meas vs Classical)")
    N, model = 6, SchwingerBoson(6, 4, m=0.5) # dim=4
    ks = np.linspace(0.5, 3.0, 16)
    
    # Particle Number Operator
    N_op_sparse = csr_matrix((4**N, 4**N))
    N_op_pauli = SparsePauliOp(["I"*(N*2)], coeffs=[0.0])
    for i in range(N):
        N_op_sparse += model.embed(model.n_op, i)
        N_op_pauli += model.embed(model.n_pop, i, False)
        
    est = AerEstimator()
    est.options.default_shots = 4000
    
    probs_c, probs_q = [], []
    
    for k in tqdm(ks):
        # Wavepacket Construction
        psi = np.zeros(4**N, dtype=complex)
        for x1 in range(N):
            for x2 in range(x1+1, N):
                amp = np.exp(-((x1-1.5)**2 + (x2-3.5)**2)) * np.exp(1j*k*(x1-x2))
                if abs(amp) > 1e-3:
                    state = np.zeros(4**N); 
                    idx = 1 * (4**(N-1-x1)) + 1 * (4**(N-1-x2))
                    state[idx] = 1.0
                    psi += amp * state
        psi /= np.linalg.norm(psi)
        
        # Classical Evolution & Meas
        psi_f_c = expm_multiply(-1j * model.get_H() * 4.0, psi)
        n_exp_c = (psi_f_c.conj().T @ N_op_sparse @ psi_f_c).real
        probs_c.append(1.0 / (1.0 + max(0, n_exp_c - 2.0)))
        
        # Quantum Evolution & Meas (Trotter + Estimator)
        qc = QuantumCircuit(N * 2) # 2 qubits per site
        qc.initialize(psi) # FIX: Using initialize
        qc.append(model.get_trotter(4.0/20, 20), range(qc.num_qubits))
        isa = transpile(qc, AerSimulator())
        n_exp_q = est.run([(isa, [N_op_pauli.apply_layout(isa.layout)])]).result()[0].data.evs[0]
        probs_q.append(1.0 / (1.0 + max(0, n_exp_q - 2.0)))

    plt.figure()
    plt.plot(ks, probs_c, 'g-', label="Classical ED")
    plt.plot(ks, probs_q, 'g--', marker='o', label="Quantum Meas")
    plt.legend()
    plt.xlabel("Initial Momentum $k$ (~ Kinetic Energy)")
    plt.ylabel("Elastic Scattering Probability Proxy")
    plt.title("Scattering Probability vs Collision Energy")
    plt.grid(True)
    plt.savefig("figures_new/plot4_scattering.png"); plt.close()

if __name__ == "__main__":
    task_1_lightcone()
    task_2_potential()
    task_3_entropy()
    task_4_scattering()