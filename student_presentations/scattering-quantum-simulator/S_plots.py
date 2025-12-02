import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq, fftshift
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit.library import QFTGate, DiagonalGate

# =============================================================================
# --- 1. PARAMETERS ---
# =============================================================================
HBAR = 1.0
M = 0.5
TWO_M = 2 * M 

# Grid Parameters
# We need a large box to contain the spreading packet over time
N_QUBITS = 10            # 1024 points
N_GRID = 2**N_QUBITS
L = 400.0                 # Very large box
dx = L / N_GRID

x_grid = np.linspace(-L/2, L/2, N_GRID, endpoint=False)
# k_grid for operators (standard FFT order)
k_grid_op = fftfreq(N_GRID, d=dx) * (2 * np.pi)

# Simulation Parameters
dt = 0.05
n_steps = 3500            # Enough time for packet to traverse and split

# Wavepacket (Broadband Source)
# Narrow spatial width (sigma=0.5) -> Broad momentum width
# Centered at k=4.0 to cover the energy range of interest
k0_incident = 4.0         
x0_start = -80.0          # Start far left
sigma_start = 0.5         

# Double Barrier Parameters
V_height = 12.0
barrier_width = 1.5
well_width = 3.0

# =============================================================================
# --- 2. FUNCTIONS ---
# =============================================================================

def get_double_barrier_potential(x):
    V = np.zeros_like(x)
    # Right Barrier
    start_right = well_width / 2.0
    end_right = start_right + barrier_width
    mask_right = (x >= start_right) & (x <= end_right)
    # Left Barrier
    start_left = -well_width / 2.0 - barrier_width
    end_left = -well_width / 2.0
    mask_left = (x >= start_left) & (x <= end_left)
    V[mask_left | mask_right] = V_height
    return V

def get_initial_state(x, k0, x0, sigma):
    norm_factor = 1.0 / (np.sqrt(sigma * np.sqrt(np.pi)))
    psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
    psi /= np.linalg.norm(psi)
    return psi

def build_step_operator(n_qubits, dt, V_x, k_grid):
    # Trotter Step: exp(-iV/2) exp(-iT) exp(-iV/2)
    qc = QuantumCircuit(n_qubits)
    
    phase_V = np.exp(-1j * V_x * (dt / 2) / HBAR)
    phase_T = np.exp(-1j * (HBAR * k_grid)**2 / TWO_M * dt / HBAR)
    
    qc.append(DiagonalGate(phase_V), range(n_qubits))
    qc.append(QFTGate(n_qubits), range(n_qubits))
    qc.append(DiagonalGate(phase_T), range(n_qubits))
    qc.append(QFTGate(n_qubits).inverse(), range(n_qubits))
    qc.append(DiagonalGate(phase_V), range(n_qubits))
    
    return Operator(qc)

# --- Analytical TMM Solver (for comparison) ---
def get_transfer_matrix_step(E, V, width):
    q = np.sqrt(2 * M * (E - V + 0j) / HBAR**2)
    if np.abs(q) < 1e-12:
        return np.array([[1.0, -width], [0.0, 1.0]], dtype=complex)
    c = np.cos(q * width)
    s = np.sin(q * width)
    # Backward propagation matrix
    return np.array([[c, -(1/q)*s], [q*s, c]], dtype=complex)

def calculate_analytical_T(energies):
    T_list = []
    for E in energies:
        if E <= 0: 
            T_list.append(0)
            continue
        k = np.sqrt(2 * M * E) / HBAR
        
        # Matrices (Right to Left)
        M2 = get_transfer_matrix_step(E, V_height, barrier_width)
        M_well = get_transfer_matrix_step(E, 0.0, well_width)
        M1 = get_transfer_matrix_step(E, V_height, barrier_width)
        M_total = M1 @ M_well @ M2
        
        # Apply to outgoing state [1, ik]
        psi_out = np.array([1.0, 1j*k])
        psi_in = M_total @ psi_out
        
        # Extract incident amplitude A from psi_in
        # 2A = psi + psi'/ik
        A = 0.5 * (psi_in[0] + psi_in[1] / (1j*k))
        T_list.append(1.0 / np.abs(A)**2)
    return np.array(T_list)

# =============================================================================
# --- 3. MAIN ANALYSIS ---
# =============================================================================

def run_analysis():
    print("1. Initializing Simulation...")
    V_x = get_double_barrier_potential(x_grid)
    psi_init = get_initial_state(x_grid, k0_incident, x0_start, sigma_start)
    
    # Compute Initial Momentum Distribution
    # We need this as the denominator for T(k)
    phi_init = fft(psi_init)
    
    print(f"2. Evolving Wavepacket ({n_steps} steps)...")
    op_step = build_step_operator(N_QUBITS, dt, V_x, k_grid_op)
    sv = Statevector(psi_init)
    
    # Fast loop (no intermediate plotting)
    for _ in range(n_steps):
        sv = sv.evolve(op_step)
        
    psi_final = sv.data
    phi_final = fft(psi_final)
    
    print("3. Analyzing Momentum Spectrum...")
    
    # --- Process FFT Data ---
    # 1. Shift FFT to center k=0
    phi_init_shifted = fftshift(phi_init)
    phi_final_shifted = fftshift(phi_final)
    
    k_vals = fftshift(k_grid_op)
    
    # 2. Map k to Energy E = h^2 k^2 / 2m
    # We only care about positive k (incident direction)
    mask_pos_k = k_vals > 0.1  # Avoid k=0 singularity
    
    k_pos = k_vals[mask_pos_k]
    E_pos = (HBAR * k_pos)**2 / TWO_M
    
    # Initial momentum density (Incident Envelope)
    # Using +k because incident wave was moving right
    rho_init = np.abs(phi_init_shifted[mask_pos_k])**2
    
    # Final momentum density
    # Positive k in final state = Transmitted
    # Negative k in final state = Reflected
    
    # Transmitted: Look at +k in final state
    rho_final_trans = np.abs(phi_final_shifted[mask_pos_k])**2
    
    # Reflected: Look at -k in final state
    # We need to map -k indices to the corresponding +k energies
    # Since the grid is symmetric, index i corresponds to -index (roughly)
    mask_neg_k = k_vals < -0.1
    # Flip to align with E_pos
    rho_final_refl = np.abs(phi_final_shifted[mask_neg_k])**2
    rho_final_refl = np.flip(rho_final_refl) # Flip to match +k ordering
    
    # Ensure shapes match (sometimes fftshift size can be off by 1)
    min_len = min(len(rho_init), len(rho_final_trans), len(rho_final_refl))
    rho_init = rho_init[:min_len]
    rho_final_trans = rho_final_trans[:min_len]
    rho_final_refl = rho_final_refl[:min_len]
    E_plot = E_pos[:min_len]
    
    # 3. Calculate Coefficients
    # Filter out regions where initial packet had zero amplitude to avoid div/0
    # We only trust results where incident intensity is significant
    valid_mask = rho_init > (np.max(rho_init) * 0.01)
    
    E_valid = E_plot[valid_mask]
    T_sim = rho_final_trans[valid_mask] / rho_init[valid_mask]
    R_sim = rho_final_refl[valid_mask] / rho_init[valid_mask]
    
    # --- Analytical Comparison ---
    T_anal = calculate_analytical_T(E_valid)
    
    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Momentum Distributions (Raw Data)
    ax1.plot(E_plot, rho_init, 'k--', label='Initial Incident Packet')
    ax1.plot(E_plot, rho_final_trans, 'b-', label='Final Transmitted (+k)')
    ax1.plot(E_plot, rho_final_refl, 'r-', label='Final Reflected (-k)')
    ax1.set_title("Momentum Distributions (Energy Space)")
    ax1.set_ylabel("Probability Density")
    ax1.set_xlabel("Energy")
    ax1.set_xlim(0, 25)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Coefficients vs Energy
    ax2.plot(E_valid, T_sim, 'bo', markersize=4, label='Simulated Transmission T(E)')
    ax2.plot(E_valid, R_sim, 'ro', markersize=4, label='Simulated Reflection R(E)')
    ax2.plot(E_valid, T_anal, 'k-', linewidth=1.5, label='Analytical Exact T(E)')
    ax2.plot(E_valid, 1-T_anal, 'k--', linewidth=1.5, label='Analytical Exact R(E)')
    
    ax2.set_title("Scattering Coefficients: Simulation vs Theory")
    ax2.set_ylabel("Probability")
    ax2.set_xlabel("Energy")
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlim(0, 25)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add barrier height marker
    ax2.axvline(V_height, color='gray', linestyle=':', label='Barrier Height')
    
    plt.tight_layout()
    plt.savefig("Packet_Scattering_Result.png")
    print("Plot saved to Packet_Scattering_Result.png")
    plt.show()

if __name__ == "__main__":
    run_analysis()