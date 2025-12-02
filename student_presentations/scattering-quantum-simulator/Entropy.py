import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit.library import QFTGate, DiagonalGate
import imageio.v2 as imageio
import os
import shutil


HBAR = 1.0
M = 0.5
TWO_M = 2 * M 

# Grid Parameters
# N_QUBITS = 10 (1024 points) covers the domain L=250 well
N_QUBITS = 10             
N_GRID = 2**N_QUBITS
L = 250.0                 
dx = L / N_GRID

x_grid = np.linspace(-L/2, L/2, N_GRID, endpoint=False)
# Standard FFT Momentum Grid
k_grid = fftfreq(N_GRID, d=dx) * (2 * np.pi)

# Simulation Parameters
dt = 0.005
n_steps = 8000           
frames_to_save = 150      

# Wavepacket (Incoming from right towards the well)
# Note: Morse potential is flat on the right, wall on the left.
k0_target = -3.5          # Negative momentum (moving Left)
x0_start = 60.0           # Start on the right side
sigma_start = 3.5

# Morse Potential Parameters
D_e = 15.0                # Well depth
alpha = 0.12              # Width parameter (controls steepness)
x_e = -10.0               # Equilibrium position (minimum of the well)
V_max_cap = 50.0          # Cap potential to prevent FFT/Phase numerical errors

# Entanglement subsystem split
SUBSYSTEM_SIZE = N_QUBITS // 2


def get_morse_potential(x):
    """
    V(x) = D_e * (1 - exp(-alpha * (x - x_e)))^2
    
    Features:
    - Minimum is 0 at x = x_e
    - Approaches D_e as x -> infinity (dissociation)
    - Shoots to infinity as x -> -infinity (repulsive wall)
    """
    # Standard Morse formula
    V = D_e * (1 - np.exp(-alpha * (x - x_e)))**2
    
    # IMPORTANT: In a grid-based FFT simulation, potentials that go to infinity
    # cause aliasing issues in the time evolution operator exp(-iV*dt).
    # We cap the repulsive wall at a reasonable height.
    V = np.minimum(V, V_max_cap)
    
    return V

def get_free_potential(x):
    """
    V(x) = D_e everywhere.
    
    Why D_e? 
    Because at large x (where the particle starts), the Morse potential 
    approaches D_e, not 0. To make a fair "Time Delay" comparison, 
    the free particle should have the same background energy as the 
    asymptotic limit of the Morse potential.
    """
    return np.full_like(x, D_e)

def get_initial_state(x, k0, x0, sigma):
    """Gaussian Wavepacket"""
    norm_factor = 1.0 / (np.sqrt(sigma * np.sqrt(np.pi)))
    psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
    psi /= np.linalg.norm(psi)
    return psi



def build_trotter_circuit(n_qubits, dt, V_x, k_grid):
    """
    Builds one Strang Splitting Step: exp(-iV/2) exp(-iT) exp(-iV/2)
    """
    qc = QuantumCircuit(n_qubits)
    
    # 1. Pre-compute Diagonal Phases
    phase_V_half = np.exp(-1j * V_x * (dt / 2) / HBAR)
    
    T_vals = (HBAR * k_grid)**2 / TWO_M
    phase_T = np.exp(-1j * T_vals * dt / HBAR)
    
    # 2. Construct Circuit
    
    # A. First Potential Half-Step (Position)
    qc.append(DiagonalGate(phase_V_half), range(n_qubits))
    
    # B. Kinetic Step (Momentum)
    qc.append(QFTGate(n_qubits), range(n_qubits))
    qc.append(DiagonalGate(phase_T), range(n_qubits))
    qc.append(QFTGate(n_qubits).inverse(), range(n_qubits))
    
    # C. Second Potential Half-Step (Position)
    qc.append(DiagonalGate(phase_V_half), range(n_qubits))
    
    return qc



def von_neumann_entropy_from_statevector(psi, n_qubits, subsystem_size=SUBSYSTEM_SIZE, base=2.0):
    """Compute von Neumann entropy of subsystem A."""
    kA = int(subsystem_size)
    kB = int(n_qubits - kA)
    dimA = 2**kA
    dimB = 2**kB
    
    psi_matrix = psi.reshape((dimA, dimB))
    s = np.linalg.svd(psi_matrix, compute_uv=False)
    p = s**2
    p = p[p > 1e-12]
    if p.size == 0:
        return 0.0
    p = p / np.sum(p)
    
    if base == 2.0:
        S = -np.sum(p * np.log2(p))
    else:
        S = -np.sum(p * np.log(p)) / np.log(base)
    return np.real_if_close(S)


def calculate_expectation_x(psi, x_grid, dx):
    prob = np.abs(psi)**2
    return np.sum(prob * x_grid) * dx


def plot_frame(x, psi_sys, psi_ref, V_sys, step, entropy_sys, filename):
    prob_sys = np.abs(psi_sys)**2
    prob_ref = np.abs(psi_ref)**2
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # --- Right Axis: Potential ---
    ax2 = ax1.twinx()
    ax2.plot(x, V_sys, 'k-', lw=2, alpha=0.5, label="Morse Potential")
    # Fill the potential well
    ax2.fill_between(x, V_sys, np.max(V_sys), color='gray', alpha=0.1)
    
    ax2.set_ylabel("Potential Energy")
    # Limit potential view so we can see the well structure
    ax2.set_ylim(0, D_e * 2.5) 
    
    # --- Left Axis: Wavefunctions ---
    # 1. System (Morse)
    ax1.plot(x, prob_sys, 'r-', lw=2.5, alpha=0.8, label="Packet (Morse)")
    
    # 2. Reference (Free Particle)
    ax1.plot(x, prob_ref, 'g--', lw=1.5, alpha=0.5, label="Packet (Free)")
    
    ax1.set_xlabel("Position (x)")
    ax1.set_ylabel("Probability Density")
    ax1.set_title(f"Scattering off Morse Potential: Step {step} — Entropy (A) = {entropy_sys:.3f} bits")
    
    # Fixed Scale for consistency
    ax1.set_ylim(0, 0.15)
    # Zoom in slightly on the interaction region
    ax1.set_xlim(-60, 90)
    
    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# =============================================================================
# --- 5. MAIN SIMULATION LOOP ---
# =============================================================================

def run_simulation():

    V_morse = get_morse_potential(x_grid)
   
    V_free = get_free_potential(x_grid)
    
    psi_init = get_initial_state(x_grid, k0_target, x0_start, sigma_start)
    

    print("Building Quantum Circuits...")
    

    qc_morse = build_trotter_circuit(N_QUBITS, dt, V_morse, k_grid)
    op_morse = Operator(qc_morse)
    
    
    qc_free = build_trotter_circuit(N_QUBITS, dt, V_free, k_grid)
    op_free = Operator(qc_free)
    

    sv_morse = Statevector(psi_init) 
    sv_free = Statevector(psi_init) 

    output_dir = "morse_qft_frames"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    frame_files = []
    steps_per_frame = max(1, n_steps // frames_to_save)
    
    print(f"Starting Simulation ({n_steps} steps, dt={dt})...")
    

    v_group = np.abs(k0_target) / M
    
    ent_morse = []
    ent_free = []
    times = []
    
    frame_idx = 1  

    for step in range(n_steps + 1):
        current_time = step * dt
        
        S_morse = von_neumann_entropy_from_statevector(sv_morse.data, N_QUBITS, SUBSYSTEM_SIZE)
        S_free = von_neumann_entropy_from_statevector(sv_free.data, N_QUBITS, SUBSYSTEM_SIZE)
        ent_morse.append(S_morse)
        ent_free.append(S_free)
        times.append(current_time)
        
        if step % steps_per_frame == 0:
            fname = os.path.join(output_dir, f"frame{frame_idx}.png")
            
            x_avg_sys = calculate_expectation_x(sv_morse.data, x_grid, dx)
            x_avg_free = calculate_expectation_x(sv_free.data, x_grid, dx)
            

            spatial_shift = x_avg_sys - x_avg_free
            
            if step % (steps_per_frame * 5) == 0:
                print(f"Step {step}: Pos(Sys)={x_avg_sys:.1f}, Entropy={S_morse:.3f}")
            
            plot_frame(x_grid, sv_morse.data, sv_free.data, V_morse, step, S_morse, fname)
            frame_files.append(fname)
            frame_idx += 1
            
        # Evolution Step
        sv_morse = sv_morse.evolve(op_morse)
        sv_free = sv_free.evolve(op_free)
        

    gif_name = "MorsePotential_Dynamics.gif"
    print(f"Creating GIF: {gif_name}...")
    images = [imageio.imread(f) for f in frame_files]
    imageio.mimsave(gif_name, images, duration=0.15)

   
    plt.figure(figsize=(8, 5))
    plt.plot(times, ent_morse, color='r', label=f"Morse (A={SUBSYSTEM_SIZE})")
    plt.plot(times, ent_free, color='g', linestyle='--', label=f"Free (A={SUBSYSTEM_SIZE})")
    plt.xlabel('Time')
    plt.ylabel('Von Neumann Entropy')
    plt.title('Entanglement Entropy: Morse Scattering')
    plt.legend()
    plt.tight_layout()
    entropy_plot = 'morse_entropy.png'
    plt.savefig(entropy_plot)
    plt.close()
    print(f"Saved entropy plot: {entropy_plot}")
    
    print("Done.")

if __name__ == "__main__":
    run_simulation()