import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit.library import QFTGate, DiagonalGate
import imageio.v2 as imageio
import os
import shutil

# =============================================================================
# --- 1. PARAMETERS & CONFIGURATION ---
# =============================================================================
HBAR = 1.0
M = 0.5
TWO_M = 2 * M 

# Grid Parameters (Must be power of 2 for Qubits)
N_QUBITS = 9              # 2^9 = 512 grid points
N_GRID = 2**N_QUBITS
L = 160.0                 # Large box size
dx = L / N_GRID

x_grid = np.linspace(-L/2, L/2, N_GRID, endpoint=False)
# Standard FFT Momentum Grid
k_grid = fftfreq(N_GRID, d=dx) * (2 * np.pi)

# Simulation Parameters
dt = 0.005
n_steps = 4000             # Increased steps to ensure full reflection capture
frames_to_save = 150      # More frames for smoother animation

# Wavepacket (Incoming from right)
k0_target = -5.0
x0_start = 20.0
sigma_start = 2.0

# Morse Potential Parameters
D_morse = 15.0
alpha_morse = 0.8
xe_morse = 0.0

# =============================================================================
# --- 2. POTENTIALS & INITIALIZATION ---
# =============================================================================

def get_morse_potential(x):
    """Calculates Morse potential with clipping."""
    V = D_morse * (1 - np.exp(-alpha_morse * (x - xe_morse)))**2
    # Clip at V_max = 200 to prevent aliasing
    V_clipped = np.minimum(V, 200.0)
    return V_clipped

def get_hard_wall_potential(x):
    """
    Creates a hard wall at x = 0.
    V = 0 for x > 0
    V = 200 for x <= 0
    """
    V = np.zeros_like(x)
    # Hard wall at x=0 (indices where x <= 0)
    V[x <= 0] = 200.0
    return V

def get_initial_state(x, k0, x0, sigma):
    """Gaussian Wavepacket"""
    norm_factor = 1.0 / (np.sqrt(sigma * np.sqrt(np.pi)))
    psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
    psi /= np.linalg.norm(psi)
    return psi

# =============================================================================
# --- 3. QUANTUM CIRCUIT CONSTRUCTION ---
# =============================================================================

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

# =============================================================================
# --- 4. DATA ANALYSIS & PLOTTING ---
# =============================================================================

def calculate_expectation_x(psi, x_grid, dx):
    """Calculates <x> = Sum |psi|^2 * x * dx"""
    prob = np.abs(psi)**2
    return np.sum(prob * x_grid) * dx

def plot_frame(x, psi_morse, psi_wall, V_morse, step, filename):
    prob_morse = np.abs(psi_morse)**2
    prob_wall = np.abs(psi_wall)**2
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # --- Right Axis: Morse Potential Only ---
    ax2 = ax1.twinx()
    ax2.plot(x, V_morse, 'k-', lw=1, alpha=0.3, label="Morse Potential")
    ax2.fill_between(x, V_morse, 200, color='gray', alpha=0.1)
    ax2.set_ylabel("Potential Energy")
    ax2.set_ylim(0, 50) 
    
    # --- Left Axis: Wavefunctions ---
    # 1. Morse System (Blue Solid)
    ax1.plot(x, prob_morse, 'b-', lw=2.5, alpha=0.8, label="Wavepacket")
    
    # 2. Hard Wall Reference (Green Dashed)
    ax1.plot(x, prob_wall, 'g--', lw=1.5, alpha=0.8, label="Free Reflection (Hard Wall)")
    
    ax1.set_xlabel("Position (x)")
    ax1.set_ylabel("Probability Density")
    ax1.set_title(f"Morse Scattering with Time Delay Analysis: Step {step}")
    
    # Dynamically scale Y to keep packets visible
    max_p = max(np.max(prob_morse), np.max(prob_wall))
    # Avoid div by zero if packets disappear (unlikely)
    if max_p < 1e-6: max_p = 1.0
    ax1.set_ylim(0, 0.2)
    ax1.set_xlim(-15, 50)
    
    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# =============================================================================
# --- 5. MAIN SIMULATION LOOP ---
# =============================================================================

def run_simulation():
    # 1. Setup Potentials
    V_morse = get_morse_potential(x_grid)
    V_wall = get_hard_wall_potential(x_grid)
    
    psi_init = get_initial_state(x_grid, k0_target, x0_start, sigma_start)
    
    # 2. Build Operators
    print("Building Quantum Circuits...")
    
    qc_morse = build_trotter_circuit(N_QUBITS, dt, V_morse, k_grid)
    op_morse = Operator(qc_morse)
    
    qc_wall = build_trotter_circuit(N_QUBITS, dt, V_wall, k_grid)
    op_wall = Operator(qc_wall)
    
    # 3. Initialize States
    sv_morse = Statevector(psi_init) 
    sv_wall = Statevector(psi_init) 
    
    # 4. Prepare Output
    output_dir = "morse_qft_frames"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    frame_files = []
    steps_per_frame = max(1, n_steps // frames_to_save)
    
    print(f"Starting Simulation ({n_steps} steps)...")
    
    # Group velocity v_g = p/m = hbar*k/m
    v_group = np.abs(k0_target) / M
    
    for step in range(n_steps + 1):
        
        # Save Frame & Analysis
        if step % steps_per_frame == 0:
            fname = os.path.join(output_dir, f"frame{len(frame_files)+1}.png")
            
            # --- Analysis: Time Delay ---
            # Calculate center of mass <x>
            x_avg_morse = calculate_expectation_x(sv_morse.data, x_grid, dx)
            x_avg_wall = calculate_expectation_x(sv_wall.data, x_grid, dx)
            
            # Spatial shift relative to the wall reference
            # Positive shift means Morse packet is to the right of the wall packet
            spatial_shift = x_avg_morse - x_avg_wall
            
            # Time delay = spatial_shift / group_velocity
            # Note: This is most meaningful after reflection (when packets separate)
            time_delay = spatial_shift / v_group
            
            print(f"Step {step}: <x>_morse={x_avg_morse:.2f}, <x>_wall={x_avg_wall:.2f}, Delay={time_delay:.2f} (approx time units)")
            
            plot_frame(x_grid, sv_morse.data, sv_wall.data, V_morse, step, fname)
            frame_files.append(fname)
            
        # Evolution Step
        sv_morse = sv_morse.evolve(op_morse)
        sv_wall = sv_wall.evolve(op_wall)
        
    # 5. Create GIF (Slower: duration=0.15s)
    gif_name = "Wigner_Time_Delay.gif"
    print(f"Creating GIF: {gif_name}...")
    images = [imageio.imread(f) for f in frame_files]
    imageio.mimsave(gif_name, images, duration=1)
    
    # Cleanup
    # shutil.rmtree(output_dir)
    # print("Done.")

if __name__ == "__main__":
    run_simulation()