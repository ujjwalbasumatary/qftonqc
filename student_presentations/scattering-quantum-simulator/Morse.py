import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.sparse as sparse
from scipy.sparse.linalg import factorized

class QuantumSimulation:
    def __init__(self, N=1000, L=40.0, mass=0.5, dt=0.05):
        self.N = N
        self.L = L
        self.mass = mass
        self.dt = dt
        
        self.x = np.linspace(-L/2, L/2, N)
        self.dx = self.x[1] - self.x[0]
        
        self.D = 15.0
        self.alpha = 0.8
        self.xe = 0.0
        
        self.V = self._morse_potential(self.x)
        self.psi = np.zeros(N, dtype=complex)
        self.H_matrix = None
        self.evolution_step = None
        
    def _morse_potential(self, x):
        return self.D * (1 - np.exp(-self.alpha * (x - self.xe)))**2

    def initialize_wavepacket(self, x0, k0, sigma):
        norm_factor = (1.0 / (np.pi * sigma**2))**0.25
        self.psi = norm_factor * np.exp(-(self.x - x0)**2 / (2 * sigma**2)) * \
                   np.exp(1j * k0 * self.x)
        self.psi /= np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx)

    def build_hamiltonian(self):
        t = 1.0 / (2 * self.mass * self.dx**2)
        
        main_diag = 2*t * np.ones(self.N) + self.V
        off_diag = -t * np.ones(self.N - 1)
        
        diagonals = [main_diag, off_diag, off_diag]
        offsets = [0, 1, -1]
        self.H_matrix = sparse.diags(diagonals, offsets, format='csc')
        
        I = sparse.eye(self.N, format='csc')
        A_matrix = I + 1j * self.H_matrix * self.dt / 2
        self.B_evolution = I - 1j * self.H_matrix * self.dt / 2
        
        self.evolution_step = factorized(A_matrix)

    def step(self):
        rhs = self.B_evolution.dot(self.psi)
        self.psi = self.evolution_step(rhs)
        return self.psi

def run_scattering_simulation():
    N_POINTS = 800
    BOX_SIZE = 30.0
    DT = 0.05
    
    sim = QuantumSimulation(N=N_POINTS, L=BOX_SIZE, mass=0.5, dt=DT)
    
    sim.initialize_wavepacket(x0=8, k0=-5.0, sigma=0.5)
    sim.build_hamiltonian()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.set_xlim(-10, 15)
    ax.set_ylim(-2, 20)
    ax.set_xlabel("Position (x)")
    ax.set_ylabel("Energy / Amplitude")
    ax.set_title("Scattering off Morse Potential (JLP Lattice Discretization)")
    
    ax.plot(sim.x, sim.V, 'k-', linewidth=2, label="Morse Potential V(x)")
    ax.axhline(sim.D, color='gray', linestyle='--', alpha=0.5, label="Dissociation Energy (D)")

    scale_factor = 15.0 
    line, = ax.plot([], [], 'b-', label=r"$|\psi(x)|^2$ (scaled)")
    fill = ax.fill_between(sim.x, 0, 0, color='blue', alpha=0.3)
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    ax.legend(loc='upper right')

    def init():
        line.set_data([], [])
        return line, time_text

    def animate(i):
        for _ in range(4):
            sim.step()
            
        prob_density = np.abs(sim.psi)**2 * scale_factor
        
        line.set_data(sim.x, prob_density)
        
        nonlocal fill
        fill.remove()
        fill = ax.fill_between(sim.x, 0, prob_density, color='blue', alpha=0.3)
        
        avg_x = np.sum(prob_density * sim.x) * sim.dx / scale_factor
        
        time_text.set_text(f"Time: {i*DT*4:.2f}\n<x>: {avg_x:.2f}")
        
        return line, fill, time_text

    ani = FuncAnimation(fig, animate, init_func=init, frames=1000, interval=1000, blit=False)
    plt.show()

if __name__ == "__main__":
    run_scattering_simulation()