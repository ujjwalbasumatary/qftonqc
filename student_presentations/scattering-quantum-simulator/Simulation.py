import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, ifft, fftfreq

# Set natural units as specified in lecture16.pdf (ħ=1, 2m=1)
# This means E = k² for a free particle.
HBAR = 1.0
TWO_M = 1.0

class DoubleBarrierSimulator:
    """
    Simulates wave packet scattering from a double barrier potential using the
    JLP (position-basis) encoding and split-step Fourier method.
    
    Compares the numerical results with the analytical transfer matrix method
    from lecture16.pdf.
    
    Uses high-precision 64-bit floats and 128-bit complex numbers.
    """
    
    def __init__(self, Vb, w, b, N_grid=2048, L=300.0):
        """
        Initializes the simulator with potential and grid parameters.
        
        Args:
            Vb (float): Barrier height.
            w (float): Width of the well between barriers.
            b (float): Width of each barrier.
            N_grid (int): Number of points in the JLP position grid.
            L (float): Total length of the spatial grid.
        """
        self.Vb = np.float64(Vb)
        self.w = np.float64(w)
        self.b = np.float64(b)
        
        # JLP Grid Setup (from lecture15.pdf)
        self.N = N_grid
        self.L = np.float64(L)
        self.dx = self.L / self.N
        # Centered grid from -L/2 to L/2
        self.x = np.linspace(-self.L/2, self.L/2 - self.dx, self.N, dtype=np.float64)
        
        # Momentum grid (reciprocal space)
        # Ensure p is float64
        self.p = fftfreq(self.N, d=self.dx).astype(np.float64) * (2.0 * np.pi * HBAR)
        
        # Kinetic Energy Operator (in momentum space)
        # T = p²/2m
        self.T_p_op = (self.p**2) / TWO_M
        self.T_p_op = self.T_p_op.astype(np.float64) # T_p_op is real
        
        # Potential Energy Operator (in position space)
        self.V_x_op = self.create_potential() # This is float64
        
        print("Simulator initialized with high precision (float64, complex128).")
        print(f"Grid: {self.N} points, L={self.L}, dx={self.dx:.3f}")
        print(f"Potential: Vb={self.Vb}, w={self.w}, b={self.b}")

    def create_potential(self):
        """
        Creates the double-barrier potential V(x) on the position grid.
        Based on Fig. 1, lecture16.pdf.
        
        This version uses periodic boundary conditions (no absorbing walls).
        """
        # V is now explicitly a float64 array
        V = np.zeros(self.N, dtype=np.float64)
        
        # Barrier 1 (left)
        barrier1_cond = (self.x >= -self.w/2.0 - self.b) & (self.x < -self.w/2.0)
        V[barrier1_cond] = self.Vb
        
        # Barrier 2 (right)
        barrier2_cond = (self.x >= self.w/2.0) & (self.x < self.w/2.0 + self.b)
        V[barrier2_cond] = self.Vb
        
        return V

    # --- Part 1: Analytical Solution (Transfer Matrix Method) ---
    
    def get_analytical_s_matrix(self, E_range):
        """
        Calculates the analytical S-matrix (T and R) over a range of energies.
        This is the "ground truth" from lecture16.pdf, steps 0-5.
        
        Args:
            E_range (np.array): Array of energies (E) to calculate for.
            
        Returns:
            T (np.array): Transmission probability |t(E)|² (float64)
            R (np.array): Reflection probability |r(E)|² (float64)
        """
        T_analytic = []
        R_analytic = []
        
        # Ensure E_range is complex128 for the calculations
        E_complex = E_range.astype(np.complex128)
        
        for E in E_complex:
            if E.real <= 0 or E.real == self.Vb:
                E += 1e-9 # Avoid division by zero or log(0)
                
            # Step 1: Wavenumbers in each region
            # k will be complex if E is complex
            k = np.sqrt(TWO_M * E) / HBAR
            
            # q can be real or imaginary
            q_squared = TWO_M * (E - self.Vb) / (HBAR**2)
            q = np.sqrt(q_squared) # np.sqrt handles complex numbers correctly
            
            # Step 2 & 3: Interface and Propagation Matrices
            
            # S_k_to_q (Interface from V=0 to V=Vb) -> k_a=k, k_b=q
            S_k_to_q = 0.5 * np.array([
                [1.0 + k/q, 1.0 - k/q],
                [1.0 - k/q, 1.0 + k/q]
            ], dtype=np.complex128)
            
            # S_q_to_k (Interface from V=Vb to V=0) -> k_a=q, k_b=k
            S_q_to_k = 0.5 * np.array([
                [1.0 + q/k, 1.0 - q/k],
                [1.0 - q/k, 1.0 + q/k]
            ], dtype=np.complex128)
            
            # P_q (Propagation through barrier of width b)
            P_q_b = np.array([
                [np.exp(1j * q * self.b), 0.0 + 0.0j],
                [0.0 + 0.0j, np.exp(-1j * q * self.b)]
            ], dtype=np.complex128)
            
            # P_k (Propagation through well of width w)
            P_k_w = np.array([
                [np.exp(1j * k * self.w), 0.0 + 0.0j],
                [0.0 + 0.0j, np.exp(-1j * k * self.w)]
            ], dtype=np.complex128)
            
            # Step 4: Total Transfer Matrix M(E)
            # All matrices are complex128, so the result is complex128
            M_total = (S_q_to_k @ P_q_b @ S_k_to_q @ P_k_w @ 
                       S_q_to_k @ P_q_b @ S_k_to_q)
            
            # Step 5: Scattering Amplitudes
            M_21 = M_total[1, 0]
            M_22 = M_total[1, 1]
            
            t = (1.0 + 0.0j) / M_22
            r = -M_21 / M_22
            
            # Probabilities are real (float64)
            T_analytic.append(np.abs(t)**2)
            R_analytic.append(np.abs(r)**2)
            
        return np.array(T_analytic, dtype=np.float64), np.array(R_analytic, dtype=np.float64)

    # --- Part 2: JLP Simulation (Wave Packet Evolution) ---

    def create_initial_state(self, k0, x0, sigma):
        """
        Creates the initial Gaussian wave packet (free particle state).
        
        Args:
            k0 (float): Central momentum (p = ħk).
            x0 (float): Initial central position.
            sigma (float): Width of the wave packet in position space.
            
        Returns:
            psi_x (np.array): Normalized wave packet (complex128).
        """
        p0 = HBAR * np.float64(k0)
        x0_f = np.float64(x0)
        sigma_f = np.float64(sigma)
        
        exponent = -(self.x - x0_f)**2 / (4.0 * sigma_f**2)
        wave_vector = 1j * p0 * (self.x - x0_f) / HBAR
        
        # Cast intermediate steps to complex128
        psi = (np.exp(exponent.astype(np.complex128)) * np.exp(wave_vector.astype(np.complex128)))
        
        # Normalize
        norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx).astype(np.float64)
        return (psi / norm).astype(np.complex128)

    def run_simulation(self, k0, x0, sigma, T_total, dt, store_every=50):
        """
        Runs the full time-domain simulation using split-step Fourier method.
        This is the core of the JLP simulation.
        
        Args:
            k0 (float): Initial central momentum.
            x0 (float): Initial central position (should be far left).
            sigma (float): Width of the packet.
            T_total (float): Total simulation time.
            dt (float): Time step.
            store_every (int): How many steps between storing a frame for animation.
            
        Returns:
            psi_initial (np.array): The initial wave function (complex128).
            psi_final (np.array): The final wave function (complex128).
            psi_frames (list): A list of snapshots (complex128).
            time_array (list): The time corresponding to each frame (float64).
        """
        psi = self.create_initial_state(k0, x0, sigma)
        psi_initial = np.copy(psi).astype(np.complex128)
        
        n_steps = int(T_total / dt)
        dt_f = np.float64(dt)
        
        # Pre-calculate evolution operators (complex128)
        exp_V = np.exp(-1j * self.V_x_op * dt_f / (2.0 * HBAR)).astype(np.complex128)
        exp_T = np.exp(-1j * self.T_p_op * dt_f / HBAR).astype(np.complex128)
        
        psi_frames = []
        time_array = []
        
        print(f"Running simulation for {n_steps} steps...")
        
        for i in range(n_steps):
            # Split-Step Fourier Method
            # 1. Half step in V (position)
            psi = exp_V * psi
            
            # 2. Full step in T (momentum)
            psi_p = fft(psi).astype(np.complex128)
            psi_p = exp_T * psi_p
            psi = ifft(psi_p).astype(np.complex128)
            
            # 3. Half step in V (position)
            psi = exp_V * psi
            
            if i % store_every == 0:
                psi_frames.append(np.copy(psi).astype(np.complex128))
                time_array.append(np.float64(i * dt_f))
        
        psi_frames.append(np.copy(psi).astype(np.complex128))
        time_array.append(np.float64(T_total))
        print("Simulation complete.")
        
        return psi_initial, psi.astype(np.complex128), psi_frames, time_array

    def extract_s_matrix(self, psi_initial, psi_final, T_total, k0):
        """
        Extracts T(E) and R(E) from the simulation results.
        
        This version uses the user's logic:
        T(E) is calculated from +p components of the transmitted packet.
        R(E) is calculated from -p components of the reflected packet.
        
        Args:
            psi_initial (np.array): Initial wave function (complex128).
            psi_final (np.array): Final wave function (complex128).
            T_total (float): Total simulation time (for phase correction).
            k0 (float): The central momentum of the initial packet.
            
        Returns:
            E_sim (np.array): Energy range from simulation (float64).
            T_sim (np.array): Transmission probability |t(E)|² (float64)
            R_sim (np.array): Reflection probability |r(E)|² (float64)
        """
        print("Extracting S-Matrix (using correct negative-p reflection logic)...")
        
        # 1. Get momentum representation of initial packet
        psi_in_p = fft(psi_initial).astype(np.complex128)
        
        # 2. Separate final packet into reflected and transmitted parts
        barrier_left_edge = -self.w/2.0 - self.b
        barrier_right_edge = self.w/2.0 + self.b
        
        refl_mask = (self.x < barrier_left_edge)
        trans_mask = (self.x > barrier_right_edge)
        
        psi_refl = np.copy(psi_final)
        psi_refl[~refl_mask] = 0.0 + 0.0j
        
        psi_trans = np.copy(psi_final)
        psi_trans[~trans_mask] = 0.0 + 0.0j
        
        # 3. Go to momentum space
        psi_refl_p = fft(psi_refl).astype(np.complex128)
        psi_trans_p = fft(psi_trans).astype(np.complex128)
        
        # 4. Correct for free-particle time evolution phase
        phase_correction = np.exp(1j * self.T_p_op * np.float64(T_total) / HBAR).astype(np.complex128)
        
        psi_refl_p_corr = psi_refl_p * phase_correction
        psi_trans_p_corr = psi_trans_p * phase_correction
        
        # 5. Get positive momentum indices and amplitudes
        # We only care about positive momentum (p > 0)
        positive_p_mask = (self.p > 0.0)
        positive_p_indices = np.where(positive_p_mask)[0]
        
        # Get initial amplitude at positive p
        psi_in_at_positive_p = psi_in_p[positive_p_indices]
        
        # 6. Calculate T(E) (Transmission)
        psi_trans_at_positive_p = psi_trans_p_corr[positive_p_indices]
        epsilon = 1e-12 + 0.0j # Use complex epsilon
        
        t_p = psi_trans_at_positive_p / (psi_in_at_positive_p + epsilon)
        T_sim = (np.abs(t_p)**2).astype(np.float64)
        
        # 7. Calculate R(E) (Reflection)
        # Find corresponding negative momentum indices
        # This uses the fftshift convention: if p[k] is +p, p[N-k] is -p
        # We must be careful about index 0
        corresponding_negative_indices = self.N - positive_p_indices
        
        psi_refl_at_negative_p = psi_refl_p_corr[corresponding_negative_indices]
        
        r_p = psi_refl_at_negative_p / (psi_in_at_positive_p + epsilon)
        R_sim = (np.abs(r_p)**2).astype(np.float64)

        # 8. Get Energy E = p²/2m (float64)
        E_sim = ((self.p[positive_p_indices])**2) / TWO_M
        
        # 9. Filter out noise where the initial packet had near-zero amplitude
        p0 = np.float64(k0 * HBAR)
        sigma_p = HBAR / (2.0 * self.initial_sigma) # Width in momentum space (float64)
        
        # Filter to 3-sigma range around the central momentum
        p_filter = (self.p[positive_p_indices] > p0 - 3.0*sigma_p) & \
                   (self.p[positive_p_indices] < p0 + 3.0*sigma_p)
                   
        return E_sim[p_filter], T_sim[p_filter], R_sim[p_filter]


    # --- Part 3: Visualization ---

    def plot_s_matrix_comparison(self, E_sim, T_sim, R_sim):
        """
        Plots the simulated T(E) and R(E) against the analytical solution.
        """
        # Get analytical solution over the same energy range
        E_min, E_max = E_sim.min(), E_sim.max()
        E_analytic = np.linspace(E_min, E_max, 400, dtype=np.float64)
        T_analytic, R_analytic = self.get_analytical_s_matrix(E_analytic)
        
        plt.figure(figsize=(12, 10))
        
        # Plot Transmission T(E)
        plt.subplot(2, 1, 1)
        plt.plot(E_analytic, T_analytic, 'r-', label=f'Analytical T(E) (Vb={self.Vb})', linewidth=2)
        plt.plot(E_sim, T_sim, 'bo', label='Simulated T(E) (JLP)', markersize=4, alpha=0.7)
        plt.title('Transmission Probability T(E)', fontsize=16)
        plt.xlabel('Energy (E)', fontsize=12)
        plt.ylabel('T(E)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Plot Reflection R(E)
        plt.subplot(2, 1, 2)
        plt.plot(E_analytic, R_analytic, 'r-', label=f'Analytical R(E) (Vb={self.Vb})', linewidth=2)
        plt.plot(E_sim, R_sim, 'go', label='Simulated R(E) (JLP)', markersize=4, alpha=0.7)
        plt.title('Reflection Probability R(E)', fontsize=16)
        plt.xlabel('Energy (E)', fontsize=12)
        plt.ylabel('R(E)', fontsize=12)
        plt.legend()
        
        # Plot R+T
        plt.subplot(2, 1, 2) # Adding to the second plot for context
        plt.plot(E_analytic, R_analytic + T_analytic, 'k--', label='Analytical R+T', linewidth=2, alpha=0.7)
        plt.legend()


        plt.tight_layout()
        plt.savefig("s_matrix_comparison.png")
        print("\nSaved S-Matrix comparison plot to 's_matrix_comparison.png'")
        # plt.show() # Commented out to not block execution

    def create_animation(self, psi_frames, time_array):
        """
        Creates and saves an animation of the scattering process.
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.set_xlim(-self.L/2.0, self.L/2.0)
        ax.set_ylim(0.0, 0.15) # Adjust ylim based on packet height
        ax.set_xlabel("Position (x)", fontsize=12)
        ax.set_ylabel(r"$|\psi(x, t)|^2$", fontsize=12)
        ax.set_title("JLP Wave Packet Scattering (Periodic Boundaries)", fontsize=16)
        
        # Plot the real part of the potential V(x)
        V_real = np.real(self.V_x_op)
        # Scale for visualization
        V_max = V_real.max()
        if V_max == 0.0: 
            V_max = 1.0
        V_plot = (V_real / V_max * 0.1).astype(np.float64)
        
        ax.plot(self.x, V_plot, 'r-', label=f'Potential V(x) (scaled)', linewidth=2)
        ax.fill_between(self.x, 0.0, V_plot, color='red', alpha=0.3)
        
        # Initialize the line for the wave packet
        line, = ax.plot([], [], 'b-', lw=2, label=r'$|\psi(x, t)|^2$')
        time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes, fontsize=12)
        
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)

        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def animate(i):
            psi = psi_frames[i]
            prob_density = (np.abs(psi)**2).astype(np.float64)
            line.set_data(self.x, prob_density)
            time_text.set_text(f'Time = {time_array[i]:.2f}')
            return line, time_text

        anim = FuncAnimation(fig, animate, init_func=init,
                             frames=len(psi_frames), interval=30, blit=True)
        
        try:
            # Save as GIF using the 'pillow' writer
            anim.save('scattering_animation.gif', writer='pillow', fps=30)
            print("\nSaved scattering animation to 'scattering_animation.gif'")
        except Exception as e:
            print(f"\nCould not save animation as GIF (pillow writer might not be installed or other error): {e}")
            print("Displaying animation instead (this may not work in all environments).")
            # plt.show() # Fallback for interactive environments
        
        plt.close(fig) # Close the figure to free up memory


if __name__ == "__main__":
    # --- Simulation Parameters (using 64-bit floats) ---
    
    # 1. Potential (from lecture16.pdf)
    Vb = 10.0
    w = 2.0
    b = 0.5
    
    # 2. Grid (JLP)
    N = 2048     # Number of grid points (power of 2 for FFT)
    L = 300.0    # Large box to prevent wrap-around
    
    # 3. Initial State (Wave Packet)
    # Target the resonance near E = 4.93
    E_target = 15
    k0_target = np.sqrt(TWO_M * E_target) / HBAR # k = sqrt(E)
    
    x0 = -100.0   # Start position (far left in the larger box)
    sigma = 8.0   # Packet width
    
    # 4. Evolution
    dt = 0.005     # Time step
    T_total = 20 # Time to allow separation without wrap-around
    
    # --- Run ---
    
    simulator = DoubleBarrierSimulator(Vb=Vb, w=w, b=b, N_grid=N, L=L)
    simulator.initial_sigma = np.float64(sigma) # Store for S-matrix extraction
    
    # Run the simulation
    psi_init, psi_final, frames, times = simulator.run_simulation(
        k0=k0_target, 
        x0=x0, 
        sigma=sigma, 
        T_total=T_total, 
        dt=dt,
        store_every=10 # Store frames more often
    )
    
    # --- S-Matrix extraction ---
    
    # Extract S-Matrix
    E_sim, T_sim, R_sim = simulator.extract_s_matrix(
        psi_init, 
        psi_final, 
        T_total, 
        k0_target
    )
    
    # Plot Comparison
    simulator.plot_s_matrix_comparison(E_sim, T_sim, R_sim)
    
    # Create Animation
    simulator.create_animation(frames, times)

    print("\nFull high-precision simulation finished.")