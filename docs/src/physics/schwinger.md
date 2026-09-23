# Bosonized Schwinger model

[`source_quench.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/models/schwinger/scripts/source_quench.jl)
prepares the ground state of a scalar chain in the presence of a local source.
At ``t=0`` the source strength changes, and the state evolves under the new
Hamiltonian. The field expectation value records the response across the
chain.

The program uses an open chain with Hamiltonian

```math
\begin{aligned}
H(J)={}&\sum_{n=1}^{L}\left[
 \frac{\pi_n^2}{2}+\frac{m^2\phi_n^2}{2}
 +\mu^2\bigl(1-\cos(\beta\phi_n-\theta)\bigr)\right]\\
&+\frac{\kappa}{2}\sum_{n=1}^{L-1}(\phi_n-\phi_{n+1})^2
 +J\sum_{n\in S}\phi_n,
\end{aligned}
```

where ``\phi_n`` and ``\pi_n`` are conjugate field variables, with
``[\phi_n,\pi_m]=i\delta_{nm}`` before the onsite Hilbert space is truncated.
The source occupies
`S = max(1,L÷2-2):min(L,L÷2+2)`, which contains five central sites for a
sufficiently long chain. The gradient term couples adjacent sites; the
quadratic and cosine terms determine the local potential.

## Parameters in the reference paper

[Belyansky et al.](https://arxiv.org/abs/2307.02522) write their bosonic lattice
Hamiltonian as

```math
H_{\rm paper}=\chi\sum_n\left[
 \frac{\pi_n^2}{2}+\frac{(\phi_n-\phi_{n-1})^2}{2}
 +\frac{\mu_{\rm paper}^2\phi_n^2}{2}
 -\lambda\cos(\beta\phi_n-\theta)\right].
```

The names of the couplings differ between that expression and the program:

| Program argument | Coefficient in the program | Paper convention at ``\chi=1`` |
| --- | --- | --- |
| `--m` | ``m^2\phi^2/2`` | ``m^2=\mu_{\rm paper}^2`` |
| `--mu` | ``-\mu^2\cos(\beta\phi-\theta)`` | ``\mu^2=\lambda`` |
| `--kappa` | ``\kappa(\phi_n-\phi_{n+1})^2/2`` | ``\kappa=1`` |
| `--beta` | cosine period | ``\beta=\sqrt{4\pi}`` for the Schwinger model |
| `--theta` | angle in the cosine | the same ``\theta`` |

The added constant ``L\mu^2`` changes the energy zero. The program has no
overall ``\chi`` factor. Its default ``\beta=1`` and ``\kappa=0.1`` therefore
describe a different parameter choice from the Schwinger normalization in the
table. The source strengths `--J0` and `--J1` are additional parameters of the
quench implemented here.

## The onsite basis

A scalar field has infinitely many onsite states. The calculation starts
with the first ``d`` harmonic-oscillator states,

```math
\phi=\frac{a+a^\dagger}{\sqrt2},\qquad
\pi=\frac{a-a^\dagger}{i\sqrt2},\qquad
|n\rangle,\quad n=0,\ldots,d-1.
```

`--d` sets this oscillator cutoff. The matrices of ``\phi^2`` and ``\pi^2``
are filled from their oscillator matrix elements. The cosine is evaluated as
a matrix function of the truncated ``\phi`` matrix. The onsite Hamiltonian
is then diagonalized, and the lowest `--d_trunc` eigenvectors form the basis
used on every site of the MPS.

If ``W`` contains these eigenvectors, the retained field matrices are
``W^\dagger\phi W`` and ``W^\dagger\phi^2W``. Projection and multiplication
do not commute: ``W^\dagger\phi^2W`` generally differs from
``(W^\dagger\phi W)^2``. The gradient term uses the projected quadratic
operator as well as the projected field.

The saved `onsite_projection_residual` measures how nearly
``W^\dagger h_{\rm onsite}W`` is diagonal with the retained eigenvalues. To
assess the truncation, increase ``d`` and ``d_{\rm trunc}`` separately and
compare the retained energies and field evolution.

## Preparation and time evolution

DMRG, the density-matrix renormalization group, minimizes the finite-chain
energy for ``H(J_0)``. `--D` sets the MPS bond dimension. The same onsite
basis is used for both source strengths, so the quench changes only the
linear source term. For ``t>0`` the desired evolution is

```math
|\psi(t)\rangle=e^{-iH(J_1)t}|\Omega(J_0)\rangle.
```

The program approximates this evolution using TDVP. Setting ``J_1=0`` removes
the source after preparation; setting ``J_1=J_0`` leaves the preparation
Hamiltonian unchanged and allows the residual time dependence of a computed
ground state to be examined.

Here `--total_time` is the final physical lattice time. `--time_step` gives
the maximum interval between samples, and the last interval is shortened
when necessary to reach the requested final time exactly.

## Field and electric flux

The saved array contains the field itself,

```math
\mathtt{field}[r,n]=\langle\psi(t_r)|\phi_n|\psi(t_r)\rangle.
```

It has one row per saved time and one column per site. `flux` is another key
for this same array. In the shifted-field convention of the paper,
``E_T/e=\beta\phi/(2\pi)``; at ``\beta=\sqrt{4\pi}`` this becomes
``E_T/e=\phi/\sqrt\pi``. The code saves ``\phi`` without this rescaling.

The PNG displays at most 21 sites around the source. The JLD2 file contains
the entire chain, the saved times, input parameters, retained onsite
energies, and the onsite and DMRG residuals. You can collect these outputs
under `results/schwinger/` or choose another directory with `--output_dir`.

The evolution describes the response to a source change. A quark or meson
collision would require prepared particle wave packets and their subsequent
evolution. The relation between outgoing states and scattering probabilities
is discussed in [Particle production](particle-production.md).
Commands and array-loading examples are in
[Running the calculations](../running.md) and [Data and figures](../data.md).
