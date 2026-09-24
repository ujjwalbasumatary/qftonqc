# Bosonized Schwinger model

[`source_quench.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/models/schwinger/scripts/source_quench.jl)
can be used to prepare the ground state of a scalar chain in the presence of a local source.
At ``t=0`` the source strength changes, resulting in a quench, and the state evolves under the new
Hamiltonian. The field expectation value records the response across the
chain.

Bosonization expresses the fermion and electric-field dynamics of the
Schwinger model in terms of a scalar field. The scalar field is proportional
to the total electric field, with the conventions given below. The
derivation in the [reference](https://arxiv.org/html/2307.02522#S1.SS1)
relates the scalar Hamiltonian to the fermionic theory.

For the purpose of the calculations, we use the bosonized version of the Schwinger model
with the Hamiltonian
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
sufficiently long chain.

The momentum term ``\pi_n^2/2`` supplies the kinetic energy of the field.
The quadratic potential favours small field values, while the cosine adds
a periodic modulation whose position depends on ``\theta``. For positive
``\kappa``, the gradient term penalizes differences between neighbouring
field values and allows a disturbance to propagate along the chain. The
source tilts the local potential on the sites in ``S``. With the sign used
here, positive ``J`` lowers the energy of negative field values there.

## Parameters in the reference paper

[Belyansky et al.](https://arxiv.org/abs/2307.02522) write their bosonic lattice
Hamiltonian as

```math
H_{\rm paper}=\chi\sum_n\left[
 \frac{\pi_n^2}{2}+\frac{(\phi_n-\phi_{n-1})^2}{2}
 +\frac{\mu_{\rm paper}^2\phi_n^2}{2}
 -\lambda\cos(\beta\phi_n-\theta)\right].
```

The following table shows the relationship of the coupling names in the program to those in the paper
at ``\chi=1``.

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

If the columns of ``W`` are these eigenvectors, the field and quadratic
operators in the retained basis are ``W^\dagger\phi W`` and
``W^\dagger\phi^2W``. The projected quadratic operator
``W^\dagger\phi^2W`` generally differs from ``(W^\dagger\phi W)^2``,
because projection and multiplication do not commute. The gradient term
therefore uses both the projected field and the separately projected
quadratic operator.

The saved `onsite_projection_residual` measures how nearly
``W^\dagger h_{\rm onsite}W`` is diagonal with the retained eigenvalues. To
assess the truncation, increase ``d`` and ``d_{\rm trunc}`` separately and
compare the retained energies and field evolution. Increasing ``d`` enlarges
the oscillator space in which the onsite Hamiltonian is diagonalized, while
increasing ``d_{\rm trunc}`` retains more of its eigenstates on each MPS site.

## Preparation and time evolution

DMRG, the density-matrix renormalization group, minimizes the finite-chain
energy for ``H(J_0)``. `--D` sets the MPS bond dimension. The same onsite
basis is used for both source strengths, so the quench changes only the
linear source term. For ``t>0`` the desired evolution is

```math
|\psi(t)\rangle=e^{-iH(J_1)t}|\Omega(J_0)\rangle.
```

The ground state prepared with ``J_0`` has a field profile adapted to that
source. After the change to ``J_1``, it is generally a superposition of
eigenstates of the new Hamiltonian. Those components acquire different
phases, so the field profile changes in time and disturbances can travel
away from the source region through the coupling between neighbouring sites.

The program approximates this evolution using TDVP. Setting ``J_1=0`` removes
the source after preparation. With ``J_1=J_0``, the preparation Hamiltonian
also governs the evolution, so any residual time dependence of the computed
ground state can be examined.

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
Here ``e`` is the gauge coupling, and ``E_T`` includes the uniform background
electric field associated with ``\theta``. Spatial variations of the
electric field are related to charge density by Gauss's law. The saved
field profile therefore describes how the electric flux changes across the
chain in this convention; the relations are given in
[Eqs. S2 and S8 of the supplement](https://arxiv.org/html/2307.02522#S1.SS1).

The PNG displays at most 21 sites around the source. The JLD2 file contains
the entire chain, the saved times, input parameters, retained onsite
energies, and the onsite and DMRG residuals. You can collect these outputs
under `results/schwinger/` or choose another directory with `--output_dir`.

## Quarks and mesons

In the deconfined regime studied in the paper, a quark is a kink joining
two degenerate vacua with different field expectation values. Its MPS has
different vacuum tensors far to the left and right of the excitation.
A meson is a neutral quark–antiquark bound state; its MPS approaches the same
vacuum on both sides. This distinction determines the boundary tensors in
the excitation calculation, as described in
[Sec. III.1 of the supplement](https://arxiv.org/html/2307.02522#S3.SS1).

The evolution describes the response to a source change. Simulating a quark
or meson collision requires preparing particle wave packets and evolving
them toward each other. The relation between outgoing states and scattering
probabilities is discussed in [Particle production](particle-production.md).
Commands and array-loading examples are in
[Running the calculations](../running.md) and [Data and figures](../data.md).
