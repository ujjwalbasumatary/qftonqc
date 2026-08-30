# Completion plan for the IFT and Schwinger projects

This is the working definition of "complete." A heatmap that looks like a
collision is a diagnostic, not a scattering result. Completion requires a
controlled incoming state, converged time evolution, an asymptotic outgoing
particle analysis, and reproducible numerical evidence.

## 1. Shared foundation

Finish these gates before a production campaign:

- Keep `Pkg.test()` green, including the exact two-particle packet-number test.
- Add a run manifest to every JLD2 output: model parameters, algorithm and
  tolerances, package versions, commit, elapsed time, and array-axis metadata.
- Save vacuum, initial window, periodic evolution checkpoints, and the latest
  complete observable row atomically; implement resume from a checkpoint.
- Report norm drift, energy drift after packet preparation, discarded weight,
  maximum bond dimension, boundary contamination, and solver residuals.
- Put parameter scans in scripts/configuration files and make plotting consume
  saved data rather than rerun dynamics.
- Add CI-sized free-theory integration tests and one deterministic interacting
  smoke test for each model.

Suggested repository acceptance thresholds are norm drift below `1e-6`, total
outgoing probability closure within `1e-3`, and less than `2%` change in quoted
observables under the final time-step, bond, window, packet-width, and
projection-grid refinements. If a target is too strict for a particular
observable, document and justify its replacement before the run.

## 2. Ising field theory

The lattice Hamiltonian used by the [IFT paper](https://arxiv.org/pdf/2411.13645)
is

```text
H = -sum_j [sigma^z_j sigma^z_(j+1) + g_x sigma^x_j + g_z sigma^z_j],
eta_latt = (g_x - 1) / |g_z|^(8/15).
```

The repository target `g_x=1.06`, `g_z=0.01` has `eta_latt` about `0.700`.
The exact `g_z=0` lattice dispersion is

```text
epsilon(k) = 2 sqrt(1 + g_x^2 - 2 g_x cos(k)).
```

### Staged IFT ladder

1. **Vacuum and spectrum.** Converge VUMPS energy, correlation length, Schmidt
   spectrum, and tangent-space dispersions over `D=12, 18, 24, 32`. The paper's
   working vacuum was around `D=24`, with the smallest retained Schmidt value
   around `1e-8`. At the integrable magnetic point, recover the first E8 mass
   ratios `m2/m1 = 1.618034` and `m3/m1 = 1.989044` to the resolution permitted
   by the lattice parameters.
2. **Free benchmark.** At `g_z=0`, compare the computed dispersion pointwise to
   the exact expression, scatter two packets, and recover unit elastic
   probability and zero time delay within numerical error.
3. **Packet validation.** Use phase-consistent `B_j(k)`, commensurate momentum
   grids, and two independently closed packets joined with the vacuum bond
   gauge. Verify one excitation in each separated support before evolution.
   Check normalization, mean momentum, position/momentum widths, total energy,
   group velocity, and exponential suppression at support boundaries.
4. **Dynamics convergence.** Progress through a cheap ladder before production:
   `N=200,Dmax=16,dt=0.1`; `N=500,Dmax=32,dt=0.05`; then
   `N=1000-2000,Dmax` up to `64`, `dt<=0.05`. Production packet supports are
   typically `400-800` sites with position widths around `70-120`. Use a
   two-site/growth stage before one-site TDVP when a closed packet bond begins
   below the desired evolution bond.
5. **Outgoing analysis.** Implement vacuum-subtracted symmetric local energy,
   asymptotic two-particle projectors with a separation gap around `100` sites,
   elastic/inelastic channel probabilities, and probability closure. Extract
   phase shifts/time delays from nearby momenta (the paper uses
   `delta-k=0.001`), not solely from heatmap peak locations.
6. **Physics deliverables.** Reproduce selected elastic and inelastic results,
   quote uncertainties from the convergence ladder, and extract resonance mass
   and width only after the projector and phase analysis pass the free test.

### IFT acceptance evidence

- Tables for vacuum and dispersion convergence, including solver residuals.
- Initial-state packet diagnostics proving two incoming particles.
- Norm/energy/truncation histories and boundary-arrival estimates for every
  quoted production run.
- Channel probabilities whose sum closes within the declared tolerance.
- A free-fermion benchmark and at least one interacting result stable under all
  final refinements.
- A script that regenerates each final table/figure from named JLD2 datasets.

## 3. Bosonized Schwinger model

The [Schwinger paper](https://arxiv.org/abs/2307.02522) studies

```text
H = chi sum_x [pi_x^2/2 + (phi_x-phi_(x-1))^2/2
               + mu^2 phi_x^2/2 - lambda cos(beta phi_x-theta)],
beta = sqrt(4*pi),  chi = 1.
```

Its main interacting parameters are `mu^2=0.1`, `lambda=0.5`, with
`theta=pi` for the deconfined case and `theta=pi-0.04` or `pi-0.07` for the
confined cases. The electric-field observable is `E_T/e = phi/sqrt(pi)`.

The current `schwinger_string_breaking.jl` instead prepares a finite-DMRG
ground state with a linear source on five sites and quenches its strength. It
is useful for testing onsite truncation and real-time evolution, but it cannot
by itself meet the paper acceptance criteria.

### Staged Schwinger ladder

1. **Hamiltonian and onsite basis.** Expose the paper variables directly and
   remove the ambiguous `m`/`mu` mapping. Diagonalize the onsite problem in a
   large oscillator basis (paper scale near `2000`) and retain `d=12`. Scan the
   large cutoff and retained dimension; report low-energy eigenvalue and
   operator-matrix convergence. Cache the projected operators once per
   parameter set.
2. **Free benchmark.** For `lambda=0`, verify
   `omega(p)=chi*sqrt(mu^2+4*sin(p/2)^2)` and convergence of local observables.
   Check that vacuum subtraction gives zero far from a prepared excitation.
3. **Infinite vacua and particles.** Replace finite DMRG with VUMPS uMPS
   vacua. Construct topological quarks between distinct vacua and mesons over
   the same vacuum using momentum-dependent tangent tensors and a consistent
   phase/gauge convention.
4. **Packets.** Use a commensurate momentum grid (paper scale `Np=400`),
   Gaussian momentum width about `0.12` for quarks and `0.06` for mesons, and
   the correct close/glue/reopen construction with the vacuum `C^-1` bond.
   Validate energy, momentum, velocity, localization, charge/flux profile, and
   one particle per incoming support.
5. **Dynamics.** Reproduce the paper's starting scales before convergence:
   quarks `D=20` grown toward `D'=50`; mesons `D=40` toward `D'=100`. Implement
   adaptive window growth so no signal reaches an artificial boundary. Scan
   `dt`, TDVP tolerances, bond dimension, onsite truncation, and initial
   separation.
6. **Outgoing analysis.** Implement particle projection with an exclusion gap
   around `r=40`; test momentum projection grids near `Mp=400` and `1300`.
   Measure channel probabilities, electric flux/string formation, and energy
   balance. Reproduce a small deconfined benchmark before the confined cases.
7. **Physics deliverables.** Reproduce selected results corresponding to the
   paper's principal scattering and string-breaking figures, with uncertainty
   bands or convergence tables and machine-readable underlying data.

### Schwinger acceptance evidence

- Onsite-cutoff and retained-basis convergence tables at every quoted coupling.
- Agreement with the analytic free dispersion and a zero-background flux test.
- Converged uMPS vacua and particle dispersion/residual data.
- Packet charge/flux and particle-number diagnostics before evolution.
- Norm/energy/truncation histories, dynamic-window history, and proof of no
  boundary contamination.
- Outgoing probability closure and stability under `Mp`, exclusion-gap, bond,
  time-step, and onsite-basis refinements.
- One-command regeneration of the selected paper-comparison figures.

## 4. Definition of done

A model track is complete when a clean checkout can instantiate the manifest,
pass tests, run a documented small benchmark, resume a production run, and
regenerate selected paper-comparison figures from versioned configuration and
raw data. Every reported number must carry convergence evidence and a clear
mapping from paper notation to code parameters.
