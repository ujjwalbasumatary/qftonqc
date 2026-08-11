using MPSKit, TensorKit, Plots, LinearAlgebra, LaTeXStrings

d = 10 # local Hilbert space dimension
D = 20 # maximum bond dimension

phi_sq = zeros(ComplexF64, (d, d))
pi_sq = zeros(ComplexF64, (d, d))
phi_4 = zeros(ComplexF64, (d, d))
phi = zeros(ComplexF64, (d, d))

for i in 2:d
  phi[i, i - 1] = sqrt((i - 1)/ 2)
  phi[i - 1, i] = sqrt((i - 1)/ 2)
end

for i in 1:d
    if i < d - 1
        phi_sq[i, i + 2] = sqrt(i * (i + 1)) / 2
        phi_sq[i + 2, i] = sqrt(i * (i + 1)) / 2
    end
    phi_sq[i, i] = (2 * i - 1) / 2
end
for i in 1:d
    if i < d - 1
        pi_sq[i, i + 2] = - sqrt(i * (i + 1)) / 2
        pi_sq[i + 2, i] = - sqrt(i * (i + 1)) / 2
    end
    pi_sq[i, i] = (2 * i - 1) / 2
end

for i in 1:d
    n = i - 1  # occupation number

    # diagonal
    phi_4[i, i] = (6 * n^2 + 6 * n + 3) / 4

    # connect |n> <-> |n+2>
    if i + 2 <= d
        val = (4 * n + 6) * sqrt((n + 1) * (n + 2)) / 4
        j = i + 2
        phi_4[i, j] = val
        phi_4[j, i] = val   # Hermitian
    end

    # connect |n> <-> |n+4>
    if i + 4 <= d
        val = sqrt((n + 1) * (n + 2) * (n + 3) * (n + 4)) / 4
        j = i + 4
        phi_4[i, j] = val
        phi_4[j, i] = val   # Hermitian
    end
end

ϕ2 = TensorMap(phi_sq, ℂ^d ← ℂ^d)
π2 = TensorMap(pi_sq, ℂ^d ← ℂ^d)
ϕ4 = TensorMap(phi_4, ℂ^d ← ℂ^d)
id = TensorMap(diagm(ComplexF64[1 for _ in 1:d]), ℂ^d ← ℂ^d)
ϕ = TensorMap(phi, ℂ^d ← ℂ^d)

μ0 = 0.3

L = 12

chain = fill(ℂ^d, L) # this is the lattice

L = 12 # number of lattice sites
max_bond_dimension = ℂ^D
physical_space = ℂ^d
ψ0 = FiniteMPS(rand, ComplexF64, L, physical_space, max_bond_dimension)

kappa_grid = 10
lam_grid = 5

kappa_max = kappa_grid * 0.1
lam_max = lam_grid * 0.2

plt = plot()

round(1.23, digits = 1)

for i in 0:lam_grid - 1
    entropy_vec = zeros(Float64, kappa_grid)
    λ = 0.2 * i
    for j in 1:kappa_grid
        κ = 0.1 * (j - 1)
        println("Currently working with λ = $λ and κ = $κ")
        phi_sq_terms = [i => (μ0^2 / 2 + κ) * ϕ2 for i in 1:L]
        phi_4_terms = [i => (λ / 24) * ϕ4 for i in 1:L]
        pi_sq_terms = [i => 0.5 * π2 for i in 1:L]
        single_site_terms = [phi_sq_terms; phi_4_terms; pi_sq_terms]
        two_site_terms = [(i, i + 1) => -κ * ϕ ⊗ ϕ for i in 1:L - 1]
        H_phi_4_finite = FiniteMPOHamiltonian(chain, single_site_terms..., two_site_terms...)
        ψ0 = FiniteMPS(L, ℂ^d, ℂ^D)
        ψ, _, _ = find_groundstate(ψ0, H_phi_4_finite, DMRG())
        entropy_vec[j] = entropy(ψ, 6)
    end
    plot!(0.1 * [0:kappa_grid - 1], entropy_vec, label=L"\lambda = %$(round(λ, digits=1))")
    xlabel!(L"\kappa")
    ylabel!(L"S_{1/2}")
end

savefig("gshalfchainEE.png", dpi=500)
