using MPSKit, TensorKit, Plots, LinearAlgebra, LaTeXStrings

d = 10
D = 64

phi_sq = zeros(ComplexF64, (d, d))
pi_sq = zeros(ComplexF64, (d, d))
phi_4 = zeros(ComplexF64, (d, d))
phi = zeros(ComplexF64, (d, d))

for i in 2:d
    phi[i, i - 1] = sqrt(i / 2)
    phi[i - 1, i] = sqrt(i / 2)
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

    # connect n and n + 2
    if i + 2 <= d
        val = (4 * n + 6) * sqrt((n + 1) * (n + 2)) / 4
        j = i + 2
        phi_4[i, j] = val
        phi_4[j, i] = val   # Hermitian
    end

    # connect n and n + 4
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
ϕ = TensorMap(phi, ℂ^d ← ℂ^d)


μ0 = 0.3


infinite_chain = PeriodicVector([ℂ^d]) # local Hilbert space dimension is D

kappa_grid = 20
lam_grid = 5

plt = plot(;dpi=600)

for i in 0:lam_grid
	# intial \kappa = 0 run does not have any two-site terms
	entropy_vec = zeros(Float64, kappa_grid + 1) # + 1 since we include 0
	λ = 0.2 * i
	for j in 1:kappa_grid + 1
		κ = 0.05 * (j - 1)
		println("Currently working with λ = $λ and κ = $κ")
		ham_onsite = μ0^2 * ϕ2 / 2 + π2 / 2 + λ * ϕ4 / 24 + κ * ϕ2 # on site terms
		ham_bond = -κ * ϕ ⊗ ϕ # bond terms
		H_phi4_infinite = InfiniteMPOHamiltonian(infinite_chain, 1 => ham_onsite, (1, 2) => ham_bond)
		ψ0 = InfiniteMPS(ℂ^d, ℂ^D)
		ψ, _, _ = find_groundstate(ψ0, H_phi4_infinite, VUMPS(;tol=1e-7))
		entropy_vec[j] = entropy(ψ)[1]
	end
	plot!(0.05 * [0:kappa_grid], entropy_vec, label=L"\lambda = %$(round(λ, digits=1))")
	xlabel!(L"\kappa")
	ylabel!(L"S_{1/2}")
end

savefig("phi4EEgsinfinite.png")
