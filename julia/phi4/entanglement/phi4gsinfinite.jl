using MPSKit, TensorKit, Plots, LinearAlgebra, LaTeXStrings

D = 20 # maximum bond dimension

function matrix_elems(d)
    phi_sq = zeros(ComplexF64, (d, d))
    pi_sq = zeros(ComplexF64, (d, d))
    phi_4 = zeros(ComplexF64, (d, d))
    phi = zeros(ComplexF64, (d, d))
    for i in 2:d
        phi[i, i-1] = sqrt((i - 1) / 2)
        phi[i-1, i] = sqrt((i - 1) / 2)
    end
    for i in 1:d
        if i < d - 1
            phi_sq[i, i+2] = sqrt(i * (i + 1)) / 2
            phi_sq[i+2, i] = sqrt(i * (i + 1)) / 2
        end
        phi_sq[i, i] = (2 * i - 1) / 2
    end
    for i in 1:d
        if i < d - 1
            pi_sq[i, i+2] = -sqrt(i * (i + 1)) / 2
            pi_sq[i+2, i] = -sqrt(i * (i + 1)) / 2
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
    ϕ = TensorMap(phi, ℂ^d ← ℂ^d)
    return ϕ, ϕ2, π2, ϕ4
end

function EEplot!(d, plt, κ_start, λ_start, κ_grid, λ_grid, κ_step, λ_step, μ0)
    ϕ, ϕ2, π2, ϕ4 = matrix_elems(d)

    κ_max = κ_grid * κ_step
    infinite_chain = PeriodicVector([ℂ^d])

    # --- custom κ grid: finer between 0 and 0.3 ---
    κ_fine_max  = 0.3
    κ_fine_step = κ_step / 20  # have to be finer near the critical point

    κ_vals_fine   = collect(0.0:κ_fine_step:κ_fine_max)
    κ_vals_coarse = collect(κ_fine_max + κ_step : κ_step : κ_max)
    κ_vals        = vcat(κ_vals_fine, κ_vals_coarse)
    # ------------------------------------------------

    for i in λ_start:λ_grid
        λ = λ_step * i
        entropy_vec = zeros(Float64, length(κ_vals))

        for (idx, κ) in enumerate(κ_vals)
            println("Currently working with λ = $λ and κ = $κ")
            ham_onsite = μ0^2 * ϕ2 / 2 + π2 / 2 + λ * ϕ4 / 24 + κ * ϕ2
            ham_bond   = -κ * ϕ ⊗ ϕ
            H_phi4_infinite = InfiniteMPOHamiltonian(infinite_chain,
                                                     1 => ham_onsite,
                                                     (1, 2) => ham_bond)
            ψ0 = InfiniteMPS(ℂ^d, ℂ^D)
            ψ, _, _ = find_groundstate(ψ0, H_phi4_infinite, VUMPS(; tol=1e-7))
            entropy_vec[idx] = entropy(ψ)[1]
        end

        plot!(plt, κ_vals, entropy_vec,
              label = L"\lambda = %$(round(λ, digits=1))")
        xlabel!(L"\kappa")
        ylabel!(L"S_{1/2}")
    end
end


plt = plot(lw=2, dpi=600)

d = 16
κ_start = 0
λ_start = 0
κ_grid = 20
λ_grid = 4
κ_step = 0.05
λ_step = 0.2
μ0 = 0.3

EEplot!(d, plt, κ_start, λ_start, κ_grid, λ_grid, κ_step, λ_step, μ0)

savefig("phi4gsentanglement_d$(d)_D$(D)_mu$(μ0).png")