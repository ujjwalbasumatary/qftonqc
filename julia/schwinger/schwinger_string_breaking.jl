using MPSKit, TensorKit, Plots, LaTeXStrings, LinearAlgebra, Plots.PlotMeasures, JLD2, ArgParse
BLAS.set_num_threads(1)

function matrix_elems(d)
    phi = zeros(ComplexF64, (d, d))
    phi_sq = zeros(ComplexF64, (d, d))
    pi_sq = zeros(ComplexF64, (d, d))
    phi_4 = zeros(ComplexF64, (d, d))

    # phi = (a + a^\dagger)/sqrt(2)
    for i in 2:d
        val = sqrt((i - 1) / 2)
        phi[i, i-1] = val
        phi[i-1, i] = val
    end

    # helper to fill phi_sq and pi_sq (same structure, sign flip on off-diagonals)
    function fill_quadratic!(M, sign)
        @inbounds for i in 1:d
            if i < d - 1
                val = sign * sqrt(i * (i + 1)) / 2
                M[i, i+2] = val
                M[i+2, i] = val  # Hermitian
            end
            M[i, i] = (2 * i - 1) / 2
        end
    end

    fill_quadratic!(phi_sq, +1)
    fill_quadratic!(pi_sq, -1)

    # phi^4 in harmonic oscillator basis
    @inbounds for i in 1:d
        n = i - 1  # occupation number

        # diagonal
        phi_4[i, i] = (6 * n^2 + 6 * n + 3) / 4

        # connect |n> <-> |n+2>
        if i + 2 <= d
            val = (4 * n + 6) * sqrt((n + 1) * (n + 2)) / 4
            j = i + 2
            phi_4[i, j] = val
            phi_4[j, i] = val  # Hermitian
        end

        # connect |n> <-> |n+4>
        if i + 4 <= d
            val = sqrt((n + 1) * (n + 2) * (n + 3) * (n + 4)) / 4
            j = i + 4
            phi_4[i, j] = val
            phi_4[j, i] = val  # Hermitian
        end
    end
    return phi, phi_sq, pi_sq, phi_4
end

function get_elems(d_trunc; d::Int=2000, β::Float64=1.0,
    μ::Float64=0.5, κ::Float64=0.1, m::Float64=1.0, θ::Float64=0.0)

    phi, phi_sq, pi_sq, phi_4 = matrix_elems(d)
    iden_mat = Array(I, d, d)

    ham_mat = (m^2 * phi_sq + pi_sq) / 2 + μ^2 * (iden_mat - cos(β * phi - θ * iden_mat))

    evals, evecs = eigen(ham_mat)

    U = evecs[:, 1:d_trunc]
    ε = evals[1:d_trunc]

    norm(U' * ham_mat * U - Diagonal(ε))

    ϕ_mat = U' * phi * U
    ϕ_sq_mat = U' * phi_sq * U
    π_sq_mat = U' * pi_sq * U
    ϕ_4_mat = U' * phi_4 * U

    ϕ = TensorMap(ϕ_mat, ℂ^d_trunc ← ℂ^d_trunc)
    H0 = TensorMap(Diagonal(ε), ℂ^d_trunc ← ℂ^d_trunc)
    # don't actually need these.
    ϕ2 = TensorMap(ϕ_sq_mat, ℂ^d_trunc ← ℂ^d_trunc)
    π2 = TensorMap(π_sq_mat, ℂ^d_trunc ← ℂ^d_trunc)
    ϕ4 = TensorMap(ϕ_4_mat, ℂ^d_trunc ← ℂ^d_trunc)
    return ϕ, ϕ2, π2, ϕ4, H0
end

function build_hamiltonian(L::Int, d_trunc::Int;
    J::Float64=1.0, κ::Float64=0.1, d::Int=2000, β::Float64=1.0,
    μ::Float64=0.5, m::Float64=1.0, θ::Float64=0.0)

    ϕ, ϕ2, π2, ϕ4, H0 = get_elems(d_trunc; d=d, β=β, μ=μ, κ=κ, m=m, θ=θ)

    single_site_terms_ham = [i => H0 for i in 1:L]
    # these two terms come from the gradient
    single_site_terms_grad1 = [i => κ * ϕ2 / 2 for i in 1:L-1]
    single_site_terms_grad2 = [i => κ * ϕ2 / 2 for i in 2:L]
    # final term comes from the two site coupling in the gradient
    two_site_terms_grad = [(i, i + 1) => -κ * ϕ ⊗ ϕ for i in 1:L-1]
    # source terms
    source_terms = [i => J * ϕ for i in L÷2-2:L÷2+2]

    chain = fill(ℂ^d_trunc, L)
    ham = FiniteMPOHamiltonian(chain, single_site_terms_grad1...,
        single_site_terms_grad2..., single_site_terms_ham...,
        two_site_terms_grad..., source_terms...)
    return ham
end


function main()
    L = 100 # number of lattice sites

    J₀ = 1.0
    J₁ = 0.2

    d_trunc = 12
    D = 20
    d = 2000
    β = 1.0
    μ = 0.5
    κ = 0.1
    m = 1.0
    θ = 3.14159

    ϕ, ϕ2, π2, ϕ4, H0 = get_elems(d_trunc; d=d, β=β, μ=μ, κ=κ, m=m, θ=θ)
    ham = build_hamiltonian(L, d_trunc, J=J₀, d=d, β=β, μ=μ, κ=κ, m=m, θ=θ)

    ψ_gs = FiniteMPS(L, ℂ^d_trunc, ℂ^D)
    ψ_gs, _, _ = find_groundstate(ψ_gs, ham, DMRG())

    ham_quench = build_hamiltonian(L, d_trunc, J=J₁, d=d, β=β, μ=μ, κ=κ, m=m, θ=θ)

    T = 2000 # total time for evolution
    dt = 0.05 # time step of evolution

    flux = zeros(Float64, T, L)

    for i in 1:L
        flux[1, i] = real(expectation_value(ψ_gs, i => ϕ))
    end

    plt = heatmap(L÷2-10:L÷2+10, dt * [1:T], flux[:, L÷2-10:L÷2+10], dpi=600, ylabel=L"Lattice time $t$", xlabel=L"Lattice site $n$",
        right_margin=15mm)

    for step in 2:T
        println("Currently at step $step")
        ψ_gs, _ = timestep(ψ_gs, ham_quench, (step - 2) * dt, dt, TDVP())
        for i in 1:L
            flux[step, i] = real(expectation_value(ψ_gs, i => ϕ))
        end
        heatmap!(plt, L÷2-10:L÷2+10, dt * [1:T], flux[:, L÷2-10:L÷2+10], dpi=600, ylabel="Lattice time", xlabel="Lattice site",
            right_margin=15mm)
        display(plt) # display progress with the plot
    end

    filename = "quench_dynamics_dtrunc_d_$(d_trunc)_D_$(D)_T_$(T)_J0_$(J₀)_J1_$(J₁)_kappa_$(κ)_m_$(m)_theta_$(θ)_mu_$(μ)"
    mkpath("./plots")
    mkpath("./data")
    savefig("./plots/$(filename).png")

    # save data
    @save "./data/$(filename).jld2" flux

end

main()