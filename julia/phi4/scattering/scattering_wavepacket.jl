using MPSKit, TensorKit, Plots, LaTeXStrings, SpecialFunctions, Plots

D = 10 # max bond dimension
d = 5 # local hilbert space dimension

function matrix_elems(d)
    phi_sq = zeros(ComplexF64, (d, d))
    pi_sq = zeros(ComplexF64, (d, d))
    phi_4 = zeros(ComplexF64, (d, d))
    phi = zeros(ComplexF64, (d, d))
    for i in 2:d
        phi[i, i - 1] = sqrt((i - 1) / 2)
        phi[i - 1, i] = sqrt((i - 1) / 2)
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
    ϕ = TensorMap(phi, ℂ^d ← ℂ^d)
    return ϕ, ϕ2, π2, ϕ4
end

function get_ham(d, μ0, λ0)
    """
    Prepares the infinte MPO Hamiltonian from μ0, λ0 and
    the local Hilbert space dimension d as inputs
    """
    ϕ, ϕ2, π2, ϕ4 = matrix_elems(d)
    chain = PeriodicVector([ℂ^d])
    single_site_term = (μ0 * ϕ2 + π2) / 2 + λ0 * ϕ4 / 24 + ϕ2
    two_site_term = - ϕ ⊗ ϕ
    ham = InfiniteMPOHamiltonian(chain, 1 => single_site_term, (1, 2) => two_site_term)
    return ham
end

function prep_gs(d, ham)
    """
    Returns the converged ground state of the uMPS
    """
    ψ0 = InfiniteMPS(ℂ^d, ℂ^D)
    ψ, _, _ = find_groundstate(ψ0, ham, VUMPS(;tol=1e-12))
    return ψ
end

λ0 = 0.2
μ0_sq = -0.078

ham = get_ham(d, μ0_sq, λ0)

ψ_gs = prep_gs(d, ham)

ϕ, ϕ2, π2, ϕ4 = matrix_elems(d)

S_gs = entropy(ψ_gs) # ground state entropy

ξ = correlation_length(ψ_gs) # ground state correlation length

# this returns a struct LeftGaugedQP with fields
# left_gs, right_gs, VLs, Xs
# the B tensors can be obtained by VLs, Xs

energies, states = excitations(ham, QuasiparticleAnsatz(), [-π/2, π/2], ψ_gs)

function get_B_tensor(state)
    """
    Returns the B tensor with VLs and Xs.
    """
    VLs = state.VLs[1]
    Xs = state.Xs[1]
    B_tensor = VLs * Xs
    B_array = convert(Array, B_tensor)
    B_array_new = B_array[:, :, 1, :]
    # B_tensor_new = TensorMap(B_array_new, ℂ^24 ⊗ ℂ^16 ← ℂ^24)
    return B_array_new # this returns a list for reasons I don't understand
end

B_1 = get_B_tensor(states[1]) # this is the tensor with momentum - π/2
B_2 = get_B_tensor(states[2]); # this is the tensor with momentum + π/2 

ψ_gs.AL

function block_upper(AL, AR, B)
    D, d, D2 = size(AL)
    @assert size(B) == (D, d, D2)
    @assert D == D2  "A must be D×d×D"

    T = zeros(eltype(AL), 2D, d, 2D)

    # top-left block = A
    T[1:D, :, 1:D] .= AL

    # top-right block = B
    T[1:D, :, D+1:2D] .= B

    # bottom-left block = 0  (already zero)

    # bottom-right block = AR
    T[D+1:2D, :, D+1:2D] .= AR

    return T
end


D = 10

function create_stacked_tensor(ψ_gs, B_1, B_2, L, offset, sigma)
    """
    Takes the ground state left and right environments and constructs
    the stacked tensors.
    """
    AL = convert(Array, ψ_gs.AL[1])
    AR = convert(Array, ψ_gs.AR[1])
    window = eltype(ψ_gs.AL)[]
    mid = L ÷ 2
    """
    B_1 is left moving and B_2 is right moving.
    So B_2 spans 1:mid and B_1 spans mid + 1 : L
    """
    mat = cat(AL, exp(im * π * (1 - (mid - offset)) / 2-(1 - (mid - offset))^2 / sigma^2) * B_2; dims=3)
    tensor = TensorMap(mat, ℂ^(D) ⊗ ℂ^d ← ℂ^(2D))
    push!(window, tensor)
    for i in 2:mid
        mat = block_upper(AL, AR, exp(im * π * (i - (mid - offset)) / 2 -(i - (mid - offset))^2 / sigma^2) * B_2)
        tensor = TensorMap(mat, ℂ^(2D) ⊗ ℂ^d ← ℂ^(2D))
        push!(window, tensor)
    end
    for i in (mid+1):(L - 1)
        mat = block_upper(AL, AR, exp(- im * π * (i - (mid + offset)) / 2-(i - (mid + offset))^2 / sigma^2) * B_1)
        tensor = TensorMap(mat, ℂ^(2D) ⊗ ℂ^d ← ℂ^(2D))
        push!(window, tensor)
    end
    mat = cat(exp(-im * π * (L - (mid + offset)) / 2-(L - (mid + offset))^2 / sigma^2) * B_1, AR; dims=1)
    tensor = TensorMap(mat, ℂ^(2D) ⊗ ℂ^d ← ℂ^(D))
    push!(window, tensor)
    return window
end
    
    

wavepacket_window = create_stacked_tensor(ψ_gs, B_1, B_2, 100, 20, 10);

ψ_window = WindowMPS(ψ_gs, wavepacket_window);

normalize!(ψ_window)

for i in 1:100
    exp_vals[i] = real(expectation_value(ψ_window, i => ϕ2))
end

L = 100

T = 1000 # grid points for time evolution
field_exp = zeros(Float64, T, L);

ψ_window

for i in 1:L
    field_exp[1, i] = real(expectation_value(ψ_window, i => ϕ2))
end

dt = 0.1

plot(field_exp[1, :])

for t_step in 2:T
    global ψ_window
    println("\rCurrently at step $t_step")
    ψ_window, _ = timestep(ψ_window, ham, t_step - 1, dt, TDVP())
    for i in 1:L
        field_exp[t_step, i] = real(expectation_value(ψ_window, i => ϕ2))
    end
end

heatmap(field_exp, c=:viridis, clims=(0.85, 0.9))

savefig("./plot.png")
