using MPSKit, TensorKit, Plots

L = 200 # the number of lattice sites
D = 10

function apply_one_site_op!(mps::FiniteMPS, operator::TensorMap, site::Int)
    AC = mps.AC[site]
    @tensor AC_new[a, b; c] := operator[b, d] * AC[a, d; c]
    mps.AC[site] = AC_new
end

ψ0 = FiniteMPS(L, ℂ^2, ℂ^D)

σ_x = TensorMap(ComplexF64[0 1; 1 0], ℂ^2 ← ℂ^2)
σ_y = TensorMap(ComplexF64[0 -im; im 0], ℂ^2 ← ℂ^2)
σ_z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2 ← ℂ^2)

J = 1.0
h = 1.1

two_site_couplings = [(i, i+1) => - J * σ_z ⊗ σ_z for i in 1:L - 1]
single_site_terms = [i => - h * σ_x for i in 1:L]
chain = fill(ℂ^2, L)
model = FiniteMPOHamiltonian(chain, two_site_couplings...)

ψ, env, eps = find_groundstate(ψ0, model, DMRG())

σ_p = σ_x + im * σ_y
σ_m = σ_x - im * σ_y

# now we will apply the S_p and S_m to create excitations
mid = L ÷ 2
offset = 30
@tensor S_p_left[a, b; c] := σ_p[b, d] * ψ.AC[mid - offset - 5][a, d; c]
@tensor S_m_left[a, b; c] := σ_m[b, d] * ψ.AC[mid - offset + 5][a, d; c]
@tensor S_p_right[a, b; c] := σ_p[b, d] * ψ.AC[mid + offset + 5][a, d; c]
@tensor S_m_right[a, b; c] := σ_m[b, d] * ψ.AC[mid + offset - 5][a, d; c]

ψ.AC[mid - offset - 5] = S_p_left
ψ.AC[mid - offset + 5] = S_m_left
ψ.AC[mid + offset + 5] = S_p_right
ψ.AC[mid + offset - 5] = S_m_right

T = 200
mag = Array{Float64, 2}(undef, L, T)

for i in 1:L
    mag[i, 1] = real(expectation_value(ψ, i => σ_z))
end

dt = 0.05

for i in 2:T
    global ψ
    print("\rCurrently at step $i")
    ψ, env = timestep(ψ, model, (i - 1) * dt, dt, TDVP())
    for j in 1:L
        mag[j, i] = real(expectation_value(ψ, j => S_z))
    end
end

heatmap(mag)