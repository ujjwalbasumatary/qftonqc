@testset "Ising spectrum boundary sector and mass ratios" begin
    args = Dict{String, Any}(
        "gx" => 1.06,
        "gz" => 0.006,
        "bond-dimension" => 2,
        "k-max" => 0.5,
        "points" => 2,
        "target-cm-ratio" => 6.0,
        "vumps-tolerance" => 1e-9,
    )

    # These fields must be rejected before constructing an MPS or solving
    # for a vacuum. Signed zero represents the same zero longitudinal field.
    for gx in (0.0, 0.5, prevfloat(1.0)), gz in (0.0, -0.0)
        args["gx"], args["gz"] = gx, gz
        failure = try
            SpectrumProgram.main(args)
            nothing
        catch err
            err
        end
        @test failure isa ArgumentError
        @test occursin("kink", sprint(showerror, failure))
    end

    for gz in (0.0, -0.0)
        args["gx"], args["gz"] = 1.0, gz
        failure = try
            SpectrumProgram.main(args)
            nothing
        catch err
            err
        end
        @test failure isa ArgumentError
        @test occursin("gap vanishes", sprint(showerror, failure))
    end
end
