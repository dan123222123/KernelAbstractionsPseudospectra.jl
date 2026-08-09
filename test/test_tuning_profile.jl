# The tuning-profile layer: which knob a solve actually resolves to, given a profile, a
# LocalPreferences entry, and an env override. This is tested because getting it wrong is silent —
# a run configured by the wrong layer produces plausible numbers, and the only symptom is a
# benchmark that disagrees with the last one for no visible reason.

using KAPseudospectra, Test
using KAPseudospectra: tune_profile, tune_profile_path, tuned_knob, tuning_keys, reload_tuning!,
                       write_tune_profile, _parse_wgs_schedule, _wgs_from_schedule

const _KEY = "trsm_tilecols_ComplexF64_gen"

# Restore whatever the caller had, so this file is order-independent within runtests.jl.
function with_profile(f, path)
    saved = get(ENV, "KAPSEUDO_TUNE_PROFILE", nothing)
    try
        path === nothing ? delete!(ENV, "KAPSEUDO_TUNE_PROFILE") :
        (ENV["KAPSEUDO_TUNE_PROFILE"] = path)
        reload_tuning!()
        f()
    finally
        saved === nothing ? delete!(ENV, "KAPSEUDO_TUNE_PROFILE") :
        (ENV["KAPSEUDO_TUNE_PROFILE"] = saved)
        reload_tuning!()
    end
end

@testset "tuning profile" begin
    mktempdir() do dir
        full = joinpath(dir, "full.toml")
        write(full, """
        [KAPseudospectra]
        $_KEY = "8"
        trsm_wgs_ComplexF64_gen = "128:32,1024:128"
        """)

        @testset "resolution order" begin
            with_profile(nothing) do
                @test tune_profile_path() === nothing
                @test tune_profile() === nothing
            end
            with_profile(full) do
                @test tuned_knob(_KEY) == "8"
                # An env scalar is an override and must still win: the probes force one candidate
                # per call that way, so a profile that outranked it would make tuning a no-op.
                withenv("KAPSEUDO_TRSM_TILECOLS" => "16") do
                    @test haskey(ENV, "KAPSEUDO_TRSM_TILECOLS")   # resolvers check ENV before tuned_knob
                end
                @test tuned_knob("trsm_warpgridpts_ComplexF64_gen") === nothing   # absent → heuristic
            end
        end

        @testset "section header is optional" begin
            bare = joinpath(dir, "bare.toml")
            write(bare, "$_KEY = 8\n")      # bare table, bare integer
            with_profile(bare) do
                @test tuned_knob(_KEY) == "8"
            end
        end

        @testset "a bad path degrades, never throws" begin
            with_profile(joinpath(dir, "nope.toml")) do
                @test (@test_logs (:warn,) match_mode = :any tune_profile()) === nothing
                @test tuned_knob(_KEY) === nothing
            end
        end

        @testset "switching profiles re-reads" begin
            other = joinpath(dir, "other.toml")
            write(other, "[KAPseudospectra]\n$_KEY = \"32\"\n")
            with_profile(full) do
                @test tuned_knob(_KEY) == "8"
            end
            with_profile(other) do
                @test tuned_knob(_KEY) == "32"
            end
        end

        @testset "write_tune_profile round-trips and flags partial" begin
            complete = Dict(k => "4" for k in tuning_keys())
            out = joinpath(dir, "written.toml")
            @test write_tune_profile(out, complete) == out
            with_profile(out) do
                @test length(tune_profile()) == length(tuning_keys())
                @test tuned_knob(_KEY) == "4"
            end
            # A tiled triple without the column schedule. It must warn and stamp itself PARTIAL,
            # because 12 of 16 keys leaves the whole column solve on `_auto_wgs`.
            partial = filter(kv -> !startswith(first(kv), "trsm_wgs_"), complete)
            pout = joinpath(dir, "partial.toml")
            @test_logs (:warn,) match_mode = :any write_tune_profile(pout, partial)
            @test occursin("# PARTIAL", read(pout, String))
        end
    end

    @testset "wgs schedules survive the round trip" begin
        sched = _parse_wgs_schedule("128:32,1024:128")
        @test _wgs_from_schedule(sched, 64) == 32     # below the first threshold: the floor
        @test _wgs_from_schedule(sched, 512) == 32
        @test _wgs_from_schedule(sched, 4096) == 128
        @test _parse_wgs_schedule("garbage") === nothing   # degrades to `_auto_wgs`
    end

    @testset "tuning_keys covers every knob × type × pencil kind" begin
        @test length(tuning_keys()) == 4 * 2 * 2
        @test allunique(tuning_keys())
        @test "trsm_wgs_ComplexF32_eye" in tuning_keys()
    end
end
