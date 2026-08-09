# Pairwise numerical-reproducibility comparison across whatever cross_vendor_sigma_*.csv files
# are present in bench/results/ (each produced by cross_vendor_repro.jl on one vendor backend).
# Pure CPU — only reads CSVs. Tolerant of a partial set: reports whichever pairs it CAN
# compare and which vendors were actually present.
#
# Usage:  julia --project=bench bench/cross_vendor_compare.jl
include(joinpath(@__DIR__, "cross_vendor_common.jl"))
using DelimitedFiles, Printf

const RESULTS = joinpath(@__DIR__, "results")
mkpath(RESULTS)   # gitignored: absent on a fresh clone, and readdir would throw before the
                  # partial-set handling below gets a chance to report "no vendors"
tags = Dict(f => match(CROSS_VENDOR_SIGMA_RX, f) for f in readdir(RESULTS))
sigmas = Dict(m.captures[1] => readdlm(joinpath(RESULTS, f), ',', Float64)
              for (f, m) in tags if m !== nothing)
vendors = sort(collect(keys(sigmas)))
println("found ", length(vendors), " vendor(s): ", isempty(vendors) ? "(none)" : join(vendors, ", "))

outpath = joinpath(RESULTS, "cross_vendor_repro.csv")
open(outpath, "w") do io
    println(io, "vendor_a,vendor_b,n_points,maxdiff,max_reldiff")
    for i in eachindex(vendors), j in (i + 1):length(vendors)
        a, b = vendors[i], vendors[j]
        A, B = sigmas[a], sigmas[b]
        if size(A) != size(B)
            @warn "grid shape mismatch — skipping pair" a b size(A) size(B)
            continue
        end
        d = abs.(A .- B)
        rd = d ./ max.(abs.(A), abs.(B), eps())
        md, mr = maximum(d), maximum(rd)
        @printf("%-12s vs %-12s  maxdiff=%.3e  max_reldiff=%.3e  (n=%d)\n",
            a, b, md, mr, length(A))
        println(io, "$a,$b,$(length(A)),$md,$mr")
    end
end
println("wrote ", outpath)
length(vendors) < 2 &&
    @warn "fewer than 2 vendor sigma files present — nothing to compare yet" vendors
