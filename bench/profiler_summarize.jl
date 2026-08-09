# Post-process a vendor-profiler CSV capture of `bench_kernels.jl --counters` into a
# per-kernel-family saturation summary — hardware-counter evidence for the
# "latency-bound, not bandwidth/compute-bound" claim. Pure CPU (only parses a CSV). ncu and
# nvprof share the parser/buckets/writer; only the metric tables and value encodings differ,
# so the FORMAT is auto-detected from the header:
#
#   ncu    (any device ≥ Volta): one row per (kernel launch, metric), columns Kernel Name /
#     Metric Name / Metric Value. `Metric Name` carries either friendly display names or raw
#     dotted identifiers depending on ncu version — matched by keyword against both.
#     dram__bytes.sum is SUM-aggregated; the % metrics are mean-aggregated.
#   nvprof (pre-Volta): one row per (kernel, metric), value in the Avg column, encoded per
#     metric as a 0-10 LEVEL, a percent string, or a 0-1 fraction — all normalized to 0-100
#     so the pre-Volta panel lines up column-for-column with the ≥ Volta one.
#
# Leading banner lines are not CSV — the header is located by content. Kernel names carry
# commas inside quotes (C++ template args), so this is a real quoted-field CSV parser.
# Missing/unrecognized metrics report NaN with a warning rather than aborting the pipeline.
#
# Usage (arg2/arg3 = eltype/m labels → per-(eltype,m) filename + leading columns):
#   julia --project=bench bench/profiler_summarize.jl <raw.csv> [eltype] [m]

# Kernel-family buckets, parallel to bench_drivers' trace summary. The vendor GEMM call
# backing tiled-gemm's trailing update has no matching name → lands in `other`.
const KERNEL_GROUPS = [
    ("column", r"column"i),
    ("tiled_panel", r"panel"i),
    ("tiled_trailing", r"trailing"i),
]
group_of(name) = something(findfirst(((_, rx),) -> occursin(rx, name), KERNEL_GROUPS), 0) |>
    i -> i == 0 ? "other" : first(KERNEL_GROUPS[i])

# One CSV row → Vector{String} fields, honoring double-quoted fields ("" = escaped quote).
function parse_csv_row(line)
    fields = String[]
    buf = IOBuffer()
    inquotes = false
    i, n = 1, lastindex(line)
    while i <= n
        c = line[i]
        if inquotes
            if c == '"'
                if i < n && line[nextind(line, i)] == '"'
                    print(buf, '"')
                    i = nextind(line, i)
                else
                    inquotes = false
                end
            else
                print(buf, c)
            end
        elseif c == '"'
            inquotes = true
        elseif c == ','
            push!(fields, String(take!(buf)))
        else
            print(buf, c)
        end
        i = nextind(line, i)
    end
    push!(fields, String(take!(buf)))
    fields
end

norm(s) = lowercase(strip(s, ['"', ' ']))

# Tolerant numeric parse (strips thousands-separator commas); `nothing` on failure so the
# caller can skip a banner/header row rather than average in a bogus 0.
function try_num(s)
    t = strip(s)
    isempty(t) && return nothing
    m = match(r"^-?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?$", replace(t, "," => ""))
    m === nothing ? nothing : parse(Float64, m.match)
end

# nvprof Avg-cell decoders, by metric kind (see header).
function decode_nvprof(kind, raw)
    s = strip(raw, ['"', ' '])
    isempty(s) && return nothing
    if kind == :level
        m = match(r"\(\s*([0-9]+(?:\.[0-9]+)?)\s*\)", s)          # "Mid (5)" → 50
        return m === nothing ? nothing : parse(Float64, m.captures[1]) * 10
    elseif kind == :pct
        m = match(r"([0-9]+(?:\.[0-9]+)?)", s)                    # "85.42%" → 85.42
        return m === nothing ? nothing : parse(Float64, m.captures[1])
    else # :frac — 0-1 fraction → percent (tolerate a stray already-%)
        v = tryparse(Float64, replace(s, r"[^0-9eE.+-]" => ""))
        return v === nothing ? nothing : (v <= 1.5 ? v * 100 : v)
    end
end

# Metric tables: short output name, matcher against the Metric Name cell, aggregation, and
# the value decoder. ncu's raw-name alternations are disjoint across metrics.
const NCU_METRICS = [
    ("achieved_occupancy_pct", r"achieved occupancy|sm__warps_active"i, :mean, try_num),
    ("dram_throughput_pct", r"memory throughput|dram throughput|dram__throughput"i, :mean, try_num),
    ("l2_hit_rate_pct", r"l2.*hit|lts__t_sector_hit"i, :mean, try_num),
    ("sm_throughput_pct", r"compute.*throughput|sm throughput|sm__throughput"i, :mean, try_num),
    ("dram_bytes_sum", r"dram__bytes"i, :sum, try_num),
]
const NVPROF_METRICS = [
    ("sm_efficiency_pct", r"^sm_efficiency$", :mean, s -> decode_nvprof(:pct, s)),
    ("achieved_occupancy_pct", r"^achieved_occupancy$", :mean, s -> decode_nvprof(:frac, s)),
    ("sp_fu_util_pct", r"^single_precision_fu_utilization$", :mean, s -> decode_nvprof(:level, s)),
    ("dram_util_pct", r"^dram_utilization$", :mean, s -> decode_nvprof(:level, s)),
]

# Locate the real header by content and identify the format from its column set:
# ncu has Kernel Name / Metric Name / Metric Value; nvprof has Kernel / Metric Name / Avg.
function find_header(lines)
    for line in lines
        hn = norm.(parse_csv_row(line))
        kcol = findfirst(h -> occursin("kernel", h), hn)
        mncol = findfirst(h -> occursin("metric", h) && occursin("name", h), hn)
        (kcol === nothing || mncol === nothing) && continue
        mvcol = findfirst(h -> occursin("metric", h) && occursin("value", h), hn)
        mvcol === nothing || return (; format = :ncu, kcol, mncol, vcol = mvcol)
        avcol = findfirst(==("avg"), hn)
        avcol === nothing || return (; format = :nvprof, kcol, mncol, vcol = avcol)
    end
    error("couldn't find an ncu (Metric Value) or nvprof (Avg) CSV header — is this a " *
          "--csv --metrics capture?")
end

function summarize(path, elt, mlab)
    lines = collect(eachline(path))
    isempty(lines) && error("profiler CSV is empty: $path")
    h = find_header(lines)
    metrics = h.format == :ncu ? NCU_METRICS : NVPROF_METRICS
    needed = max(h.kcol, h.mncol, h.vcol)
    println("format: ", h.format)

    groups = Dict{String, Any}()   # group => (n Ref, per-short sums, per-short counts)
    nskipped = 0
    for line in lines
        isempty(strip(line)) && continue
        f = parse_csv_row(line)
        length(f) < needed && (nskipped += 1; continue)
        mname = strip(f[h.mncol], ['"', ' '])
        mi = findfirst(((_, rx, _, _),) -> occursin(rx, mname), metrics)
        mi === nothing && (nskipped += 1; continue)   # banner/header/untracked metric
        short, _, _, decode = metrics[mi]
        v = decode(f[h.vcol])
        v === nothing && (nskipped += 1; continue)
        g = get!(groups, group_of(f[h.kcol])) do
            (; n = Ref(0), sums = Dict(s => 0.0 for (s, _, _, _) in metrics),
                counts = Dict(s => 0 for (s, _, _, _) in metrics))
        end
        g.sums[short] += v
        g.counts[short] += 1
    end
    # n_launches per group = the max metric count seen (metrics can drop rows independently).
    for g in values(groups)
        g.n[] = maximum(values(g.counts); init = 0)
    end
    for (short, rx, _, _) in metrics
        any(g.counts[short] > 0 for g in values(groups)) ||
            @warn "metric not found in profiler output — NaN for every group" short rx
    end

    tag = String(h.format)
    outpath = joinpath(dirname(path),
        isempty(elt) ? "$(tag)_summary.csv" :
        isempty(mlab) ? "$(tag)_summary_$(elt).csv" : "$(tag)_summary_$(elt)_m$(mlab).csv")
    hpfx = isempty(elt) ? "" : isempty(mlab) ? "eltype," : "eltype,m,"
    rpfx = isempty(elt) ? "" : isempty(mlab) ? "$elt," : "$elt,$mlab,"
    open(outpath, "w") do io
        println(io, hpfx, "kernel_group,n_launches," * join(first.(metrics), ","))
        for gname in sort(collect(keys(groups)))
            g = groups[gname]
            cells = [g.counts[s] == 0 ? "NaN" :
                     string(agg == :sum ? g.sums[s] : g.sums[s] / g.counts[s])
                     for (s, _, agg, _) in metrics]
            println(io, rpfx, gname, ",", g.n[], ",", join(cells, ","))
        end
    end
    println("wrote ", outpath, " (", length(groups), " kernel group(s), ", nskipped,
        " non-data row(s) skipped)")
    isempty(groups) &&
        @warn "profiler_summarize: no kernel rows matched in $path — no usable counters " *
              "(blocked perf-counter access?) or an unfamiliar layout. Wrote empty $outpath."
end

path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "results", "ncu_raw.csv")
summarize(path, length(ARGS) >= 2 ? ARGS[2] : "", length(ARGS) >= 3 ? ARGS[3] : "")
