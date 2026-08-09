# Derive the native-FLOP expansion of each MultiFloat type by counting the float instructions
# emitted for one scalar `*` and one `+`, from optimized LLVM IR.
# Run it and paste the table into `MF_FLOP_COST` in ../bench_common.jl.
using MultiFloats, InteractiveUtils, Printf

# (mul_flops, add_flops) emitted for `op(x::T, y::T)`, summing SIMD lanes.
function native_cost(op, T)
    ir = sprint((io, f, ts) -> code_llvm(io, f, ts; debuginfo = :none, optimize = true),
        op, (T, T))
    mul = add = 0
    for ln in split(ir, '\n')
        # SIMD width: `<4 x double>` → 4, scalar → 1 (first vector operand on the line).
        wm = match(r"<(\d+) x (?:double|float)>", ln)
        w = wm === nothing ? 1 : parse(Int, wm.captures[1])
        if occursin(r"\bfmul\b", ln)
            mul += w
        elseif occursin(r"\bfadd\b", ln) || occursin(r"\bfsub\b", ln)
            add += w
        elseif occursin(r"@llvm\.(?:fma|fmuladd)\.", ln)
            mul += w
            add += w
        end
    end
    (mul, add)
end

const TYPES = (; f32x2 = Float32x2, f32x4 = Float32x4, f64x2 = Float64x2, f64x4 = Float64x4)

println("MultiFloats native-FLOP cost per real-limb operation")
println("(mul, add) = native f32/f64 FLOPs emitted; E = (μ_mul + μ_add)/2 = per-logical-flop expansion\n")
# μ_mul = TOTAL native flops in one `T*T` (its fmuls + fadds + fmas); μ_add likewise for `T+T`.
@printf("%-7s  %-14s %-14s  %-8s\n", "type", "mul: (m,a)→μ", "add: (m,a)→μ", "E")
rows = String[]
for tok in keys(TYPES)
    T = TYPES[tok]
    mm, ma = native_cost(*, T)
    am, aa = native_cost(+, T)
    μmul, μadd = mm + ma, am + aa
    E = (μmul + μadd) / 2
    @printf("%-7s  (%2d,%3d)→%-4d  (%2d,%3d)→%-4d  %-8.2f\n", tok, mm, ma, μmul, am, aa, μadd, E)
    push!(rows, "    :$tok => ($μmul, $μadd),")
end
println("\n# paste into bench_common.jl:")
println("const MF_FLOP_COST = Dict(   # (native μ_mul, μ_add) per real-limb op; see bench/multifloat_flop_costs.jl")
foreach(println, rows)
println(")")
