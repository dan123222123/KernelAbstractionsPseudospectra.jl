#!/usr/bin/env julia
#
# Build a sysimage that preloads the heavy GPU stack (CUDA, GPUCompiler, LLVM, …) and
# KAPseudospectra, so `using KAPseudospectra` + the first `ihlpsa` start in seconds instead
# of ~100 s. This targets the ~80 s of *package load + compiler-infra warmup* — the larger,
# version-independent part of the cost. The GPU *kernel* JIT still happens on the first solve
# until Julia ships the foreign-CodeInstance precompile fix (JuliaLang/julia#60747); pair this
# with `enable_gpu_kernel_cache!()` on a Julia that has that fix to also persist the kernels.
#
# Usage — run with the project that has KAPseudospectra + your GPU backend active:
#     julia --project=. sysimage/build_sysimage.jl [output.so]
#
# Then launch Julia with the image (it is NOT applied automatically):
#     julia --sysimage /abs/output.so
#     alias jl='julia --sysimage /abs/output.so'                       # ~/.bashrc
#     # VS Code settings.json: "julia.additionalArgs": ["--sysimage", "/abs/output.so"]
#
# Dev note: to keep KAPseudospectra's source editable, remove :KAPseudospectra from `bake`
# below — its heavy deps still load fast, while your package edits stay live.

import Pkg
const USER_PROJ = dirname(Base.active_project())
const OUT = abspath(get(ARGS, 1, joinpath(USER_PROJ, "kapseudo_sys.so")))

deps = collect(keys(get(Pkg.TOML.parsefile(joinpath(USER_PROJ, "Project.toml")), "deps", Dict())))
bake = Symbol[]
for p in ("CUDA", "AMDGPU", "oneAPI", "Metal", "KAPseudospectra")
    p in deps && push!(bake, Symbol(p))
end
isempty(bake) &&
    error("active project has none of CUDA/AMDGPU/oneAPI/Metal/KAPseudospectra " *
          "as direct deps — activate the project that uses KAPseudospectra")

# PackageCompiler lives in a throwaway env so the user's Project.toml is left untouched.
let t = mktempdir()
    Pkg.activate(t)
    Pkg.add("PackageCompiler")
end
using PackageCompiler

@info "Building sysimage (~10–20 min, ~1 GB output)" project=USER_PROJ output=OUT bake
create_sysimage(bake;
    sysimage_path = OUT,
    project = USER_PROJ,
    precompile_execution_file = joinpath(@__DIR__, "precompile_workload.jl")
)

println("\n✔ sysimage built: ", OUT, "  (", round(filesize(OUT) / 2^30, digits = 2), " GB)")
println("\nLaunch with one of:")
println("  julia --sysimage $OUT")
println("  alias jl='julia --sysimage $OUT'        # add to ~/.bashrc")
println("  VS Code settings.json: \"julia.additionalArgs\": [\"--sysimage\", \"$OUT\"]")
