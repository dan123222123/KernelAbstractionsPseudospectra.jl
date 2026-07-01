# Serve the Pseudospectra Explorer as a warm, GPU-backed web app.
#
# One-time setup — make a small project for the app (package + Pluto + plotting + CUDA).
# From the repo root:
#   julia --project=pluto -e 'using Pkg; Pkg.develop(path="."); \
#       Pkg.add(["Pluto","PlutoUI","Plots","MatrixDepot","KernelAbstractions","CUDA"]); Pkg.instantiate()'
#
# Serve (on the CUDA box):
#   julia --project=pluto pluto/serve.jl
#   # for a much faster server start, layer a sysimage:
#   julia --project=pluto --sysimage kapseudo_sys.so pluto/serve.jl
#
# Security: a Pluto server is a live remote Julia REPL. Keep it behind an SSH tunnel
#   (ssh -L 1234:localhost:1234 user@gpuserver) or an authenticating reverse proxy — never
#   expose it raw. A per-session access secret is required by default. Set PSA_HOST=0.0.0.0
#   only when a proxy in front handles auth/TLS.

import Pluto
Pluto.run(;
    host = get(ENV, "PSA_HOST", "127.0.0.1"),
    port = parse(Int, get(ENV, "PSA_PORT", "1234")),
    notebook = joinpath(@__DIR__, "pseudospectra_explorer.jl"),
    launch_browser = false,
)
