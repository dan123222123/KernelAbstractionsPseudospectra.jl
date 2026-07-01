# Pseudospectra Explorer (Pluto)

An interactive, GPU-accelerated pseudospectra front-end — the EigTool-style "pick a matrix,
move a slider, see the contour" experience, backed by `KAPseudospectra.ihlpsa`. Because the
Julia process stays warm, the ~startup cost is paid **once** when the notebook first runs; every
slider change after that recomputes interactively.

## Files
- `pseudospectra_explorer.jl` — the Pluto notebook (matrix/size/grid/depth controls → `log₁₀ σ_min` contour).
- `serve.jl` — launches Pluto as a web app serving that notebook.

## Run locally
```julia
julia --project=pluto -e 'using Pkg; Pkg.develop(path="."); \
    Pkg.add(["Pluto","PlutoUI","Plots","MatrixDepot","KernelAbstractions","CUDA"]); Pkg.instantiate()'
julia --project=pluto pluto/serve.jl    # then open the printed http://127.0.0.1:1234/?secret=… URL
```
(Drop `CUDA` from the add list on a machine without an NVIDIA GPU — the notebook falls back to CPU.)

## Serve from a CUDA GPU server

The notebook detects `CUDA.functional()` and runs `ihlpsa` on the GPU automatically. The Pluto
session lives on the **server**, so the warmup happens there once and clients just see a fast UI.

1. **On the server**, do the one-time setup above (ideally build a sysimage — `sysimage/build_sysimage.jl` —
   and start with `--sysimage` so the server itself comes up in seconds).
2. **Start it** bound to localhost: `julia --project=pluto pluto/serve.jl`.
3. **Reach it safely** — a Pluto server is a remote Julia REPL, so don't expose it raw:
   - **SSH tunnel (simplest):** `ssh -L 1234:localhost:1234 user@gpuserver`, then browse `http://localhost:1234`.
   - **Reverse proxy:** set `PSA_HOST=0.0.0.0`, put nginx/Caddy in front with TLS + auth (Pluto needs
     WebSocket upgrade proxied). The per-session secret token is still required by default.

### Multiple users / multiple GPUs
- A single Pluto process holds **one** notebook state and **one** GPU; concurrent heavy computes
  serialize on it. For a few collaborators that's fine.
- For many viewers, use **PlutoSliderServer.jl** (serves the reactive notebook read-only-ish to many
  clients). For throughput, run **one Pluto/SliderServer process per GPU** (pin with
  `CUDA.device!`/`CUDA_VISIBLE_DEVICES`) behind a load balancer.

### Why this is the EigTool-like answer
Clients never install Julia, download a sysimage, or pay a TTFP — they hit a URL and interact.
All the startup/compile cost we discussed lives on the warm server, paid once at boot (and cut
further by the sysimage). It also sidesteps the GPU-kernel-cache wait (Julia #60747): everything
is server-side and already warm.
