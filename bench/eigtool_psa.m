function eigtool_psa(m, npts, ax, outprefix)
% Headless EigTool pseudospectra run for bench_cpu_gpu.jl: times the σ-field computation on
% gallery('grcar', m) and dumps the grid + σ values for the Julia-side agreement check.
%
% Uses EigTool's documented command-line mode ([x,y,sigs] = eigtool(A,opts) with
% opts.no_graphics) — the GUI figure is created invisible and closed. Notes for reading the
% timing against ours:
%   * gallery('grcar', m) is the SAME integer Toeplitz matrix as MatrixDepot.grcar (k = 3
%     superdiagonals), so no matrix needs to cross the MATLAB/Julia boundary.
%   * the timed call INCLUDES EigTool's internal Schur reduction (the Julia side reports its
%     own factor_s separately so totals compare like-for-like).
%   * EigTool iterates its inverse Lanczos to a 1e-5 relative tolerance (psacore's tol
%     argument), not to a fixed depth — read the σ-agreement column against ~5 digits.
%   * grcar is real and the box spans the real axis, so EigTool's mirror-symmetry
%     optimization computes roughly half the rows and reflects them. The returned y grid is
%     NOT a plain linspace in that case; the Julia side compares σ on the RETURNED x/y
%     rather than assuming our qgrid.
%
% Writes <outprefix>_meta.csv (m, npts, nx, ny, t_compute_s), _x.csv, _y.csv, _sigs.csv
% (sigs is ny-by-nx, rows follow y).
  warm.npts = 10; warm.no_graphics = 1; warm.no_waitbar = 1; warm.scale_equal = 0;
  [~, ~, ~] = eigtool(gallery('grcar', 16), warm);   % JIT/figure warmup, excluded from timing

  A = gallery('grcar', m);
  opts.npts = npts;
  opts.ax = ax(:)';                    % [xmin xmax ymin ymax]
  opts.no_graphics = 1;
  opts.no_waitbar = 1;
  opts.scale_equal = 0;                % npts in BOTH directions (not longer-axis-only)
  opts.direct = 1;                     % dense path (Schur + inverse Lanczos), like ours
  tic;
  [x, y, sigs] = eigtool(A, opts);
  t = toc;
  writematrix([m, npts, numel(x), numel(y), t], [outprefix '_meta.csv']);
  writematrix(reshape(x, 1, []), [outprefix '_x.csv']);
  writematrix(reshape(y, 1, []), [outprefix '_y.csv']);
  writematrix(sigs, [outprefix '_sigs.csv']);
end
