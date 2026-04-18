# SRHD 1D Postprocess Pipeline (Case 1/2/3)

This folder provides a reproducible workflow to:

1. Generate reference solutions with `Exact_Riemann_Solver` (RMHD) under SRHD-degenerate settings:
   - `Bx=By=Bz=0`
   - `vy=vz=0`
   - `u <-> vx`
2. Plot chapter-ready comparison figures:
   - **black solid line** = reference solution
   - **dense blue scatter markers** = numerical SRHD-GSPH result (uniformly sampled for visualization)

## Scripts

- `generate_exact_solutions.py`
  - Writes `RInput.txt` for each case
  - Builds and runs `riemann_rmhd`
  - Saves `case{1,2,3}_exact.npz` in `postprocess/exact_riemann`

- `plot_case1_case2_case3.py`
  - Reads your numerical `.npz` outputs:
    - `sod_shocktube_rel_output/results_rel.npz`
    - `riemann_problem2_rel_output/results_rel.npz`
    - `riemann_problem4_rel_output/results_rel.npz`
  - Loads reference `.npz`
  - Produces:
    - `fig5-1-case1-1d.pdf/png`
    - `fig5-2-case2-1d.pdf/png`
    - `fig5-3-case3-1d.pdf/png`

- `run_case123_pipeline.sh`
  - one-click full pipeline

## Important note on sampling

Sampling is used **only for visualization** (to avoid cluttered markers and match paper style).
It does **not** modify raw numerical data and is **not** used in error computations.

### Recommended plotting style for GSPH

- Keep reference as black continuous line.
- Plot numerical solution as dense scatter (`--num-marker .` by default), not connected lines.
- Use `--plot-stride` or `--max-plot-points` to control density reproducibly.
- For cleaner reflection-test panels, use `--case3-point-scale` (default `0.70`).
- To avoid title overlap, adjust `--fig-height`, `--subplot-top`, `--suptitle-y`.
