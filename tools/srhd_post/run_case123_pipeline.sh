#!/usr/bin/env bash
set -euo pipefail

# One-click pipeline:
# 1) generate exact/reference solutions with Exact_Riemann_Solver
# 2) plot chapter-ready 1x3 figures for case1/case2/case3

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

python tools/srhd_post/generate_exact_solutions.py \
  --solver-dir third_party/Exact_Riemann_Solver \
  --out-dir postprocess/exact_riemann \
  --cases case1 case2 case3 \
  --nx-ref 4000 \
  --accuracy 1e-10 \
  --verbose 0

python tools/srhd_post/plot_case1_case2_case3.py \
  --repo-root . \
  --exact-dir postprocess/exact_riemann \
  --fig-dir postprocess/figures \
  --cases case1 case2 case3 \
  --min-plot-points 140 \
  --max-plot-points 280 \
  --num-marker . \
  --num-marker-size 4.0 \
  --num-color blue \
  --dpi 600

echo "[done] Outputs in postprocess/figures/"
