#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

ts="$(date +%Y%m%d_%H%M%S)"
log_dir="${repo_root}/postprocess/logs"
mkdir -p "${log_dir}"
log_file="${log_dir}/case4_2d_final_${ts}.log"

echo "[info] repo_root: ${repo_root}" | tee "${log_file}"
echo "[info] start time: $(date -Iseconds)" | tee -a "${log_file}"
echo "[info] running final 2D SRHD-GSPH case ..." | tee -a "${log_file}"

python pysph/examples/gas_dynamics/riemann2d_problem2_rel.py \
  --nx 320 \
  --hdx 1.12 \
  --cfl-rel 0.18 \
  --tf 0.4 \
  --pfreq 100 \
  --nplot 900 \
  --nlevels 30 \
  --logrho-level-min -1.38417 \
  --logrho-level-max 0.0 \
  --plot-smooth-passes 1 \
  --export-fig-dir postprocess/figures \
  2>&1 | tee -a "${log_file}"

echo "[info] finished at: $(date -Iseconds)" | tee -a "${log_file}"
echo "[info] expected final publication figure:" | tee -a "${log_file}"
echo "       ${repo_root}/postprocess/figures/fig5-4-case4-2d.pdf" | tee -a "${log_file}"
echo "       ${repo_root}/postprocess/figures/fig5-4-case4-2d.png" | tee -a "${log_file}"
echo "[info] run log saved at: ${log_file}" | tee -a "${log_file}"

