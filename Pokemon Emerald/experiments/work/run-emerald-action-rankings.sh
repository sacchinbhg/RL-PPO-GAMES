#!/bin/bash
set -euo pipefail
base=/mnt/c/Users/sacch/Documents/Codex/2026-11-24/new-chat
root="$base/outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
folder="$root/controls/heldout-action-rewards"
[[ -f "$folder/real-outcomes.json" ]] || exit 0
exec 7>"$folder/probe.lock"
flock -n 7 || exit 0
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
for arm in seed-71299-h15-reward-replay seed-71299-h15-retained-rewards; do
  [[ ! -f "$folder/$arm-ranking.json" ]] || continue
  /opt/emerald-rl/venv/bin/python -u "$base/work/score-emerald-action-reward-forecasts.py" "$arm" > "$folder/$arm.log" 2>&1
done
