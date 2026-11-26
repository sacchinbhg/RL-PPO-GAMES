#!/bin/bash
set -euo pipefail
base="$(cd "$(dirname "$0")/.." && pwd)"
root="$base/outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
out="$root/controls/heldout-rewards"
exec 9>"$out/earned-probe.lock"
flock -n 9 || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
while [[ ! -f "$root/earned-experience-comparison/diverse-opening/adaptation/result.json" ]]; do sleep 30; done
while [[ ! -f "$root/earned-experience-comparison/shared-opening/fresh-65536/status.json" ]]; do sleep 30; done
for arm in earned-official-updates earned-experience-comparison/shared-opening earned-experience-comparison/diverse-opening; do
  label="${arm//\//-}"
  target="$out/$label.json"
  [[ ! -e "$target" ]] || continue
  while [[ ! -f "$root/$arm/adaptation/result.json" ]] || (( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) < 1850000 )); do sleep 30; done
  /opt/emerald-rl/venv/bin/python -u "$base/work/probe-emerald-reward-model.py" seed-71299-h15 eval-65536 \
    --adapted "$root/$arm/adaptation" --output "$target" >> "$out/earned-probe.log" 2>&1
done
