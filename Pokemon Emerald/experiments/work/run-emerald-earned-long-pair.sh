#!/bin/bash
set -euo pipefail
base="$(cd "$(dirname "$0")/.." && pwd)"
project="$base/outputs/EmeraldRL"
root="$project/runs/pixels-only-01/official-wsl"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export EMERALD_ROM="$base/work/pokeemerald/pokeemerald.gba"
export EMERALD_CORE=/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so
parent="$root/seed-71299-h15/eval-65536"
out="$root/earned-official-updates/long-pair"
mkdir -p "$out"
exec 9>"$out/job.lock"
flock -n 9 || exit 1
while [[ ! -f "$root/earned-official-updates/fresh-65536/result.json" ]]; do sleep 15; done
while [[ ! -f "$root/earned-experience-comparison/diverse-opening/adaptation/result.json" ]]; do sleep 15; done
while [[ ! -f "$root/earned-experience-comparison/shared-opening/fresh-65536/status.json" ]]; do sleep 15; done
for arm in adapted unchanged; do
  check="$parent/frozen-policy"
  if [[ "$arm" == adapted ]]; then check="$root/earned-official-updates/adaptation/frozen-policy"; fi
  if [[ ! -f "$out/$arm/result.json" ]]; then
    while (( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) < 1650000 )); do sleep 15; done
    /opt/emerald-rl/venv/bin/python -u "$project/play_frozen_dreamer.py" "$check" "$parent" "$out/$arm" \
      --upstream /opt/emerald-rl/source --limit 131072 --seed 271342 >> "$out/run.log" 2>&1
  fi
done
