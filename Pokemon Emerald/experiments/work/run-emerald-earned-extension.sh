#!/bin/bash
set -euo pipefail
base="$(cd "$(dirname "$0")/.." && pwd)"
project="$base/outputs/EmeraldRL"
root="$project/runs/pixels-only-01/official-wsl"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export EMERALD_ROM="$base/work/pokeemerald/pokeemerald.gba"
export EMERALD_CORE=/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so
out="$root/earned-route-extension"
mkdir -p "$out"
exec 9>"$out/job.lock"
flock -n 9 || exit 1
while (( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) < 1300000 )); do sleep 15; done
/opt/emerald-rl/venv/bin/python -u "$project/extend_earned_route.py" --model "$root/earned-screen-memory/model" \
  --checkpoint "$root/earned-official-updates/adaptation/frozen-policy" --parent "$root/seed-71299-h15/eval-65536" \
  --output "$out/trial-197461" --seed 197461 --patience 512 --limit 65536 --upstream /opt/emerald-rl/source > "$out/run.log" 2>&1
