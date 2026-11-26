#!/bin/bash
set -euo pipefail
base="$(cd "$(dirname "$0")/.." && pwd)"
root="$base/outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
out="$root/controls/dreamer-training-forks"
mkdir -p "$out"
exec 9>"$out/job.lock"
flock -n 9 || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export EMERALD_ROM="$base/work/pokeemerald/pokeemerald.gba"
export EMERALD_CORE=/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so
py=/opt/emerald-rl/venv/bin/python
while (( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) < 1400000 )); do sleep 15; done
"$py" -u "$base/work/check-emerald-real-fork-policy.py" > "$out/probe.log" 2>&1
for arm in dreamer uniform; do
  printf '%s starting %s proposal search\n' "$(date -Is)" "$arm" >> "$out/search.log"
  "$py" -u "$base/work/run-emerald-option-comparison.py" --motion --landmarks 64 --proposal "$arm" >> "$out/search.log" 2>&1
done
