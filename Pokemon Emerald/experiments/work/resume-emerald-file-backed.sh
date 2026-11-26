#!/bin/bash
set -euo pipefail
base="$(cd "$(dirname "$0")/.." && pwd)"
official="$base/outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
out="$official/controls/file-backed-replay"
exec 9>"$out/resume.lock"
flock -n 9 || exit 1
for item in "staggered-screen-collection 524288" "retained-entropy 262144"; do
  read -r arm budget <<< "$item"
  source_log="$official/seed-71299-h15-$arm/train-$budget.log"
  if [[ ! -e "$out/$arm-before-fix.log" ]]; then
    cp "$source_log" "$out/$arm-before-fix.log"
  fi
done
while [[ ! -s "$out/verified.json" ]]; do sleep 10; done
export EMERALD_ROM="$base/work/pokeemerald/pokeemerald.gba"
export EMERALD_CORE=/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$base/outputs/EmeraldRL"
bash "$base/work/run-emerald-script.sh" "$base/work/emerald-variant-block.sh" retained-entropy 262144 >>"$out/resume.log" 2>&1
