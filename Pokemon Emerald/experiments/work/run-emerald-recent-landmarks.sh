#!/bin/bash
set -euo pipefail
base="$(cd "$(dirname "$0")/.." && pwd)"
cd "$base"
root="$base/outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
family="$root/paired-self-earned-frontiers"
export EMERALD_CORE=/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so
export EMERALD_ROM="$base/work/pokeemerald/pokeemerald.gba"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec 9>"$family/recent-landmarks-queue.lock"
flock -n 9 || exit 1
while [[ ! -f "$family/learned-visual-motion-landmarks-128-systematic-options/completed.json" ]]; do sleep 30; done
while (( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) < 1500000 )); do sleep 30; done
exec nice -n 8 /opt/emerald-rl/venv/bin/python -u work/run-emerald-option-comparison.py --motion --landmarks 128 \
  --systematic-options --reward-landmarks --recent-landmarks 32
