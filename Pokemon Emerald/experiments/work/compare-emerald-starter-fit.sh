#!/bin/bash
set -euo pipefail
base=/mnt/c/Users/sacch/Documents/Codex/2026-11-24/new-chat
root="$base/outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
parent="$root/seed-71299-h15/eval-65536"
candidate="$root/reward-only-training-forks/actor-fits/$1/fresh-65536"
control="$root/controls/direct-parent-65536"
mkdir -p "$control"
exec 8>"$control/run.lock"
flock -n 8 || exit 1
export EMERALD_CORE=/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so
export EMERALD_ROM="$base/work/pokeemerald/pokeemerald.gba"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
while [[ ! -f "$candidate/result.json" ]]; do sleep 20; done
while (( $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) < 2050000 )); do sleep 15; done
py=/opt/emerald-rl/venv/bin/python
nice -n 14 "$py" "$base/outputs/EmeraldRL/play_frozen_dreamer.py" "$parent/frozen-policy" "$parent" "$control" --limit 65536 --upstream /opt/emerald-rl/source > "$control/process.log" 2>&1
"$py" "$base/work/compare-emerald-fitted-policy.py" "$1" 65536 > "$control/comparison.log"
