#!/bin/bash
set -euo pipefail
base=/mnt/c/Users/sacch/Documents/Codex/2026-11-24/new-chat
root="$base/outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
forks="$root/reward-only-training-forks"
fit="$forks/actor-fits/$1"
limit=${2:-8192}
dest="$fit/fresh-$limit"
parent="$root/seed-71299-h15/eval-65536"
mkdir -p "$dest"
exec 8>"$dest/run.lock"
flock -n 8 || exit 1
export EMERALD_CORE=/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so
export EMERALD_ROM="$base/work/pokeemerald/pokeemerald.gba"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# The four-game comparison fits alongside the primary frozen evaluation after the large replay is released.
while [[ ! -f "$root/seed-71299-h15-screen-collection/train/final-checkpoint.json" ]]; do sleep 15; done
while (( $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) < 2200000 )); do sleep 15; done
exec nice -n 14 /opt/emerald-rl/venv/bin/python -u "$base/outputs/EmeraldRL/play_frozen_dreamer.py" "$fit/frozen-policy" "$parent" "$dest" --limit "$limit" --upstream /opt/emerald-rl/source > "$dest/process.log" 2>&1
