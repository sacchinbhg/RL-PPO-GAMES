#!/bin/bash
set -euo pipefail
base=/mnt/c/Users/sacch/Documents/Codex/2026-11-24/new-chat
project="$base/outputs/EmeraldRL"
root="$project/runs/pixels-only-01/official-wsl"
parent="$root/seed-71299-h15/eval-65536"
series="$root/seed-71299-h15/unbounded-seed-271341"
mkdir -p "$series"
exec 8>"$series/run.lock"
flock -n 8 || exit 1
while [[ ! -f "$root/controls/direct-parent-65536/result.json" ]]; do sleep 10; done
while (( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) < 1000000 )); do sleep 15; done
if [[ -f "$root/seed-71299-h15/eval-65536-game-262144/verified-stone-badge.json" ]]; then exit 0; fi
attempt=$(mktemp -d "$series/fresh-XXXXXXXX")
printf '%s\n' "$attempt" > "$series/current.tmp"
mv "$series/current.tmp" "$series/current"
export EMERALD_ROM="$base/work/pokeemerald/pokeemerald.gba"
export EMERALD_CORE=/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /opt/emerald-rl/venv/bin/python -u "$project/play_frozen_dreamer.py" \
  "$parent/frozen-policy" "$parent" "$attempt" --seed 271341 --until-badge \
  --upstream /opt/emerald-rl/source > "$attempt/run.log" 2>&1
