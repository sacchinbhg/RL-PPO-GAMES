#!/bin/bash
set -euo pipefail
base=/mnt/c/Users/sacch/Documents/Codex/2026-11-24/new-chat
export EMERALD_CORE=/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so
export EMERALD_ROM="$base/work/pokeemerald/pokeemerald.gba"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
root="$base/outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
arm="$root/reward-only-training-forks"
mkdir -p "$arm"
exec 8>"$arm/run.lock"
flock -n 8 || exit 1
py=/opt/emerald-rl/venv/bin/python
"$py" "$base/work/test-emerald-training-forks.py" >"$arm/check.log" 2>&1
exec nice -n 14 "$py" "$base/outputs/EmeraldRL/reward_branch_search.py" --seed 69731 --steps 5000000 --logdir "$arm" >"$arm/search.log" 2>&1
