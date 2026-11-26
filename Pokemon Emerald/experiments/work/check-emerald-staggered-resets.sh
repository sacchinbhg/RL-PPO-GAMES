#!/bin/bash
set -euo pipefail
base=/mnt/c/Users/sacch/Documents/Codex/2026-11-24/new-chat
export EMERALD_CORE=/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so
export EMERALD_ROM="$base/work/pokeemerald/pokeemerald.gba"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
while (( $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) < 2200000 )); do sleep 15; done
exec nice -n 12 /opt/emerald-rl/venv/bin/python "$base/work/check-emerald-staggered-resets.py"
