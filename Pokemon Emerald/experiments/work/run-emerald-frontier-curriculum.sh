#!/bin/bash
set -euo pipefail
base="$(cd "$(dirname "$0")/.." && pwd)"
cd "$base"
export EMERALD_ROM="$base/work/pokeemerald/pokeemerald.gba"
export EMERALD_CORE=/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
dir="$base/outputs/EmeraldRL/runs/pixels-only-01/official-wsl/earned-frontier-curriculum"
mkdir -p "$dir"
mode=selection
if [[ "${1:-}" == --world-model ]]; then mode=official; fi
exec /opt/emerald-rl/venv/bin/python -u work/promote-emerald-earned-frontiers.py "$@" >>"$dir/$mode.log" 2>&1
