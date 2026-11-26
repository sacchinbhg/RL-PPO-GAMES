#!/bin/bash
set -euo pipefail
base=/mnt/c/Users/sacch/Documents/Codex/2026-11-24/new-chat
export EMERALD_CORE=/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so
export EMERALD_ROM="$base/work/pokeemerald/pokeemerald.gba"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
study="$base/outputs/EmeraldRL/runs/pixels-only-01/official-wsl/controls/screen-collection-smoke"
mkdir -p "$study"
cd "$base/outputs/EmeraldRL"
exec /opt/emerald-rl/venv/bin/python "$base/work/test-emerald-screen-collection.py" >"$study/smoke.log" 2>&1
