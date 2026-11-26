#!/bin/bash
set -euo pipefail
base=/mnt/c/Users/sacch/Documents/Codex/2026-11-24/new-chat
export EMERALD_CORE=/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so
export EMERALD_ROM="$base/work/pokeemerald/pokeemerald.gba"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
study="$base/outputs/EmeraldRL/runs/pixels-only-01/official-wsl/controls/screen-exploration-32768"
exec 8>"$study/continuation.lock"
flock -n 8 || exit 1
while [[ ! -f "$study/selected-131072/game-01/result.json" ]]; do sleep 10; done
if [[ ! -f "$study/selected-131072/diagnostics/verified.json" ]]; then
  nice -n 15 /opt/emerald-rl/venv/bin/python "$base/work/diagnose-emerald-explorer-tape.py" >"$study/diagnostics.log" 2>&1 &
  diagnosis=$!
fi
nice -n 12 /opt/emerald-rl/venv/bin/python "$base/work/emerald-screen-exploration.py" --extend-best-further >"$study/continuation.log" 2>&1
if [[ -n "${diagnosis:-}" ]]; then wait "$diagnosis"; fi
