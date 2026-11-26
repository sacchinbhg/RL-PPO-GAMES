#!/bin/bash
set -euo pipefail
base=/mnt/c/Users/sacch/Documents/Codex/2026-11-24/new-chat
root="$base/outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
run="$root/reward-only-training-forks"
record="$run/discoveries/$1/record.json"
export EMERALD_CORE=/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so
export EMERALD_ROM="$base/work/pokeemerald/pokeemerald.gba"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
py=/opt/emerald-rl/venv/bin/python
tape=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["tape_sha256"])' "$record")
mkdir -p "$run/actor-fits/$tape"
exec 9>"$run/actor-fits/$tape/pipeline.lock"
flock -n 9 || exit 1
exec >> "$run/actor-fits/$tape/pipeline.log" 2>&1
nice -n 15 "$py" "$base/outputs/EmeraldRL/export_reward_branch.py" "$run" "$run/verified-training-tapes" --record "$record"
while (( $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) < 1500000 )); do sleep 15; done
nice -n 12 "$py" "$base/work/fit-emerald-self-earned-actor.py" "$tape"
if [[ -n "${3:-}" ]]; then
  while [[ ! -f "$run/actor-fits/$3/fresh-8192/result.json" ]]; do sleep 15; done
fi
exec bash "$base/work/run-emerald-script.sh" "$base/work/evaluate-emerald-self-earned-actor.sh" "$tape" "$2"
