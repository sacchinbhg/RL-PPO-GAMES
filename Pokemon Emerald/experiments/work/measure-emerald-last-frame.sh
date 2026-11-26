#!/bin/bash
set -euo pipefail
base="$(cd "$(dirname "$0")/.." && pwd)"
cd "$base"
official="$base/outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
out="$official/controls/last-frame-render"
exec 9>"$out/paired-tests.lock"
flock -n 9 || exit 1
export EMERALD_CORE=/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so
export EMERALD_ROM="$base/work/pokeemerald/pokeemerald.gba"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=/opt/emerald-rl/venv/bin/python
while [[ ! -f "$official/earned-frontier-curriculum/stage-003847-01f5c6813216/fresh-001/result.json" ]]; do sleep 30; done
nice -n 10 "$py" -u work/test-emerald-render-cadence.py --steps 2048 >>"$out/measurements.log" 2>&1
if "$py" -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r["all_images_states_and_rewards_match"] and r["cpu_speedup"]>1.03 else 1)' "$out/comparison-2048.json"; then
  nice -n 10 "$py" -u work/test-emerald-render-cadence.py --steps 36058 --long >>"$out/measurements.log" 2>&1
fi
