#!/bin/bash
set -euo pipefail
base="$(cd "$(dirname "$0")/.." && pwd)"
project="$base/outputs/EmeraldRL"
root="$project/runs/pixels-only-01/official-wsl"
export EMERALD_CORE=/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so
export EMERALD_ROM="$base/work/pokeemerald/pokeemerald.gba"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
py=/opt/emerald-rl/venv/bin/python
check="$root/controls/outcome-collection-smoke"
short=("$root"/reward-only-training-forks/verified-training-tapes/99e579*)
[[ ${#short[@]} == 1 && -f "${short[0]}/manifest.json" ]]
"$py" "$project/collect_outcome_branches.py" --dataset "${short[0]}" --output "$check" --seed 4421 --branches 2 --steps-per-branch 32
"$py" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); report=json.loads((p/"completed.json").read_text()); assert report["new_emulator_buttons"]==64 and len(set(r["tape_sha256"] for r in report["records"]))==2; assert all(json.loads((p/r["audit"]/"game.json").read_text())["whiteouts"]==[] for r in report["records"])' "$check"
while (( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) < 1400000 )); do sleep 30; done
data="$root/paired-self-earned-frontiers/visual-slots-0/verified-training-tapes/e30822c294f0d5469126f3e250c9ad4ac83b232534b37d6f54b24b56078e37c5"
exec nice -n 8 "$py" -u "$project/collect_outcome_branches.py" --dataset "$data" --output "$root/paired-self-earned-frontiers/all-outcome-attempts" \
  --seed 260695 --branches 8 --steps-per-branch 8192 --inherited-search-buttons 321586
