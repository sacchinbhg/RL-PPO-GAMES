#!/bin/bash
set -euo pipefail
base="$(cd "$(dirname "$0")/.." && pwd)"
project="$base/outputs/EmeraldRL"
root="$project/runs/pixels-only-01/official-wsl"
out="$root/earned-experience-comparison"
mkdir -p "$out"
exec 9>"$out/job.lock"
flock -n 9 || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export EMERALD_ROM="$base/work/pokeemerald/pokeemerald.gba"
export EMERALD_CORE=/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so
py=/opt/emerald-rl/venv/bin/python
upstream=/opt/emerald-rl/source
parent="$root/seed-71299-h15/eval-65536"
tree="$root/reward-only-training-forks/verified-training-tapes/24479561ff0b9fe07b3932aeb611e70d3b7b7f4b42a9f6fe428cea84e19fa412"
torch="$root/paired-self-earned-frontiers/visual-slots-0/verified-training-tapes/e30822c294f0d5469126f3e250c9ad4ac83b232534b37d6f54b24b56078e37c5"
online="$root/earned-online-datasets/c43fbf0200a1e10d70e0cf3f93bdc64547005e3c7547372619a0041dc3146df5"
log="$out/run.log"
for arm in shared-opening diverse-opening; do
  first="$tree"
  if [[ "$arm" == diverse-opening ]]; then
    first="$online"
    while [[ ! -f "$first/manifest.json" ]]; do sleep 15; done
    "$py" "$base/work/validate-earned-training-stream.py" "$first" "$out/online-source-audit.json" >> "$log"
  fi
  if [[ ! -f "$out/$arm/adaptation/result.json" ]]; then
    while (( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) < 1700000 )); do sleep 15; done
    printf '%s fitting %s\n' "$(date -Is)" "$arm" >> "$log"
    "$py" -u "$project/train_earned_dreamer.py" --dataset "$first" --dataset "$torch" \
      --source-run "$parent" --upstream "$upstream" --output "$out/$arm/adaptation" --updates 2176 >> "$log" 2>&1
  fi
done
for arm in shared-opening diverse-opening; do
  if [[ ! -f "$out/$arm/fresh-65536/result.json" ]]; then
    printf '%s playing %s\n' "$(date -Is)" "$arm" >> "$log"
    "$py" -u "$project/play_frozen_dreamer.py" "$out/$arm/adaptation/frozen-policy" "$parent" \
      "$out/$arm/fresh-65536" --limit 65536 --seed 271340 --upstream "$upstream" >> "$log" 2>&1
  fi
done
