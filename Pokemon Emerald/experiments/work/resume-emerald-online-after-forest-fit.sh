#!/bin/bash
set -euo pipefail
base="$(cd "$(dirname "$0")/.." && pwd)"
project="$base/outputs/EmeraldRL"
official="$project/runs/pixels-only-01/official-wsl"
out="$official/controls/overnight-resource-recovery"
exec 9>"$out/online-resume.lock"
flock -n 9 || exit 1
export EMERALD_ROM="$base/work/pokeemerald/pokeemerald.gba"
export EMERALD_CORE=/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
for arm in staggered-screen-collection retained-entropy; do
  run="$official/seed-71299-h15-$arm"
  latest=$(cat "$run/train/ckpt/latest")
  [[ -s "$run/train/ckpt/$latest/agent.pkl" && -s "$run/train/ckpt/$latest/replay.pkl" && -f "$run/train/ckpt/$latest/done" ]]
  if [[ "$arm" == retained-entropy ]]; then budget=262144; else budget=524288; fi
  if [[ ! -f "$out/$arm-first-attempt.log" ]]; then cp "$run/train-$budget.log" "$out/$arm-first-attempt.log"; fi
done
printf '%s waiting for focused visit-reward fit and online memory slot\n' "$(date -Is)" >>"$out/online-resume.log"
while [[ ! -f "$official/earned-official-updates/uncapped-training/adaptation/result.json" ]]; do sleep 60; done
cd "$project"
# The retained-entropy launcher first resumes the pending staggered arm. Both use the memory-gated optimizer slot.
bash "$base/work/run-emerald-script.sh" "$base/work/emerald-variant-block.sh" retained-entropy 262144 >>"$out/online-resume.log" 2>&1
