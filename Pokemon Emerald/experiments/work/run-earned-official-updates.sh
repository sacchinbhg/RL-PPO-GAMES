#!/bin/bash
set -euo pipefail
base=/mnt/c/Users/sacch/Documents/Codex/2026-11-24/new-chat
project="$base/outputs/EmeraldRL"
root="$project/runs/pixels-only-01/official-wsl"
py=/opt/emerald-rl/venv/bin/python
parent="$root/seed-71299-h15/eval-65536"
tape_a="$root/reward-only-training-forks/verified-training-tapes/24479561ff0b9fe07b3932aeb611e70d3b7b7f4b42a9f6fe428cea84e19fa412"
tape_b="$root/paired-self-earned-frontiers/visual-slots-0/verified-training-tapes/e30822c294f0d5469126f3e250c9ad4ac83b232534b37d6f54b24b56078e37c5"
out="$root/earned-official-updates"
mkdir -p "$out"
exec 8>"$out/job.lock"
flock -n 8 || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export EMERALD_ROM="$base/work/pokeemerald/pokeemerald.gba"
export EMERALD_CORE=/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so
while [[ ! -f "$tape_b/manifest.json" ]]; do sleep 10; done
"$py" "$base/work/validate-earned-training-stream.py" "$tape_b" "$root/controls/earned-recurrent-stream/torchic.json" > "$out/validation.log"
[[ -f "$root/controls/earned-recurrent-stream/treecko.json" ]]
# The online learner, frozen control and original-ROM collectors share this WSL instance.
while (( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) < 1700000 )); do sleep 15; done
printf '%s starting pinned Dreamer updates on earned training data\n' "$(date -Is)" >> "$out/run.log"
"$py" -u "$project/train_earned_dreamer.py" --dataset "$tape_a" --dataset "$tape_b" \
  --source-run "$parent" --upstream /opt/emerald-rl/source --output "$out/adaptation" \
  --updates 1152 >> "$out/run.log" 2>&1
[[ -f "$out/adaptation/result.json" ]]
"$py" -u "$project/play_frozen_dreamer.py" "$out/adaptation/frozen-policy" "$parent" "$out/fresh-65536" \
  --limit 65536 --upstream /opt/emerald-rl/source >> "$out/run.log" 2>&1
