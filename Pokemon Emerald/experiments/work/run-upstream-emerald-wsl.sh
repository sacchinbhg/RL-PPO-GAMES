#!/bin/bash
set -euo pipefail
project=/mnt/c/Users/sacch/Documents/Codex/2026-11-24/new-chat/outputs/EmeraldRL
export EMERALD_CORE=/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so
export EMERALD_ROM=/mnt/c/Users/sacch/Documents/Codex/2026-11-24/new-chat/work/pokeemerald/pokeemerald.gba
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
export TF_CPP_MIN_LOG_LEVEL=1
cd "$project"
python=/opt/emerald-rl/venv/bin/python
official="$project/runs/pixels-only-01/official-wsl"
variant() {
  bash "$project/../../work/run-emerald-script.sh" "$project/../../work/emerald-variant-block.sh" "$@"
}
if [[ "${1:-}" == --logdir && "$2" == "$official/seed-19831-h15/eval-65536" ]]; then
  bash "$project/../../work/run-emerald-script.sh" "$project/../../work/emerald-reward-probes.sh" || echo "Reward-model diagnostics deferred; starting frozen games."
fi
if [[ "${1:-}" == --logdir && "$2" == "$official/seed-47363-h15/eval-65536" && -f "$official/seed-71299-h15-reward-control/train/branch-origin.json" ]]; then
  variant reward-control 262144
  bash "$project/../../work/run-emerald-script.sh" "$project/../../work/emerald-reward-probes.sh" || echo "Reward forecast check deferred; continuing frozen games."
  printf '%s evaluate seed=47363 horizon=15 budget=65536\n' "$(date -Is)" >> "$official/../wsl-campaign.log"
fi
if [[ "${1:-}" == --logdir && "$2" == "$official/seed-19831-h1/eval-65536" && -f "$official/seed-71299-h15-reward-replay/block-65536.done" ]]; then
  printf '%s extending frozen reward-replay games to 16384 decisions\n' "$(date -Is)" >> "$official/../wsl-campaign.log"
  "$python" "$project/../../work/extend-emerald-frozen-eval.py" seed-71299-h15-reward-replay
fi
if [[ "${1:-}" == --logdir && "$2" == "$official/seed-19831-h15/train" && -f "$official/seed-71299-h15-reward-replay/block-65536.done" ]]; then
  printf '%s extending matched ordinary frozen games to 16384 decisions\n' "$(date -Is)" >> "$official/../wsl-campaign.log"
  "$python" "$project/../../work/extend-emerald-frozen-eval.py" seed-71299-h15
  variant reward-replay 262144
  printf '%s running scheduled seed=19831 horizon=15\n' "$(date -Is)" >> "$official/../wsl-campaign.log"
fi
if [[ "${1:-}" == --logdir && "$2" == "$official/seed-19831-h1/train" && -f "$official/seed-71299-h1/block-65536.done" ]]; then
  if [[ ! -f "$official/controls/parallel-frozen-evaluation/verified-equivalence.json" ]]; then
    printf '%s validating parallel driver on frozen evaluation\n' "$(date -Is)" >> "$official/../wsl-campaign.log"
    "$python" "$project/../../work/validate-emerald-parallel-eval.py" || printf '%s parallel evaluation needs review, retaining serial driver\n' "$(date -Is)" >> "$official/../wsl-campaign.log"
  fi
  variant reward-replay
  printf '%s running scheduled seed=19831 horizon=1\n' "$(date -Is)" >> "$official/../wsl-campaign.log"
fi
if [[ "${1:-}" == --logdir && "$2" == "$official/seed-47363-h15/train" && -f "$official/seed-19831-h15/block-65536.done" ]]; then
  if [[ -f "$official/seed-71299-h15-retained-rewards/train/branch-origin.json" ]]; then
    variant retained-rewards 262144
  fi
  if [[ -f "$official/seed-71299-h15/highlights/first-frozen-rival/verification.json" ]]; then
    variant ordinary 262144
  fi
  if [[ -f "$official/controls/parallel-frozen-evaluation/verified-equivalence.json" ]]; then
    variant parallel-baseline
  fi
  variant uncapped
  variant long-discount
  printf '%s running scheduled seed=47363 horizon=15\n' "$(date -Is)" >> "$official/../wsl-campaign.log"
fi
driver=()
if [[ -f "$official/controls/parallel-frozen-evaluation/verified-equivalence.json" ]]; then driver=(--parallel-envs); fi
longer=()
if [[ "${1:-}" == --logdir && "$2" == "$official"/seed-*/* ]]; then
  phase=$(basename "$2")
  if [[ "$phase" =~ ^eval-([0-9]+)$ ]] && (( BASH_REMATCH[1] > 65536 )); then
    longer=(--episode 32768 --steps 131076)
  elif [[ "$phase" == train ]]; then
    arguments=("$@")
    for ((i=0; i+1<${#arguments[@]}; i++)); do
      if [[ "${arguments[i]}" == --steps ]] && (( arguments[i+1] > 65536 )); then longer=(--episode 32768); fi
    done
  fi
fi
if [[ "${1:-}" == --logdir && "$2" == */train ]]; then
  source "$project/../../work/enter-emerald-optimizer.sh"
fi
exec "$python" -u upstream_emerald.py --upstream /opt/emerald-rl/source --platform cuda "${driver[@]}" "$@" "${longer[@]}"
