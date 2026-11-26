#!/bin/bash
set -euo pipefail

base=/mnt/c/Users/sacch/Documents/Codex/2026-11-24/new-chat
official="$base/outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
case "$1" in
  ordinary|parallel-baseline|screen-collection|staggered-screen-collection) option=() ;;
  reward-replay|reward-control) option=(--reward-replay-fraction 0.25) ;;
  retained-rewards) option=(--reward-replay-fraction 0.25 --retain-reward-windows) ;;
  retained-entropy) option=(--reward-replay-fraction 0.25 --retain-reward-windows --actor-entropy 0.003) ;;
  long-discount) option=(--discount-horizon 4096) ;;
  uncapped) option=(--uncapped-exploration) ;;
  *) exit 2 ;;
esac
arm="$official/seed-71299-h15-$1"
if [[ "$1" == ordinary ]]; then arm="$official/seed-71299-h15"; fi
mkdir -p "$arm"
exec 8>"$arm/run.lock"
flock -n 8 || { echo "Variant already running: $arm"; exit 1; }
budget=${2:-65536}
train_budget=$budget
behavior=()
train_episode=16384
eval_episode=8192
eval_steps=32780
if [[ "$1" == long-discount || "$1" == parallel-baseline || "$1" == uncapped ]]; then
  eval_episode=16384
  eval_steps=65544
fi
if (( budget > 65536 )); then
  train_episode=32768
  eval_episode=32768
  eval_steps=131076
fi
if [[ "$budget" == 524288 && ( "$1" == ordinary || "$1" == screen-collection || "$1" == staggered-screen-collection ) ]]; then
  # Four episodes from the 262152-step parent, including each reset observation.
  train_budget=524300
  train_episode=65536
  eval_episode=65536
  eval_steps=262152
fi
if [[ "$1" == screen-collection || "$1" == staggered-screen-collection ]]; then
  [[ -f "$arm/train/branch-origin.json" && -f "$official/controls/screen-collection-smoke/checked/verification.json" ]]
  behavior=(--screen-exploration-envs 2)
fi
if [[ "$1" == staggered-screen-collection ]]; then
  behavior+=(--short-episode-envs 2 --short-episode-length 8192)
fi
[[ ! -f "$arm/block-$budget.done" ]] || exit 0
campaign_log="$official/../wsl-campaign.log"
if [[ "$1" == retained-entropy && -f "$official/seed-71299-h15-staggered-screen-collection/train/branch-origin.json" && -f "$official/controls/staggered-reset-smoke/verification.json" ]]; then
  bash "$base/work/run-emerald-script.sh" "$base/work/emerald-variant-block.sh" staggered-screen-collection 524288 || echo "$(date -Is) Staggered resets need review; continuing scheduled work." >> "$campaign_log"
fi
if [[ "$1" == ordinary && "$budget" == 524288 && -f "$official/controls/screen-collection-smoke/checked/verification.json" ]]; then
  bash "$base/work/run-emerald-script.sh" "$base/work/emerald-variant-block.sh" screen-collection 524288 || echo "$(date -Is) Screen collection needs review; continuing matched ordinary training." >> "$campaign_log"
fi
if [[ "$1" == parallel-baseline && -f "$official/controls/heldout-action-rewards/real-outcomes.json" ]]; then
  bash "$base/work/run-emerald-script.sh" "$base/work/run-emerald-action-rankings.sh" || echo "$(date -Is) Action ranking needs review; continuing GPU campaign." >> "$campaign_log"
fi
if [[ "$1" == parallel-baseline && -f "$official/seed-71299-h15/eval-65536-long/prefix-verification.json" && ! -f "$official/seed-71299-h15/eval-65536-game-65536/prefix-verification.json" ]]; then
  printf '%s evaluate ordinary Dreamer seed=71299 horizon=15 from boot for 65536 decisions per game\n' "$(date -Is)" >> "$campaign_log"
  /opt/emerald-rl/venv/bin/python "$base/work/extend-emerald-frozen-eval.py" seed-71299-h15 65536 || echo "$(date -Is) Longer frozen evaluation needs review; continuing GPU campaign." >> "$campaign_log"
fi
if [[ "$1" == parallel-baseline && -f "$official/seed-71299-h15/block-262144.done" ]]; then
  if /opt/emerald-rl/venv/bin/python "$base/work/select-emerald-next-budget.py" >> "$campaign_log"; then
    bash "$base/work/run-emerald-script.sh" "$base/work/emerald-variant-block.sh" ordinary 524288
  fi
fi
if [[ "$1" == parallel-baseline && -f "$official/seed-71299-h15-retained-entropy/train/branch-origin.json" ]]; then
  bash "$base/work/run-emerald-script.sh" "$base/work/emerald-variant-block.sh" retained-entropy 262144
fi
if [[ "$1" == ordinary || "$1" == parallel-baseline ]]; then
  bash "$base/work/run-emerald-script.sh" "$base/work/emerald-reward-probes.sh" || echo "$(date -Is) Reward forecast check deferred; continuing GPU campaign." >> "$campaign_log"
fi
if [[ "$1" == ordinary && "$budget" == 262144 && -f "$official/seed-71299-h15-retained-rewards/block-262144.done" ]]; then
  study="$official/controls/frozen-action-mixture"
  mkdir -p "$study"
  for condition in "0 256" "0.20 8192"; do
    read -r fraction buttons <<< "$condition"
    printf '%s diagnose frozen retained-rewards policy, uniform button fraction %s, %s decisions\n' "$(date -Is)" "$fraction" "$buttons" >> "$campaign_log"
    if ! /opt/emerald-rl/venv/bin/python -u "$base/work/test-frozen-emerald-action-mixture.py" "$fraction" "$buttons" > "$study/run-$fraction.log" 2>&1; then
      echo "$(date -Is) Frozen button diagnostic needs review; continuing GPU training." >> "$campaign_log"
      break
    fi
  done
fi
source "$base/work/enter-emerald-optimizer.sh"
printf '%s train seed=71299 horizon=15 variant=%s steps=%s\n' "$(date -Is)" "$1" "$train_budget" >> "$campaign_log"
command=(/opt/emerald-rl/venv/bin/python -u upstream_emerald.py
  --upstream /opt/emerald-rl/source --platform cuda --envs 4
  --batch 8 --length 32 --ratio 16 --imag-length 15 "${option[@]}")
if [[ -f "$official/controls/parallel-frozen-evaluation/verified-equivalence.json" ]]; then
  command+=(--parallel-envs)
fi
"${command[@]}" "${behavior[@]}" --logdir "$arm/train" --seed 71299 --steps "$train_budget" --episode "$train_episode" > "$arm/train-$budget.log" 2>&1
flock -u 7
exec 7>&-
checkpoint="$arm/train/ckpt/$(cat "$arm/train/ckpt/latest")"
[[ -f "$checkpoint/done" ]]
printf '%s evaluate seed=71299 horizon=15 variant=%s\n' "$(date -Is)" "$1" >> "$campaign_log"
"${command[@]}" --logdir "$arm/eval-$budget" --seed 271340 --evaluate "$checkpoint" \
    --steps "$eval_steps" --episode "$eval_episode" > "$arm/eval-$budget.log" 2>&1
touch "$arm/block-$budget.done"
if [[ "$1" == screen-collection && "$budget" == 524288 && -f "$official/seed-71299-h15/eval-65536-game-65536/prefix-verification.json" ]]; then
  printf '%s extend the starter-earning frozen ordinary policy to 131072 buttons per game, seed=271340\n' "$(date -Is)" >> "$campaign_log"
  /opt/emerald-rl/venv/bin/python "$base/work/extend-emerald-frozen-eval.py" seed-71299-h15 131072 || echo "$(date -Is) Extended starter policy evaluation needs review; continuing training." >> "$campaign_log"
fi
