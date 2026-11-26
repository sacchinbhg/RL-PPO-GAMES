#!/bin/bash
set -euo pipefail
base=/mnt/c/Users/sacch/Documents/Codex/2026-11-24/new-chat
out="$base/outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
launcher="$base/work/run-upstream-emerald-wsl.sh"
mkdir -p "$out"
exec >>"$out/../wsl-campaign.log" 2>&1
exec 9>/opt/emerald-rl/campaign.lock
flock -n 9 || { echo "Another campaign process owns this GPU run"; exit 1; }
for budget in 65536 262144 1048576 4194304 16777216 67108864 268435456; do
  for seed in 71299 19831 47363; do
    if [[ "$seed" == 19831 ]]; then horizons=(1 15); else horizons=(15 1); fi
    for horizon in "${horizons[@]}"; do
      run="$out/seed-$seed-h$horizon"
      mkdir -p "$run"
      if [[ -f "$run/block-$budget.done" ]]; then continue; fi
      echo "$(date -Is) train seed=$seed horizon=$horizon steps=$budget"
      bash "$base/work/run-emerald-script.sh" "$launcher" --logdir "$run/train" --seed "$seed" --imag-length "$horizon" \
        --steps "$budget" --episode 16384 --envs 4 --batch 8 --length 32 --ratio 16 \
        >"$run/train-$budget.log" 2>&1
      checkpoint=$(find "$run/train/ckpt" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
      [[ -n "$checkpoint" ]] || { echo "missing checkpoint"; exit 1; }
      echo "$(date -Is) evaluate $checkpoint"
      bash "$base/work/run-emerald-script.sh" "$launcher" --logdir "$run/eval-$budget" --seed "$((seed+200041))" --imag-length "$horizon" \
        --evaluate "$checkpoint" --steps 32780 --episode 8192 --envs 4 --batch 8 --length 32 --ratio 16 \
        >"$run/eval-$budget.log" 2>&1
      touch "$run/block-$budget.done"
      if grep -rl --include='*.json' '"stone_badge"' "$run/eval-$budget/external-audit" >"$run/winning-evaluations.txt"; then
        echo "$(date -Is) fresh policy reached badge; independent replay required"
        exit 0
      fi
    done
  done
done
