#!/bin/bash
set -euo pipefail
base=/mnt/c/Users/sacch/Documents/Codex/2026-11-24/new-chat
root="$base/outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
out="$root/controls/heldout-rewards"
[[ -f "$out/sources.json" ]] || exit 0
for pair in "seed-71299-h15 eval-65536" "seed-71299-h15-reward-replay eval-65536" "seed-71299-h15-reward-replay eval-262144" \
            "seed-71299-h15-retained-rewards eval-262144" "seed-71299-h15-reward-control eval-262144"; do
  read -r arm evaluation <<< "$pair"
  if [[ -f "$out/$arm-$evaluation.json" || ! -f "$root/$arm/$evaluation/provenance.json" ]]; then continue; fi
  printf '%s checking frozen reward predictions for %s %s\n' "$(date -Is)" "$arm" "$evaluation" >> "$root/../wsl-campaign.log"
  /opt/emerald-rl/venv/bin/python -u "$base/work/probe-emerald-reward-model.py" "$arm" "$evaluation" \
    > "$out/$arm-$evaluation.log" 2>&1 || { echo "Reward-model check needs review: $arm $evaluation" >> "$root/../wsl-campaign.log"; exit 1; }
done
