#!/usr/bin/env bash
set -u
base="$(cd "$(dirname "$0")/.." && pwd)"
cd "$base" || exit 1
root="$base/outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
out="$root/controls/morning-host-recovery"
mkdir -p "$out"
exec 9>"$out/gpu.lock"
flock -n 9 || exit 0
export EMERALD_REPLAY_MMAP_DIR=/opt/emerald-rl/replay-cache
export EMERALD_ROM="$base/work/pokeemerald/pokeemerald.gba"
export EMERALD_CORE=/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
log="$out/gpu.log"
variant="$root/seed-71299-h15-staggered-screen-collection"
for attempt in 1 2 3; do
    if [[ -f "$variant/block-524288.done" ]]; then break; fi
    if [[ -s "$variant/train-524288.log" ]]; then
        cp "$variant/train-524288.log" "$out/before-gpu-$(date +%s).log"
    fi
    printf '%s resuming online training and evaluation from %s\n' "$(date -Is)" "$(cat "$variant/train/ckpt/latest")" >>"$log"
    (cd "$base/outputs/EmeraldRL" && bash "$base/work/emerald-variant-block.sh" staggered-screen-collection 524288) >>"$log" 2>&1 && break
    printf '%s online attempt %s exited with error\n' "$(date -Is)" "$attempt" >>"$log"
    sleep 30
done
printf '%s starting paired official earned-reset and boot training\n' "$(date -Is)" >>"$log"
exec /opt/emerald-rl/venv/bin/python -u "$base/work/train-emerald-earned-resets.py" >>"$log" 2>&1
