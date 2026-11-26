# Pokémon Emerald with official DreamerV3

The experiment trains the DreamerV3 author's JAX learner to earn the Stone Badge in Pokémon Emerald. Every episode starts at boot on the English revision 0 ROM. The policy receives RGB frames and chooses one of nine buttons, including wait. A separate emulator worker computes scalar rewards from game events and writes an action tape for verification. It sends the learner the image, reward and episode boundary. All training data comes from the current policy interacting with the game.

The paired experiment changes DreamerV3's imagination horizon from 15 to 1. Both policies train in a learned world model. Record each run's actual training step, elapsed time and fresh-game milestone counts to measure what the additional imagined steps buy.

## Run on a WSL2 NVIDIA GPU

The tested system uses Ubuntu 24.04, Python 3.12 and an RTX 4060 Laptop GPU. From this directory in WSL, install the emulator and pinned Python environment, then check out the upstream revision.

```bash
sudo apt-get update
sudo apt-get install -y git python3-venv libretro-mgba ffmpeg
python3 -m venv .venv-dreamer
source .venv-dreamer/bin/activate
python -m pip install -r dreamer-wsl-requirements.txt
git clone https://github.com/danijar/dreamerv3.git upstream-emerald
git -C upstream-emerald checkout --detach e01491fad6434b2245a3b8ca201dd7faedcc458c
export EMERALD_CORE=/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so
export EMERALD_ROM="/path/to/your/Pokemon Emerald.gba"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
python -c 'import jax, jax.numpy as jnp; x = jnp.ones((512, 512)); y = jax.jit(lambda a: a @ a)(x).block_until_ready(); print(y.devices(), float(y[0, 0])); assert next(iter(y.devices())).platform == "gpu"'
```

The emulator checks the ROM SHA-1, `f3ae088181bf583e55daf962a92bb46f4f1d07b7`. The launcher checks the upstream Git revision and source tree. Train one arm, then evaluate its frozen weights in four fresh games.

```bash
run=runs/pixels-only/seed-71299-h15
common=(--upstream upstream-emerald --platform cuda --envs 4 --batch 8 --length 32 --ratio 16 --imag-length 15)
python upstream_emerald.py "${common[@]}" --seed 71299 --logdir "$run/train" --steps 65536 --episode 16384
checkpoint="$run/train/ckpt/$(cat "$run/train/ckpt/latest")"
python upstream_emerald.py "${common[@]}" --seed 271340 --logdir "$run/eval-65536" --evaluate "$checkpoint" --steps 32780 --episode 8192
```

For the paired control, set `--imag-length 1` and use a separate run directory with the same seeds and budgets. To extend training, keep the training directory and increase `--steps` to the desired cumulative driver step. The driver includes each game's initial observation in its counter and rounds up to a whole group. Full game tapes count actual button decisions.

The first training blocks used the sequential upstream driver. Add `--parallel-envs` to enable its process-based driver. Each game keeps its own emulator and input history. When comparing wall-clock training times, record the driver alongside the GPU and model configuration.

## Exploration experiments

Additional 15-step runs use seed 71299 and fresh weights. Give each experiment its own directory and pass the chosen option during training and evaluation. Compare frozen policies at the same decision cutoff.

- `--reward-replay-fraction 0.25` changes how often the learner revisits its own experience. Once a rewarded sequence exists, on a quarter of offline training draws it samples sequences in proportion to their largest absolute scalar reward during the 32 training steps. The other draws are uniform. The upstream online queue still supplies newly collected sequences, and report batches use uniform sampling. This is a local replay extension to the pinned DreamerV3 learner. The policy still chooses every button. The result to compare is fresh-game progress.
- `--uncapped-exploration` pays 0.02 for each distinct location cell visited during an episode. The baseline stops after 50 cells. A cell covers two by two game tiles, every cell has equal value, and repeat visits earn zero. This varies the scalar game reward while keeping ordinary upstream replay.
- `--discount-horizon 4096` changes the official learner's reward discount from `1 - 1/333` to `1 - 1/4096`. Use ordinary replay and 15 imagined steps in both arms. At 65,536 training steps, evaluate each for four fresh games of 16,384 button decisions. The change gives distant rewards more weight in the return. The emulator observations and game rewards are identical.

## Inspect the evidence

- `train/metrics.jsonl` contains throughput, model losses and replay statistics. Both arms use a 131,072-transition in-memory replay limit.
- `train/world-model-predictions/` stores the official model's prediction grids as compressed NumPy arrays. Each column is a sampled training sequence. Rows show game pixels, predicted pixels, and signed error. Green borders mark the observed part. Red borders mark open-loop prediction conditioned on recorded buttons.
- Each evaluation's `provenance.json` identifies the actual training step and SHA-256 of its frozen policy. The directory also retains the weights so a later training checkpoint can rotate safely.
- `external-audit/` contains one JSON milestone log and raw button tape per completed episode. Use the evaluation directory to assess frozen-policy performance. The training audit describes episodes during which the weights could change.
- A claimed evaluation badge triggers a second emulator process. It replays every button from boot, matches the milestone timings, checks Roxanne's trainer ID and both victory flags in Rustboro Gym, and renders video. A successful verification writes `verified-stone-badge.json` under that evaluation. The action tape covers the full playthrough. The video is a short clip around the badge event.

[Pinned DreamerV3 source](https://github.com/danijar/dreamerv3/tree/e01491fad6434b2245a3b8ca201dd7faedcc458c)
