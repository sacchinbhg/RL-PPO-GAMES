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

For a mix of short and long training games, pass `--short-episode-envs 2 --short-episode-length 8192` with `--envs 4 --episode 65536`. The final two workers return to boot after 8,192 decisions. The first two can continue for 65,536. This keeps opening screens arriving in replay while longer games can reach subsequent events. Short-worker settings are recorded in provenance and rejected during frozen evaluation.

With four environments, `--screen-exploration-envs 2` gives the first two training games to an online pixel explorer. It estimates which screens each button leads to and favors unfamiliar outcomes and past scalar rewards. Its memory starts empty at each fresh boot. The other two games use the upstream Dreamer actor. The official learner trains its world model and imagined actor from the real actions taken in all four games. The collector has access to RGB, scalar rewards and episode boundaries. Training seeds are separate from the discovery and evaluation seeds.

For a matched comparison, resume two copies of the same ordinary learner and replay, use the same episode length and interaction budget, and enable the collector on just one copy. For frozen gameplay evaluation omit the collection option on both arms. The evaluator enforces zero exploration workers. A training milestone on a screen-controlled worker establishes that the learner had access to that experience. A fresh evaluation establishes how far the neural policy gets on its own.

Additional 15-step runs use seed 71299 and fresh weights. Give each experiment its own directory and pass the chosen option during training and evaluation. Compare frozen policies at the same decision cutoff.

- `--reward-replay-fraction 0.25` changes how often the learner revisits its own experience. Once a rewarded sequence exists, on a quarter of offline training draws it samples sequences in proportion to their largest absolute scalar reward during the 32 training steps. The other draws are uniform. The upstream online queue still supplies newly collected sequences, and report batches use uniform sampling. This is a local replay extension to the pinned DreamerV3 learner. The policy still chooses every button. The result to compare is fresh-game progress.
- `--uncapped-exploration` pays 0.02 for each distinct location cell visited during an episode. The baseline stops after 50 cells. A cell covers two by two game tiles, every cell has equal value, and repeat visits earn zero. This varies the scalar game reward while keeping ordinary upstream replay.
- `--discount-horizon 4096` changes the official learner's reward discount from `1 - 1/333` to `1 - 1/4096`. Use ordinary replay and 15 imagined steps in both arms. At 65,536 training steps, evaluate each for four fresh games of 16,384 button decisions. The change gives distant rewards more weight in the return. The emulator observations and game rewards are identical.
- `--reward-replay-fraction 0.25 --retain-reward-windows` preserves a few of the agent's own recorded sequences ending in an absolute scalar reward of at least one. It keeps up to eight sequences in each of eight power-of-two reward bands, with the highest band including all magnitudes of 128 and above. Once the live buffer contains zero eligible windows reaching that threshold, the existing reward-focused share samples the retained sequences. Fresh online experience and uniform reporting keep their upstream behavior. Retained sequences travel with the training checkpoint. This changes replay selection only. Frozen games start from boot.

Run `PYTHONPATH=upstream-emerald python -m unittest test_reward_replay` to check retention, checkpoint restoration and sampling boundaries against the pinned replay implementation.

To test how strongly the imagined actor is encouraged to explore, set `--actor-entropy 0.003` in a separate training directory. The pinned release defaults to 0.0003. This option changes the official `agent.imag_loss.actent` coefficient. It assigns the same entropy incentive to the nine buttons. Evaluate the trained checkpoint from boot using its ordinary sampled policy. The evaluation records the coefficient used in training. [Official imagined actor loss](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/dreamerv3/agent.py#L411-L415)

## Self-generated training forks

`python reward_branch_search.py --logdir runs/reward-forks --steps 5000000` starts a separate collector from the original boot. Its pixel transition model chooses a button and explores different hold durations. It can return to points in its own training game using opaque emulator slots. It selects those points by accumulated scalar reward, screen diversity and how often it has tried them. The environment pays the existing story and level rewards and 0.02 for every first cell visit within a path. Restoring a point also restores the rewards already collected on that path. The controller receives pixels, payoffs and the episode boundary. The argument to `--steps` counts newly simulated button decisions over every branch.

`python export_reward_branch.py runs/reward-forks runs/reward-forks/verified-training-tapes` plays the current best training tape from boot in an independent emulator. It checks the exact buttons, milestone times, visited-cell count and total reward. Each exported chunk contains the RGB frame before every action, the action, the following frame and the resulting scalar reward. The collector and exporter are separate from the official Dreamer training and its frozen evaluation. A replay verifies a discovery. Fresh closed-loop play by a frozen policy measures what the policy learned.

To test whether an earned route can teach Dreamer, use `fit_branch_actor.py` in the WSL CUDA environment. Pass the pinned checkout to `--upstream`, the verified dataset directory to `--dataset`, an official frozen evaluation directory to `--source-run`, and a new directory to `--output`. The script feeds each frame and the previous executed button through the upstream encoder and recurrent state. It fits the final action-logit kernel and bias and checks exactly which parameters changed. The fitted checkpoint is for gameplay evaluation. Its source checkpoint supplies the world model. Blocks withheld from the same training trajectory select the amount of fitting. Fresh games provide the gameplay comparison.

Use `python play_frozen_dreamer.py CHECKPOINT SOURCE_RUN OUTPUT --upstream UPSTREAM --limit 65536` to play four fresh games on the GPU. The process loads the upstream inference modules and keeps each game's recurrent state. It executes each button in the original ROM. To validate an unchanged policy against historical gameplay, add `--reference EXISTING_EVALUATION`. The runner checks each button only after the policy chooses it. It stops on a mismatch. A Stone Badge triggers a second full replay and battle and flag verification.

The optional training search argument `--unrewarded-screen-slots 160` keeps visually distinct points after zero-reward actions. This lets a new training attempt start partway across an already rewarded area. These slots use the same RGB observations and scalar-return history as the rest of the collector.

## Inspect the evidence

- `train/metrics.jsonl` contains throughput, model losses and replay statistics. Both arms use a 131,072-transition in-memory replay limit.
- `train/world-model-predictions/` stores the official model's prediction grids as compressed NumPy arrays. Each column is a sampled training sequence. Rows show game pixels, predicted pixels, and signed error. Green borders mark the observed part. Red borders mark open-loop prediction conditioned on recorded buttons.
- Each evaluation's `provenance.json` identifies the actual training step and SHA-256 of its frozen policy. The directory also retains the weights so a later training checkpoint can rotate safely.
- `external-audit/` contains one JSON milestone log and raw button tape per completed episode. Use the evaluation directory to assess frozen-policy performance. The training audit describes episodes during which the weights could change.
- `in-progress/` saves the current button prefix every 8,192 decisions and whenever a decision earns at least four reward units, in both training and frozen evaluation. These are read-only observations of the same ongoing game. Use the complete tape for the final result.
- The JSONL logger skips its unused full-episode image stack. With four environments and 32,768 decisions, the original per-image views can retain 1.5 GiB of batch storage. The action tapes reproduce gameplay footage, and model prediction reports still save their image grids.
- A claimed evaluation badge triggers a second emulator process. It replays every button from boot, matches the milestone timings, checks Roxanne's trainer ID and both victory flags in Rustboro Gym, and renders video. A successful verification writes `verified-stone-badge.json` under that evaluation. The action tape covers the full playthrough. The video is a short clip around the badge event.

[Pinned DreamerV3 source](https://github.com/danijar/dreamerv3/tree/e01491fad6434b2245a3b8ca201dd7faedcc458c)

### Fixed budgets and the badge attempt

Each agent decision advances 24 GBA frames. Use `--limit 65536` on the direct frozen player to compare checkpoints at the same button budget. Use `--until-badge --seed 271341` to start four fresh games with another sampling seed and keep playing across reporting milestones until a victory is independently verified. The player writes its actual progress every 1,024 decisions. In an uncapped game it saves full tapes at power-of-two reporting checkpoints and major rewards. Use a fresh output directory for each uncapped attempt so an interrupted game's evidence is preserved.

Map names appear in human-facing game audits. The policy receives RGB, scalar reward and episode signals. The direct player also saves the input prefix at each new map. To verify a forest arrival independently, run `python pixel_replay.py AUDIT_JSON OUTPUT --map PetalburgWoods --video`. The verifier checks the original ROM, replays the recorded buttons from boot, matches the story and map timings and saves the entry frame and a short clip.

### Training on an agent's own successful games

`export_reward_branch.py` replays a discovery from an agent's training archive and writes aligned images, actions, rewards, a complete input tape and a verification manifest. Each action connects two successive images. The scalar reward belongs to the image reached by that action.

Use exported training trajectories to update the pinned Dreamer world model, reward and continuation predictors, imagined actor and critic. Each trajectory occupies one continuous batch lane and restarts at boot when its recorded episode ends. Choose the update budget so the longest lane receives at least one pass. With length 32, 1,152 updates cover 36,864 observation positions per lane. The output records optimizer updates separately from emulator decisions.

```bash
python train_earned_dreamer.py --upstream /opt/emerald-rl/source \
  --source-run PARENT_EVALUATION --dataset EARNED_GAME_A --dataset EARNED_GAME_B \
  --output ADAPTATION --updates 1152
python play_frozen_dreamer.py ADAPTATION/frozen-policy PARENT_EVALUATION FRESH_GAMES \
  --upstream /opt/emerald-rl/source --limit 65536
```

The adaptation inherits its parent's exploration reward by default. To test the visit cap on self-earned trajectories that logged all visit rewards, add `--uncapped-training-exploration` to the training command and use a separate output directory. This retains the 0.02 reward for new cells after the first 50. The original images, actions, story rewards, optimization budget and frozen evaluation environment can be held identical.

### Learning from battle losses

`collect_outcome_branches.py` replays a verified self-earned game from boot, saves its final state inside the emulator worker, and runs independently seeded screen-based exploration from that state. It keeps every completed attempt in collection order. An external game log records when Emerald invokes its whiteout routine. The chooser learns from pixels, buttons and the original scalar reward.

```bash
python collect_outcome_branches.py --dataset EARNED_GAME --output ALL_ATTEMPTS \
  --seed 260695 --branches 8 --steps-per-branch 8192
python export_reward_branch.py ALL_ATTEMPTS VERIFIED_GAMES \
  --record ALL_ATTEMPTS/discoveries/branch-000/record.json
```

The exporter independently replays each complete game from boot and checks the loss timings. To ablate a half-point training penalty, run two adaptations with the same verified datasets and update budget. Add `--whiteout-cost-cents 50` to one adaptation. Its learner receives the lower scalar reward on the observation after the triggering button. Pass `--log-whiteouts` to both frozen evaluations to measure losses in the original game. For review footage around a known decision, use `python pixel_replay.py GAME_JSON OUTPUT --decision BUTTON_NUMBER --video`.

To measure how much an explicit visual memory can recover from the same games, run a separate self-imitation controller. It checks its expected next screen against the real image and retrieves a matching stored screen when possible. Using a different output directory and adding `--wait-every 257` tests recovery after input timing changes. Compare its resulting action tape to the training tapes to distinguish exact reproduction from new decisions.

```bash
python earned_memory.py --dataset EARNED_GAME_A --dataset EARNED_GAME_B \
  --model MEMORY --output MEMORY_FRESH --limit 65536
```

### Learning button durations and screen motion

The duration explorer estimates the reward and the next screen after repeating each button for 1, 2, 4, 8 or 16 decisions. Start both comparison arms from the same self-earned discovery and choose a separate output directory for each. The extra budget counts every newly simulated button decision across the branches. The run records the inherited exploration cost and the cost of replaying the starting history separately.

```bash
python option_explorer.py DURATION_RUN --from-search OWN_TRAINING_SEARCH --additional-steps 1000000
python fit_visual_motion.py VERIFIED_BEST_GAME MOTION_PROBE.json
python option_explorer.py MOTION_RUN --from-search OWN_TRAINING_SEARCH --additional-steps 1000000 \
  --motion-probe MOTION_PROBE.json
```

The motion fit estimates displacement from RGB frame pairs in the first three quarters of the training game. It reports accuracy and coverage on the final quarter. The motion explorer tests predicted displacements against new frames and favors less visited locations in its estimated coordinates. The probe must match the source discovery's action-tape hash. Use fresh games and equal budgets to compare downstream gameplay.

Add `--stage-reservoir 64` in a new motion run to test a different way to pick training saves. Half the starts then come from a uniform sample of visual appearances since the greatest number of rewards of at least one. The other starts use the existing return-ranked archive. The visual signatures use image color and texture. This comparison asks whether the constant reward for new cells makes the ordinary archive too concentrated on the end of its own history.

To test a neural proposal during this training search, add `--dreamer-proposal ADAPTATION/frozen-policy --proposal-parent PARENT_EVALUATION --upstream /opt/emerald-rl/source`. At half of option choices, the frozen Dreamer chooses one button. The ordinary explorer chooses the other options. The recurrent model observes every actual action and saves its state alongside each emulator fork. Policy sampling continues after a restore, allowing different attempts from the same state. Use `--uniform-proposal` with the same settings in a separate run for the matched control. A discovery made in either arm is a training result. Evaluate a frozen controller from boot to measure independent play.

Use `--systematic-options --stage-reservoir 128` to test different first actions from the same exact training save. This variant starts 90% of its branches from visual landmarks, considers a new landmark every 16 decisions, and gives each branch 16, 32 or 64 decisions. It picks a least-tested combination of button and duration for the first choice. Its ordinary learned explorer controls the rest of the branch. The run logs the count of newly tested state-action-duration combinations and the full real-emulator budget.

Add `--reward-landmarks` to offer a visual landmark only on a positive scalar reward. Large rewards still advance the reservoir to a new stage. A paired test uses the same seed, original training history, 128 slots and 262,144 additional controller decisions for both conditions. On the verified 36,058-button training history, periodic sampling offered 2,261 candidate states and reward-triggered sampling offered 376. Each reservoir retained 128. The controller receives the image and scalar reward at each step.

With `--recent-landmarks 32`, the same 128 slots keep the 32 most recently discovered distinct appearances plus a uniform reservoir from earlier ones at the same reward stage. Restarts still sample uniformly among stored landmarks. On the original 36,058 actions, the reward-triggered archive kept 23 of the newest 32 appearances. Reserving the 32 slots kept all 32. These checks measure coverage of training saves. Fresh games score whether the resulting experience teaches the policy to travel farther.

### Can Dreamer extend a route that memory can repeat?

Run two fresh boots with the same frozen visual memory. When both games reach the end of a stored sequence or make no progress through the known screens for 512 decisions, the trained Dreamer controls the first game. Uniform random buttons control the second. Dreamer receives its actual previous input even while the memory chooses the buttons. The run checks that the two games have identical images and rewards until the switch. Read `handoff.json` for the first decision made independently in each game.

```bash
python extend_earned_route.py --model MEMORY --checkpoint ADAPTATION/frozen-policy \
  --parent PARENT_EVALUATION --upstream /opt/emerald-rl/source --output EXTENSION --limit 65536
```
