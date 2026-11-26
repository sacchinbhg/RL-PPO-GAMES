# Pokémon Emerald agent handoff

**Stopped on November 26, 2026, at the user's request.** The autonomous agent reached Route 103. A self-generated training branch challenged May and lost. Petalburg Woods, a win over Roxanne, and the Stone Badge remain unverified. The Emerald WSL distro was stopped, the Windows restart task disabled, and the 15-minute Codex automation deleted. Resume when the user requests it.

## What the results establish

| Controller and setting | Last verified result |
| --- | --- |
| Frozen screen-memory policy, fresh game | Got a starter at decision 15,367, entered Oldale at 27,916 and Route 103 at 37,258. It reproduced 55,711 buttons from its own training history and played to 65,536 with zero in-game restores. This demonstrates repeatability on this opening. |
| Screen-based explorer, training forks | Reached May's fight on Route 103 at 40,073. An independent original-ROM emulator replayed all 40,548 buttons from boot and recorded the loss at 40,452. Torchic entered at 3/23 HP. Mudkip still had 20/20 when Torchic fainted. |
| Official DreamerV3, frozen after training with self-earned resets | All four fresh games set the clock and received the instruction to meet the rival. Zero of four got a starter in 65,536 decisions each. The matched boot-only checkpoint finished training. Its fresh evaluation was interrupted at 3,072 decisions per game. |
| Retry policy trained at positions preceding battle losses | At shutdown, 144 training attempts yielded seven battle wins, 12 losses and 125 attempts that reached the 768-decision limit. The opponents in the seven wins still need independent identification. The frozen and uniform comparison phases had zero completed results. |

The route to May came from autonomous search with emulator snapshots. The frozen approach to Route 103 came from imitating self-play. A faster Gym 1 win from Dreamer than from PPO remains an open question. [Original-ROM fight audit](results/route-103-first-rival-loss.json) · [Frozen policy result](results/handoff/frozen-memory-result.json) · [Shutdown counters, seeds and hashes](results/handoff/stop-snapshot.json) · [Retry results](results/handoff/retry-counts.json)

## Experiment contract

Success requires a frozen agent playing a fresh English Emerald revision 0 game, beating Roxanne and earning the Stone Badge with zero in-game restores. Preserve its weights and its entire action tape. Independently replay the tape on the original ROM and check both the trainer result and badge. Petalburg Woods means first entry into the named map. Its later story reward has a different trigger.

In the pixels-only experiments the learner sees RGB, scalar reward and episode signals. Each choice advances mGBA by 24 frames. The selected button is held for 14 frames and released for ten. Nine choices include a wait. The reward and audit code reads RAM. Policy selection must keep coordinates, collision maps, NPC identities, directions, demonstrations and evaluation tapes out of training. Training can restore a position the agent earned. Count the restored prefix separately from new actions. Always identify the controller, whether it trained or evaluated, and what was replayed.

## Which implementation to use

- **The original Windows baselines** implement PPO, archive exploration, self-imitation and a small action-conditioned world model. These earlier experiments include structured game observations. SelfMimic is the project's proposal to learn from the agent's own discoveries. SuperMimic supplied inspiration from humanoid control. The custom Pokémon implementation and the official Dreamer learner are separate experiments. [Baseline measurements](WORLD_MODEL_RESULTS.md) · [SuperMimic](https://github.com/hybridrobotics/supermimic)
- **The historical map-assisted experiment** gave an explorer collision tiles, exits, objects and interaction triggers. Its scripts include the `gym_*.py` family. Its deeper discoveries belong to the assisted category. The earlier guided starter diagnostic also has separate provenance. [Assisted run notes](experiments/ASSISTED_RUN_NOTES.md)
- **The official learner** uses pinned upstream DreamerV3 commit `e01491fad6434b2245a3b8ca201dd7faedcc458c`, JAX/CUDA in WSL2 and CPU mGBA workers. The GPU is an NVIDIA RTX 4060 Laptop with 8 GB VRAM. The 1- and 15-step conditions both use a world model. Check matched budgets and fresh games before attributing an improvement to imagination. The staggered screen-collection variant reached training step 524,304. Its screen-explorer workers and neural workers have different action sources. [Official adapter and setup](OFFICIAL_DREAMERV3.md) · [Upstream DreamerV3](https://github.com/danijar/dreamerv3)
- **The reward-only search and screen memory** produced the strongest clean gameplay result. The explorer follows scalar rewards and observed pixel transitions. The memory retrieves its own prior actions and checks the live frame against the predicted next frame. Keep the native-engine speed experiments separate from original-ROM verification. [Explorer](explore_reward_frontiers.py) · [Self-imitation policy](earned_memory.py) · [Native engine results](NATIVE_RESULTS.md)

## Why progress stalled

Two earlier hits reduced Torchic from 10 HP to 5 and then to 3 at decisions 38,652 and 38,792. The second hit came 1,281 decisions before May. Many retries started after that damage. In another fresh game, the memory controller repeated Select 25,626 times on a wild Wurmple encounter. These are two distinct failure modes. More retries from a wounded save cannot teach earlier survival, and recognizing a familiar screen does not guarantee the selected action advances the game. [Prebattle health trace](results/handoff/before-rival-damage.json) · [Training diary and footage index](YOUTUBE_NOTES.md)

Alongside the retry study, sustained-button experiments found zero winning options at 11 fully tested starts. The health-preserving route study completed three of four intervals and 65,989 alternative decisions without reproducing the target screen and milestones with more HP and equal or higher return. The new combat reward was verified on both a wild win and the rival loss. It rewards damage dealt, penalizes damage taken and pays for resolved battle outcomes. Its effect on fresh gameplay is unmeasured. [Sustained-button results](results/handoff/button-options.json) · [Health-preserving route results](results/handoff/healthier-routes.json) · [Combat reward check](results/combat-feedback-check.json)

## Files and checkpoints

The original workspace on this PC is `C:\Users\sacch\Documents\Codex\2026-11-24\new-chat`. In WSL it is `/mnt/c/Users/sacch/Documents/Codex/2026-11-24/new-chat`. The earlier Desktop copy is historical. The current run directory, abbreviated below as `RUN`, is `outputs/EmeraldRL/runs/pixels-only-01/official-wsl` under the original workspace. Weights, replay buffers, snapshots, detailed game logs and footage are local. The repository includes the authored source, native integration patches, experiment launchers, diagnostics, key evidence and two complete button tapes.

| Local path under `RUN` | What another agent needs |
| --- | --- |
| `earned-frontier-curriculum/stage-003948-c2da70ebe5f2/` | Frozen self-imitation model and `fresh-002` recording. |
| `earned-reset-online-comparison/{earned,boot}/training-result.json` | Exact checkpoint locations and SHA-256 values. Both started from identical weights and collected 32,774 real actions each. Training seed 71299, fresh evaluation seed 271348, imagination horizon 15. |
| `reward-only-training-forks/` and `paired-self-earned-frontiers/` | Self-generated discoveries and ancestry. The primary search recorded 1,386,112 search decisions. The first positive retry was extended into nine branches, totaling 4,608 new decisions. |
| `controls/all-losses-practice/` | Registered training cohort, trial tapes and partial counts. A positive battle outcome alone cannot identify the opponent. |
| `outcome-curriculum/` | New matched combat-feedback experiment. Its bank-building replay had checked 34,816 of 40,597 buttons when stopped. The 65,536-step-per-arm online comparison had yet to start. A visual approach model was fitted on 40,192 buttons, with 405 withheld. Its independent bank check and fresh evaluation were pending. |
| `battle-win-comparison/win-reward/fit/` | A separate offline official Dreamer fit completed 1,408 updates using a win bonus. Its fresh evaluation was pending. |
| `controls/endpoint-button-sweep/observed-interactions/original-from-boot/decision-40030.mp4` | Verified footage of May's challenge. The sibling `first-encounter-followup/` has battle stills, the HP trace and full replay audit. |

The historical launcher and diagnostic sources live in [the experiment archive](experiments/work). On another machine, `python experiments/materialize.py PATH_TO_WORKSPACE` copies the runtime into `outputs/EmeraldRL` and helpers into its sibling `work` directory. It only stages files. Older orchestration scripts include paths specific to this PC and depend on local run artifacts. Use the official setup guide for a new installation. Native probe intermediates come from their build scripts. The two SuperMimic learner reference files, mGBA, the game/toolchain sources, the ROM and installed runtimes are external dependencies.

## Reproduce the evidence

The published tapes contain one controller **action index** per byte. The adjacent JSON is the game audit. Hashes and lengths are recorded in the shutdown snapshot. On the original machine the WSL interpreter is `/opt/emerald-rl/venv/bin/python`. Supply mGBA and an Emerald ROM matching SHA-1 `f3ae088181bf583e55daf962a92bb46f4f1d07b7`. These commands are for a later session, from the runtime directory in the materialized workspace.

```bash
export EMERALD_CORE=/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so
export EMERALD_ROM=/path/to/your/verified/pokeemerald.gba
python pixel_replay.py results/handoff/route103-frozen.json replay-route103 --map Route103
python ../../work/diagnose-emerald-battles.py results/handoff/rival-training.json replay-rival.json
```

## Where to resume

1. Finish the boot-only evaluation using the existing checkpoints and `work/train-emerald-earned-resets.py`. It preserves the partial fresh game and starts another attempt. Compare four equal-length fresh games per checkpoint. Only the earned-reset side has a complete denominator today.
2. Use `work/diagnose-emerald-battles.py` on the seven positive training tapes. Record opponent and outcome from an independent full replay before claiming a rival win. The canonical verified loss provides a control. [Seven candidate recordings and checksums](results/handoff/practice-wins/index.json)
3. Finish `work/prepare-emerald-battle-curriculum.py` and the original-ROM checks in `work/check-emerald-combat-feedback.py`. Then run `work/train-emerald-battle-curriculum.py` for the matched official comparison. Both arms use the same weights, reset positions and step budget. Score their frozen combat probes separately from fresh play. The hybrid fresh-game experiment is `work/play-emerald-outcome-approach.py`.
4. Judge the next milestone by a verified win over May and whether a frozen policy can repeat it from boot. Then measure Petalburg Woods and ultimately the Stone Badge. Save the full episode, exact weights and verification artifacts at each first. A reliable comparison to PPO requires matched observations, rewards, action timing and fresh-start success rates.
