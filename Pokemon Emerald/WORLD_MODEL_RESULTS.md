# Emerald world-model experiment

Each seed trained from scratch for 32,768 real controller decisions on the RTX 4060 Laptop GPU. The primary outcome was setting the bedroom clock from a fresh game within 2,048 decisions. The paired PPO control used the same pixel policy, archive rules, rewards, episode length, worker count and optimizer. World runs added a learned action-conditioned dynamics and reward model and an actor update through three imagined button presses. Training used eight mGBA processes, 64 decisions per rollout and 512 per training episode.

## Real game results

| Training seed | Learner | First clock during training, aggregate real decisions | Independent clock successes / 8 | Starter successes / 8 | Training wall time | Active decisions/s |
|---:|---|---:|---:|---:|---:|---:|
| 19 | Archive-PPO | 22265 | 6 | 0 | 128.6 s | 272.5 |
| 19 | World model + archive-PPO | 30414 | 3 | 0 | 150.8 s | 230.8 |
| 37 | Archive-PPO | 14687 | 1 | 0 | 133.2 s | 263.6 |
| 37 | World model + archive-PPO | 23110 | 1 | 0 | 135.6 s | 256.9 |
| 71 | Archive-PPO | 23139 | 4 | 0 | 142.2 s | 243.9 |
| 71 | World model + archive-PPO | 23158 | 6 | 0 | 150.5 s | 230.8 |

The world-assisted policies set the clock in **10/24** fresh boots. Archive-PPO achieved **11/24**. The uniform random controller achieved **1/8** using the same evaluation budget. The world-assisted policies obtained a starter in 0/24 trials and archive-PPO in 0/24. Trial outcomes within a training seed share the same policy. The experiment has three independent training seeds per arm.

The two arms each used 98,304 real decisions in total. The world-assisted arm also used 70,272 imagined transitions for policy improvement. Mean elapsed time per training job was 145.6 seconds with the world model and 134.7 seconds for archive-PPO. Timing includes startup and checkpoint writes. One computer ran the jobs sequentially, controls first. The first-clock training metric includes archive restores. Frozen-policy evaluations always started at boot with an empty goal and no restores. The experiment measures success at a fixed training budget. A reliable time-to-proficiency curve requires fresh-game evaluations at multiple training budgets.

## Predictions on independent gameplay

Eight new games used uniformly random buttons and seed 82631. The audit collected 12,288 real transitions and replayed all nine buttons from 192 saved positions. All 192 sampled continuation choices reproduced their corresponding fork outcomes. These data were excluded from model fitting and policy updates. The persistence baseline copies the last observed state. Lower prediction error is better.

| Seed | 1-step RGB RMSE / persistence | 3-step RGB RMSE / persistence | 1-step RAM RMSE / persistence | Action-specific RGB variance explained | Goal-directed button success / uniform expectation |
|---:|---:|---:|---:|---:|---:|
| 19 | 0.1034 / 0.1588 | 0.1664 / 0.2350 | 0.0302 / 0.0419 | 49.9% | 33/35 / 11.4% |
| 37 | 0.1071 / 0.1588 | 0.1744 / 0.2350 | 0.0310 / 0.0419 | 49.1% | 25/35 / 11.4% |
| 71 | 0.1036 / 0.1588 | 0.1665 / 0.2350 | 0.0312 / 0.0419 | 46.8% | 31/35 / 11.4% |

The button-effect metric subtracts each state's mean outcome across buttons. A model predicting the same outcome for every button scores zero. Negative values indicate that its predicted differences among buttons increase squared error. The local goal test selects one action using the model to reach a different attainable coarse tile on the same map. Each target was scored against the emulator's actual response to all nine buttons. The raw audit also includes reward errors, three-step RAM errors and a rotated-button control.

## Verification and scope

- The six component tests cover action encoding and gradients, zero actor gradient when the model ignores actions, simulator transition storage, ring wraparound, boundary-safe sequence sampling, goal reward and archive replacement, checkpoint round trip, and frozen critic/model weights during imagination.
- From frame one, replaying 826 archived inputs from the world-model seed 19 run reached LittlerootTown_BrendansHouse_2F. The final descriptor matched the stored endpoint. Fresh-game policy trials selected their own buttons.
- The runner reads the original English Emerald game flag for setting the clock. The game sets it inside `InitTimeBasedEvents`. The world model predicts coarse RGB and structured game variables. All archive entries and reported gameplay milestones come from the real emulator. [Emerald clock initialization](https://github.com/pret/pokeemerald/blob/8cf518839e529eeb2477905ede1df7213ac134b8/src/clock.c)
- The actor objective follows SuperMimic's short differentiable rollout and value bootstrap, adapted to discrete GBA controls. The game dynamics, reward model and training code here were written for Emerald. The ongoing PPO update anchors the actor on real observations. [SuperMimic policy learner](https://github.com/HybridRobotics/supermimic/blob/5fb92297ab4fe3d86629d9f0c635334791bb0d5f/src/supermimic/learners/policy_learner.py)

Full per-trial records, fixed-budget timings, checkpoints and prediction diagnostics are under `runs/world-comparison`. Use the commands in the README to rerun training or the independent audit.

## Can the policy follow a named destination?

The probe chose distinct overworld destinations whose archived path extended a same-map start by 3–24 recorded decisions. Each controller received the same emulator state, 96 decisions and the same uniform random numbers for selecting buttons. The goal-conditioned and empty-goal arms used identical frozen weights. The destinations came from training archives. This is a test of whether the supplied goal helps control, with one draw per destination.

| Training seed | Reachable targets | Supplied goal | Empty goal | Uniform buttons |
|---:|---:|---:|---:|---:|
| 19 | 8 | 3 | 4 | 1 |
| 37 | 8 | 5 | 5 | 3 |
| 71 | 8 | 4 | 4 | 2 |

The stored suffix reproduced every chosen destination at its recorded decision. [Per-target evidence](runs/world-comparison/goal-probe.json)

## Why the first actor update can be weak

The first implementation computed a direct actor gradient only for the first imagined action. Later policy calls had detached weights, although state derivatives flowed through the three-step forecast. The update also reused the Adam optimizer just used for real PPO. We isolated its moving-average effect on three saved policies. Each pair used identical weights, optimizer state, 128 sampled archive observations, model and action randomness. In the control, hooks zeroed the current imagined gradients before Adam stepped.

| Seed | Size of zero-gradient update / normal update | Cosine of both parameter updates | Relative change when the current gradient was removed |
|---:|---:|---:|---:|
| 19 | 100.0% | 0.9984 | 5.7% |
| 37 | 100.0% | 1.0000 | 0.4% |
| 71 | 100.1% | 0.9997 | 2.4% |

These measurements compare parameter displacements on selected archive states and use the PPO optimizer state saved after the checkpoint's final imagined update. They do not apportion the causes of an entire run. A second implementation uses separate Adam moments and backpropagates through every imagined action. A seventh component test checks all three action gradients, the independent optimizer and restoring its history. [Raw optimizer control](runs/world-comparison/gradient-control.json) · [SuperMimic actor training](https://github.com/HybridRobotics/supermimic/blob/5fb92297ab4fe3d86629d9f0c635334791bb0d5f/src/supermimic/learners/policy_learner.py)

A complete checkpoint was extended from 32,768 to 33,280 real decisions. The learner continued to 2,080 predictor updates and 23,808 imagined transitions. [Restart record](runs/world-comparison/restart-check.json)

Only 226 of 12,288 independent real decisions had a positive shaped reward. Reward RMSE ranged from 0.0577 to 0.0623, versus approximately 0.0654 when always predicting the training reward mean. The held-out movement test provides much stronger evidence of usable local dynamics than this aggregate metric provides about recognizing rare milestones.
