# Longer Emerald training and actor-gradient check

The revised actor set the bedroom clock in **13/16** fresh games at 65,536 real training decisions and **10/16** at 98,304. The original world update scored 3/16 and 4/16. Archive-PPO scored 0/16 and 3/16. The win at both later checkpoints repeated across two evaluation seeds. One training seed supplied this comparison. Archive-PPO had already reached 6/8 on its initial checkpoint, and performance varied as training continued. Establishing a convergence-speed multiplier requires more independent training seeds and a sustained success threshold.

The pilot extended seed 19 from 32,768 to 98,304 real decisions. The original archive-PPO and world algorithms continued from their own saved weights, optimizer and archive. A third branch started from the same world checkpoint and switched to full actor backpropagation with an independent Adam optimizer. All three used identical new process seeds (17019 and 27019) for the two extension blocks, eight workers and the original three-step model horizon. Both extension boundaries began new worker episodes. The world learner also carried its real replay and model optimizer between blocks.

| Total real decisions | Learner | Fresh clock / 8 | Fresh starter / 8 | Fresh badge / 8 | Cumulative training wall time |
|---:|---|---:|---:|---:|---:|
| 32,768 | Archive-PPO | 6 | 0 | 0 | 2.14 min |
| 32,768 | World model, original update | 3 | 0 | 0 | 2.51 min |
| 65,536 | Archive-PPO | 0 | 0 | 0 | 4.64 min |
| 65,536 | World model, original update | 2 | 0 | 0 | 5.39 min |
| 65,536 | World model, full actor gradient | 6 | 0 | 0 | 5.58 min |
| 98,304 | Archive-PPO | 1 | 0 | 0 | 7.09 min |
| 98,304 | World model, original update | 2 | 0 | 0 | 8.45 min |
| 98,304 | World model, full actor gradient | 5 | 0 | 0 | 8.62 min |

The denominator is eight fresh game boots at each point. Evaluation used the same seed and 2,048 decisions per boot throughout. Each group of eight shares one set of trained weights. Separate seeds 37 and 71 were evaluated only at the initial training budget. Wall time includes training startup and checkpoint writes and excludes evaluation. Cumulative wall time for both world branches includes their shared initial training. The observed milestone is the bedroom clock. A starter and a badge are also tracked.

![Learning curves](learning-curve.png)

## What the controller learns from imagination

The stored-checkpoint control zeroed the imagined gradient and measured how far Adam still moved the actor. The live control repeats that comparison immediately after a real PPO update, using the actual rollout at the final update of each measured block. A cosine of one means the two parameter changes point in the same direction. The relative difference measures how far apart those changes are, divided by the size of the normal update.

| Cumulative real decisions | Size of zero-gradient update / normal | Direction cosine | Relative difference |
|---:|---:|---:|---:|
| 65,536 | 99.98% | 0.99997 | 0.81% |
| 98,304 | 99.61% | 0.99991 | 1.41% |

The new branch gives direct credit to all three predicted actions. It also switches the optimizer and its learning rate and moments, so the comparison tests those changes together. The critic bootstrap remains frozen during the imagined update. The predictor sees an 8 × 12 RGB summary, current game variables, button and goal. The episode's visited cells, best prior progress and previous goal hits affect the training reward but are hidden from that predictor. Its estimate of those rewards must average over indistinguishable histories. The clock requires many decisions and the reward is sparse. A correct one-step movement prediction alone does not teach that sequence.

The [short-run report](WORLD_MODEL_RESULTS.md) covers the three independent seeds, model prediction, the exact archived-goal test and replay verification. The [YouTube diary](YOUTUBE_NOTES.md) links to native frames and controller recordings.

The Emerald pilot used 128 starts × 3 imagined buttons per 512 real decisions after warmup. SuperMimic's published learner configuration defaults to 4,096 starts and 32 model steps, with a dedicated policy optimizer and actor gradients at every step. Those are software defaults for a continuous-control robotics system. Predicting nine discrete buttons, sparse story events and emulator state from a small screen entails a different prediction problem. Each of the Emerald pilot's arms still collected the same full real PPO rollout. Any elapsed-time advantage has to come from reaching the target in fewer such rollouts. [Original SuperMimic learner configuration](https://github.com/HybridRobotics/supermimic/blob/5fb92297ab4fe3d86629d9f0c635334791bb0d5f/src/supermimic/learners/policy_learner.py)

Both world branches began from identical checkpoint weights. The first 512-decision rollout produced matching logged PPO loss (-0.02665) and predictor loss (0.120452). At this update, the mean policy KL was 0.00001703 for the original imagined update and 0.00032561 for the full-gradient update. This measures how much the update changes action probabilities. Its gameplay value is measured by the fresh-boot trials.

## Replication on a second evaluation seed

The six saved extension policies each played eight additional games with evaluation seed 92741. These test games used the same 2,048-decision limit, zero restores and an empty goal input. The two evaluation batches provide 16 trials per frozen policy, all from the same training seed.

| Real training decisions | Learner | Clocks on original 8 | Clocks on new 8 | Combined / 16 | Combined starters / 16 |
|---:|---|---:|---:|---:|---:|
| 65,536 | Archive-PPO | 0 | 0 | 0 | 0 |
| 65,536 | World model, original update | 2 | 1 | 3 | 0 |
| 65,536 | World model, full actor gradient | 6 | 7 | 13 | 0 |
| 98,304 | Archive-PPO | 1 | 2 | 3 | 0 |
| 98,304 | World model, original update | 2 | 2 | 4 | 0 |
| 98,304 | World model, full actor gradient | 5 | 5 | 10 | 0 |

## Identical inputs with different rewards

The 32,768-row replay buffers contained exactly identical encoded state, goal and button combinations associated with more than one real reward. Partial observability and differences in episode history can both contribute. These counts are a lower bound on ambiguity in the model's stored inputs.

- world-19 contained 61 such inputs, accounting for 231 rows and 100 of 657 positive reward rows.
- world-37 contained 67 such inputs, accounting for 244 rows and 111 of 674 positive reward rows.
- world-71 contained 72 such inputs, accounting for 241 rows and 102 of 672 positive reward rows.
