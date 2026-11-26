# Native Emerald on the RTX 4060 laptop

The Windows native engine passed two independent checks before training. It matched all six upstream framebuffer hashes in each of two benchmark passes. It also reproduced the upstream browser engine’s RGB pixels at the clock, outside the rival’s house, and during the Mudkip battle. Replaying the starter recording confirmed the rival story flag, Route 101, and a one-Pokémon party. The recording was used only for verification.

## Timing

| Eight-worker test | Native Emerald | mGBA |
|---|---:|---:|
| Median real decisions per second | 11,838 | 369 |
| Median time to restore a 512-decision branch in a second worker | 217 ms | 2.2 ms |

Each sample used 4,096 controller decisions, 24 game frames per decision, a 60 × 40 RGB observation, and game-state extraction. The order was native, mGBA, mGBA, native using the same early-game inputs. The original training run and a local build shared the laptop during this measurement. All eight subsequent observations and game descriptors matched between the restored worker and the source worker. Native saves use compact button histories, so restoring a longer route takes more time.

The **complete native world-model pilot averaged 532.3 real decisions per second** over 16,384 decisions. It performed 1,024 dynamics updates, generated 11,136 imagined transitions, and restored 15 archived routes. Training used eight workers and the RTX 4060 Laptop GPU. The policy weights came from the 65,536-decision mGBA checkpoint. The native archive, dynamics model, and optimizer state started fresh. The 32.1× environment throughput result and the pilot training rate measure different workloads.

## Overnight experiment

Two supervisors run the original mGBA experiment and the native continuation. Each takes recovery checkpoints and evaluates frozen policies in fresh games. After the final native evaluation, the runner evaluates those same weights from boot in the original mGBA ROM for eight games of 32,768 decisions. The morning report tracks independent gameplay milestones and the transfer result.

The first transfer test used the frozen 16,384-decision pilot policy. Four fresh native games and four fresh mGBA games used the same sampling seed and 8,192 decisions per game. The policy set the clock in four native games and three mGBA games. Two mGBA games also reached Professor Birch’s lab. Neither group met the rival, reached Route 101, or acquired a party Pokémon. This checkpoint inherited policy weights that had already set the clock on mGBA. These trials establish which behaviors carried back to the original ROM. A comparison of learning speed requires matched training runs. [Paired trial results](results/native-pilot-transfer.json)

The Windows port fixes two problems revealed by the replay. The first failure occurred while reading an erased flash save. The next build could set the rival and party flags and draw the player, while the overworld background remained black. A valid background tilemap lived outside the address range used by the original game’s renderer. The corrected build draws the same scene as the reference.

[Worker benchmark](results/native-backend-check.json) · [Native pilot](results/native-pilot.json) · [Visual checks](results/native-scripted-replay.json) · [Upstream framebuffer verification](results/native-upstream-check.json)

### Restoring long self-generated input histories

`NativeEmerald.load(state, render_intermediate=False)` advances fourteen frames with each button held, releases it for ten frames, and renders the final restored screen. Gameplay after the restore uses ordinary rendering. Intermediate rendering remains enabled by default.

We compared five prefixes of two player input histories ranging from 4,096 to 44,393 buttons. After each restore we checked the native framebuffer and all thirteen audit values, then checked every observation through 256 recorded buttons and 128 independently sampled buttons. All 1,925 comparisons matched. The five restores used 18.84 CPU seconds with ordinary rendering and 2.83 CPU seconds with final-screen rendering. This measures the time to restore an archive on this native build. [Per-prefix times and replay hashes](results/native-restore-rendering.json)
