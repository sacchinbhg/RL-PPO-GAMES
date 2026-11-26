# Emerald pilot results

Two 16,384-decision runs trained a 55,434-parameter policy on the RTX 4060 Laptop GPU. Each used seed 19, four independent GBA processes, 24 frames per controller decision and a 512-decision training episode.

| Training | Distinct descriptor cells | Maps observed | Most badges |
|---|---:|---|---:|
| Archive-conditioned PPO | 69 | InsideOfTruck, LittlerootTown, LittlerootTown_BrendansHouse_1F, LittlerootTown_BrendansHouse_2F | 0 |
| Fresh-start PPO | 33 | InsideOfTruck, LittlerootTown, LittlerootTown_BrendansHouse_1F, LittlerootTown_MaysHouse_1F | 0 |

The archive recorded 74 shorter paths to known cells and restored checkpoints 15 times. It recorded 6 hits from 19 assigned endpoints. Some assignments can still be active at the cutoff. A shorter stored path can arise from exploration alone.

## Independent fresh-game trials

Each controller received four fresh boots, 2,048 decisions per boot and zero checkpoint interventions. The two trained policies were frozen. The random controller sampled buttons uniformly. The trial counts below describe one training seed. The original 1,024-decision trials are preserved in the run directories ending in `-eval`. The longer follow-up gives each controller enough decisions to cover the length of the archived clock route.

| Controller | Trials | Reached truck | Reached Littleroot outdoors | Set clock | Got starter | Most badges |
|---|---:|---:|---:|---:|---:|---:|
| Archive-conditioned PPO | 4 | 4 | 4 | 0 | 0 | 0 |
| Plain PPO | 4 | 4 | 4 | 0 | 0 | 0 |
| Uniform random | 4 | 4 | 4 | 0 | 0 | 0 |

## Compute and Red comparison

The PC has an Intel Core i7-13650HX (14 cores, 20 logical processors), about 16 GB of system RAM and an RTX 4060 Laptop GPU with 8,188 MiB of dedicated memory. PyTorch used CUDA 12.8. Peak allocated PyTorch tensor memory during the archive run was 33.2 MiB. The driver, display and CUDA context also use memory.

The archive run took 130.9 seconds end to end (125.2 decisions/s). Inside the training loop it sustained 166.9 decisions/s. Collection, including action inference and worker communication, took 91.26 seconds. PPO updates took 6.89 seconds. Estimates below assume the active rate holds during longer training and exclude startup, periodic writes and pauses.

| Interaction budget | Estimated continuous training time |
|---|---:|
| 1,000,000 decisions | 1.7 hours |
| 10,000,000 decisions | 16.6 hours |
| 100,000,000 decisions | 6.9 days |
| 252,000,000 decisions | 17.5 days |

Rubinstein's team reports seven hours for a tuned scripted Red run and a separate peak of 10,000 decisions/s. The 252 million row is the product of these two figures. The team's reported peak gives an illustrative upper-end workload for that time interval. The winning run's average throughput and exact decision count are unspecified in that account. Emerald introduces different exploration and learning demands. A time to the Hall of Fame requires its own milestone and success data. [Red training report](https://drubinstein.github.io/pokerl/docs/chapter-3/running/) · [Red results](https://drubinstein.github.io/pokerl/docs/chapter-4/results/)

A second worker-count check used the first 8,192 decisions in each configuration. Four workers averaged 169.6 active decisions/s. Eight workers averaged 283.5. At the eight-worker rate, one million decisions would take 0.98 hours and 252 million would take 10.3 days. Different numbers of workers also change which states are visited, so this comparison measures practical throughput.

## Archive replay

The highest-progress archived endpoint was replayed from boot using 1,207 recorded controller decisions. The recorded destination was LittlerootTown_BrendansHouse_2F. The replay reached LittlerootTown_BrendansHouse_2F and matched the stored descriptor. The image gallery in `runs/archive/replayed-route.png` contains emulator frames collected during this replay. The independent trials above used the policy's own decisions and had 2,048 decisions to act.

The next milestone is repeatable clock setting and starter acquisition from boot across independent training seeds. Rock Badge evaluation follows once the agent can reach Rustboro.

## Desktop RTX 4090 estimate

The eight-worker timing run collected 16,384 decisions in 60.75 seconds of active training (269.7 decisions/s). It spent 43.84 seconds stepping workers and moving their results, 7.77 seconds preparing observations, running the actor and retrieving actions, and 5.37 seconds updating PPO. The remaining time covered archive, reward and rollout bookkeeping. The model used 36.8 MiB of peak allocated tensor memory.

Policy calls and updates together occupied 21.6% of active time. If a desktop RTX 4090 made both of those sections 2× faster, the projected end-to-end speedup would be 1.12×. At a 4× improvement in both sections, it would be 1.19×. Their zero-time ceiling with all other sections held fixed is 1.28×. These are scenarios based on timings from the laptop. Both sections include host-side work.

A desktop RTX 4090 has 24 GB of VRAM. A larger world model, bigger batch or many more CPU emulator workers could change this profile. On a rental, run the same training workload across several worker counts and compare dollars per million real decisions. At an observed 1,000 decisions/s, 100 million decisions would take 27.8 hours. At 2,000 decisions/s, the same budget would take 13.9 hours. Those rates are rental selection targets. [NVIDIA RTX 4090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)
