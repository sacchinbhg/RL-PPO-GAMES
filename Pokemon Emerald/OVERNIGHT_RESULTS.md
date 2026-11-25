# Emerald overnight report

Updated Nov 25 09:26 AM Pacific Standard Time.

Native pipeline: **Completed** as of Nov 25 09:26 AM Pacific Standard Time.

## Original mGBA run

**Completed** as of Nov 25 09:08 AM Pacific Standard Time. 12,373,504 total real decisions, including 12,307,968 in this continuation. The latest logged rate is 285.2 real decisions/s. The world model records 9,278,976 imagined transitions and 773,344 dynamics updates, including inherited work.

Frozen policies played from a fresh boot. Counts are independent games that reached each event at least once.

| Checkpoint | Decisions | Games | Clock | Met rival | Route 101 | Actual party |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 65,536 | 4 | 3 | — | — | — |
| step-855552 | 855,552 | 4 | 2 | 1 | 0 | 0 |
| step-1783808 | 1,783,808 | 4 | 0 | — | — | — |
| step-2781696 | 2,781,696 | 4 | 0 | 0 | 0 | 0 |
| step-3857920 | 3,857,920 | 4 | 0 | 0 | 0 | 0 |
| step-4933632 | 4,933,632 | 4 | 0 | 0 | 0 | 0 |
| step-6009856 | 6,009,856 | 4 | 1 | 1 | 0 | 0 |
| step-7074304 | 7,074,304 | 4 | 0 | 0 | 0 | 0 |
| step-8136704 | 8,136,704 | 4 | 0 | 0 | 0 | 0 |
| step-9171456 | 9,171,456 | 4 | 1 | 1 | 0 | 0 |
| step-10230272 | 10,230,272 | 4 | 0 | 0 | 0 | 0 |
| step-11298304 | 11,298,304 | 4 | 0 | 0 | 0 | 0 |
| final | 12,373,504 | 8 | 0 | 0 | 0 | 0 |

A dash means that checkpoint used an older evaluator and the event lacks a verified count.

Verified moments from the updated evaluator:

- step-6009856, fresh game 1: met the rival at decision 785.

## Native Emerald run

**Completed** as of Nov 25 09:16 AM Pacific Standard Time. 27,060,736 total real decisions, including 27,044,352 in this continuation. The latest logged rate is 764.2 real decisions/s. The world model records 20,294,400 imagined transitions and 1,691,296 dynamics updates, including inherited work.

Frozen policies played from a fresh boot. Counts are independent games that reached each event at least once.

| Checkpoint | Decisions | Games | Clock | Met rival | Route 101 | Actual party |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 16,384 | 4 | 4 | 0 | 0 | 0 |
| step-2707456 | 2,707,456 | 4 | 4 | 0 | 0 | 0 |
| step-5426176 | 5,426,176 | 4 | 4 | 0 | 0 | 0 |
| step-8210432 | 8,210,432 | 4 | 4 | 0 | 0 | 0 |
| step-10986496 | 10,986,496 | 4 | 4 | 0 | 0 | 0 |
| step-13731328 | 13,731,328 | 4 | 0 | 0 | 0 | 0 |
| step-16464896 | 16,464,896 | 4 | 0 | 0 | 0 | 0 |
| step-19048448 | 19,048,448 | 4 | 0 | 0 | 0 | 0 |
| step-21804544 | 21,804,544 | 4 | 0 | 0 | 0 | 0 |
| step-24582656 | 24,582,656 | 4 | 0 | 0 | 0 | 0 |
| final | 27,060,736 | 8 | 0 | 0 | 0 | 0 |

A dash means that checkpoint used an older evaluator and the event lacks a verified count.

## Backend and transfer checks

- native: 11,837.5 decisions/s across 2 timed sample(s), eight workers, with RGB and state observations. Median route restore 0.2174 s.
- mgba: 368.6 decisions/s across 2 timed sample(s), eight workers, with RGB and state observations. Median route restore 0.0022 s.

The upstream scripted starter sequence checks compatibility. Its inputs are excluded from policy training. Both backends share the laptop with the original training run during these measurements.

The native worker benchmark delivered 32.1× the real-game decisions per second. It times game stepping, observation extraction, and worker communication; model updates are timed in the trainer.

All six upstream framebuffer hashes matched over two deterministic benchmark passes. The native policy observations also matched the upstream browser engine at the clock, rival’s house, and Mudkip battle.

The native pilot trained for 16,384 real decisions at 532.3 decisions/s with the complete learner. It ran 1,024 world-model updates, generated 11,136 imagined transitions, and restored 15 archived routes. It inherited the original 65,536-decision policy weights and began with a fresh native archive and dynamics model.

The 16,384-decision native pilot also played 4 independent games in the original mGBA ROM. Clock 3/4, rival 0/4, Route 101 0/4, occupied party 0/4.

The frozen native-trained policy tested in the original ROM reached the clock in 0/8, the rival in 0/8, Route 101 in 0/8, and an occupied party in 0/8 fresh games.

## What to do next

Compare fresh-game milestones at matched real-decision budgets. Use the faster verified backend for algorithm tests, and evaluate its best frozen policy from boot on the original mGBA ROM. If reaching the rival stays rare across seeds, test one general doorway or NPC interaction bonus against the same archive/world learner.

The policy can discover routes during archive-assisted training before it can repeat them independently. Use confirmed fresh-game dialogue, Route 101, and occupied-party events for video milestones. The original raw room log includes transient map IDs during loading.
