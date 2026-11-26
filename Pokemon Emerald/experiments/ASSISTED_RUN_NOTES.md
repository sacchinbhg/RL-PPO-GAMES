# Stone Badge experiment

The target is a frozen policy beating Roxanne in Pokémon Emerald from a fresh original-ROM boot. Training branches may rewind. Evaluation uses ordinary controller buttons, with no restorations or game-memory writes.

## What the learner was given

The opening seed consists of 596 inputs discovered in the earlier mGBA training run. The current-map exploration policy can read the map's collision tiles, exits, objects and active interaction triggers. It branches on possible interactions and prioritizes named map and story milestones and party development. This uses human knowledge of which outcomes matter, including the first gym. It receives no fixed target coordinates or human controller inputs. The earlier waypoint-guided starter diagnostic is excluded. Once training finds a successful sequence, a separate neural policy learns to imitate it from screen pixels, game variables and its own button history.

The tests report three different outcomes: a saved training discovery, a frozen policy's new game, and an independent replay of that policy's actual button log. The tests also identify whether the policy used episodic action lookups.

## Notable moments

- **The starter survives the removal of the explorer.** The GPU-trained neural clone set the clock at decision 303, met the rival at 518, gained a party member at 711 and completed Birch's lab handoff at 884. The game started from boot and action lookups were disabled. It was trained on a successful self-discovered sequence and evaluated at the same deterministic starting condition. [Frozen neural run](starter-policy/neural-only/evaluation.json) · [Starter frame](starter-policy/neural-only/starter.png)
- **One extra job for the world model did not win the clock test.** On five held-out clock onsets amid 9,472 real transitions, both small recurrent predictors detected all five at a threshold of 0.5. The standard reward predictor raised 16 false alarms, and the model with the additional event loss raised 27. Their average precisions were 0.943 and 0.876. The models had one training seed, and the clock task has few held-out successes. These models are small Dreamer-inspired experiments. Their imagined actors did not generate the map-assisted discovery. [Paired predictor test](dream/pilot.json)
