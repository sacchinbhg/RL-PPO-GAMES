"""Read completed battles by independently replaying a verified self-generated training episode."""
import argparse
from collections import Counter
from pathlib import Path

from emerald import Emerald, ROM_SHA1
from pixel_env import Score, game_paths
from train_earned_dreamer import Trajectory, write_json


def replay(dataset, output):
    history = Trajectory(dataset)
    game = Emerald(*game_paths())
    score = Score(history.audit["uncapped_exploration"])
    wins = []
    outcomes = Counter()
    active = zero_seen = recorded = False
    total = 0
    try:
        game.load(game.boot)
        for decision, button in enumerate(history.tape, 1):
            _, key = game.step(button)
            reward, _ = score.read(game, key)
            total += round(100 * reward)
            battle = key[0] == 6
            field = key[0] == 5 and game.verified_story(key) is not None
            if battle and not active:
                active, zero_seen, recorded = True, False, False
            if active and (battle or field):
                outcome = game.ram(0x0202433a, 1)[0]
                if outcome == 0 and battle:
                    zero_seen = True
                if zero_seen and outcome and not recorded:
                    outcomes[outcome] += 1
                    recorded = True
                    if outcome == 1:
                        wins.append(decision)
                if field:
                    active = False
        if (score.events != history.audit["events"] or score.first_maps != history.audit["maps_first_reached"]
                or len(score.cells) != history.audit["first_visit_cells"] or total != history.source["return_cents"]):
            raise ValueError("Original-ROM battle replay disagrees with training progress or rewards")
    finally:
        game.close()
    output.mkdir(parents=True, exist_ok=True)
    sha = history.source["source_sha256"]
    write_json(output / (sha + ".json"), dict(training_tape=sha, decisions=len(history.tape), rom_sha1=ROM_SHA1,
               in_game_restores=0, original_training_return_cents=total, battle_outcomes=outcomes,
               win_decisions=wins, frozen_evaluation_data_used=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.dataset, args.output)
