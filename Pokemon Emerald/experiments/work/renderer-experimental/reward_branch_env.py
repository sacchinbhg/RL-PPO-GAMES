"""Original-ROM training forks. The controller receives pixels, scalar return and opaque slot IDs."""
import copy
import json
import multiprocessing as mp
from pathlib import Path
import traceback


class VisitReward:
    """Pay a location's existing exploration reward once across this training worker's branches."""
    def __init__(self):
        self.paid = set()

    def adjust(self, cents, location, new_in_game):
        if not new_in_game:
            return cents
        if location in self.paid:
            return cents - 2
        self.paid.add(location)
        return cents


def _worker(pipe, map_audits=None, log_whiteouts=False, shared_visit_reward=False, draw_decision_frames_only=False,
            trainer_audits=None):
    from emerald import Emerald
    from pixel_env import Score, WhiteoutLog, first_map_audit, game_paths

    game = Emerald(*game_paths(), draw_decision_frames_only=draw_decision_frames_only)
    try:
        game.load(game.boot)
        score = Score(uncapped_exploration=True)
        visit_reward = VisitReward() if shared_visit_reward else None
        picture, tape, total, terminal = game.boot_image, bytearray(), 0, False
        slots = {}
        losses=WhiteoutLog() if log_whiteouts else None
        if map_audits:
            map_audits.mkdir(parents=True, exist_ok=True)
        saved_maps = set()
        saved_trainers = set()
        def audit(target, cents):
            target.mkdir(parents=True, exist_ok=True)
            (target / "game.bin").write_bytes(tape)
            report = dict(steps=len(tape), events=score.events, first_visit_cells=len(score.cells),
                          maps_first_reached=score.first_maps, uncapped_exploration=True,
                          return_cents=cents, source="self-generated training forks")
            if losses is not None:
                report["whiteouts"]=losses.events
            if visit_reward is not None:
                report["distinct_cells_across_training"]=len(visit_reward.paid)
            if draw_decision_frames_only:
                report["frameskip_option_queries"]=game.frameskip_queries
            (target / "game.json").write_text(json.dumps(report, indent=2))
        pipe.send(picture)
        while True:
            command, arg = pipe.recv()
            if command == "close":
                break
            if command == "step":
                button, duration = arg
                if terminal or not 0 <= button < 9 or not 1 <= duration <= 32:
                    raise ValueError("Invalid action option or completed game")
                reward = original_reward = 0
                for used in range(1, duration + 1):
                    picture, descriptor = game.step(button)
                    n_maps = len(score.first_maps)
                    cells_before = len(score.cells)
                    amount, terminal = score.read(game, descriptor)
                    cents = round(100 * amount)
                    original_reward += cents
                    if visit_reward is not None:
                        cents = visit_reward.adjust(cents, descriptor[1:5], len(score.cells) > cells_before)
                    reward += cents
                    tape.append(button)
                    if losses is not None:
                        losses.observe(game,len(tape))
                    if map_audits and len(score.first_maps) != n_maps:
                        name = next(reversed(score.first_maps))
                        if name not in saved_maps:
                            first_map_audit(map_audits, name, lambda target: audit(target,total+original_reward))
                            saved_maps.add(name)
                    if trainer_audits is not None and descriptor[0] == 6:
                        battle_type = int.from_bytes(game.ram(0x02022fec, 4), "little")
                        if battle_type & 8:
                            opponent = int.from_bytes(game.ram(0x02038bca, 2), "little")
                            if opponent not in saved_trainers:
                                first_map_audit(trainer_audits, f"trainer-{opponent:05d}", lambda target: audit(target,total+original_reward))
                                saved_trainers.add(opponent)
                    if reward or terminal:
                        break
                total += original_reward
                response = picture, reward, terminal, used
            elif command == "save":
                slots[arg] = game.save(), copy.deepcopy(score), picture.copy(), bytes(tape), total, terminal, copy.deepcopy(losses)
                response = True
            elif command == "restore":
                blob, old_score, picture, history, total, terminal, old_losses = slots[arg]
                game.load(blob)
                score, tape = copy.deepcopy(old_score), bytearray(history)
                losses=copy.deepcopy(old_losses)
                response = picture
            elif command == "audit":
                target = Path(arg)
                audit(target,total)
                response = True
            else:
                raise ValueError(command)
            pipe.send(response)
    except BaseException:
        pipe.send({"error": traceback.format_exc()})
    finally:
        game.close()
        pipe.close()


class ForkEnv:
    def __init__(self, map_audits=None, log_whiteouts=False, shared_visit_reward=False, draw_decision_frames_only=False,
                 trainer_audits=None):
        ctx = mp.get_context("spawn")
        self.pipe, back = ctx.Pipe()
        self.process = ctx.Process(target=_worker, args=(back,map_audits,log_whiteouts,shared_visit_reward,draw_decision_frames_only,
                                                       trainer_audits), daemon=True)
        self.process.start()
        back.close()
        self.initial = self.receive()

    def receive(self):
        if not self.pipe.poll(90):
            raise TimeoutError("No response from the training fork emulator")
        value = self.pipe.recv()
        if isinstance(value, dict):
            raise RuntimeError(value["error"])
        return value

    def command(self, name, arg):
        self.pipe.send((name, arg))
        return self.receive()

    def close(self):
        try:
            if self.process.is_alive():
                self.pipe.send(("close", None))
        finally:
            self.pipe.close()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()
