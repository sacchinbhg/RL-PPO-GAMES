"""Original-ROM training forks. The controller receives pixels, scalar return and opaque slot IDs."""
import copy
import json
import multiprocessing as mp
from pathlib import Path
import traceback


def _worker(pipe):
    from emerald import Emerald
    from pixel_env import Score, game_paths

    game = Emerald(*game_paths())
    try:
        game.load(game.boot)
        score = Score(uncapped_exploration=True)
        picture, tape, total, terminal = game.boot_image, bytearray(), 0, False
        slots = {}
        pipe.send(picture)
        while True:
            command, arg = pipe.recv()
            if command == "close":
                break
            if command == "step":
                button, duration = arg
                if terminal or not 0 <= button < 9 or not 1 <= duration <= 32:
                    raise ValueError("Invalid action option or completed game")
                reward = 0
                for used in range(1, duration + 1):
                    picture, descriptor = game.step(button)
                    amount, terminal = score.read(game, descriptor)
                    reward += round(100 * amount)
                    tape.append(button)
                    if reward or terminal:
                        break
                total += reward
                response = picture, reward, terminal, used
            elif command == "save":
                slots[arg] = game.save(), copy.deepcopy(score), picture.copy(), bytes(tape), total, terminal
                response = True
            elif command == "restore":
                blob, old_score, picture, history, total, terminal = slots[arg]
                game.load(blob)
                score, tape = copy.deepcopy(old_score), bytearray(history)
                response = picture
            elif command == "audit":
                target = Path(arg)
                target.mkdir(parents=True, exist_ok=True)
                (target / "game.bin").write_bytes(tape)
                report = dict(steps=len(tape), events=score.events, first_visit_cells=len(score.cells),
                              uncapped_exploration=True, return_cents=total,
                              source="self-generated training forks")
                (target / "game.json").write_text(json.dumps(report, indent=2))
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
    def __init__(self):
        ctx = mp.get_context("spawn")
        self.pipe, back = ctx.Pipe()
        self.process = ctx.Process(target=_worker, args=(back,), daemon=True)
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
