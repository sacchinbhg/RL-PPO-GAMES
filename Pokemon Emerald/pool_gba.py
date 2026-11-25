"""Process isolation keeps each libretro core's global state independent."""

import multiprocessing as mp
import traceback

from emerald import Emerald


def worker(pipe, core, rom, backend):
    env = None
    try:
        if backend == "native":
            from native_emerald import NativeEmerald
            env = NativeEmerald(core, rom)
        else:
            env = Emerald(core, rom)
        pipe.send((env.boot_image, env.descriptor()))
        while True:
            cmd, arg = pipe.recv()
            if cmd == "close":
                break
            if cmd == "step":
                reply = env.step(arg)
            elif cmd == "step_verified":
                observation = env.step(arg)
                reply = (observation, env.verified_story(observation[1]))
            elif cmd == "save":
                reply = env.save()
            elif cmd == "frame":
                reply = env.screen.copy()
            elif cmd == "inspect" and backend == "native":
                reply = env.inspect()
            elif cmd == "load":
                env.load(env.boot if arg is None else arg)
                reply = env.descriptor()
            else:
                raise ValueError(cmd)
            pipe.send(reply)
    except EOFError:
        pass
    except Exception:
        pipe.send(traceback.format_exc())
    finally:
        if env is not None:
            env.close()
        pipe.close()


class Pool:
    def __init__(self, count, core, rom, backend="mgba"):
        ctx = mp.get_context("spawn")
        self.pipes, self.workers = [], []
        try:
            for _ in range(count):
                front, back = ctx.Pipe()
                proc = ctx.Process(target=worker, args=(back, core, rom, backend), daemon=True)
                proc.start()
                back.close()
                self.pipes.append(front)
                self.workers.append(proc)
            self.initial = [self.recv(i) for i in range(count)]
        except BaseException:
            self.close()
            raise

    def recv(self, i):
        if not self.pipes[i].poll(45):
            raise TimeoutError(f"Emulator {i} timed out")
        reply = self.pipes[i].recv()
        if isinstance(reply, str):
            raise RuntimeError(reply)
        return reply

    def request(self, i, cmd, arg=None):
        self.pipes[i].send((cmd, arg))
        return self.recv(i)

    def step(self, actions, verified=False):
        for pipe, act in zip(self.pipes, actions):
            pipe.send(("step_verified" if verified else "step", int(act)))
        return [self.recv(i) for i in range(len(actions))]

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
            pipe.close()
        for proc in self.workers:
            proc.join(2)
            if proc.is_alive():
                proc.terminate()
                proc.join()
