"""Exercise a real rewarded transition, deliberately detour, then check the restored suffix."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile

project = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL"
sys.path.insert(0, str(project))
from reward_branch_env import ForkEnv


def run():
    root = project / "runs/pixels-only-01/official-wsl"
    fixture = root / "controls/self-earned-prefix"
    source = fixture / json.loads((fixture / "result.json").read_text())["final_audit"]
    tape = source.with_suffix(".bin").read_bytes()
    output = root / "controls/training-fork-check"
    output.mkdir(exist_ok=True)
    env = ForkEnv()
    try:
        restore_at = len(tape) - 83
        for button in tape[:restore_at]:
            env.command("step", (button, 1))
        env.command("save", 7)
        def replay_suffix():
            result = []
            for button in tape[restore_at:]:
                pic, reward, terminal, used = env.command("step", (button, 1))
                result.append((hashlib.sha256(pic.tobytes()).hexdigest(), reward, terminal, used))
            return result
        reference = replay_suffix()
        assert reference[-1][1] >= 100
        with tempfile.TemporaryDirectory(dir=output) as scratch:
            folder = Path(scratch)
            env.command("audit", str(folder / "one"))
            env.command("restore", 7)
            for button in reversed(range(9)):
                env.command("step", (button, 8))
            env.command("restore", 7)
            assert replay_suffix() == reference
            env.command("audit", str(folder / "two"))
            for filename in ("game.json", "game.bin"):
                assert (folder / "one" / filename).read_bytes() == (folder / "two" / filename).read_bytes()
            assert (folder / "two/game.bin").read_bytes() == tape
        (output / "verification.json").write_text(json.dumps(dict(
            restored_at=restore_at, compared_frames=len(reference), compared_rewards=len(reference),
            final_reward_cents=reference[-1][1], matching_tape_sha256=hashlib.sha256(tape).hexdigest(),
            fixture_only=True, mixed_training_or_evaluation_data_used=False), indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    run()
