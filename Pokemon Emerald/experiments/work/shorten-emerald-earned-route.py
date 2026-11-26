"""Remove visual loops from a self-earned training path, accepting only emulator-tested reward sequences."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

project = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL"
sys.path.insert(0, str(project))
from reward_branch_env import ForkEnv
from reward_branch_search import visual_code, write_json

base = project / "runs/pixels-only-01/official-wsl"
parent = base / "reward-only-training-forks"
stride = 512


def trace(env, tape):
    frame = env.command("restore", 0)
    env.command("save", 1)
    sequences, phases = [], [0]
    groups = defaultdict(list)
    def register(pic, index, phase):
        groups[phase, "exact", hashlib.blake2b(pic.tobytes(), digest_size=12).digest()].append(index)
        groups[phase, "pooled", visual_code(pic)].append(index)
    register(frame, 0, 0)
    for n, button in enumerate(tape, 1):
        frame, cents, _, _ = env.command("step", (button, 1))
        if cents >= 25:
            sequences.append(cents // 25)
        phases.append(len(sequences))
        register(frame, n, len(sequences))
        if n % stride == 0:
            env.command("save", 1 + n // stride)
    # Reward-neutral cuts with the same observed screen. Rank by removed buttons per replay cost.
    candidates = {}
    for (_, kind, _), locations in groups.items():
        if len(locations) < 2:
            continue
        for left, right in [(locations[0],locations[-1]), (locations[len(locations)//2],locations[-1])]:
            gap = right-left
            if gap < 32:
                continue
            bucket = left//128, right//128
            priority = gap / (196 + len(tape)-right + left%stride) * (1.50 if kind=="exact" else 1)
            if bucket not in candidates or priority > candidates[bucket][0]:
                candidates[bucket] = (priority, left, right, kind)
    return sequences, phases, sorted(candidates.values(), reverse=True)


def try_cut(env, tape, left, right, phases, signature):
    start = (left//stride)*stride
    frame = env.command("restore", 1 + start//stride)
    phase, work = phases[start], 0
    prefix = tape[start:left]
    changes = []
    for buttons in (prefix, tape[right:]):
        for button in buttons:
            frame, cents, done, _ = env.command("step", (button,1))
            work += 1
            if cents >= 25:
                if phase == len(signature) or cents//25 != signature[phase]:
                    return None, work
                phase += 1
            if phase == len(signature):
                return tape[:start] + prefix[:min(work,len(prefix))] + tape[right:right+max(0,work-len(prefix))], work
            if done:
                return None, work
    return None, work


def run(tape_id, budget):
    src = parent / "verified-training-tapes" / tape_id
    manifest = json.loads((src/"manifest.json").read_text())
    assert manifest["source_sha256"] == tape_id and not manifest["frozen_evaluation_data_used"]
    source, = (src/"audit").glob("*.bin")
    original = source.read_bytes()
    assert hashlib.sha256(original).hexdigest()==tape_id
    out = base / "shortened-self-earned-routes" / tape_id
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/"settings.json", dict(seed=manifest["source_seed"], training_forks=True,
        receives_saves_from_other_runs=False, parent_training_tape=tape_id,
        inherited_search_buttons=manifest["search_buttons_at_discovery"],
        criterion="preserve every scalar reward of at least 0.25, in order and in quarter units; remove screen loops"))
    target = out/"result.json"
    if target.exists():
        return
    previous = json.loads((out/"best.json").read_text()) if (out/"best.json").exists() else None
    current = (out/previous["audit"]/"game.bin").read_bytes() if previous else original
    work = previous["work"] if previous else 0
    trials = previous["branches"] if previous else 0
    signature = None
    with_original = json.loads((parent/"discoveries"/f"work-{manifest['search_buttons_at_discovery']:010d}"/"game.json").read_text())
    env = ForkEnv()
    try:
        env.command("save",0)
        while work < budget:
            sig, phases, candidates = trace(env,current)
            work += len(current)
            if signature is None:
                import numpy as np
                signature = []
                for f in sorted(src.glob("chunk-*.npz")):
                    with np.load(f) as ds:
                        signature += [round(100*float(r))//25 for r in ds["reward"] if r>=.25]
            assert sig == signature
            changed = False
            for _, left, right, kind in candidates:
                if work >= budget:
                    break
                shorter, cost = try_cut(env,current,left,right,phases,signature)
                work, trials = work+cost, trials+1
                write_json(out/"status.json",dict(work=work,trials=trials,original_length=len(original),
                    accepted_length=len(current),last_test=[left,right],accepted=shorter is not None))
                if shorter is None:
                    continue
                relative = f"discoveries/work-{work:010d}"
                dest = out/relative
                env.command("audit",str(dest))
                assert (dest/"game.bin").read_bytes()==shorter
                audit = json.loads((dest/"game.json").read_text())
                # Human-facing verification catches different game events with the same scalar value.
                # Acceptance used only the scalar signature. Record any semantic difference explicitly.
                same_events = list(audit["events"]) == list(with_original["events"])
                record = dict(work=work,branches=trials,audit=relative,return_cents=audit["return_cents"],
                    length=len(shorter),tape_sha256=hashlib.sha256(shorter).hexdigest(),training_only=True,
                    terminal="stone_badge" in audit["events"],removed_interval=[left,right],screen_match=kind,
                    parent_length=len(current), matching_event_names=same_events, quarter_reward_sequence=signature)
                write_json(dest/"record.json",record)
                write_json(out/"best.json",record)
                current = shorter
                changed = True
                break
            if not changed:
                break
        write_json(target,dict(original_length=len(original),final_length=len(current),work=work,trials=trials,
            original_tape_sha256=tape_id, final_tape_sha256=hashlib.sha256(current).hexdigest()))
    finally:
        env.close()


if __name__ == "__main__":
    run(sys.argv[1],int(sys.argv[2]))
