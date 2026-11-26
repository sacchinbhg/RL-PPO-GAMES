"""Resolve changes to the retained bank in the last, incomplete training games."""
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location("scene_trace", Path(__file__).with_name("trace-retained-scenes.py"))
scene = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scene)
dest = scene.root/"controls/retained-last-training-games"


def verify(job):
    env, actions, expected = job
    game = scene.Emerald(*scene.game_paths())
    score = scene.Score()
    frames, phases, rewards = (deque(maxlen=33) for _ in range(3))
    menu = None
    matches = {}
    out = dest/env
    out.mkdir(parents=True, exist_ok=True)
    try:
        game.load(game.boot)
        for i, button in enumerate(actions[:-1], 1):
            pic, key = game.step(int(button))
            r, _ = score.read(game,key)
            assert np.float32(r) == expected[i], (env,i,r,expected[i])
            frames.append(pic)
            phase = scene.PHASES[key[0]]
            phases.append(phase)
            if phase in ("main-menu","title"):
                menu = i
            rewards.append(r)
            if r >= 1:
                fingerprint = scene.digest(np.stack(frames), actions[i-32:i+1], rewards)
                matches[fingerprint] = dict(source=(out/"episode.json").relative_to(scene.root).as_posix(),
                    end_decision=i, events=[k for k,v in score.events.items() if v == i],
                    reward=r, phases=Counter(phases), last_title_or_menu=menu)
        tape = np.asarray(actions[:-1],np.uint8).tobytes()
        (out/"episode.bin").write_bytes(tape)
        result = dict(steps=len(tape),events=score.events,first_visit_cells=len(score.cells),
                    uncapped_exploration=False,source="incomplete training game reconstructed from final replay",
                    matched_original_audit_prefix=8192,all_scalar_rewards_exact=True,
                    input_sha256=hashlib.sha256(tape).hexdigest())
        (out/"episode.json").write_text(json.dumps(result,indent=2))
        return matches
    finally:
        game.close()


if __name__ == "__main__":
    src = scene.branch
    files = list((src/"replay").glob("*.npz"))
    lookup = {p.stem.split("-")[1]:p for p in files}
    following = {p.stem.split("-")[2] for p in files}
    refs = {p.read_bytes():p.parents[1].name for p in (src/"in-progress").glob("*/*/step-008192/*00002.bin")}
    assert len(refs) == 4
    jobs = []
    for path in (p for k,p in lookup.items() if k not in following):
        chunks = []
        while path:
            chunks.append(path)
            path = lookup.get(path.stem.split("-")[2])
        if sum(int(p.stem.split("-")[3]) for p in chunks) != 49152:
            continue
        buffers = {k:[] for k in ("action","reward","is_first")}
        for chunk in chunks:
            with np.load(chunk) as z:
                for k,v in buffers.items():
                    v.append(z[k])
        data = {k:np.concatenate(v) for k,v in buffers.items()}
        assert np.flatnonzero(data["is_first"]).tolist() == [0,32769]
        actions = data["action"][32769:]
        env = refs[np.asarray(actions[:8192],np.uint8).tobytes()]
        jobs.append((env, actions, data["reward"][32769:]))
    assert len({env for env,_,_ in jobs}) == 4
    intermediate = json.loads((scene.root/"controls/retained-scene-coverage-17.json").read_text())
    matched = dict(intermediate["examples"])
    with ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(verify,jobs):
            matched.update(result)
    cp = src/"ckpt"
    final = cp/(cp/"latest").read_text().strip()
    assert final.name == Path(json.loads((src/"final-checkpoint.json").read_text())["checkpoint"]).name
    with (final/"replay.pkl").open("rb") as f:
        bank = pickle.load(f)["retained_v1"]
    windows = [scene.digest(x["image"][:,12:52,2:62],x["action"],x["reward"]) for x in bank]
    assert len(windows) == len(set(windows)) == 17
    rows = {k:matched[k] for k in windows}
    counts, events = Counter(), Counter()
    for row in rows.values():
        counts.update(row["phases"])
        events.update(row["events"])
    report = dict(checkpoint=final.name,training_step=262152,windows=len(windows),frames=sum(counts.values()),
                changed_examples=len(set(windows)-set(intermediate["examples"])),
                observed_phases=counts, ending_events=events,examples=rows,
                minimum_gap_from_last_title_or_menu=min(v["end_decision"]-32-v["last_title_or_menu"] for v in rows.values()))
    (dest/"final-scene-coverage.json").write_text(json.dumps(report,indent=2))
    print("Verified all four unfinished games and all final 17 windows",counts,events,flush=True)
