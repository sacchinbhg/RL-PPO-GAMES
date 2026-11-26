"""Closed-loop evaluation only: blend all nine button probabilities equally on frozen weights."""
from collections import Counter
import json
from pathlib import Path
import sys
from unittest.mock import patch

work = Path(__file__).resolve().parent
project = work.parent / "outputs/EmeraldRL"
sys.path[:0] = [str(project), "/opt/emerald-rl/source"]
from dreamerv3 import main as official
from embodied.jax import heads, outs
import upstream_emerald as bridge
from emerald import BUTTONS

root = project / "runs/pixels-only-01/official-wsl"
source = root / "seed-71299-h15-retained-rewards/eval-262144"
study = root / "controls/frozen-action-mixture"


def evaluate(fraction, horizon):
    # Run each condition in a fresh Python process to reset JAX and upstream counters.
    dest = study / (f"mix-{round(fraction*100):02}-steps-{horizon}")
    dest.mkdir(parents=True, exist_ok=True)
    result = dest/"comparison.json"
    if result.is_file():
        return
    parent = json.loads((source/"provenance.json").read_text())
    assert parent["training_step"] == 262152 and parent["platform"] == "cuda"
    if fraction:
        check = json.loads((study/"mix-00-steps-256/comparison.json").read_text())
        assert len(check["games"]) == 4 and all(g["prefix_matches"] for g in check["games"])
    original = heads.Head.categorical
    main = official.main
    observed = set()

    def actor_distribution(head, inputs):
        output = original(head, inputs)
        if head.path.split("/")[-3:] == ["pol", "head", "action"]:
            assert output.logits.shape[-1] == 9
            observed.add(head.path)
            if fraction:
                mixed = outs.Categorical(output.logits, unimix=fraction)
                mixed.minent, mixed.maxent = output.minent, output.maxent
                return mixed
        return output

    def marked_main(options):
        assert "--script" in options and options[options.index("--script") + 1] == "eval_only"
        provenance = dest/"provenance.json"
        info = json.loads(provenance.read_text())
        assert info["agent_sha256"] == parent["agent_sha256"]
        info.update(purpose="frozen action-distribution diagnostic",
                    evaluation_uniform_button_fraction=fraction,
                    original_evaluation=str(source), optimizer_updates_during_evaluation=0)
        provenance.write_text(json.dumps(info, indent=2))
        return main(options)

    sys.argv = [str(project/"upstream_emerald.py"), "--upstream", "/opt/emerald-rl/source",
                "--platform","cuda","--envs","4","--parallel-envs","--batch","8","--length","32",
                "--ratio","16","--imag-length","15","--seed",str(parent["seed"]),
                "--evaluate",str(source/"frozen-policy"),"--episode",str(horizon),
                "--steps",str(4*(horizon+1)+8),"--logdir",str(dest),
                "--reward-replay-fraction","0.25","--retain-reward-windows"]
    with patch.object(heads.Head, "categorical", actor_distribution), patch.object(official,"main",marked_main):
        bridge.main()
    assert len(observed) == 1, f"Actor distribution hook saw {observed}"
    games = []
    for ref in sorted((source/"external-audit").glob("*/*/*episode-00001.bin")):
        env = ref.parent.name
        audit, = (dest/"external-audit").glob(f"*/{env}/*episode-00001.json")
        original_actions = ref.read_bytes()[:horizon]
        actions = audit.with_suffix(".bin").read_bytes()
        assert len(actions) == horizon
        prefix_matches = actions == original_actions
        if not fraction:
            assert prefix_matches, env
        games.append(dict(env=env, prefix_matches=prefix_matches, events=json.loads(audit.read_text())["events"],
                          button_counts=Counter(BUTTONS[i] for i in actions)))
    assert len(games) == 4
    result.write_text(json.dumps(dict(fraction=fraction, steps_per_game=horizon, games=games,
                    frozen_sha256=parent["agent_sha256"], traced_heads=sorted(observed)), indent=2))
    print("Verified", fraction, [g["events"] for g in games], flush=True)


if __name__ == "__main__":
    fraction, horizon = (float(sys.argv[1]), int(sys.argv[2]))
    assert (fraction, horizon) in ((0.,256),(.20,8192))
    evaluate(fraction, horizon)
