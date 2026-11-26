"""Exercise the adapter's argument path and the pinned Dreamer configuration parser."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "outputs/EmeraldRL"))
import upstream_emerald


def check(upstream, options, elements, log, steps):
    from ruamel.yaml import YAML
    presets = YAML(typ="safe").load((Path(upstream.__file__).parent / "configs.yaml").read_text())
    parsed, remaining = elements.Flags(configs=["defaults"]).parse_known(options)
    config = elements.Config(presets["defaults"])
    for name in parsed.configs:
        config = config.update(presets[name])
    config = elements.Flags(config).parse(remaining)
    provenance = json.loads((log / "provenance.json").read_text())
    assert config.agent.horizon == provenance["discount_horizon"]
    assert config.agent.imag_loss.actent == provenance["actor_entropy"]
    assert config.agent.imag_length == 15 and config.jax.platform == "cuda"
    observed.append(config.agent.horizon)
    entropies.append(config.agent.imag_loss.actent)


observed = []
entropies = []
with tempfile.TemporaryDirectory(dir=root / "work") as scratch:
    base = ["upstream_emerald.py", "--upstream", "/opt/emerald-rl/source", "--platform", "cuda"]
    def run(path, *options):
        with patch.object(sys, "argv", base + ["--logdir", str(Path(scratch) / path), *options]), \
                patch.object(upstream_emerald, "train_through_budget", check), \
                contextlib.redirect_stdout(io.StringIO()):
            upstream_emerald.main()
    run("ordinary")
    run("long", "--discount-horizon", "4096")
    checkpoint = Path(scratch) / "long/ckpt/version"
    frozen = Path(scratch) / "long/frozen-policy"
    assert upstream_emerald.resolve_discount_horizon(None, checkpoint) == 4096
    assert upstream_emerald.resolve_discount_horizon(None, frozen) == 4096
    try:
        upstream_emerald.resolve_discount_horizon(333, checkpoint)
    except ValueError as error:
        assert "Checkpoint was trained" in str(error)
    else:
        raise AssertionError("Allowed a mislabeled frozen evaluation")
    try:
        run("ordinary", "--discount-horizon", "4096")
    except ValueError as error:
        assert "discount horizon" in str(error)
    else:
        raise AssertionError("Allowed an incompatible training resume")
    # Legacy provenance files omitted the default setting.
    file = Path(scratch) / "ordinary/provenance.json"
    old = json.loads(file.read_text())
    del old["discount_horizon"]
    del old["actor_entropy"]
    file.write_text(json.dumps(old))
    run("ordinary")
    assert observed == [333, 4096, 333], observed
    run("entropy", "--actor-entropy", "0.003")
    ec = Path(scratch)/"entropy/ckpt/version"
    assert upstream_emerald.checkpoint_setting(None, ec, "actor_entropy", 0.0003) == .003
    try:
        run("entropy")
    except ValueError as error:
        assert "entropy" in str(error)
    else:
        raise AssertionError("Allowed a different entropy coefficient when resuming")
    for invalid in ("-1", "nan", "inf"):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                run("invalid", "--actor-entropy", invalid)
        except SystemExit as error:
            assert error.code == 2
        else:
            raise AssertionError(f"Accepted {invalid}")
    assert entropies == [.0003,.0003,.0003,.003], entropies
print("Verified official discount, actor entropy, frozen settings, legacy default and isolated runs.")
