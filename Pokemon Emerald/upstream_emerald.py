"""Environment-only adapter for the pinned danijar/dreamerv3 release. No upstream agent edits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

PIN="e01491fad6434b2245a3b8ca201dd7faedcc458c"
REPLAY_CAPACITY = 131_072


def freeze_for_evaluation(checkpoint, directory, elements):
    """Keep the evaluated weights after upstream rotates its training checkpoint."""
    counter = elements.Counter()
    elements.checkpoint.load(str(checkpoint), {"step": counter.load})
    frozen = directory / "frozen-policy"
    frozen.mkdir(exist_ok=True)
    source = checkpoint / "agent.pkl"
    with source.open("rb") as reader:
        digest = hashlib.file_digest(reader, "sha256").hexdigest()
    if (frozen / "done").exists():
        with (frozen / "agent.pkl").open("rb") as reader:
            if hashlib.file_digest(reader, "sha256").hexdigest() != digest:
                raise ValueError(f"Evaluation already contains a different policy: {frozen}")
    else:
        shutil.copyfile(source, frozen / "agent.pkl")
        shutil.copyfile(checkpoint / "step.pkl", frozen / "step.pkl")
        (frozen / "done").touch()
    return frozen, dict(training_step=int(counter), agent_sha256=digest,
                        source_checkpoint=str(checkpoint), frozen_policy=str(frozen))


def verify_evaluation_win(log, checkpoint_info):
    failed = []
    for audit in sorted((log / "external-audit").glob("*/*/*.json")):
        if "stone_badge" not in json.loads(audit.read_text())["events"]:
            continue
        out = log / "verified-wins" / audit.relative_to(log / "external-audit").with_suffix("")
        command = [sys.executable, str(Path(__file__).with_name("pixel_replay.py")),
                   str(audit), str(out), "--event", "stone_badge", "--video"]
        if subprocess.run(command).returncode:
            failed.append(str(audit))
            continue
        result = json.loads((out / "verification.json").read_text())
        result.update(checkpoint_info)
        (log / "verified-stone-badge.json").write_text(json.dumps(result, indent=2))
        return
    if failed:
        raise RuntimeError(f"Stone Badge audit could not be independently reproduced: {failed}")


def train_through_budget(upstream, options, elements, log, budget):
    # The upstream loop saves on a wall clock. Save its live objects once more
    # after the last learner update so both imagination horizons can be compared.
    constructor = elements.Checkpoint
    logger_factory = upstream.make_logger
    training_checkpoint = None

    def make_checkpoint(directory=None, *args, **kwargs):
        nonlocal training_checkpoint
        instance = constructor(directory, *args, **kwargs)
        if directory and Path(str(directory)).resolve() == log / "ckpt":
            training_checkpoint = instance
        return instance

    def logger_with_predictions(config):
        import numpy as np
        logger = logger_factory(config)

        def save_predictions(metrics):
            for step, name, frames in metrics:
                if name == "report/openloop/image":
                    folder = log / "world-model-predictions"
                    folder.mkdir(exist_ok=True)
                    np.savez_compressed(folder / f"step-{step:012d}.npz", frames=frames)

        logger.outputs.append(save_predictions)
        return logger

    with (patch.object(elements, "Checkpoint", make_checkpoint),
          patch.object(upstream, "make_logger", logger_with_predictions)):
        upstream.main(options)
    if training_checkpoint is None or int(training_checkpoint.step) < budget:
        raise RuntimeError("Training ended before reaching the requested checkpoint")
    training_checkpoint.save()
    final = dict(requested_steps=budget, training_step=int(training_checkpoint.step),
                 checkpoint=str(training_checkpoint.latest()))
    (log / "final-checkpoint.json").write_text(json.dumps(final, indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--upstream",type=Path,required=True)
    parser.add_argument("--logdir",type=Path,required=True)
    parser.add_argument("--platform",choices=["cpu","cuda"],default="cuda")
    parser.add_argument("--imag-length",type=int,choices=[1,15],default=15)
    parser.add_argument("--steps",type=int,default=2048)
    parser.add_argument("--envs",type=int,default=1)
    parser.add_argument("--parallel-envs", action="store_true",
                        help="Use the upstream parallel emulator driver")
    parser.add_argument("--seed",type=int,default=71299)
    parser.add_argument("--episode",type=int,default=8192)
    parser.add_argument("--batch",type=int,default=4)
    parser.add_argument("--length",type=int,default=16)
    parser.add_argument("--ratio",type=float,default=8.)
    parser.add_argument("--uncapped-exploration", action="store_true",
                        help="Continue rewarding a first cell visit after the first 50 cells in an episode")
    parser.add_argument("--reward-replay-fraction", type=float, default=0.,
                        help="Fraction of offline training samples drawn in proportion to peak absolute reward")
    parser.add_argument("--evaluate",type=Path,help="Evaluate frozen upstream checkpoint")
    args=parser.parse_args()
    if not 0 <= args.reward_replay_fraction <= 1:
        parser.error("reward replay fraction must be between zero and one")
    checkout=args.upstream.resolve()
    rev=subprocess.check_output(["git","-C",str(checkout),"rev-parse","HEAD"],text=True).strip()
    if rev!=PIN:raise ValueError(f"Expected {PIN}, found {rev}")
    subprocess.run(["git","-C",str(checkout),"diff","--exit-code","HEAD","--","dreamerv3","embodied"],check=True)
    sys.path.insert(0,str(checkout))
    import elements
    import embodied
    import numpy as np
    from dreamerv3 import main as upstream
    from pixel_env import PixelPool
    if args.reward_replay_fraction and not args.evaluate:
        from reward_replay import install_reward_sampling
        make_replay = upstream.make_replay
        def make_reward_replay(config, folder, mode="train"):
            replay = make_replay(config, folder, mode)
            if mode == "train":
                install_reward_sampling(replay, config.replay_context, args.reward_replay_fraction, args.seed)
            return replay
        upstream.make_replay = make_reward_replay
    class Adapter(embodied.Env):
        def __init__(self,index):
            self.pool=PixelPool(1, args.uncapped_exploration)
            self.index=index
            self.done=True
            self.age=0
            self.session=f"{os.getpid()}-{__import__('time').time_ns()}"
        @property
        def obs_space(self):
            return dict(image=elements.Space(np.uint8,(64,64,3)),reward=elements.Space(np.float32),
                        is_first=elements.Space(bool),is_last=elements.Space(bool),is_terminal=elements.Space(bool))
        @property
        def act_space(self):
            return dict(action=elements.Space(np.int32,(),0,9),reset=elements.Space(bool))
        def step(self,act):
            first=bool(act["reset"]) or self.done
            if first:
                pic=self.pool.command(0,"reset")
                self.age=0
                reward,terminal=0.,False
            else:
                pics,rews,ends=self.pool.step([int(act["action"])])
                pic,reward,terminal=pics[0],float(rews[0]),bool(ends[0])
                self.age+=1
            self.done=terminal or self.age>=args.episode
            if self.done:
                self.pool.command(0,"audit",str(args.logdir.resolve()/"external-audit"/self.session/f"env-{self.index}"))
            elif not args.evaluate and self.age and (self.age % 8192 == 0 or reward >= 4):
                folder = args.logdir.resolve()/"in-progress"/self.session/f"env-{self.index}"/f"step-{self.age:06}"
                self.pool.command(0, "audit", str(folder))
            padded=np.zeros((64,64,3),np.uint8)
            padded[12:52,2:62]=pic
            return dict(image=padded,reward=np.float32(reward),is_first=np.bool_(first),
                        is_last=np.bool_(self.done),is_terminal=np.bool_(terminal))
        def close(self):
            self.pool.close()
    upstream.make_env=lambda config,index,**kw:upstream.wrap_env(Adapter(index),config)
    log=args.logdir.resolve()
    log.mkdir(parents=True,exist_ok=True)
    settings=dict(upstream_revision=rev,upstream_diff="clean",adapter="Emerald RGB + scalar reward",
                  terminal="verified Stone Badge",truncation=args.episode,platform=args.platform,
                  imagination_horizon=args.imag_length,seed=args.seed,batch=args.batch,length=args.length,
                  sample_ratio=args.ratio,agent_size="upstream size1m",run_steps=args.steps,envs=args.envs,
                  replay_capacity=REPLAY_CAPACITY,uncapped_exploration=args.uncapped_exploration,
                  reward_replay_fraction=args.reward_replay_fraction,parallel_envs=args.parallel_envs)
    provenance = log / "provenance.json"
    if provenance.exists():
        previous = json.loads(provenance.read_text())
        if previous.get("uncapped_exploration", False) != args.uncapped_exploration:
            raise ValueError("Use a new run directory when changing exploration rewards")
        if previous.get("reward_replay_fraction", 0.) != args.reward_replay_fraction:
            raise ValueError("Use a new run directory when changing the replay sampler")
    if args.evaluate:
        frozen, checkpoint_info = freeze_for_evaluation(args.evaluate.resolve(), log, elements)
        settings.update(checkpoint_info)
    provenance.write_text(json.dumps(settings,indent=2))
    print("Launch",settings,flush=True)
    options=["--configs","size1m","--task","emerald_pixels","--logdir",str(log),
                   "--seed",str(args.seed),"--jax.platform",args.platform,"--jax.prealloc","False",
                   "--jax.compute_dtype","float32","--run.steps",str(args.steps),"--run.envs",str(args.envs),
                   "--batch_size",str(args.batch),"--batch_length",str(args.length),"--report_length",str(args.length),
                   "--run.train_ratio",str(args.ratio),"--agent.imag_length",str(args.imag_length),
                   "--replay.size",str(REPLAY_CAPACITY),
                   "--run.log_every","20","--run.report_every","600","--run.save_every","180",
                   "--run.usage.nvsmi","False","--logger.outputs","jsonl",
                   "--run.debug",str(not args.parallel_envs)]
    if args.evaluate:
        options+=["--script","eval_only","--run.from_checkpoint",str(frozen)]
        upstream.main(options)
        verify_evaluation_win(log, checkpoint_info)
    else:
        train_through_budget(upstream, options, elements, log, args.steps)


if __name__=="__main__":
    main()
