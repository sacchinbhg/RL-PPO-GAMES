"""Check the frozen Dreamer world's action rankings against independent emulator outcomes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import numpy as np
import elements
import jax
from ruamel.yaml import YAML

work=Path(__file__).resolve().parent
sys.path.insert(0,"/opt/emerald-rl/source")
from dreamerv3 import main as upstream
from dreamerv3.agent import Agent
from embodied.jax import internal
spec=importlib.util.spec_from_file_location("reward_probe",work/"probe-emerald-reward-model.py")
probe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
directory=probe.root/"controls/heldout-action-rewards"
H=15


def imagine_rewards(self, initial, data):
    carry, obs, action, _=self._apply_replay_context(initial,data)
    cut=lambda xs,s:jax.tree.map(lambda x:x[:,s],xs)
    history=cut(obs,slice(None,-H))
    enc,_,tokens=self.enc(carry[0],history,history["is_first"],training=False)
    state,_,_=self.dyn.observe(carry[1],tokens,cut(action,slice(None,-H)),history["is_first"],training=False)
    _,feats,_=self.dyn.imagine(state,cut(action,slice(-H,None)),H,training=False)
    return initial,dict(estimate=self.rew(self.feat2tensor(feats),2).pred())


def main(arm,phase="eval-262144",adapted=None,label=None,training=None):
    path=probe.root/arm/phase
    info=json.loads((path/"provenance.json").read_text())
    frozen=path/"frozen-policy"
    sha=info["agent_sha256"]
    fit=None
    if adapted:
        fit=json.loads((adapted/"result.json").read_text())
        assert fit["parent_sha256"]==sha and fit["upstream_revision"]==info["upstream_revision"] and label
        sha=fit["trained_sha256"]
        frozen=adapted/"frozen-policy"
    with (frozen/"agent.pkl").open("rb") as f:
        assert hashlib.file_digest(f,"sha256").hexdigest()==sha
    source="/opt/emerald-rl/source"
    assert subprocess.check_output(["git","-C",source,"rev-parse","HEAD"],text=True).strip()==info["upstream_revision"]
    subprocess.run(["git","-C",source,"diff","--exit-code","HEAD","--","dreamerv3","embodied"],check=True)
    raw=YAML(typ="safe").load((path/"config.yaml").read_text())
    assert raw["jax"]["platform"]=="cuda"
    raw["jax"]["precompile"]=False
    ctx=[]; past=[]; future=[]; actual=[]
    aligned=[]
    if training:
        source_games=json.loads((training/"completed.json").read_text())["games"]
        files=[training/g["training_tape"]/"windows.npz" for g in source_games]
        truth=[None]*sum(g["distinct_training_states"] for g in source_games)
    else:
        truth=json.loads((directory/"real-outcomes.json").read_text())["windows"]
        files=[]
        for doc in sorted(directory.glob("env-*.json")):
            aligned.extend(json.loads(doc.read_text())["windows"])
            files.append(doc.with_suffix(".npz"))
        assert [(r["env"],r["button_at"]) for r in aligned]==[(r["env"],r["button_at"]) for r in truth]
        assert len(truth)==39
    for file in files:
        with np.load(file) as ds:
            ctx.extend(ds["context"]); past.extend(ds["past"]); future.extend(ds["future"]); actual.extend(ds["real_rewards"])
    ctx,past,future,actual=map(np.asarray,(ctx,past,future,actual))
    assert future.shape==actual.shape==(len(truth),9,H)
    assert np.all(future[:,:,0]==np.arange(9)[None,:]) and np.all(future[:,:,1:]==future[:,:1,1:])
    forecast=np.zeros_like(actual)
    with patch.object(upstream,"make_env",lambda *a,**kw:probe.Spaces()),patch.object(Agent,"report",imagine_rewards):
        agent=upstream.make_agent(elements.Config(raw))
        devices=[d.device_kind for d in jax.devices() if d.platform=="gpu"]
        assert devices
        cp=elements.Checkpoint()
        cp.agent=agent
        cp.load(str(frozen),keys=["agent"])
        updates=int(agent.n_updates)
        batch=raw["batch_size"]
        def predict(indices,button,seed,future_pixels=0):
            inputs=agent._zeros(agent.spaces,(batch,33+H))
            inputs["image"][:,:33,12:52,2:62]=ctx[indices]
            inputs["image"][:,33:]=future_pixels
            inputs["action"][:,:32]=past[indices]
            inputs["action"][:,32:32+H]=future[indices,button]
            inputs["consec"][:]=1
            assert not inputs["reward"].any()
            data=internal.device_put(inputs,agent.train_sharded)
            data["seed"]=agent._seeds(seed,agent.train_mirrored)
            _,metrics=agent.report(agent.init_report(batch),data)
            return np.asarray(metrics["estimate"])
        for start in range(0,len(truth),batch):
            ix=np.minimum(np.arange(start,start+batch),len(truth)-1)
            length=min(batch,len(truth)-start)
            for button in range(9):
                # Identical random seeds for all nine alternatives within each window.
                draws=[predict(ix,button,81703+i) for i in range(8)]
                if not start and not button:
                    assert np.array_equal(draws[0],predict(ix,button,81703,255))
                forecast[start:start+length,button]=np.mean(draws,axis=0)[:length]
        assert int(agent.n_updates)==updates
    if training:
        np.savez_compressed(training/"world-model-forecasts.npz",forecasts=forecast)
        (training/"world-model-forecasts.json").write_text(json.dumps(dict(weights=sha,
            training_tapes=[g["training_tape"] for g in source_games],windows=len(truth),real_reward_input_zero=True,
            future_image_invariance_verified=True,matched_latent_noise_draws=8,optimizer_updates=0),indent=2))
        return
    rows=[]
    for meta,re,pred in zip(truth,actual.sum(-1),forecast.sum(-1)):
        best=np.flatnonzero(np.isclose(pred,pred.max(),rtol=0,atol=1e-8))
        pairs=[]
        for a in range(9):
            for b in range(9):
                if re[a]>re[b]+.001:
                    pairs.append(float(pred[a]>pred[b])+0.5*float(pred[a]==pred[b]))
        rows.append(dict(env=meta["env"],event=meta["event"],offset=meta["offset"],
            original_button=meta["original_button"],actual_total=re.tolist(),forecast_total=pred.tolist(),
            model_choice=best.tolist(),real_return_at_model_choice=float(re[best].mean()),
            real_return_uniform=float(re.mean()),real_return_best=float(re.max()),
            action_sensitive_pairs=len(pairs),correct_pairs=sum(pairs)))
    sensitive=[r for r in rows if r["action_sensitive_pairs"]]
    sensitive_returns={key:float(np.mean([r[original] for r in sensitive])) for key,original in
        (("model_choice","real_return_at_model_choice"),("uniform","real_return_uniform"),("best_possible","real_return_best"))}
    result=dict(arm=arm,training_step=info["training_step"],weights=sha,
        gpu=devices,optimizer_updates=0,horizon=H,matched_latent_noise_draws=8,
        future_image_invariance_verified=True,real_reward_input_zero=True,
        windows=len(rows),sensitive_windows=len(sensitive),
        primary_metric="Sum of 15 real scalar rewards; only the first button differs",
        actual_choice_mean=float(np.mean([r["real_return_at_model_choice"] for r in rows])),
        actual_uniform_mean=float(np.mean([r["real_return_uniform"] for r in rows])),
        actual_oracle_mean=float(np.mean([r["real_return_best"] for r in rows])),
        correct_pairs=sum(r["correct_pairs"] for r in rows),
        comparable_pairs=sum(r["action_sensitive_pairs"] for r in rows),records=rows,
        mean_return_denominator="all 39 windows",sensitive_window_returns=sensitive_returns)
    if fit:
        result.update(additional_optimizer_updates=fit["optimizer_updates"],training_tapes=fit["training_tapes"])
    name=label or arm
    np.savez_compressed(directory/f"{name}-forecasts.npz",forecasts=forecast)
    (directory/f"{name}-ranking.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print(arm,"sensitive",len(sensitive),"choice",result["actual_choice_mean"],
          "uniform",result["actual_uniform_mean"],"oracle",result["actual_oracle_mean"],flush=True)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("--phase",default="eval-262144")
    parser.add_argument("--adapted",type=Path)
    parser.add_argument("--label")
    parser.add_argument("--training",type=Path,help="Only compute forecasts on sealed self-generated training alternatives")
    args=parser.parse_args()
    main(args.arm,args.phase,args.adapted,args.label,args.training)
