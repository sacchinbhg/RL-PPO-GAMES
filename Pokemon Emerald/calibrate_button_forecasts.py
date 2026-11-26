"""Fit a small action-ranking model on returns measured from the agent's own emulator forks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from earned_memory import encode
from train_earned_dreamer import digest, write_json


def features(forecasts):
    if forecasts.ndim != 3 or forecasts.shape[1:] != (9,15):
        raise ValueError("Expected nine actions and fifteen predicted rewards per origin")
    bins=np.stack([forecasts[:,:,0], forecasts[:,:,1:5].mean(-1),
                   forecasts[:,:,5:10].mean(-1), forecasts[:,:,10:15].mean(-1)],axis=-1).astype(np.float64)
    return bins-bins.mean(axis=1,keepdims=True)


def scenes(contexts):
    return np.stack([np.concatenate([encode(frame[-9])[1],encode(frame[-1])[1]]) for frame in contexts]).astype(np.float32)/255


def context_hash(image,buttons):
    h=hashlib.sha256(image)
    h.update(np.asarray(buttons,dtype="<i4").tobytes())
    return h.hexdigest()


def design(centered, scale):
    local=np.clip(centered/scale,-5,5)
    action_bias=np.broadcast_to(np.eye(9)-1/9,(len(local),9,9))
    return np.concatenate([local,action_bias],axis=-1)


def fit(directory,output):
    collection=json.loads((directory/"completed.json").read_text())
    metadata=json.loads((directory/"world-model-forecasts.json").read_text())
    if metadata["training_tapes"]!=[g["training_tape"] for g in collection["games"]]:
        raise ValueError("Training forecasts and simulation records came from different games")
    forecast_file=directory/"world-model-forecasts.npz"
    if (digest(forecast_file)!=metadata["forecasts_sha256"]
            or metadata["window_sha256"]!=[g["data_sha256"] for g in collection["games"]]):
        raise ValueError("Imagined rewards or source windows changed after inference")
    with np.load(forecast_file) as data:
        predictions=data["forecasts"]
    contexts,hashes,returns=[],[],[]
    for source in collection["games"]:
        file=directory/source["training_tape"]/"windows.npz"
        if digest(file)!=source["data_sha256"]:
            raise ValueError("A training window changed after collection")
        with np.load(file) as data:
            contexts.extend(data["context"])
            hashes.extend(context_hash(im,btn) for im,btn in zip(data["context"],data["past"]))
            returns.extend(data["real_rewards"].sum(-1))
    reward=np.asarray(returns,np.float64)
    if reward.shape!=predictions.shape[:2]:
        raise ValueError("The number of training returns and forecasts differs")
    spread=np.ptp(reward,axis=1)
    active=spread>.001
    if not active.any():
        raise ValueError("No available action changed a training return")
    target=(reward[active]-reward[active].mean(1,keepdims=True))/spread[active,None]
    centered=features(predictions)[active]
    scale=np.maximum(np.sqrt(np.mean(centered**2,axis=(0,1))),1e-8)
    matrix=design(centered,scale).reshape(-1,13)
    penalty=.1*len(target)
    coefficients=np.linalg.solve(matrix.T@matrix+penalty*np.eye(13),matrix.T@target.ravel())
    output.mkdir(parents=True,exist_ok=True)
    model=output/"model.npz"
    temp=output/"model.tmp.npz"
    np.savez_compressed(temp,scale=scale,coef=coefficients,buttons=target.mean(0),
                       training_scenes=scenes(np.asarray(contexts)[active]),action_advantages=target,
                       training_context_hashes=np.asarray(hashes))
    temp.replace(model)
    write_json(output/"fit.json",dict(weights=metadata["weights"],model_sha256=digest(model),
        training_tapes=metadata["training_tapes"],windows=len(reward),action_sensitive_windows=int(active.sum()),
        ridge_penalty=penalty,forecast_features="four time bins centered across the nine possible first buttons",
        target="action return relative to the state mean, scaled by that state's measured range",evaluation_data_used=False))


def score(model,contexts,forecasts):
    centered=features(forecasts)
    result=dict(world_model=forecasts.sum(-1),action_only=np.broadcast_to(model["buttons"],centered.shape[:2]),
                calibrated_world_model=design(centered,model["scale"])@model["coef"])
    estimates=[]
    for image in scenes(contexts):
        distance=np.sqrt(np.mean((model["training_scenes"]-image)**2,axis=1))
        nearest=np.argsort(distance,kind="stable")[:4]
        estimates.append(np.average(model["action_advantages"][nearest],axis=0,weights=1/np.maximum(.02,distance[nearest])))
    result["similar_screens"]=np.asarray(estimates)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    fit(args.training,args.output)
