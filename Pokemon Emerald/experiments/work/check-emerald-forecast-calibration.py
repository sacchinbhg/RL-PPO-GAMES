"""Check ranking direction, action alignment and data separation on synthetic state-dependent returns."""
import json
from pathlib import Path
import sys
import tempfile
import numpy as np

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from calibrate_button_forecasts import fit,score
from train_earned_dreamer import digest,write_json
out=project/"runs/pixels-only-01/official-wsl/earned-button-alternatives"
rng=np.random.default_rng(125134)
def batch(n):
    images=np.zeros((n,33,40,60,3),np.uint8)
    images[:]=rng.integers(0,255,(n,1,1,1,3),dtype=np.uint8)
    past=rng.integers(0,9,(n,32),dtype=np.int32)
    best=np.arange(n)%9
    rewards=np.zeros((n,9,15),np.float32)
    rewards[np.arange(n),best,8]=1
    forecast=-rewards+.0005*rng.normal(size=rewards.shape)
    return images,past,rewards,forecast,best

with tempfile.TemporaryDirectory() as tmp:
    train=Path(tmp)/"train"
    source=train/"fixture"
    source.mkdir(parents=True)
    images,past,rewards,forecast,best=batch(27)
    np.savez(source/"windows.npz",context=images,past=past,real_rewards=rewards)
    np.savez(train/"world-model-forecasts.npz",forecasts=forecast)
    write_json(train/"completed.json",dict(games=[dict(training_tape="fixture",data_sha256=digest(source/"windows.npz"))]))
    write_json(train/"world-model-forecasts.json",dict(training_tapes=["fixture"],weights="synthetic",
        forecasts_sha256=digest(train/"world-model-forecasts.npz"),window_sha256=[digest(source/"windows.npz")]))
    fitted=Path(tmp)/"fitted"
    fit(train,fitted)
    images,past,rewards,forecast,best=batch(18)
    with np.load(fitted/"model.npz") as model:
        scores=score(model,images,forecast)
    assert (scores["calibrated_world_model"].argmax(1)==best).all()
    assert not (scores["world_model"].argmax(1)==best).any()
    out.mkdir(exist_ok=True)
    write_json(out/"calibration-fixture.json",dict(source="synthetic sign-reversal fixture",
        unseen_states=18,correct_calibrated=18,correct_uncalibrated=0,external_game_data_used=False))
