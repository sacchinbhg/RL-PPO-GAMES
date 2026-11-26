"""Check public forecasting against stored official GPU forecasts before releasing it."""
import fcntl
import json
from pathlib import Path
import sys
import time
import numpy as np

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from forecast_button_alternatives import predict_buttons
from train_earned_dreamer import write_json
root=project/"runs/pixels-only-01/official-wsl"
external=root/"controls/heldout-action-rewards"
first=sorted(external.glob("env-*.npz"))[0]
marker=root/"earned-button-alternatives/public-inference-check.json"

if __name__=="__main__":
    with (root/"controls/large-optimizer.lock").open("a") as lock:
        while True:
            available=int(next(s.split()[1] for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:")))
            if available>2800000:
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    pass
            time.sleep(30)
        with np.load(first) as origins:
            forecasts,info=predict_buttons(*(origins[k][:4] for k in ("context","past","future")),
                root/"seed-71299-h15/eval-65536",root/"earned-official-updates/adaptation/frozen-policy",Path("/opt/emerald-rl/source"))
        with np.load(external/"earned-1152-forecasts.npz") as previous:
            compared=previous["forecasts"][:len(forecasts)]
        assert np.array_equal(forecasts,compared), np.max(np.abs(forecasts-compared))
        write_json(marker,dict(**info,reproduced_reward_values=forecasts.size,bitwise_equal=True))
