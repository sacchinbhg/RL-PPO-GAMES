import json
import sys
import time
from pathlib import Path

import torch

folder = Path(__file__).resolve().parents[1] / "outputs" / "EmeraldRL"
sys.path.insert(0, str(folder))
from policy import N_FEATURES, Policy

torch.manual_seed(19)
torch.set_num_threads(2)
results = {}
for device in ("cuda", "cpu"):
    net = Policy().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=.0003)
    results[device] = {}
    for batch, train in ((4, False), (128, True)):
        img = torch.randint(256, (batch, 3, 40, 60), dtype=torch.uint8, device=device)
        ram = torch.rand(batch, 2*N_FEATURES, device=device)
        target = torch.randn(batch, device=device)
        action = torch.randint(9, (batch,), device=device)
        def step():
            with torch.set_grad_enabled(train):
                dist, v = net(img, ram)
                if train:
                    loss = (v-target).square().mean() - dist.log_prob(action).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                else:
                    dist.sample()
        for _ in range(12):
            step()
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            step()
        if device == "cuda":
            torch.cuda.synchronize()
        results[device][f"{'train' if train else 'inference'}_batch{batch}_ms"] = round((time.perf_counter()-start)*10, 3)
out = dict(device=torch.cuda.get_device_name(), torch=torch.__version__, cpu_threads=2,
           parameters=sum(p.numel() for p in net.parameters()), peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20,1),
           results=results, protocol="Synthetic tensors already on device, 12 warmup and 100 timed iterations, FP32. Concurrent source build.")
(folder / "hardware.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out))
