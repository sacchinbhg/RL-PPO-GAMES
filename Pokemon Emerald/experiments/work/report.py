import json
from pathlib import Path

folder = Path(__file__).resolve().parents[1] / "outputs" / "EmeraldRL"
runs = folder / "runs"
def read(name, kind):
    return json.loads((runs / name / (kind + ".json")).read_text(encoding="utf-8-sig"))

a = read("archive", "training")
b = read("ppo", "training")
wall = read("archive", "timing")["wall_seconds"]
wall_sps = a["steps"] / wall
rows = []
for name in ("archive-eval-2048", "ppo-eval-2048", "random-eval-2048"):
    e = read(name, "evaluation")
    trials = e["trials"]
    rows.append((name, len(trials), sum("InsideOfTruck" in t["maps"] for t in trials),
                 sum("LittlerootTown" in t["maps"] for t in trials),
                 sum(t["clock"] for t in trials), sum(t["starter"] for t in trials),
                 max(t["badges"] for t in trials)))

def order(r):
    return ", ".join(r["maps"]) or "Pregame only"

def elapsed(n):
    hours = n / a["active_steps_per_second"] / 3600
    if hours < 1:
        return f"{hours * 60:.0f} minutes"
    if hours < 36:
        return f"{hours:.1f} hours"
    return f"{hours/24:.1f} days"

text = f"""# Emerald pilot results

Two {a['steps']:,}-decision runs trained a {a['parameters']:,}-parameter policy on the RTX 4060 Laptop GPU. Each used seed {a['seed']}, four independent GBA processes, 24 frames per controller decision and a 512-decision training episode.

| Training | Distinct descriptor cells | Maps observed | Most badges |
|---|---:|---|---:|
| Archive-conditioned PPO | {a['cells']} | {order(a)} | {a['badges']} |
| Fresh-start PPO | {b['cells']} | {order(b)} | {b['badges']} |

The archive recorded {a.get('shorter_paths',0)} shorter paths to known cells and restored checkpoints {a.get('archive_resets',0)} times. It recorded {a.get('goals_reached',0)} hits from {a.get('goals_assigned',0)} assigned endpoints. Some assignments can still be active at the cutoff. A shorter stored path can arise from exploration alone.

## Independent fresh-game trials

Each controller received four fresh boots, 2,048 decisions per boot and zero checkpoint interventions. The two trained policies were frozen. The random controller sampled buttons uniformly. The trial counts below describe one training seed. The original 1,024-decision trials are preserved in the run directories ending in `-eval`. The longer follow-up gives each controller enough decisions to cover the length of the archived clock route.

| Controller | Trials | Reached truck | Reached Littleroot outdoors | Set clock | Got starter | Most badges |
|---|---:|---:|---:|---:|---:|---:|
"""
for name, *values in rows:
    label = {"archive-eval-2048": "Archive-conditioned PPO", "ppo-eval-2048": "Plain PPO",
             "random-eval-2048": "Uniform random"}[name]
    text += "| " + " | ".join([label] + list(map(str, values))) + " |\n"
text += f"""
## Compute and Red comparison

The PC has an Intel Core i7-13650HX (14 cores, 20 logical processors), about 16 GB of system RAM and an RTX 4060 Laptop GPU with 8,188 MiB of dedicated memory. PyTorch used CUDA 12.8. Peak allocated PyTorch tensor memory during the archive run was {a['peak_gpu_allocated_mib']} MiB. The driver, display and CUDA context also use memory.

The archive run took {wall:.1f} seconds end to end ({wall_sps:.1f} decisions/s). Inside the training loop it sustained {a['active_steps_per_second']} decisions/s. Collection, including action inference and worker communication, took {a['collect_seconds']} seconds. PPO updates took {a['ppo_update_seconds']} seconds. Estimates below assume the active rate holds during longer training and exclude startup, periodic writes and pauses.

| Interaction budget | Estimated continuous training time |
|---|---:|
"""
for n in (1_000_000, 10_000_000, 100_000_000, 252_000_000):
    text += f"| {n:,} decisions | {elapsed(n)} |\n"
text += """
Rubinstein's team reports seven hours for a tuned scripted Red run and a separate peak of 10,000 decisions/s. The 252 million row is the product of these two figures. The team's reported peak gives an illustrative upper-end workload for that time interval. The winning run's average throughput and exact decision count are unspecified in that account. Emerald introduces different exploration and learning demands. A time to the Hall of Fame requires its own milestone and success data. [Red training report](https://drubinstein.github.io/pokerl/docs/chapter-3/running/) · [Red results](https://drubinstein.github.io/pokerl/docs/chapter-4/results/)
"""
s = json.loads((folder / "worker-scaling.json").read_text(encoding="utf-8-sig"))
r = read("archive", "replay")
text += f"""
A second worker-count check used the first 8,192 decisions in each configuration. Four workers averaged {s['four_workers_active_sps']:.1f} active decisions/s. Eight workers averaged {s['eight_workers_active_sps']:.1f}. At the eight-worker rate, one million decisions would take {1e6/s['eight_workers_active_sps']/3600:.2f} hours and 252 million would take {252e6/s['eight_workers_active_sps']/86400:.1f} days. Different numbers of workers also change which states are visited, so this comparison measures practical throughput.

## Archive replay

The highest-progress archived endpoint was replayed from boot using {r['decisions']:,} recorded controller decisions. The recorded destination was {r['target_map']}. The replay reached {r['actual_map']} and {'matched' if r['matched'] else 'differed from'} the stored descriptor. The image gallery in `runs/archive/replayed-route.png` contains emulator frames collected during this replay. The independent trials above used the policy's own decisions and had 2,048 decisions to act.

The next milestone is repeatable clock setting and starter acquisition from boot across independent training seeds. Rock Badge evaluation follows once the agent can reach Rustboro.
"""
p = json.loads((folder / "profile-eight.json").read_text(encoding="utf-8-sig"))
total = p['collect_seconds'] + p['ppo_update_seconds']
f = (p['actor_call_seconds'] + p['ppo_update_seconds']) / total
text += f"""
## Desktop RTX 4090 estimate

The eight-worker timing run collected {p['steps']:,} decisions in {total:.2f} seconds of active training ({p['active_steps_per_second']:.1f} decisions/s). It spent {p['emulator_step_seconds']:.2f} seconds stepping workers and moving their results, {p['actor_call_seconds']:.2f} seconds preparing observations, running the actor and retrieving actions, and {p['ppo_update_seconds']:.2f} seconds updating PPO. The remaining time covered archive, reward and rollout bookkeeping. The model used {p['peak_gpu_allocated_mib']:.1f} MiB of peak allocated tensor memory.

Policy calls and updates together occupied {100*f:.1f}% of active time. If a desktop RTX 4090 made both of those sections 2× faster, the projected end-to-end speedup would be {1/(1-f+f/2):.2f}×. At a 4× improvement in both sections, it would be {1/(1-f+f/4):.2f}×. Their zero-time ceiling with all other sections held fixed is {1/(1-f):.2f}×. These are scenarios based on timings from the laptop. Both sections include host-side work.

A desktop RTX 4090 has 24 GB of VRAM. A larger world model, bigger batch or many more CPU emulator workers could change this profile. On a rental, run the same training workload across several worker counts and compare dollars per million real decisions. At an observed 1,000 decisions/s, 100 million decisions would take 27.8 hours. At 2,000 decisions/s, the same budget would take 13.9 hours. Those rates are rental selection targets. [NVIDIA RTX 4090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)
"""
(folder / "RESULTS.md").write_text(text, encoding="utf-8")
print(text)
