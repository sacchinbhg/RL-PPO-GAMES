$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot
$project = Join-Path $root 'outputs\EmeraldRL'
$cfg = Get-Content (Join-Path $project 'local.json') -Raw | ConvertFrom-Json
$runDir = Join-Path $project 'runs\archive'
$ready = Join-Path $project 'runs\ppo-eval\timing.json'
for ($i=0; $i -lt 180 -and !(Test-Path -LiteralPath $ready); $i++) { Start-Sleep -Seconds 2 }
if (!(Test-Path -LiteralPath $ready)) { throw 'Main pilot has not finished' }
& $cfg.python (Join-Path $project 'replay.py') (Join-Path $runDir 'checkpoint.pt') --out (Join-Path $runDir 'replay.json') --gallery (Join-Path $runDir 'replayed-route.png')
if ($LASTEXITCODE -ne 0) { throw 'Replay failed' }
$bench = Join-Path $PSScriptRoot 'eight-workers'
$timer = [Diagnostics.Stopwatch]::StartNew()
& $cfg.python (Join-Path $project 'experiment.py') train --core $cfg.core --rom $cfg.rom --envs 8 --steps 8192 --episode 512 --seed 19 --out $bench
if ($LASTEXITCODE -ne 0) { throw 'Eight-worker benchmark failed' }
$timer.Stop()
$row = Get-Content (Join-Path $bench 'training.json') -Raw | ConvertFrom-Json
$baseline = Get-Content (Join-Path $runDir 'metrics.jsonl') | ForEach-Object {$_ | ConvertFrom-Json} | Where-Object steps -eq 8192
@{ steps_each=8192; four_workers_active_sps=$baseline.active_steps_per_second; eight_workers_active_sps=$row.active_steps_per_second; eight_workers_wall_seconds=$timer.Elapsed.TotalSeconds; eight_workers_peak_allocated_mib=$row.peak_gpu_allocated_mib } | ConvertTo-Json | Set-Content (Join-Path $project 'worker-scaling.json')
