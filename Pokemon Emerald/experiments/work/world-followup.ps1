$ErrorActionPreference='Stop'
$project=(Resolve-Path (Join-Path $PSScriptRoot '..\outputs\EmeraldRL')).Path
$cfg=Get-Content (Join-Path $project 'local.json') -Raw | ConvertFrom-Json
$runs=Join-Path $project 'runs\world-comparison'
$ready=Join-Path $runs 'world-71-eval\evaluation.json'
for ($i=0; $i -lt 360 -and !(Test-Path -LiteralPath $ready); $i++){Start-Sleep -Seconds 2}
if (!(Test-Path -LiteralPath $ready)) {throw 'Timing jobs still active'}
$weights=@(19,37,71) | ForEach-Object {Join-Path $runs "world-$_\checkpoint.pt"}
& $cfg.python (Join-Path $project 'verify_goals.py') --rom $cfg.rom --core $cfg.core --checkpoints @weights --out (Join-Path $runs 'goal-probe.json')
if ($LASTEXITCODE -ne 0) { throw 'Goal probe failed' }
$extra=Join-Path $PSScriptRoot 'world-extension-check'
& $cfg.python (Join-Path $project 'experiment.py') train --rom $cfg.rom --core $cfg.core --out $extra --mode world --seed 19 --steps 512 --envs 8 --restart-from $weights[0]
if ($LASTEXITCODE -ne 0) {throw 'World checkpoint extension failed'}
$t=Get-Content (Join-Path $extra 'training.json') -Raw | ConvertFrom-Json
if ($t.parent_steps -ne 32768 -or $t.total_training_steps -ne 33280 -or $t.dynamics_updates -ne 2080 -or $t.imagined_steps -ne 23808) {throw 'World counters failed to continue'}
Copy-Item -LiteralPath (Join-Path $extra 'training.json') -Destination (Join-Path $runs 'restart-check.json')
$capture=Join-Path $runs 'world-19-capture'
& $cfg.python (Join-Path $project 'experiment.py') eval --rom $cfg.rom --core $cfg.core --out $capture --mode world --seed 4709 --episode 2048 --envs 8 --weights (Join-Path $runs 'world-19\policy.pt') --capture-moments
if ($LASTEXITCODE -ne 0) {throw 'Independent gameplay capture failed'}
