$ErrorActionPreference='Stop'
$p=(Resolve-Path (Join-Path $PSScriptRoot '..\outputs\EmeraldRL')).Path
$cfg=Get-Content (Join-Path $p 'local.json') -Raw | ConvertFrom-Json
$base=Join-Path $p 'runs\world-comparison'
$ready=Join-Path $base 'world-71-eval\evaluation.json'
for ($n=0; $n -lt 1000 -and !(Test-Path -LiteralPath $ready); $n++){Start-Sleep -Seconds 2}
if (!(Test-Path -LiteralPath $ready)) {throw 'World-model runs incomplete'}
& $cfg.python (Join-Path $p 'experiment.py') eval --rom $cfg.rom --core $cfg.core --envs 8 --episode 2048 --seed 4709 --out (Join-Path $base 'random-eval')
if ($LASTEXITCODE -ne 0) {throw 'Random evaluation failed'}
$ckpts = @(19,37,71) | ForEach-Object { Join-Path $base "world-$_\checkpoint.pt" }
& $cfg.python (Join-Path $p 'verify_dynamics.py') --checkpoints @ckpts --rom $cfg.rom --core $cfg.core --out (Join-Path $base 'heldout.json')
if ($LASTEXITCODE -ne 0) {throw 'Dynamics audit failed'}
& $cfg.python (Join-Path $p 'replay.py') (Join-Path $base 'world-19\checkpoint.pt') --out (Join-Path $base 'world-19\replay.json') --gallery (Join-Path $base 'world-19\route.png')
if ($LASTEXITCODE -ne 0) {throw 'Archived route replay failed'}
