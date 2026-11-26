$ErrorActionPreference='Stop'
$p=(Resolve-Path (Join-Path $PSScriptRoot '..\outputs\EmeraldRL')).Path
$cfg=Get-Content (Join-Path $p 'local.json') -Raw | ConvertFrom-Json
$base=Join-Path $p 'runs\world-comparison'
$long=Join-Path $base 'long'
$ready=Join-Path $long 'world-98304-eval\evaluation.json'
for ($i=0; $i -lt 1200 -and !(Test-Path -LiteralPath $ready); $i++){Start-Sleep -Seconds 2}
if (!(Test-Path -LiteralPath $ready)){throw 'Matched extensions still incomplete'}
foreach ($total in @(65536,98304)) {
    $src=if($total -eq 65536){Join-Path $base 'world-19'}else{Join-Path $long 'full-65536'}
    $stream=if($total -eq 65536){17019}else{27019}
    $dest=Join-Path $long "full-$total"
    $eval=Join-Path $long "full-$total-eval"
    $timer=[Diagnostics.Stopwatch]::StartNew()
    & $cfg.python -u (Join-Path $p 'experiment.py') train --rom $cfg.rom --core $cfg.core --out $dest --mode world --imagination-update full-bptt --seed $stream --steps 32768 --envs 8 --episode 512 --restart-from (Join-Path $src 'checkpoint.pt')
    if ($LASTEXITCODE -ne 0){throw "Full BPTT training failed at $total"}
    @{wall_seconds=$timer.Elapsed.TotalSeconds} | ConvertTo-Json | Set-Content (Join-Path $dest 'elapsed.json')
    & $cfg.python -u (Join-Path $p 'experiment.py') eval --rom $cfg.rom --core $cfg.core --out $eval --mode world --seed 4709 --episode 2048 --envs 8 --weights (Join-Path $dest 'policy.pt')
    if ($LASTEXITCODE -ne 0){throw "Full BPTT eval failed at $total"}
}
