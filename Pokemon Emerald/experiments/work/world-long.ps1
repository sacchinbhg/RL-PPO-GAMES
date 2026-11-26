$ErrorActionPreference='Stop'
$p=(Resolve-Path (Join-Path $PSScriptRoot '..\outputs\EmeraldRL')).Path
$cfg=Get-Content (Join-Path $p 'local.json') -Raw | ConvertFrom-Json
$base=Join-Path $p 'runs\world-comparison'
foreach($ready in @('heldout.json','goal-probe.json','world-19-capture\evaluation.json','world-19\replay.json')) {
    for ($n=0; $n -lt 650 -and !(Test-Path -LiteralPath (Join-Path $base $ready)); $n++){Start-Sleep -Seconds 2}
    if (!(Test-Path -LiteralPath (Join-Path $base $ready))) {throw "Missing $ready"}
}
$weights=@(19,37,71) | ForEach-Object {Join-Path $base "world-$_\checkpoint.pt"}
& $cfg.python (Join-Path $p 'verify_update.py') --checkpoints @weights --out (Join-Path $base 'gradient-control.json')
if ($LASTEXITCODE -ne 0){throw 'Optimizer control failed'}
$long=Join-Path $base 'long'
# Extend both from their paired seed-19 checkpoints using identical fresh RNG seeds per block.
foreach ($total in @(65536,98304)) {
    $modes=if($total -eq 65536){@('world','archive')}else{@('archive','world')}
    $stream=if($total -eq 65536){17019}else{27019}
    foreach($mode in $modes){
        $orig=if($total -eq 65536){Join-Path $base "$mode-19"}else{Join-Path $long "$mode-65536"}
        $dest=Join-Path $long "$mode-$total"
        $test=Join-Path $long "$mode-$total-eval"
        $timer=[Diagnostics.Stopwatch]::StartNew()
        & $cfg.python -u (Join-Path $p 'experiment.py') train --rom $cfg.rom --core $cfg.core --out $dest --mode $mode --seed $stream --steps 32768 --envs 8 --episode 512 --restart-from (Join-Path $orig 'checkpoint.pt')
        if ($LASTEXITCODE -ne 0){throw "Long train failed $mode $total"}
        @{wall_seconds=$timer.Elapsed.TotalSeconds} | ConvertTo-Json | Set-Content (Join-Path $dest 'elapsed.json')
        & $cfg.python -u (Join-Path $p 'experiment.py') eval --rom $cfg.rom --core $cfg.core --out $test --mode $mode --seed 4709 --episode 2048 --envs 8 --weights (Join-Path $dest 'policy.pt')
        if ($LASTEXITCODE -ne 0){throw "Long eval failed $mode $total"}
    }
}
