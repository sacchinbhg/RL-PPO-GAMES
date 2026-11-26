param([ValidateSet('archive','world')][string]$Mode, [int[]]$Seeds=@(19,37,71))
$ErrorActionPreference = 'Stop'
$p = (Resolve-Path (Join-Path $PSScriptRoot '..\outputs\EmeraldRL')).Path
$cfg = Get-Content (Join-Path $p 'local.json') -Raw | ConvertFrom-Json
$entry = Join-Path $p 'experiment.py'
$parent = Join-Path $p 'runs\world-comparison'
foreach($seed in $Seeds) {
    $dest = Join-Path $parent "$Mode-$seed"
    $test = Join-Path $parent "$Mode-$seed-eval"
    if (!(Test-Path -LiteralPath (Join-Path $dest 'elapsed.json'))) {
        $t = [Diagnostics.Stopwatch]::StartNew()
        & $cfg.python -u $entry train --rom $cfg.rom --core $cfg.core --out $dest --mode $Mode --seed $seed --steps 32768 --envs 8 --episode 512
        if ($LASTEXITCODE -ne 0) { throw "Training failed $Mode $seed" }
        @{ wall_seconds=$t.Elapsed.TotalSeconds } | ConvertTo-Json | Set-Content (Join-Path $dest 'elapsed.json')
    }
    if (!(Test-Path -LiteralPath (Join-Path $test 'evaluation.json'))) {
        & $cfg.python -u $entry eval --rom $cfg.rom --core $cfg.core --out $test --mode $Mode --seed 4709 --episode 2048 --envs 8 --weights (Join-Path $dest 'policy.pt')
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed $Mode $seed" }
    }
}
