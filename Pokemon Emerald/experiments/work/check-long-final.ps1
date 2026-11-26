param([ValidateSet(65536,98304)][int]$Budget=98304)
$ErrorActionPreference='Stop'
$proj=(Resolve-Path (Join-Path $PSScriptRoot '..\outputs\EmeraldRL')).Path
$cfg=Get-Content (Join-Path $proj 'local.json') -Raw|ConvertFrom-Json
$path=Join-Path $proj 'runs\world-comparison\long'
$ready=Join-Path $path 'full-98304-eval\evaluation.json'
for ($i=0; $i -lt 1400 -and !(Test-Path -LiteralPath $ready); $i++){Start-Sleep -Seconds 2}
if (!(Test-Path -LiteralPath $ready)){throw 'The final training checkpoint was not evaluated'}
foreach($name in @('archive','world','full')){
    $trained=Join-Path $path "$name-$Budget"
    $mode=if($name -eq 'archive'){'archive'}else{'world'}
    & $cfg.python -u (Join-Path $proj 'experiment.py') eval --rom $cfg.rom --core $cfg.core --out (Join-Path $path "$name-$Budget-confirm") --seed 92741 --mode $mode --episode 2048 --envs 8 --weights (Join-Path $trained 'policy.pt') --no-capture-moments
    if($LASTEXITCODE -ne 0){throw "Fresh evaluation seed failed for $name"}
}
