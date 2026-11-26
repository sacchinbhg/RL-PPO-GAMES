$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot
$project = Join-Path $root 'outputs\EmeraldRL'
$cfg = Get-Content (Join-Path $project 'local.json') -Raw | ConvertFrom-Json
$rom = $cfg.rom
$finished = $false
for ($i=0; $i -lt 480; $i++) {
    if (Test-Path -LiteralPath $rom) {
        if ((Get-FileHash -LiteralPath $rom -Algorithm SHA1).Hash -eq 'F3AE088181BF583E55DAF962A92BB46F4F1D07B7') {
            $finished = $true
            break
        }
    }
    Start-Sleep -Seconds 2
}
if (!$finished) { throw 'Timed out waiting for the verified Emerald build.' }
& $cfg.python -X faulthandler (Join-Path $PSScriptRoot 'smoke_core.py') $rom
if ($LASTEXITCODE -ne 0) { throw 'Real Emerald replay check failed.' }
$runs = Join-Path $project 'runs'
foreach ($name in 'archive','archive-eval','random-eval','ppo','ppo-eval') {
    $out = Join-Path $runs $name
    $isEval = $name.EndsWith('-eval')
    $isRandom = $name -eq 'random-eval'
    $kind = if ($isEval) { 'eval' } else { 'train' }
    $argList = @($kind, '--out', $out, '--core', $cfg.core, '--rom', $rom, '--envs', 4)
    if ($isEval) {
        $argList += @('--episode', 1024, '--seed', 901)
        if (!$isRandom) {
            $mode = $name.Replace('-eval','')
            $argList += @('--weights', (Join-Path $runs "$mode\policy.pt"))
        }
    } else {
        $argList += @('--mode', $name, '--steps', 16384, '--episode', 512, '--seed', 19)
    }
    $timer = [Diagnostics.Stopwatch]::StartNew()
    & $cfg.python -u (Join-Path $project 'experiment.py') @argList
    if ($LASTEXITCODE -ne 0) { throw "Run failed for $name" }
    $timer.Stop()
    @{ wall_seconds=$timer.Elapsed.TotalSeconds } | ConvertTo-Json | Set-Content (Join-Path $out 'timing.json')
}
