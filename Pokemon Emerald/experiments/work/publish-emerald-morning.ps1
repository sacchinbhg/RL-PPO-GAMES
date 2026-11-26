# Publish only the completed, portable Markdown summary to the user's requested repo.
$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$project = Join-Path $workspace 'outputs/EmeraldRL'
$night = Join-Path $project 'runs/native-night-20261124'
$local = Get-Content -Raw -LiteralPath (Join-Path $project 'local.json') | ConvertFrom-Json
$env:GIT_TERMINAL_PROMPT = '0'
$limit = [DateTimeOffset]::Parse('2026-11-25T16:00:00-08:00')
function Read-State ($path) {
    if (Test-Path -LiteralPath $path) {
        try { return (Get-Content -Raw -LiteralPath $path | ConvertFrom-Json) } catch { return $null }
    }
}
do {
    $a = Read-State (Join-Path $project 'runs/world-12h-20261124/status.json')
    $b = Read-State (Join-Path $night 'pipeline.json')
    if ($a.phase -like 'Completed*' -and $b.phase -eq 'Completed' -and
        (Test-Path -LiteralPath (Join-Path $night 'transfer/evaluation.json'))) { break }
    if ($b.phase -eq 'Interrupted' -or $a.phase -eq 'Interrupted' -or [DateTimeOffset]::Now -gt $limit) {
        throw 'A run needs review. Its live local report and raw results are available.'
    }
    Start-Sleep -Seconds 45
} while ($true)

$snapshot = Join-Path $project 'results/OVERNIGHT_RESULTS.md'
Push-Location $project
try {
    & $local.python morning_report.py --original runs/world-12h-20261124 --native runs/native-night-20261124/long --verification runs/native-night-20261124/verification --pilot runs/native-night-20261124/pilot --transfer runs/native-night-20261124/transfer --out $snapshot
    if ($LASTEXITCODE -ne 0) { throw 'Could not generate the final report' }
} finally { Pop-Location }
$text = [IO.File]::ReadAllText($snapshot)
if ($text -match '(?i)[A-Z]:[\\/]|/Users/|/home/|ghp_') { throw 'Local-only details in report; kept locally for review.' }
$check = Join-Path $PSScriptRoot 'emerald-final-publication'
for ($attempt=1; $attempt -le 6; $attempt++) {
    if (!(Test-Path (Join-Path $check '.git'))) {
        git -c credential.interactive=never clone --depth 1 https://github.com/sacchinbhg/RL-PPO-GAMES.git $check
        if ($LASTEXITCODE -ne 0) { Start-Sleep -Seconds 45; continue }
    }
    $target = Join-Path $check 'Pokemon Emerald/OVERNIGHT_RESULTS.md'
    Copy-Item -LiteralPath $snapshot -Destination $target
    git -C $check add -- 'Pokemon Emerald/OVERNIGHT_RESULTS.md'
    git -C $check diff --cached --quiet
    if ($LASTEXITCODE -eq 1) {
        git -C $check -c user.name=Codex -c user.email=codex@users.noreply.github.com commit -m 'Record completed Emerald overnight experiments and ROM transfer evaluation'
        if ($LASTEXITCODE -ne 0) { throw 'Could not commit report' }
    }
    git -C $check -c credential.interactive=never pull --rebase origin main
    if ($LASTEXITCODE -ne 0) { throw 'Remote changed; local final report preserved.' }
    git -C $check -c credential.interactive=never push origin HEAD:main
    if ($LASTEXITCODE -eq 0) {
        git -C $check rev-parse HEAD
        exit 0
    }
    Start-Sleep -Seconds 45
}
throw 'Final report saved locally; remote publication retries exhausted.'
