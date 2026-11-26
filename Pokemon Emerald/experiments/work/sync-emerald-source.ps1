$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$src = Join-Path $root 'outputs/EmeraldRL'
$repo = Join-Path $PSScriptRoot 'publish-emerald'
$dst = Join-Path $repo 'Pokemon Emerald'
$desktop = 'C:\Users\sacch\OneDrive\Desktop\Pokemon RL\EmeraldRL'
New-Item -ItemType Directory -Path $dst -Force | Out-Null
$sources = @('emerald.py','experiment.py','overnight.py','policy.py','pool_gba.py',
    'world_model.py','test_world_model.py','verify_dynamics.py','verify_goals.py','verify_update.py',
    'replay.py','replay_discovery.py','run.ps1','maps.json','native_emerald.py','verify_native.py',
    'morning_report.py','native_night.py','NATIVE_RESULTS.md','local.example.json','.gitignore','.gitattributes')
foreach ($f in $sources) {
    Copy-Item -LiteralPath (Join-Path $src $f) -Destination (Join-Path $dst $f)
    Copy-Item -LiteralPath (Join-Path $src $f) -Destination (Join-Path $desktop $f)
}
foreach ($dest in @($dst,$desktop)) {
    $dir = Join-Path $dest 'native'
    New-Item -Path $dir -ItemType Directory -Force | Out-Null
    Get-ChildItem -LiteralPath (Join-Path $src 'native') -File | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $dir $_.Name)
    }
    $results = Join-Path $dest 'results'
    New-Item -Path $results -ItemType Directory -Force | Out-Null
    Get-ChildItem -LiteralPath (Join-Path $src 'results') -Filter 'native-*.json' -File | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $results $_.Name)
    }
}
Copy-Item -LiteralPath (Join-Path $src 'TRAINING.md') -Destination (Join-Path $dst 'README.md')
Copy-Item -LiteralPath (Join-Path $src 'TRAINING.md') -Destination (Join-Path $desktop 'TRAINING.md')
Copy-Item -LiteralPath (Join-Path $src 'README.md') -Destination (Join-Path $desktop 'README.md')
Copy-Item -LiteralPath (Join-Path $src 'YOUTUBE_NOTES.md') -Destination (Join-Path $desktop 'YOUTUBE_NOTES.md')
$readme = Join-Path $repo 'README.md'
$original = [IO.File]::ReadAllText($readme)
if (!$original.Contains('Pokemon%20Emerald')) {
    [IO.File]::AppendAllText($readme, "`n`n# 3. Pokémon Emerald`n`n[Pokémon Emerald training code](Pokemon%20Emerald/README.md) includes the PPO and world-model learners, mGBA and native game backends, and fresh-game evaluation.`n")
}
