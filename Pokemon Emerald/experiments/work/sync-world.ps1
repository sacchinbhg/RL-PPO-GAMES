param([switch]$Initial)
$ErrorActionPreference='Stop'
$p=(Resolve-Path (Join-Path $PSScriptRoot '..\outputs\EmeraldRL')).Path
$d=(Resolve-Path 'C:\Users\sacch\OneDrive\Desktop\Pokemon RL\EmeraldRL').Path
$runs=Join-Path $p 'runs\world-comparison'
$longPrefix=Join-Path $runs 'long\'
$files=@(Get-ChildItem -LiteralPath $p -File) + @(Get-ChildItem -LiteralPath $runs -File -Recurse |
    Where-Object {$_.Extension -ne '.tmp' -and (!$Initial -or !$_.FullName.StartsWith($longPrefix,[StringComparison]::OrdinalIgnoreCase))})
$copied=0
foreach($f in $files){
    $target=Join-Path $d ([IO.Path]::GetRelativePath($p,$f.FullName))
    $sha=(Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash
    if((Test-Path -LiteralPath $target) -and (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -eq $sha){continue}
    $parent=Split-Path $target
    if(!(Test-Path -LiteralPath $parent)){New-Item -ItemType Directory -Path $parent -Force | Out-Null}
    Copy-Item -LiteralPath $f.FullName -Destination $target -Force
    if((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $sha){throw "Checksum differs at $target"}
    $copied++
}
"Verified $($files.Count) files in $d ($copied copied)"
