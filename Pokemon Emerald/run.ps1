param(
    [Parameter(Mandatory, Position=0)][ValidateSet('train','eval')][string]$Command,
    [Parameter(Mandatory, Position=1)][string]$Out,
    [ValidateSet('archive','ppo','world')][string]$Mode = 'archive',
    [ValidateSet('mgba','native')][string]$Backend = 'mgba',
    [int]$Steps = 16384, [int]$Envs = 4, [int]$Episode = 512,
    [int]$Seed = 19, [ValidateSet('cuda','cpu')][string]$Device = 'cuda',
    [string]$Weights, [string]$RestartFrom, [ValidateSet('root','full-bptt')][string]$ImaginationUpdate
)
$ErrorActionPreference = 'Stop'
$cfg = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'local.json') -Raw | ConvertFrom-Json
$core = if ($Backend -eq 'native') { $cfg.native_core } else { $cfg.core }
$runArgs = @((Join-Path $PSScriptRoot 'experiment.py'), $Command, '--rom', $cfg.rom,
    '--backend', $Backend, '--core', $core, '--out', $Out, '--steps', $Steps, '--envs', $Envs,
    '--episode', $Episode, '--seed', $Seed, '--device', $Device, '--mode', $Mode)
if ($Weights) { $runArgs += @('--weights', $Weights) }
if ($RestartFrom) { $runArgs += @('--restart-from', $RestartFrom) }
if ($ImaginationUpdate) { $runArgs += @('--imagination-update', $ImaginationUpdate) }
& $cfg.python @runArgs
exit $LASTEXITCODE
