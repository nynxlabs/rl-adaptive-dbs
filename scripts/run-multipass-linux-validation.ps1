# Launch Multipass Ubuntu, bootstrap git+uv, run validate-fresh.sh. Run on Windows desktop PowerShell.
#   pwsh -ExecutionPolicy Bypass -File scripts/run-multipass-linux-validation.ps1
#
# Options:
#   -RepoPath \\wsl.localhost\Ubuntu\home\...\rl-adaptive-dbs   # mount working tree (preferred)
#   -Memory 2G                                                # default; use 4G if host has free RAM
#   -KeepVm                                                   # do not delete VM when finished

param(
    [string]$VmName = 'rl-dbs-linux',
    [string]$RepoPath = '',
    [string]$Memory = '2G',
    [switch]$KeepVm
)

$ErrorActionPreference = 'Stop'

function Get-Multipass {
    $cmd = Get-Command multipass.exe -ErrorAction SilentlyContinue
    if (-not $cmd) { $cmd = Get-Command multipass -ErrorAction SilentlyContinue }
    if (-not $cmd) {
        $default = 'C:\Program Files\Multipass\bin\multipass.exe'
        if (Test-Path $default) { return $default }
        Write-Error 'multipass not found. Run scripts/prepare-desktop-host.ps1 or install from https://multipass.run'
    }
    return $cmd.Source
}

$mp = Get-Multipass
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$bootstrapSh = Join-Path $scriptDir 'bootstrap-fresh-linux.sh'
if (-not (Test-Path $bootstrapSh)) {
    Write-Error "Missing $bootstrapSh"
}

Write-Host "=== Multipass Linux validation ($VmName) ==="

$prevEap = $ErrorActionPreference
$ErrorActionPreference = 'SilentlyContinue'
& $mp delete $VmName --purge 2>$null | Out-Null
$ErrorActionPreference = $prevEap
& $mp launch 24.04 --name $VmName --cpus 2 --memory $Memory --disk 20G --timeout 1200

$mountArgs = @()
if ($RepoPath -and (Test-Path $RepoPath)) {
    Write-Host "Mounting repo: $RepoPath -> /mnt/rl-adaptive-dbs"
    & $mp mount $RepoPath "${VmName}:/mnt/rl-adaptive-dbs"
    $remoteCmd = @'
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
if ! command -v git >/dev/null; then sudo apt-get update && sudo apt-get install -y git curl ca-certificates; fi
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
source "$HOME/.local/bin/env" 2>/dev/null || true
cd /mnt/rl-adaptive-dbs
bash scripts/validate-fresh.sh
'@
} else {
    $remoteB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes((Get-Content -Raw -LiteralPath $bootstrapSh)))
    $remoteCmd = "echo $remoteB64 | base64 -d | bash"
}

Write-Host 'Running validation inside VM (may take several minutes)...'
& $mp exec $VmName -- bash -lc $remoteCmd
$code = $LASTEXITCODE

if (-not $KeepVm) {
    Write-Host "Removing VM $VmName..."
    & $mp delete $VmName --purge | Out-Null
}

exit $code
