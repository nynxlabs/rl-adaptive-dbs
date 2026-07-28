# Launch Multipass Ubuntu, clone from GitHub, run validate-fresh.sh. Run on Windows desktop PowerShell.
#   pwsh -ExecutionPolicy Bypass -File scripts/validation/run-multipass-linux-validation.ps1
#
# Options:
#   -Memory 3G     # default; raise to 4G if host has plenty of free RAM
#   -KeepVm        # do not delete VM when finished
#
# Host log: <WSL repo>/.validation-logs/multipass.log (gitignored)

param(
    [string]$VmName = 'rl-dbs-linux',
    [string]$Memory = '3G',
    [switch]$KeepVm
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'validation-repo.ps1')

function Get-Multipass {
    $cmd = Get-Command multipass.exe -ErrorAction SilentlyContinue
    if (-not $cmd) { $cmd = Get-Command multipass -ErrorAction SilentlyContinue }
    if (-not $cmd) {
        $default = 'C:\Program Files\Multipass\bin\multipass.exe'
        if (Test-Path $default) { return $default }
        Write-Error 'multipass not found. Run: pwsh -File scripts/validation/install-fresh-validation-host.ps1 -Multipass'
    }
    return $cmd.Source
}

$mp = Get-Multipass
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-ValidationRepoPath -CallerScriptDir $scriptDir
$logDir = Get-ValidationLogDir -RepoRoot $repoRoot
$hostLog = Join-Path $logDir 'multipass.log'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$bootstrapSh = Join-Path $scriptDir 'bootstrap-fresh-linux.sh'
if (-not (Test-Path $bootstrapSh)) {
    Write-Error "Missing $bootstrapSh"
}

Write-Host "=== Multipass Linux validation ($VmName) ==="
Write-Host "  Mode:      git clone inside VM (fresh Linux)"
Write-Host "  Memory:    $Memory"
Write-Host "  Host log:  $hostLog"
Write-Host ''

$prevEap = $ErrorActionPreference
$ErrorActionPreference = 'SilentlyContinue'
& $mp delete $VmName --purge 2>$null | Out-Null
$ErrorActionPreference = $prevEap

& $mp launch 24.04 --name $VmName --cpus 2 --memory $Memory --disk 20G

$remoteB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes((Get-Content -Raw -LiteralPath $bootstrapSh)))
$remoteCmd = "echo $remoteB64 | base64 -d | bash"

Write-Host 'Running validation inside VM (clone + validate; may take several minutes)...'
"=== multipass validation started $(Get-Date -Format o) ===" | Set-Content -Path $hostLog -Encoding UTF8
# multipass writes progress to stderr; do not let PowerShell treat that as a terminating error.
$prevEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $mp exec $VmName -- bash -lc $remoteCmd 2>&1 | Tee-Object -FilePath $hostLog -Append -Encoding utf8
$code = $LASTEXITCODE
$ErrorActionPreference = $prevEap
"=== multipass validation finished exit=$code $(Get-Date -Format o) ===" | Add-Content -Path $hostLog

if (-not $KeepVm) {
    Write-Host "Removing VM $VmName..."
    & $mp delete $VmName --purge | Out-Null
}

exit $code
