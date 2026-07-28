# Run Multipass Linux + Windows Sandbox fresh validation in parallel.
#   pwsh -ExecutionPolicy Bypass -File scripts/validation/run-parallel-fresh-validation.ps1
#
# Multipass memory is sized from host available RAM before Sandbox starts (3G floor for parallel).
# Sandbox has no MemoryInMB cap; Windows manages memory dynamically.
# Multipass launches first; Sandbox opens once the VM is Running (avoids boot RAM contention).

param(
    [string]$Memory = '',   # override auto (e.g. 4G)
    [string]$LogDir = ''
)

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $scriptDir 'validation-repo.ps1')

$repoRoot = Resolve-ValidationRepoPath -CallerScriptDir $scriptDir

if (-not $LogDir) {
    $LogDir = Get-ValidationLogDir -RepoRoot $repoRoot
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Get-AvailableRamGb {
    [math]::Round((Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue / 1024, 1)
}

function Get-MultipassMemory {
    param([double]$AvailGb, [string]$Override)
    if ($Override) { return $Override }
    if ($AvailGb -ge 14) { return '4G' }
    return '3G'   # parallel floor: no 2G cap (torch + uv need headroom inside VM)
}

function Get-MultipassExe {
    $cmd = Get-Command multipass.exe -ErrorAction SilentlyContinue
    if (-not $cmd) { $cmd = Get-Command multipass -ErrorAction SilentlyContinue }
    if ($cmd) { return $cmd.Source }
    $default = 'C:\Program Files\Multipass\bin\multipass.exe'
    if (Test-Path $default) { return $default }
    Write-Error 'multipass not found'
}

$availGb = Get-AvailableRamGb
$mem = Get-MultipassMemory -AvailGb $availGb -Override $Memory
$mpExe = Get-MultipassExe
$vmName = 'rl-dbs-linux'
$mpLog = Join-Path $LogDir 'multipass-launcher.log'
$mpErr = Join-Path $LogDir 'multipass-launcher.err.log'
$mpValidationLog = Join-Path $LogDir 'multipass.log'
$sandboxValidationLog = Join-Path $LogDir 'sandbox.log'
$statusFile = Join-Path $LogDir 'status.txt'
$sandboxNote = Join-Path $LogDir 'sandbox.txt'

Write-Host "=== Parallel fresh validation ==="
Write-Host "  Log dir:   $LogDir"
Write-Host "  Host RAM:  ${availGb} GB available (before Sandbox)"
Write-Host "  Multipass: $mem (fixed VM allocation; Sandbox uncapped, dynamic host memory)"
Write-Host "  Order:     Multipass launch first, Sandbox after VM is Running"
Write-Host ""

$mpScript = Join-Path $scriptDir 'run-multipass-linux-validation.ps1'
$sandboxScript = Join-Path $scriptDir 'launch-windows-sandbox-validation.ps1'

if ($availGb -lt 6) {
    Write-Warning "Low host RAM (${availGb} GB). Parallel run may hang or OOM. Consider closing apps."
}

Write-Host "Starting Multipass validation (background process)..."
$mpArgs = @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $mpScript,
    '-Memory', $mem
)
$mpProc = Start-Process -FilePath 'pwsh' -ArgumentList $mpArgs `
    -RedirectStandardOutput $mpLog -RedirectStandardError $mpErr `
    -PassThru -WindowStyle Hidden

Write-Host 'Waiting for Multipass VM to reach Running (no timeout)...'
$running = $false
while (-not $mpProc.HasExited) {
    $list = & $mpExe list 2>&1 | Out-String
    if ($list -match "${vmName}\s+Running") {
        $running = $true
        Write-Host 'Multipass VM is Running.'
        break
    }
    if ($list -match "${vmName}\s+Stopped") {
        Write-Error "Multipass VM stopped unexpectedly during launch. See $mpLog"
    }
    Start-Sleep -Seconds 15
}

if (-not $running -and -not $mpProc.HasExited) {
    Write-Warning 'Multipass still not Running; launching Sandbox anyway (logs may contend).'
}

Write-Host "Starting Windows Sandbox validation..."
& pwsh -NoProfile -ExecutionPolicy Bypass -File $sandboxScript

@"
Started: $(Get-Date -Format o)
Log dir: $LogDir
Multipass memory: $mem
Host available RAM at start: ${availGb} GB
Multipass PID: $($mpProc.Id)
Multipass launcher log: $mpLog
Multipass validation log: $mpValidationLog
Sandbox validation log: $sandboxValidationLog
"@ | Set-Content -Path $sandboxNote

Write-Host ""
Write-Host "Sandbox launched. Multipass PID $($mpProc.Id) running."
Write-Host "  Multipass log: $mpValidationLog"
Write-Host "  Sandbox log:   $sandboxValidationLog"
Write-Host "  Waiting for Multipass validation to finish (may take 10-25 min)..."
Write-Host ""

$mpProc.WaitForExit()
$mpExit = $mpProc.ExitCode

@"
Finished: $(Get-Date -Format o)
Multipass exit code: $mpExit
"@ | Set-Content -Path $statusFile

Write-Host "=== Multipass finished (exit $mpExit) ==="
if (Test-Path $mpLog) {
    Get-Content $mpLog -Tail 25 | ForEach-Object { Write-Host $_ }
}

Write-Host ""
Write-Host "=== Summary ==="
Write-Host "  Multipass exit: $mpExit - validation log: $mpValidationLog"
Write-Host "  Sandbox:        $sandboxValidationLog"
Write-Host "  Status:         $statusFile"

exit $mpExit
