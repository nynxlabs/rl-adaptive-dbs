# Check Windows host prerequisites for Multipass + Windows Sandbox validation.
# Run on Windows (PowerShell as Administrator for full feature status):
#   pwsh -File scripts/validation/check-windows-host.ps1
# From WSL:
#   /mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -ExecutionPolicy Bypass -File scripts/validation/check-windows-host.ps1

$ErrorActionPreference = 'Continue'

function Get-OptionalFeatureState {
    param([string]$Name)
    try {
        $feat = Get-WindowsOptionalFeature -Online -FeatureName $Name -ErrorAction Stop
        return $feat.State
    } catch {
        return $null
    }
}

Write-Host '=== Windows host check (Multipass + Sandbox) ==='
Write-Host ''

$ci = Get-ComputerInfo -ErrorAction SilentlyContinue
if ($ci) {
    Write-Host "OS: $($ci.OsName) $($ci.OsVersion)"
} else {
    Write-Host "OS: $((Get-CimInstance Win32_OperatingSystem).Caption)"
}

Write-Host ''
Write-Host '=== Firmware virtualization ==='
try {
    $virt = (Get-CimInstance Win32_Processor).VirtualizationFirmwareEnabled
    Write-Host "VirtualizationFirmwareEnabled: $virt"
} catch {
    Write-Host 'VirtualizationFirmwareEnabled: (unknown)'
}

Write-Host ''
Write-Host '=== Hyper-V (heuristic + optional feature) ==='
$hyperInfo = systeminfo | Select-String -Pattern 'Hyper-V'
if ($hyperInfo) { $hyperInfo | ForEach-Object { Write-Host $_.Line.Trim() } }
$hyperFeature = Get-OptionalFeatureState 'Microsoft-Hyper-V-All'
if ($hyperFeature) {
    Write-Host "Microsoft-Hyper-V-All (DISM): $hyperFeature"
} else {
    Write-Host 'Microsoft-Hyper-V-All (DISM): (run PowerShell as Administrator to read)'
}
$hypervisorRunning = [bool](systeminfo | Select-String -Pattern 'A hypervisor has been detected')
Write-Host "Hypervisor active: $hypervisorRunning"

Write-Host ''
Write-Host '=== Windows Sandbox ==='
$sandboxExe = 'C:\Windows\System32\WindowsSandbox.exe'
$sandboxPresent = Test-Path $sandboxExe
Write-Host "WindowsSandbox.exe: $(if ($sandboxPresent) { 'present' } else { 'MISSING' })"
$sandboxFeature = Get-OptionalFeatureState 'Containers-DisposableClientVM'
if ($sandboxFeature) {
    Write-Host "Containers-DisposableClientVM (DISM): $sandboxFeature"
} else {
    Write-Host 'Containers-DisposableClientVM (DISM): (run PowerShell as Administrator to read)'
}

Write-Host ''
Write-Host '=== Multipass ==='
$mp = Get-Command multipass.exe -ErrorAction SilentlyContinue
if (-not $mp) { $mp = Get-Command multipass -ErrorAction SilentlyContinue }
if (-not $mp) {
    $defaultMp = 'C:\Program Files\Multipass\bin\multipass.exe'
    if (Test-Path $defaultMp) {
        Write-Host "multipass: $defaultMp"
        & $defaultMp version
    } else {
        Write-Host 'multipass: NOT installed — https://multipass.run'
    }
} else {
    Write-Host "multipass: $($mp.Source)"
    & $mp.Source version
}

Write-Host ''
Write-Host '=== Summary ==='
$hyperOk = $hypervisorRunning -or ($hyperFeature -eq 'Enabled')
$sandboxOk = $sandboxPresent -or ($sandboxFeature -eq 'Enabled')
$mpOk = $null -ne $mp -or (Test-Path 'C:\Program Files\Multipass\bin\multipass.exe')
Write-Host ("Hyper-V:   {0}" -f $(if ($hyperOk) { 'OK (hypervisor or feature enabled)' } else { 'ENABLE — Microsoft-Hyper-V-All' }))
Write-Host ("Sandbox:   {0}" -f $(if ($sandboxOk) { 'OK' } else { 'ENABLE — Containers-DisposableClientVM; reboot' }))
Write-Host ("Multipass: {0}" -f $(if ($mpOk) { 'OK' } else { 'INSTALL — https://multipass.run' }))

if (-not $sandboxOk) {
    Write-Host ''
    Write-Host 'Install Sandbox (Administrator PowerShell):'
    Write-Host '  pwsh -ExecutionPolicy Bypass -File scripts/validation/install-fresh-validation-host.ps1 -Sandbox'
}

if (-not $mpOk) {
    Write-Host ''
    Write-Host 'Install Multipass (Administrator PowerShell):'
    Write-Host '  pwsh -ExecutionPolicy Bypass -File scripts/validation/install-fresh-validation-host.ps1 -Multipass'
}

Write-Host ''
Write-Host 'Next: docs/development/fresh-validation.md'
Write-Host '      bash scripts/validation/validate-fresh.sh (inside Multipass or Sandbox Git Bash)'
