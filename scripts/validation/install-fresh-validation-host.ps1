# Install Windows Sandbox and/or Multipass for fresh-validation (Phase 4).
# Run as Administrator on the Windows desktop (not inside WSL).
#
#   pwsh -ExecutionPolicy Bypass -File scripts/validation/install-fresh-validation-host.ps1
#   pwsh -ExecutionPolicy Bypass -File scripts/validation/install-fresh-validation-host.ps1 -Sandbox
#   pwsh -ExecutionPolicy Bypass -File scripts/validation/install-fresh-validation-host.ps1 -Multipass
#
# After reboot (if prompted): pwsh -File scripts/validation/check-windows-host.ps1

#Requires -RunAsAdministrator
param(
    [switch]$Sandbox,
    [switch]$Multipass
)

$ErrorActionPreference = 'Stop'

$installSandbox = $Sandbox.IsPresent
$installMultipass = $Multipass.IsPresent
if (-not $installSandbox -and -not $installMultipass) {
    $installSandbox = $true
    $installMultipass = $true
}

function Test-WindowsSandboxReady {
    if (Test-Path "$env:WINDIR\System32\WindowsSandbox.exe") { return $true }
    try {
        $feat = Get-WindowsOptionalFeature -Online -FeatureName Containers-DisposableClientVM -ErrorAction Stop
        return $feat.State -eq 'Enabled'
    } catch {
        return $false
    }
}

function Test-MultipassInstalled {
    if (Get-Command multipass.exe -ErrorAction SilentlyContinue) { return $true }
    if (Get-Command multipass -ErrorAction SilentlyContinue) { return $true }
    return Test-Path 'C:\Program Files\Multipass\bin\multipass.exe'
}

function Enable-HyperVFeature {
    Write-Host 'Ensuring Hyper-V (Microsoft-Hyper-V-All)...'
    $hv = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All
    if ($hv.State -eq 'Enabled') {
        Write-Host '  Already enabled.'
        return $false
    }
    $result = Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All -All -NoRestart
    Write-Host "  Enabled (restart needed: $($result.RestartNeeded))"
    return $result.RestartNeeded -eq 'Yes'
}

function Install-WindowsSandboxFeature {
    if (Test-WindowsSandboxReady) {
        Write-Host 'Windows Sandbox: already available.'
        return $false
    }
    Write-Host 'Enabling Windows Sandbox (Containers-DisposableClientVM)...'
    $sb = Enable-WindowsOptionalFeature -Online -FeatureName Containers-DisposableClientVM -All -NoRestart
    Write-Host "  State: $($sb.RestartNeeded)"
    return $sb.RestartNeeded -eq 'Yes'
}

function Install-MultipassPackage {
    if (Test-MultipassInstalled) {
        Write-Host 'Multipass: already installed.'
        $mp = (Get-Command multipass.exe -ErrorAction SilentlyContinue).Source
        if (-not $mp) { $mp = 'C:\Program Files\Multipass\bin\multipass.exe' }
        if (Test-Path $mp) { & $mp version }
        return $false
    }
    Write-Host 'Installing Multipass (winget)...'
    winget install --id Canonical.Multipass -e `
        --accept-source-agreements --accept-package-agreements --disable-interactivity
    return $false
}

Write-Host '=== install-fresh-validation-host.ps1 ==='
Write-Host ("  Sandbox:   {0}" -f $(if ($installSandbox) { 'install' } else { 'skip' }))
Write-Host ("  Multipass: {0}" -f $(if ($installMultipass) { 'install' } else { 'skip' }))
Write-Host ''

$rebootNeeded = $false

if ($installSandbox -or $installMultipass) {
    if (Enable-HyperVFeature) { $rebootNeeded = $true }
}

if ($installSandbox) {
    if (Install-WindowsSandboxFeature) { $rebootNeeded = $true }
}

if ($rebootNeeded) {
    Write-Host ''
    Write-Host 'Reboot required before remaining steps work.'
    if ($installMultipass -and -not (Test-MultipassInstalled)) {
        Write-Host 'After reboot, install Multipass:'
        Write-Host '  pwsh -ExecutionPolicy Bypass -File scripts/validation/install-fresh-validation-host.ps1 -Multipass'
        Write-Host 'Or rerun without flags to finish both.'
    }
    Write-Host 'Then: pwsh -File scripts/validation/check-windows-host.ps1'
    exit 3010
}

if ($installMultipass) {
    Install-MultipassPackage | Out-Null
}

Write-Host ''
Write-Host '=== done ==='
Write-Host 'Verify: pwsh -File scripts/validation/check-windows-host.ps1'
Write-Host 'Guide:  docs/development/fresh-validation.md'
if ($installMultipass) {
    Write-Host 'Linux validation:  pwsh -File scripts/validation/run-multipass-linux-validation.ps1'
}
if ($installSandbox) {
    Write-Host 'Windows validation: pwsh -File scripts/validation/launch-windows-sandbox-validation.ps1'
}
