# Enable Windows Sandbox + install Multipass. Run as Administrator on the Windows desktop.
#   pwsh -ExecutionPolicy Bypass -File scripts/prepare-desktop-host.ps1
# Reboot if prompted, then: pwsh -File scripts/check-windows-host.ps1

#Requires -RunAsAdministrator
$ErrorActionPreference = 'Stop'

Write-Host '=== prepare-desktop-host.ps1 ==='
Write-Host ''

Write-Host 'Enabling Windows Sandbox (Containers-DisposableClientVM)...'
$sb = Enable-WindowsOptionalFeature -Online -FeatureName Containers-DisposableClientVM -All -NoRestart
Write-Host "  State: $($sb.RestartNeeded)"

Write-Host 'Ensuring Hyper-V (Microsoft-Hyper-V-All)...'
$hv = Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All -All -NoRestart
Write-Host "  State: $($hv.RestartNeeded)"

if ($sb.RestartNeeded -eq 'Yes' -or $hv.RestartNeeded -eq 'Yes') {
    Write-Host ''
    Write-Host 'Reboot required. After reboot, run:'
    Write-Host '  pwsh -File scripts/check-windows-host.ps1'
    Write-Host '  winget install --id Canonical.Multipass -e --accept-source-agreements --accept-package-agreements'
    exit 3010
}

Write-Host ''
Write-Host 'Installing Multipass (winget)...'
winget install --id Canonical.Multipass -e --accept-source-agreements --accept-package-agreements --disable-interactivity

Write-Host ''
Write-Host '=== done ==='
Write-Host 'Run: pwsh -File scripts/check-windows-host.ps1'
Write-Host 'Linux validation: pwsh -File scripts/run-multipass-linux-validation.ps1'
Write-Host 'Windows validation: pwsh -File scripts/launch-windows-sandbox-validation.ps1'
