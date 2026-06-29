# Refresh Multipass image catalog (fixes "Remote is unknown or unreachable" on launch).
# Run on Windows desktop — Administrator PowerShell:
#   pwsh -ExecutionPolicy Bypass -File scripts/refresh-multipass-catalog.ps1
#
# See docs/setup.md § Fresh machine validation — Troubleshooting.

#Requires -RunAsAdministrator
$ErrorActionPreference = 'Stop'

$mp = 'C:\Program Files\Multipass\bin\multipass.exe'
if (-not (Test-Path $mp)) {
    Write-Error 'multipass not found. Install from https://multipass.run or run scripts/prepare-desktop-host.ps1'
}

Write-Host 'Stopping multipassd...'
Get-Process multipassd -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

$cache = 'C:\ProgramData\Multipass\cache\network-cache'
if (Test-Path $cache) {
    Write-Host "Removing $cache"
    Remove-Item -Recurse -Force $cache
}

Write-Host 'Starting multipassd...'
Start-Process "$env:ProgramFiles\Multipass\bin\multipassd.exe" -WindowStyle Hidden
Start-Sleep -Seconds 5

Write-Host 'Refreshing catalog (multipass find --force-update)...'
& $mp find --force-update
