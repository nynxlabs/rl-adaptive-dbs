# Enable Windows Sandbox + install Multipass. Run as Administrator on the Windows desktop.
# Prefer: pwsh -ExecutionPolicy Bypass -File scripts/validation/install-fresh-validation-host.ps1
#   (supports -Sandbox / -Multipass flags)
#
#   pwsh -ExecutionPolicy Bypass -File scripts/validation/prepare-desktop-host.ps1

#Requires -RunAsAdministrator
$ErrorActionPreference = 'Stop'

$forward = Join-Path $PSScriptRoot 'install-fresh-validation-host.ps1'
if (-not (Test-Path -LiteralPath $forward)) {
    Write-Error "Missing $forward"
}
& $forward @args
