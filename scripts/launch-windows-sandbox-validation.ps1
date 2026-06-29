# Generate and launch Windows Sandbox validation for the current repo.
#   pwsh -ExecutionPolicy Bypass -File scripts/launch-windows-sandbox-validation.ps1
#
# Requires: Windows Sandbox enabled (scripts/prepare-desktop-host.ps1)
# Uses mapped host folder — no git clone inside Sandbox; tests your working tree.

$ErrorActionPreference = 'Stop'

$sandboxExe = "$env:WINDIR\System32\WindowsSandbox.exe"
if (-not (Test-Path $sandboxExe)) {
    Write-Error @"
Windows Sandbox is not available (WindowsSandbox.exe missing).
Run as Administrator: pwsh -File scripts/prepare-desktop-host.ps1
Then reboot if prompted.
"@
}

$repoRoot = (Get-Item (Join-Path $PSScriptRoot '..')).FullName
$wsbPath = Join-Path $env:TEMP 'rl-adaptive-dbs-sandbox.wsb'

$wsb = @"
<Configuration>
  <MappedFolders>
    <MappedFolder>
      <HostFolder>$repoRoot</HostFolder>
      <SandboxFolder>C:\rl-adaptive-dbs</SandboxFolder>
      <ReadOnly>false</ReadOnly>
    </MappedFolder>
  </MappedFolders>
  <LogonCommand>
    <Command>powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\rl-adaptive-dbs\scripts\bootstrap-fresh-windows.ps1</Command>
  </LogonCommand>
</Configuration>
"@

Set-Content -Path $wsbPath -Value $wsb -Encoding UTF8
Write-Host "Launching Windows Sandbox validation..."
Write-Host "  Repo: $repoRoot"
Write-Host "  Log inside Sandbox: C:\validation-log.txt"
Write-Host "  WSB config: $wsbPath"
Start-Process -FilePath $wsbPath
