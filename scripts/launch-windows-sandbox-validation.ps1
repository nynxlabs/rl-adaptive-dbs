# Generate and launch Windows Sandbox validation.
#   pwsh -ExecutionPolicy Bypass -File scripts/launch-windows-sandbox-validation.ps1
#   pwsh -ExecutionPolicy Bypass -File scripts/launch-windows-sandbox-validation.ps1 -Clone
#
# Requires: Windows Sandbox enabled (scripts/install-fresh-validation-host.ps1 -Sandbox)
# Default: maps WSL working tree. -Clone: git clone inside Sandbox (tests GitHub main).
# Logs -> <repo>/.validation-logs/sandbox.log (gitignored).

param(
    [string]$RepoPath = '',
    [switch]$Clone
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'validation-repo.ps1')

$sandboxExe = "$env:WINDIR\System32\WindowsSandbox.exe"
if (-not (Test-Path $sandboxExe)) {
    Write-Error @"
Windows Sandbox is not available (WindowsSandbox.exe missing).
Run as Administrator: pwsh -File scripts/install-fresh-validation-host.ps1 -Sandbox
Then reboot if prompted.
"@
}

$repoRoot = Resolve-ValidationRepoPath -RepoPath $RepoPath -CallerScriptDir $PSScriptRoot
$logDir = Get-ValidationLogDir -RepoRoot $repoRoot
$hostLog = Join-Path $logDir 'sandbox.log'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (-not (Test-Path -LiteralPath $repoRoot)) {
    Write-Error @"
WSL repo path is not reachable from Windows: $repoRoot
Start WSL (open a WSL shell) and retry. From WSL:
  bash scripts/install-fresh-validation-host.sh --check
Or use -Clone (maps only scripts + logs; clones repo inside Sandbox).
"@
}

if (-not (Test-Path -LiteralPath $logDir)) {
    Write-Error "Validation log dir missing: $logDir"
}

Write-Host "Preflight OK: repo and log dir reachable from Windows."

$gitCache = Ensure-SandboxGitInstallerCache -LogDir $logDir
Write-Host "  Git cache:   $gitCache"

$wsbPath = Join-Path $env:TEMP 'rl-adaptive-dbs-sandbox.wsb'

if ($Clone) {
    $scriptsDir = Join-Path $repoRoot 'scripts'
    if (-not (Test-Path -LiteralPath $scriptsDir)) {
        Write-Error "Missing scripts dir: $scriptsDir"
    }
    $wsb = @"
<Configuration>
  <MappedFolders>
    <MappedFolder>
      <HostFolder>$logDir</HostFolder>
      <SandboxFolder>C:\host-logs</SandboxFolder>
      <ReadOnly>false</ReadOnly>
    </MappedFolder>
    <MappedFolder>
      <HostFolder>$scriptsDir</HostFolder>
      <SandboxFolder>C:\rl-scripts</SandboxFolder>
      <ReadOnly>true</ReadOnly>
    </MappedFolder>
  </MappedFolders>
  <LogonCommand>
    <Command>powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\rl-scripts\bootstrap-fresh-windows.ps1 -Clone -LogDir C:\host-logs</Command>
  </LogonCommand>
</Configuration>
"@
} else {
    $wsb = @"
<Configuration>
  <MappedFolders>
    <MappedFolder>
      <HostFolder>$repoRoot</HostFolder>
      <SandboxFolder>C:\rl-adaptive-dbs</SandboxFolder>
      <ReadOnly>false</ReadOnly>
    </MappedFolder>
    <MappedFolder>
      <HostFolder>$logDir</HostFolder>
      <SandboxFolder>C:\host-logs</SandboxFolder>
      <ReadOnly>false</ReadOnly>
    </MappedFolder>
  </MappedFolders>
  <LogonCommand>
    <Command>powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\rl-adaptive-dbs\scripts\bootstrap-fresh-windows.ps1 -LogDir C:\host-logs</Command>
  </LogonCommand>
</Configuration>
"@
}

Set-Content -Path $wsbPath -Value $wsb -Encoding UTF8
Write-Host "Launching Windows Sandbox validation..."
if ($Clone) {
    Write-Host '  Mode:      git clone inside Sandbox (fresh Windows)'
} else {
    Write-Host '  Mode:      WSL repo mapped into Sandbox'
}
Write-Host "  Repo (WSL):  $repoRoot"
Write-Host "  WSL log:     ~/neuroengineering/rl-adaptive-dbs/.validation-logs/sandbox.log"
Write-Host "  Host log:    $hostLog"
Write-Host "  WSB:         $wsbPath"
Start-Process -FilePath $wsbPath
