# Bootstrap Windows Sandbox session: install Git + uv, run validate-fresh.sh.
# Invoked by launch-windows-sandbox-validation.ps1 LogonCommand.
#
# Default (mapped repo): validate from C:\rl-adaptive-dbs; logs -> .validation-logs/ in repo.
# -Clone: git clone inside Sandbox; logs via -LogDir (host .validation-logs mapped to C:\host-logs).
param(
    [switch]$Clone,
    [string]$LogDir = '',
    [string]$RepoUrl = 'https://github.com/wilzen3476/rl-adaptive-dbs.git',
    [string]$RepoDir = 'C:\rl-adaptive-dbs'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'validation-repo.ps1')

$RepoMount = if ($Clone) { $RepoDir } else { 'C:\rl-adaptive-dbs' }

if (-not $LogDir) {
    if ($Clone) {
        Write-Error 'Clone mode requires -LogDir (e.g. C:\host-logs mapped from host .validation-logs)'
    }
    $LogDir = Join-Path $RepoMount '.validation-logs'
}

$log = Join-Path $LogDir 'sandbox.log'

function Log($msg) {
    $line = "$(Get-Date -Format o) $msg"
    Add-Content -Path $log -Value $line
    Write-Host $line
}

function Get-GitBashPath {
    @(
        "${env:ProgramFiles}\Git\bin\bash.exe",
        "${env:ProgramFiles(x86)}\Git\bin\bash.exe"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
}

function Install-GitFromGitHub {
    Log 'Installing Git for Windows (GitHub release installer)...'
    Log "Pinned release: $script:SandboxGitTag ($script:SandboxGitInstallerName)"

    $cached = Get-SandboxGitInstallerCachePath -LogDir $LogDir
    $installer = Join-Path $env:TEMP $script:SandboxGitInstallerName

    if (Test-Path -LiteralPath $cached) {
        Log "Using host-cached installer: $cached"
        Copy-Item -LiteralPath $cached -Destination $installer -Force
        $sizeMb = [math]::Round((Get-Item $installer).Length / 1MB, 1)
        Log "Copied to Sandbox temp ($sizeMb MB)"
    } else {
        try {
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        } catch {
            Log "Note: could not set TLS 1.2: $_"
        }
        $downloadUrl = Get-SandboxGitInstallerUrl
        Log "Cache miss; downloading $downloadUrl ..."
        Invoke-WebRequest -Uri $downloadUrl -OutFile $installer -UseBasicParsing `
            -Headers @{ 'User-Agent' = 'rl-adaptive-dbs-bootstrap' } -TimeoutSec 600
        $sizeMb = [math]::Round((Get-Item $installer).Length / 1MB, 1)
        Log "Download complete ($sizeMb MB)"
    }

    Log 'Running silent installer (/VERYSILENT)...'
    $p = Start-Process -FilePath $installer -ArgumentList '/VERYSILENT', '/NORESTART', '/NOCANCEL', '/SP-' -PassThru -Wait
    Remove-Item $installer -Force -ErrorAction SilentlyContinue
    if ($p.ExitCode -ne 0) {
        Log "ERROR: Git installer exit $($p.ExitCode)"
        exit $p.ExitCode
    }
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
"=== sandbox validation started $(Get-Date -Format o) ===" | Set-Content -Path $log -Encoding UTF8

if ($Clone) {
    Log '=== bootstrap-fresh-windows.ps1 (clone from GitHub) ==='
} else {
    Log '=== bootstrap-fresh-windows.ps1 (mapped repo) ==='
}

if (-not $Clone -and -not (Test-Path $RepoMount)) {
    Log "ERROR: mapped repo not found at $RepoMount"
    exit 1
}

Log "Host log file: $log"
if ($Clone) {
    Log "Repo URL: $RepoUrl"
    Log "Clone dir: $RepoDir"
}

$gitBash = Get-GitBashPath
if (-not $gitBash) {
    Install-GitFromGitHub
    $gitBash = Get-GitBashPath
}

if (-not $gitBash) {
    Log 'ERROR: Git Bash not found after install'
    exit 1
}

Log "Git Bash: $gitBash"

Log 'Installing uv...'
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" 2>&1 |
    ForEach-Object { Log $_ }

if ($Clone) {
    if (-not (Test-Path (Join-Path $RepoDir '.git'))) {
        Log "Cloning $RepoUrl -> $RepoDir"
        $cloneCmd = "git clone '$RepoUrl' '$($RepoDir -replace '\\', '/')'"
        & $gitBash -lc $cloneCmd 2>&1 | ForEach-Object { Log $_ }
        if ($LASTEXITCODE -ne 0) {
            Log "ERROR: git clone failed (exit $LASTEXITCODE)"
            exit $LASTEXITCODE
        }
    } else {
        Log "Repo already present at $RepoDir; skipping clone"
    }
}

$bashRepo = $RepoMount -replace '\\', '/'
if ($bashRepo -match '^C:') { $bashRepo = '/c' + $bashRepo.Substring(2) }
$validateCmd = @"
export PATH=\"\$HOME/.local/bin:\$PATH\"
cd '$bashRepo'
bash scripts/validate-fresh.sh
"@

Log 'Running validate-fresh.sh...'
& $gitBash -lc $validateCmd 2>&1 | ForEach-Object { Log $_ }
$code = $LASTEXITCODE
Log "=== finished (exit $code) ==="
exit $code
