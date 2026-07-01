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

function Invoke-GitCloneInSandbox {
    param(
        [string]$GitBash,
        [string]$RepoUrl,
        [string]$RepoDir,
        [string]$LogDir,
        [int]$TimeoutSeconds = 900
    )

    $bashRepoDir = $RepoDir -replace '\\', '/'
    if ($bashRepoDir -match '^C:') { $bashRepoDir = '/c' + $bashRepoDir.Substring(2) }

    Log "Cloning $RepoUrl -> $RepoDir (shallow, --progress, ${TimeoutSeconds}s timeout)"
    Log 'git: GIT_TERMINAL_PROMPT=0, http.version=HTTP/1.1, low-speed timeout 60s'

    $cloneOneLiner = @"
export GIT_TERMINAL_PROMPT=0
export GIT_HTTP_LOW_SPEED_LIMIT=1000
export GIT_HTTP_LOW_SPEED_TIME=60
git config --global http.version HTTP/1.1
git config --global core.longpaths true
git clone --depth 1 --single-branch --progress '$RepoUrl' '$bashRepoDir'
"@ -replace "`r`n", ' ; '

    $job = Start-Job -ScriptBlock {
        param($GitBash, $cmd)
        $lines = & $GitBash -lc $cmd 2>&1 | ForEach-Object { "$_" }
        [pscustomobject]@{
            Exit = $LASTEXITCODE
            Lines = $lines
        }
    } -ArgumentList $GitBash, $cloneOneLiner

    if (-not (Wait-Job -Job $job -Timeout $TimeoutSeconds)) {
        Log "git clone timed out after ${TimeoutSeconds}s; stopping job..."
        Stop-Job -Job $job -Force
        Remove-Job -Job $job -Force
        return $false
    }

    $result = Receive-Job -Job $job
    Remove-Job -Job $job -Force
    foreach ($line in $result.Lines) {
        if ($line) { Log $line }
    }
    Log "git clone exit $($result.Exit)"

    if ($result.Exit -eq 0 -and (Test-Path (Join-Path $RepoDir '.git'))) {
        Log 'git clone succeeded inside Sandbox'
        return $true
    }
    return $false
}

function Copy-RepoFromHostCache {
    param(
        [string]$RepoDir,
        [string]$LogDir
    )

    $cacheDir = Get-SandboxRepoCacheDir -LogDir $LogDir
    if (-not (Test-Path (Join-Path $cacheDir '.git'))) {
        Log "ERROR: host repo cache missing at $cacheDir"
        exit 1
    }

    Log "Falling back to host-cached shallow clone: $cacheDir"
    if (Test-Path -LiteralPath $RepoDir) {
        Remove-Item -LiteralPath $RepoDir -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $RepoDir | Out-Null

    & robocopy $cacheDir $RepoDir /E /NFL /NDL /NJH /NJS /nc /ns /np 2>&1 | ForEach-Object { Log $_ }
    if ($LASTEXITCODE -gt 7) {
        Log "ERROR: robocopy from cache failed (exit $LASTEXITCODE)"
        exit $LASTEXITCODE
    }
    if (-not (Test-Path (Join-Path $RepoDir '.git'))) {
        Log 'ERROR: cache copy did not produce a .git directory'
        exit 1
    }
    Log 'Host cache copy complete'
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

function Install-VcRedist {
    $name = 'vc_redist.x64.exe'
    $cached = Join-Path (Join-Path $LogDir 'cache') $name
    $installer = Join-Path $env:TEMP $name

    Log 'Installing Microsoft VC++ Redistributable (PyTorch / torch DLLs)...'
    if (Test-Path -LiteralPath $cached) {
        Log "Using host-cached installer: $cached"
        Copy-Item -LiteralPath $cached -Destination $installer -Force
    } else {
        $url = 'https://aka.ms/vs/17/release/vc_redist.x64.exe'
        Log "Cache miss; downloading $url ..."
        try {
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        } catch { }
        Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing `
            -Headers @{ 'User-Agent' = 'rl-adaptive-dbs-bootstrap' } -TimeoutSec 600
    }

    $p = Start-Process -FilePath $installer -ArgumentList '/install', '/quiet', '/norestart' -PassThru -Wait
    Remove-Item $installer -Force -ErrorAction SilentlyContinue
    if ($p.ExitCode -ne 0 -and $p.ExitCode -ne 1638) {
        # 1638 = newer version already installed
        Log "ERROR: VC++ redist installer exit $($p.ExitCode)"
        exit $p.ExitCode
    }
    Log 'VC++ Redistributable ready'
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

Install-VcRedist

if ($Clone) {
    if (Test-Path (Join-Path $RepoDir '.git')) {
        Log "Repo already present at $RepoDir; skipping clone"
    } else {
        $cloned = Invoke-GitCloneInSandbox -GitBash $gitBash -RepoUrl $RepoUrl -RepoDir $RepoDir -LogDir $LogDir
        if (-not $cloned) {
            Log 'In-Sandbox git clone failed or timed out; using host cache fallback'
            Copy-RepoFromHostCache -RepoDir $RepoDir -LogDir $LogDir
        }
    }
}

$bashRepo = $RepoMount -replace '\\', '/'
if ($bashRepo -match '^C:') { $bashRepo = '/c' + $bashRepo.Substring(2) }
$logBash = Convert-ToGitBashPath -Path $log
$hostUvCache = Get-SandboxUvCacheDir -LogDir $LogDir
$sandboxUvCache = Join-Path $env:TEMP 'rl-adaptive-dbs-uv-cache'
if ((Test-Path $hostUvCache) -and (Get-ChildItem -LiteralPath $hostUvCache -ErrorAction SilentlyContinue)) {
    Log 'Copying uv wheel cache to Sandbox temp (avoids mapped-folder ACL issues)...'
    if (Test-Path -LiteralPath $sandboxUvCache) {
        Remove-Item -LiteralPath $sandboxUvCache -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $sandboxUvCache | Out-Null
    & robocopy $hostUvCache $sandboxUvCache /E /NFL /NDL /NJH /NJS /nc /ns /np 2>&1 |
        ForEach-Object { Log $_ }
    $uvCacheBash = Convert-ToGitBashPath -Path $sandboxUvCache
} else {
    Log 'No host uv cache; Sandbox will download wheels during uv sync'
    $uvCacheBash = Convert-ToGitBashPath -Path $hostUvCache
}
# Use backtick-escaped `$ so PowerShell does not expand $HOME / $PATH / $? into bash.
$validateCmd = @"
export PATH="`$HOME/.local/bin:`$PATH"
export PYTHONUNBUFFERED=1
export UV_PYTHON=3.12
export UV_CACHE_DIR='$uvCacheBash'
export UV_LINK_MODE=copy
set -o pipefail
cd '$bashRepo'
bash scripts/validate-fresh.sh 2>&1 | tee -a '$logBash'
exit `$?
"@

Log 'Running validate-fresh.sh (tee -> host log; UV_CACHE_DIR=host cache)...'
& $gitBash -lc $validateCmd
$code = $LASTEXITCODE
Log "=== finished (exit $code) ==="
exit $code
