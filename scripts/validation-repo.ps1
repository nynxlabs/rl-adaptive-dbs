# Resolve WSL checkout path for host-side logs (Sandbox map, .validation-logs/).

function Resolve-ValidationRepoPath {
    param(
        [string]$RepoPath = '',
        [string]$CallerScriptDir = ''
    )

    if ($RepoPath -and (Test-Path -LiteralPath $RepoPath)) {
        return (Get-Item -LiteralPath $RepoPath).FullName
    }

    # Preferred: WSL checkout (nynxbox default; override with -RepoPath on other machines).
    $wslUnc = '\\wsl.localhost\Ubuntu\home\nynxbox\neuroengineering\rl-adaptive-dbs'
    if (Test-Path -LiteralPath $wslUnc) {
        return (Get-Item -LiteralPath $wslUnc).FullName
    }

    if (-not $CallerScriptDir) {
        throw 'Resolve-ValidationRepoPath: set CallerScriptDir or pass -RepoPath'
    }
    return (Get-Item (Join-Path $CallerScriptDir '..')).FullName
}

function Get-ValidationLogDir {
    param([string]$RepoRoot)
    Join-Path $RepoRoot '.validation-logs'
}

# Pinned Git for Windows installer for Sandbox bootstrap (bump when validating against newer Git).
$script:SandboxGitTag = 'v2.55.0.windows.1'
$script:SandboxGitInstallerName = 'Git-2.55.0-64-bit.exe'

function Get-SandboxGitInstallerUrl {
    "https://github.com/git-for-windows/git/releases/download/$($script:SandboxGitTag)/$($script:SandboxGitInstallerName)"
}

function Get-SandboxGitInstallerCachePath {
    param([string]$LogDir)
    Join-Path (Join-Path $LogDir 'cache') $script:SandboxGitInstallerName
}

function Ensure-SandboxGitInstallerCache {
    param([string]$LogDir)

    $cacheDir = Join-Path $LogDir 'cache'
    $cachePath = Get-SandboxGitInstallerCachePath -LogDir $LogDir
    if (Test-Path -LiteralPath $cachePath) {
        return $cachePath
    }

    New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
    $url = Get-SandboxGitInstallerUrl
    Write-Host "Prefetching $($script:SandboxGitInstallerName) to validation cache..."
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    } catch { }
    Invoke-WebRequest -Uri $url -OutFile $cachePath -UseBasicParsing `
        -Headers @{ 'User-Agent' = 'rl-adaptive-dbs' } -TimeoutSec 600
    return $cachePath
}
