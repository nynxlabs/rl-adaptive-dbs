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

$script:SandboxValidationRepoUrl = 'https://github.com/wilzen3476/rl-adaptive-dbs.git'

function Get-SandboxRepoCacheDir {
    param([string]$LogDir)
    Join-Path (Join-Path $LogDir 'cache') 'rl-adaptive-dbs-shallow'
}

function Invoke-HostShallowGitClone {
    param(
        [string]$RepoUrl,
        [string]$DestDir
    )

    if (Test-Path (Join-Path $DestDir '.git')) {
        return
    }
    if (Test-Path -LiteralPath $DestDir) {
        Remove-Item -LiteralPath $DestDir -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $DestDir) | Out-Null

    $git = Get-Command git -ErrorAction SilentlyContinue
    if ($git) {
        & git clone --depth 1 --single-branch $RepoUrl $DestDir
        if ($LASTEXITCODE -ne 0) {
            throw "git clone failed (exit $LASTEXITCODE)"
        }
        return
    }

    $wsl = Get-Command wsl -ErrorAction SilentlyContinue
    if (-not $wsl) {
        throw 'git not found on host PATH and wsl.exe unavailable for repo prefetch'
    }

    $destForWsl = Convert-WslUncToLinuxPath -UncPath $DestDir
    $bashCmd = "git clone --depth 1 --single-branch '$RepoUrl' '$destForWsl'"
    & wsl bash -lc $bashCmd
    if ($LASTEXITCODE -ne 0) {
        throw "wsl git clone failed (exit $LASTEXITCODE)"
    }
}

function Ensure-SandboxRepoCache {
    param(
        [string]$LogDir,
        [string]$RepoUrl = $script:SandboxValidationRepoUrl
    )

    $cacheDir = Get-SandboxRepoCacheDir -LogDir $LogDir
    if (Test-Path (Join-Path $cacheDir '.git')) {
        Write-Host "  Repo cache:  $cacheDir (shallow clone)"
        return $cacheDir
    }

    Write-Host "Prefetching shallow git clone to validation cache..."
    Invoke-HostShallowGitClone -RepoUrl $RepoUrl -DestDir $cacheDir
    Write-Host "  Repo cache:  $cacheDir"
    return $cacheDir
}

function Get-SandboxUvCacheDir {
    param([string]$LogDir)
    Join-Path (Join-Path $LogDir 'cache') 'uv'
}

function Convert-WslUncToLinuxPath {
    param([string]$UncPath)

    if ($UncPath -match '^\\\\wsl(?:\.localhost|\$)\\[^\\]+\\(.+)$') {
        return '/' + ($Matches[1] -replace '\\', '/')
    }

    if ($UncPath -match '^([A-Za-z]):\\(.*)$') {
        $drive = $Matches[1].ToLower()
        return "/mnt/$drive/" + ($Matches[2] -replace '\\', '/')
    }

    $wslOut = & wsl.exe wslpath -a "$UncPath" 2>$null
    if ($wslOut) {
        return "$wslOut".Trim()
    }

    throw "cannot convert to WSL path: $UncPath"
}

function Ensure-WindowsHostUv {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        return
    }
    Write-Host 'Installing uv on Windows host (wheel prefetch for Sandbox)...'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -Command `
        "irm https://astral.sh/uv/install.ps1 | iex"
    $uvBin = Join-Path $env:USERPROFILE '.local\bin'
    if (Test-Path $uvBin) {
        $env:Path = "$uvBin;$env:Path"
    }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw 'uv not on PATH after Windows install'
    }
}

function Ensure-SandboxUvWheelCache {
    param(
        [string]$LogDir,
        [string]$RepoCacheDir
    )

    $uvCache = Get-SandboxUvCacheDir -LogDir $LogDir
    New-Item -ItemType Directory -Force -Path $uvCache | Out-Null

    $lockFile = Join-Path $RepoCacheDir 'uv.lock'
    if (-not (Test-Path -LiteralPath $lockFile)) {
        throw "missing uv.lock in repo cache: $RepoCacheDir"
    }

    $marker = Join-Path $uvCache '.prefetch-ok'
    $lockHash = (Get-FileHash -LiteralPath $lockFile -Algorithm SHA256).Hash
    if ((Test-Path $marker) -and ((Get-Content -LiteralPath $marker -Raw) -eq $lockHash)) {
        Write-Host "  UV cache:    $uvCache (warm)"
        return $uvCache
    }

    Write-Host 'Prefetching uv wheel cache on host (uv sync --group dev)...'

    # Windows Sandbox needs Windows wheels — prefetch with native Windows uv on NTFS paths.
    if ($LogDir -match '^[A-Za-z]:\\') {
        Ensure-WindowsHostUv
        Push-Location $RepoCacheDir
        try {
            $env:UV_CACHE_DIR = $uvCache
            & uv sync --group dev
            if ($LASTEXITCODE -ne 0) {
                throw "Windows host uv sync failed (exit $LASTEXITCODE)"
            }
        } finally {
            Pop-Location
        }
    } else {
        $repoWsl = Convert-WslUncToLinuxPath -UncPath $RepoCacheDir
        $uvWsl = Convert-WslUncToLinuxPath -UncPath $uvCache
        & wsl bash -lc "cd '$repoWsl' && UV_CACHE_DIR='$uvWsl' uv sync --group dev"
        if ($LASTEXITCODE -ne 0) {
            throw "host uv sync failed (exit $LASTEXITCODE)"
        }
    }

    Set-Content -LiteralPath $marker -Value $lockHash -NoNewline
    Write-Host "  UV cache:    $uvCache"
    return $uvCache
}

function Convert-ToGitBashPath {
    param([string]$Path)
    $p = $Path -replace '\\', '/'
    if ($p -match '^C:') { return '/c' + $p.Substring(2) }
    return $p
}
