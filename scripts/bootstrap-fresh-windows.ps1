# Bootstrap Windows Sandbox session: install Git + uv, run validate-fresh.sh from mapped repo.
# Invoked by windows-sandbox-validation.wsb LogonCommand.
$ErrorActionPreference = 'Stop'

$RepoMount = 'C:\rl-adaptive-dbs'
$log = 'C:\validation-log.txt'

function Log($msg) {
    $line = "$(Get-Date -Format o) $msg"
    Add-Content -Path $log -Value $line
    Write-Host $line
}

Log '=== bootstrap-fresh-windows.ps1 ==='

if (-not (Test-Path $RepoMount)) {
    Log "ERROR: mapped repo not found at $RepoMount"
    exit 1
}

Log 'Installing Git for Windows (winget)...'
winget install --id Git.Git -e --accept-source-agreements --accept-package-agreements --disable-interactivity

$gitBash = @(
    "${env:ProgramFiles}\Git\bin\bash.exe",
    "${env:ProgramFiles(x86)}\Git\bin\bash.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $gitBash) {
    Log 'ERROR: Git Bash not found after install'
    exit 1
}

Log "Git Bash: $gitBash"

Log 'Installing uv...'
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"

$bashRepo = $RepoMount -replace '\\', '/'
if ($bashRepo -match '^C:') { $bashRepo = '/c' + $bashRepo.Substring(2) }

$validateCmd = @"
export PATH=\"\$HOME/.local/bin:\$PATH\"
cd '$bashRepo'
bash scripts/validate-fresh.sh
"@

Log 'Running validate-fresh.sh...'
& $gitBash -lc $validateCmd 2>&1 | Tee-Object -FilePath $log -Append
exit $LASTEXITCODE
