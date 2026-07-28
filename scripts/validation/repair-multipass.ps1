# Repair wedged Multipass on Windows (hung CLI, stuck VM, stale clients).
# Run as Administrator on the Windows desktop:
#   pwsh -ExecutionPolicy Bypass -File scripts/validation/repair-multipass.ps1
#
# From WSL (UAC prompt):
#   bash scripts/validation/repair-multipass.ps1

#Requires -RunAsAdministrator
$ErrorActionPreference = 'Continue'

$mp = 'C:\Program Files\Multipass\bin\multipass.exe'
$logDir = Join-Path $env:TEMP 'rl-adaptive-dbs-multipass-repair'
$log = Join-Path $logDir 'repair.log'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
"=== repair-multipass.ps1 $(Get-Date -Format o) ===" | Set-Content -Path $log -Encoding UTF8

function Log($msg) {
    $line = "$(Get-Date -Format o) $msg"
    Add-Content -Path $log -Value $line
    Write-Host $line
}

function Stop-MultipassHyperVInstances {
    $names = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    [void]$names.Add('rl-dbs-linux')

    $instancesDir = 'C:\ProgramData\Multipass\data\multipassd\vault\instances'
    if (Test-Path $instancesDir) {
        Get-ChildItem $instancesDir -Directory -ErrorAction SilentlyContinue |
            ForEach-Object { [void]$names.Add($_.Name) }
    }

    foreach ($name in $names) {
        $vm = Get-VM -Name $name -ErrorAction SilentlyContinue
        if (-not $vm) { continue }
        Log "Hyper-V VM '$name' state=$($vm.State) — force stop + remove"
        if ($vm.State -ne 'Off') {
            Stop-VM -Name $name -Force -TurnOff -ErrorAction SilentlyContinue | Out-Null
            Start-Sleep -Seconds 2
        }
        Remove-VM -Name $name -Force -ErrorAction SilentlyContinue | Out-Null
    }
}

function Remove-StaleHostsIcsEntries {
  param([string[]]$Hostnames = @('rl-dbs-linux.mshome.net'))

  $hostsIcs = Join-Path $env:WINDIR 'System32\drivers\etc\hosts.ics'
  if (-not (Test-Path -LiteralPath $hostsIcs)) {
    Log "hosts.ics not present (nothing to clean)"
    return
  }

  $lines = Get-Content -LiteralPath $hostsIcs -ErrorAction SilentlyContinue
  if (-not $lines) { return }

  $kept = [System.Collections.Generic.List[string]]::new()
  $removed = 0
  foreach ($line in $lines) {
    $drop = $false
    foreach ($hostname in $Hostnames) {
      if ($line -match [regex]::Escape($hostname)) { $drop = $true; break }
    }
    if ($drop) { $removed++ } else { [void]$kept.Add($line) }
  }

  if ($removed -eq 0) {
    Log 'hosts.ics: no stale rl-dbs-linux entries'
    return
  }

  Log "hosts.ics: removing $removed stale entr$(if ($removed -eq 1) { 'y' } else { 'ies' })"
  Set-Content -LiteralPath $hostsIcs -Value $kept -Encoding ASCII
  ipconfig.exe /flushdns 2>&1 | ForEach-Object { Log "  $_" }
}

function Stop-MultipassDaemon {
    Log 'Stopping hung multipass clients...'
    Get-Process multipass -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1

    # VMs must go before the daemon, or Stop-Service blocks forever.
    Log 'Stopping Multipass Hyper-V instances (before daemon)...'
    Stop-MultipassHyperVInstances

    Log 'Stopping Multipass service (sc stop + taskkill fallback)...'
    $svc = Get-Service Multipass -ErrorAction SilentlyContinue
    if ($svc -and $svc.Status -eq 'Running') {
        sc.exe stop Multipass | ForEach-Object { Log "  sc: $_" }
        $deadline = (Get-Date).AddSeconds(20)
        while ((Get-Service Multipass).Status -ne 'Stopped' -and (Get-Date) -lt $deadline) {
            Start-Sleep -Seconds 1
        }
    }

    $svcState = (Get-Service Multipass -ErrorAction SilentlyContinue).Status
    Log "  Service status after sc stop: $svcState"

    if ($svcState -ne 'Stopped') {
        $procId = (Get-CimInstance Win32_Service -Filter "Name='Multipass'" -ErrorAction SilentlyContinue).ProcessId
        if ($procId -and $procId -gt 0) {
            Log "  taskkill /F /PID $procId"
            taskkill.exe /F /PID $procId 2>&1 | ForEach-Object { Log "    $_" }
        }
        Log '  taskkill /F /IM multipassd.exe'
        taskkill.exe /F /IM multipassd.exe 2>&1 | ForEach-Object { Log "    $_" }
        Start-Sleep -Seconds 2
        sc.exe stop Multipass | ForEach-Object { Log "  sc (retry): $_" }
        Start-Sleep -Seconds 2
    }

    $final = (Get-Service Multipass -ErrorAction SilentlyContinue).Status
    Log "  Final service status: $final"
}

Log '=== repair-multipass.ps1 ==='
Remove-StaleHostsIcsEntries
Stop-MultipassDaemon

$cache = 'C:\ProgramData\Multipass\cache\network-cache'
if (Test-Path $cache) {
    Log "Removing catalog cache: $cache"
    Remove-Item -Recurse -Force $cache -ErrorAction SilentlyContinue
}

Log 'Starting Multipass service...'
try {
    Start-Service Multipass -ErrorAction Stop
} catch {
    Log "Start-Service failed: $_ — launching multipassd.exe directly"
    Start-Process "$env:ProgramFiles\Multipass\bin\multipassd.exe" -WindowStyle Hidden
}
Start-Sleep -Seconds 6

if (-not (Test-Path $mp)) {
    Log "ERROR: $mp not found"
    exit 1
}

Log 'Testing multipass version...'
& $mp version 2>&1 | ForEach-Object { Log "  $_" }

Log 'Testing multipass list...'
& $mp list 2>&1 | ForEach-Object { Log $_ }

Log 'Purging rl-dbs-linux (if registered)...'
& $mp delete rl-dbs-linux --purge 2>&1 | ForEach-Object { Log "  $_" }

Log "=== repair complete; log: $log ==="
