# Resize the Windows Sandbox host window (WindowsSandboxClient).
# Microsoft .wsb files do not support display size; this adjusts outer window chrome via Win32.
#
# Standalone (running Sandbox instance — does not launch or restart Sandbox):
#   pwsh -File scripts/validation/sandbox-window.ps1
#   pwsh -File scripts/validation/sandbox-window.ps1 -WindowWidth 1024 -WindowHeight 768
#
# Dot-sourced by launch-windows-sandbox-validation.ps1 after Start-Process on the .wsb file.

param(
    [int]$WindowWidth = 1280,
    [int]$WindowHeight = 960,
    [int]$WaitSeconds = 120,
    [int]$SettleSeconds = 3
)

$script:SandboxWin32Loaded = $false

function Initialize-SandboxWindowWin32 {
    if ($script:SandboxWin32Loaded) { return }
    Add-Type @"
using System;
using System.Runtime.InteropServices;

public class SandboxWin32 {
    [DllImport("user32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool GetWindowRect(IntPtr hWnd, out SandboxRect lpRect);

    [DllImport("user32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool GetClientRect(IntPtr hWnd, out SandboxRect lpRect);

    [DllImport("user32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool MoveWindow(IntPtr hWnd, int X, int Y, int nWidth, int nHeight, bool bRepaint);
}

public struct SandboxRect {
    public int Left;
    public int Top;
    public int Right;
    public int Bottom;
}
"@
    $script:SandboxWin32Loaded = $true
}

function Set-SandboxClientWindowSize {
    param(
        [Parameter(Mandatory)]
        [IntPtr]$MainWindowHandle,
        [int]$Width = 1280,
        [int]$Height = 960
    )

    Initialize-SandboxWindowWin32

    $rcWindow = New-Object SandboxRect
    $rcClient = New-Object SandboxRect
    [void][SandboxWin32]::GetWindowRect($MainWindowHandle, [ref]$rcWindow)
    [void][SandboxWin32]::GetClientRect($MainWindowHandle, [ref]$rcClient)

    $dx = ($rcWindow.Right - $rcWindow.Left) - $rcClient.Right
    $dy = ($rcWindow.Bottom - $rcWindow.Top) - $rcClient.Bottom

    $ok = [SandboxWin32]::MoveWindow($MainWindowHandle, $rcWindow.Left, $rcWindow.Top, $Width + $dx, $Height + $dy, $true)
    if (-not $ok) {
        throw "MoveWindow failed for Sandbox client handle $MainWindowHandle"
    }
}

function Wait-Set-SandboxClientWindowSize {
    param(
        [int]$Width = 1280,
        [int]$Height = 960,
        [int]$WaitSeconds = 120,
        [int]$SettleSeconds = 3
    )

    $deadline = (Get-Date).AddSeconds($WaitSeconds)
    $client = $null
    while ((Get-Date) -lt $deadline) {
        $client = Get-Process -Name WindowsSandboxClient -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($client -and $client.MainWindowHandle -ne [IntPtr]::Zero) { break }
        Start-Sleep -Milliseconds 500
        $client = $null
    }

    if (-not $client) {
        Write-Warning "WindowsSandboxClient did not appear within ${WaitSeconds}s; window size unchanged."
        return $false
    }

    if ($SettleSeconds -gt 0) {
        Start-Sleep -Seconds $SettleSeconds
    }

    Set-SandboxClientWindowSize -MainWindowHandle $client.MainWindowHandle -Width $Width -Height $Height
    Write-Host "Sandbox window resized to ${Width}x${Height} (client area, 4:3 default)."
    return $true
}

# Direct execution (not dot-sourced).
if ($MyInvocation.InvocationName -ne '.') {
    $ratio = $WindowWidth / $WindowHeight
    if ([math]::Abs($ratio - (4.0 / 3.0)) -gt 0.02) {
        Write-Warning "Window is ${WindowWidth}x${WindowHeight} (ratio $([math]::Round($ratio, 3))), not 4:3."
    }

    $client = Get-Process -Name WindowsSandboxClient -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $client) {
        Write-Error "No running Windows Sandbox (WindowsSandboxClient not found). This script resizes an existing instance; it does not launch Sandbox."
    }

    if ($client.MainWindowHandle -eq [IntPtr]::Zero) {
        Write-Error "Sandbox client has no main window handle yet. Retry in a few seconds."
    }

    Set-SandboxClientWindowSize -MainWindowHandle $client.MainWindowHandle -Width $WindowWidth -Height $WindowHeight
    Write-Host "Sandbox window resized to ${WindowWidth}x${WindowHeight} (client area)."
}
