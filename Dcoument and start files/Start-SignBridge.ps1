param(
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'

function Write-Step([string]$message) {
    Write-Host "[SignBridge] $message" -ForegroundColor Cyan
}

Set-Location -Path $PSScriptRoot

if (-not (Test-Path -LiteralPath "app.py")) {
    if (Test-Path -LiteralPath "..\app.py") {
        Set-Location -Path ".."
    } else {
        throw "app.py not found in $PSScriptRoot or parent directory"
    }
}

Write-Step "Checking Python availability"
$pythonExe = $null
$pythonPrefixArgs = @()
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonExe = 'python'
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonExe = 'py'
    $pythonPrefixArgs = @('-3')
} else {
    throw "Python is not installed or not available in PATH."
}

Write-Step "Verifying required packages"
$checkCode = @"
import importlib.util
import sys
mods = ['flask','cv2','mediapipe','joblib','numpy']
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print('MISSING:' + ','.join(missing))
    sys.exit(1)
print('OK')
"@

$null = & $pythonExe @pythonPrefixArgs -c $checkCode
if ($LASTEXITCODE -ne 0) {
    Write-Step "Installing missing packages"
    & $pythonExe @pythonPrefixArgs -m pip install flask opencv-python mediapipe joblib numpy
    if ($LASTEXITCODE -ne 0) {
        throw "Package installation failed."
    }
}

# Resolve potential port collision
$portConnections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
if ($portConnections) {
    Write-Step "Port $Port is already in use."
    foreach ($conn in $portConnections) {
        $procId = $conn.OwningProcess
        if ($procId -gt 0) {
            $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
            if ($proc -and ($proc.Name -like "*python*" -or $proc.Name -like "*py*")) {
                Write-Step "Stopping previous Python instance (PID: $procId) listening on port $Port..."
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
                Start-Sleep -Seconds 1
            }
        }
    }
}

Write-Step "Launching browser in the background..."
$ScriptBlock = {
    param($Port)
    $maxWait = 30
    $waited = 0
    while ($waited -lt $maxWait) {
        try {
            $tcp = New-Object System.Net.Sockets.TcpClient
            $connect = $tcp.BeginConnect("127.0.0.1", $Port, $null, $null)
            $success = $connect.AsyncWaitHandle.WaitOne(200, $false)
            if ($success -and $tcp.Connected) {
                $tcp.Close()
                Start-Process "http://127.0.0.1:$Port"
                break
            }
        } catch {}
        Start-Sleep -Milliseconds 250
        $waited += 0.25
    }
}
$null = Start-Job -ScriptBlock $ScriptBlock -ArgumentList $Port

Write-Step "Starting Flask backend server..."
& $pythonExe @pythonPrefixArgs "app.py"

