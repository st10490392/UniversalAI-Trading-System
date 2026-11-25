$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
Set-Location -LiteralPath $projectRoot

function Read-SecretValue([string]$Prompt, [bool]$Required) {
    while ($true) {
        $secure = Read-Host -Prompt $Prompt -AsSecureString
        $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
        try { $value = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) }
        finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
        if (-not $Required -or -not [string]::IsNullOrWhiteSpace($value)) { return $value }
        Write-Host 'A non-empty value is required.' -ForegroundColor Yellow
    }
}

function Set-EnvValue([string]$Name, [string]$Value) {
    $path = Join-Path $projectRoot '.env'
    $lines = [System.Collections.Generic.List[string]]::new()
    if (Test-Path -LiteralPath $path) { foreach ($line in [IO.File]::ReadAllLines($path)) { $lines.Add($line) } }
    $pattern = '^\s*' + [regex]::Escape($Name) + '='
    $found = $false
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match $pattern) { $lines[$i] = $Name + '=' + $Value; $found = $true; break }
    }
    if (-not $found) { $lines.Add($Name + '=' + $Value) }
    [IO.File]::WriteAllLines($path, $lines, [Text.UTF8Encoding]::new($false))
}

if (-not (Test-Path -LiteralPath '.env')) { Copy-Item -LiteralPath '.env.example' -Destination '.env' }
$envLines = Get-Content -LiteralPath '.env'
$adminLine = $envLines | Where-Object { $_ -match '^\s*ADMIN_PIN=' } | Select-Object -First 1
$adminValue = if ($adminLine) { ($adminLine -replace '^\s*ADMIN_PIN=', '') } else { '' }
if ([string]::IsNullOrWhiteSpace($adminValue)) {
    $adminValue = Read-SecretValue 'Set the private ADMIN_PIN for license administration' $true
    Set-EnvValue 'ADMIN_PIN' $adminValue
}
$groqLine = $envLines | Where-Object { $_ -match '^\s*GROQ_API_KEY=' } | Select-Object -First 1
$groqValue = if ($groqLine) { ($groqLine -replace '^\s*GROQ_API_KEY=', '') } else { '' }
if ([string]::IsNullOrWhiteSpace($groqValue)) {
    $groqValue = Read-SecretValue 'Groq API key (press Enter to continue without live Qwen decisions)' $false
    if (-not [string]::IsNullOrWhiteSpace($groqValue)) { Set-EnvValue 'GROQ_API_KEY' $groqValue }
}
Remove-Variable adminValue, groqValue -ErrorAction SilentlyContinue

$pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
if (-not $pythonLauncher) { throw 'Python Launcher (py.exe) was not found. Install Python 3.11 or 3.12 x64 first.' }
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $pythonLauncher.Source -3 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed.' }
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }

$hostValue = '127.0.0.1'; $portValue = 8000
foreach ($line in (Get-Content -LiteralPath '.env')) {
    if ($line -match '^\s*HOST=(.*)$') { $hostValue = $Matches[1].Trim() }
    if ($line -match '^\s*PORT=(\d+)$') { $portValue = [int]$Matches[1] }
}
$healthUrl = "http://127.0.0.1:$portValue/api/health"
try {
    $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
    if ($health.status -eq 'ok') { Write-Host "UniversalAI is already responding at $healthUrl" -ForegroundColor Green; exit 0 }
} catch { }

$logDir = Join-Path $projectRoot 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$stdout = Join-Path $logDir 'brain.stdout.log'
$stderr = Join-Path $logDir 'brain.stderr.log'
$process = Start-Process -FilePath $python -ArgumentList @('brain.py') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$ready = $false
for ($attempt = 0; $attempt -lt 20; $attempt++) {
    Start-Sleep -Seconds 1
    try { $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2; if ($health.status -eq 'ok') { $ready = $true; break } } catch { }
}
if (-not $ready) {
    if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force }
    Write-Host "Server did not become healthy. Review $stderr and $stdout" -ForegroundColor Red
    exit 1
}
Set-Content -LiteralPath (Join-Path $logDir 'brain.pid') -Value $process.Id -NoNewline
Write-Host "UniversalAI started (PID $($process.Id)) at $healthUrl" -ForegroundColor Green
Write-Host "Logs: $logDir" -ForegroundColor DarkGray
