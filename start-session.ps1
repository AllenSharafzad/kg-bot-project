# Run this before starting work on this laptop
# Usage: .\start-session.ps1

$projectPath = $PSScriptRoot
$lockFile = Join-Path $projectPath ".session.lock"
$hostname = $env:COMPUTERNAME

# Check if another session is active
if (Test-Path $lockFile) {
    $lock = Get-Content $lockFile | ConvertFrom-Json
    Write-Host ""
    Write-Host "WARNING: Project is locked by: $($lock.machine)" -ForegroundColor Yellow
    Write-Host "  Locked at: $($lock.started)" -ForegroundColor Yellow
    Write-Host ""
    $answer = Read-Host "Override and take the lock? (yes/no)"
    if ($answer -ne "yes") {
        Write-Host "Aborted. Make sure the other laptop is done first." -ForegroundColor Red
        exit 1
    }
}

# Check for uncommitted changes from last session
Push-Location $projectPath
$status = git status --porcelain 2>&1
if ($status) {
    Write-Host ""
    Write-Host "WARNING: There are uncommitted changes from the last session:" -ForegroundColor Yellow
    git status --short
    Write-Host ""
    Write-Host "You should commit or review these before continuing." -ForegroundColor Yellow
    $answer = Read-Host "Continue anyway? (yes/no)"
    if ($answer -ne "yes") {
        Pop-Location
        exit 1
    }
}
Pop-Location

# Write lock file
$lock = @{
    machine = $hostname
    started = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
} | ConvertTo-Json
Set-Content $lockFile $lock

Write-Host ""
Write-Host "Session started on $hostname" -ForegroundColor Green
Write-Host "Remember to run .\end-session.ps1 when you're done!" -ForegroundColor Cyan
Write-Host ""
