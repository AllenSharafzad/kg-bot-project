# Run this when you're done working on this laptop
# Usage: .\end-session.ps1 "brief description of what you did"

param(
    [string]$Message = ""
)

$projectPath = $PSScriptRoot
$lockFile = Join-Path $projectPath ".session.lock"
$hostname = $env:COMPUTERNAME

Push-Location $projectPath

# Check for changes to commit
$status = git status --porcelain 2>&1
if ($status) {
    Write-Host ""
    Write-Host "Uncommitted changes detected:" -ForegroundColor Yellow
    git status --short
    Write-Host ""

    if ($Message -eq "") {
        $Message = Read-Host "Enter a commit message (or press Enter to skip commit)"
    }

    if ($Message -ne "") {
        git add -A
        git commit -m $Message
        Write-Host "Changes committed." -ForegroundColor Green
    } else {
        Write-Host "Skipped commit. Changes left unstaged." -ForegroundColor Yellow
    }
} else {
    Write-Host "No uncommitted changes." -ForegroundColor Green
}

# Show log of this session
Write-Host ""
Write-Host "Recent commits:" -ForegroundColor Cyan
git log --oneline -5

# Remove lock file
if (Test-Path $lockFile) {
    Remove-Item $lockFile
    Write-Host ""
    Write-Host "Session ended on $hostname. Lock released." -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "No lock file found (already released or never locked)." -ForegroundColor Yellow
}

Pop-Location
