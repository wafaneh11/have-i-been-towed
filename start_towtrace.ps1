$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 .\start_towtrace.py
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python .\start_towtrace.py
} else {
    Write-Host "Python 3 was not found." -ForegroundColor Red
    Write-Host "Install it from https://www.python.org/downloads/ and enable Add Python to PATH."
    Read-Host "Press Enter to close"
    exit 1
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "TowTrace could not start. Read the message above, then try again." -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit $LASTEXITCODE
}
