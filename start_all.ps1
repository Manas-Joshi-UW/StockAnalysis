# start_all.ps1
# ─────────────────────────────────────────────────────────────────────────────
# Launches both the chat service and the main Dash app in separate terminals.
# ─────────────────────────────────────────────────────────────────────────────

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Launching chat service in a new window…" -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-File", (Join-Path $ScriptDir "start_chat_service.ps1")

# Brief pause so the model starts loading before the UI comes up
Start-Sleep -Seconds 2

Write-Host "Launching main Dash app in a new window…" -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-File", (Join-Path $ScriptDir "start_main_app.ps1")

Write-Host "Both processes started. Check the two new terminal windows." -ForegroundColor Green
