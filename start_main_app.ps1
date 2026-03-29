# start_main_app.ps1
# ─────────────────────────────────────────────────────────────────────────────
# Starts the Dash stock-analysis app using whatever Python is active
# (the repo's main venv or conda environment).
# ─────────────────────────────────────────────────────────────────────────────

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "[main_app] Starting Dash app…" -ForegroundColor Cyan
python (Join-Path $ScriptDir "app.py")
