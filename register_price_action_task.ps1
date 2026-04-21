# register_price_action_task.ps1
# ─────────────────────────────────────────────────────────────────────────────
# One-time setup: registers a Windows Scheduled Task that runs the daily
# price-history update every evening. Run this PowerShell script once.
#
# Defaults:
#   - Task name: StockAnalysis-DailyPriceUpdate
#   - Trigger:   Daily at 5:30 PM local time (after US equities close)
#   - Runs even if the computer was asleep at trigger time (StartWhenAvailable)
#
# To change the time, edit the -At value below and re-run this script.
# To remove the task:  Unregister-ScheduledTask -TaskName "StockAnalysis-DailyPriceUpdate" -Confirm:$false
# ─────────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$wrapperPath = Join-Path $scriptDir "update_price_action.ps1"

if (-not (Test-Path $wrapperPath)) {
    throw "Wrapper not found: $wrapperPath"
}

$taskName = "StockAnalysis-DailyPriceUpdate"
$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$wrapperPath`"" `
    -WorkingDirectory $scriptDir

$trigger = New-ScheduledTaskTrigger -Daily -At "5:30PM"

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6)

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Incrementally updates price_history/*.parquet files with latest OHLCV bars." `
    -Force | Out-Null

Write-Host "Registered scheduled task '$taskName' (daily @ 5:30 PM local)."
Write-Host "Logs will be written to: $(Join-Path $scriptDir 'logs')"
