# update_price_action.ps1
# ─────────────────────────────────────────────────────────────────────────────
# Wrapper for the daily price-history update job. Intended to be run by
# Windows Task Scheduler (see register_price_action_task.ps1) or manually.
#
# - cd's into the repo root so relative paths work
# - activates .venv if present
# - runs update_price_action.py
# - propagates exit code so Task Scheduler can report failure
# ─────────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptDir

$venvActivate = Join-Path $scriptDir ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    . $venvActivate
    $python = "python"
} else {
    $python = "python"
}

& $python (Join-Path $scriptDir "update_price_action.py")
exit $LASTEXITCODE
