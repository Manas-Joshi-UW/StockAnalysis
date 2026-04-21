# start_chat_service.ps1
# ─────────────────────────────────────────────────────────────────────────────
# Starts the Flask chat microservice using the isolated LLM venv.
# Run this BEFORE (or alongside) start_main_app.ps1.
#
# First-time setup (run once from repo root):
# ─────────────────────────────────────────────────────────────────────────────
#   py -3.11 -m venv chat_model\.venv_llm
#   & .\chat_model\.venv_llm\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
#
#   # CPU-only (always works):
#   & .\chat_model\.venv_llm\Scripts\python.exe -m pip install flask python-dotenv llama-cpp-python
#
#   # CUDA 12.4 wheel (faster if you have an NVIDIA GPU):
#   # & .\chat_model\.venv_llm\Scripts\python.exe -m pip install flask python-dotenv `
#   #     llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
#
#   # Build from source with CUDA (if the wheel above doesn't match your Python/CUDA):
#   # $env:CMAKE_ARGS="-DGGML_CUDA=on"; $env:FORCE_CMAKE="1"
#   # & .\chat_model\.venv_llm\Scripts\python.exe -m pip install --no-cache-dir --force-reinstall llama-cpp-python
# ─────────────────────────────────────────────────────────────────────────────

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ScriptDir "chat_model\.venv_llm\Scripts\python.exe"
$ServiceScript = Join-Path $ScriptDir "chat_model\chat_service.py"

if (-not (Test-Path $VenvPython)) {
    $msg = "LLM venv not found at $VenvPython`n" +
        "Create it first - see the setup instructions at the top of this script."
    Write-Error $msg
    exit 1
}

Write-Host ('[chat_service] Starting with venv: ' + $VenvPython) -ForegroundColor Cyan
& $VenvPython $ServiceScript
