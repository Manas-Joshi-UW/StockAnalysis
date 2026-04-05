"""
financial_chatbot.py
--------------------
Singleton loader for the Llama GGUF model.
Import get_llm() from this module; it initialises the model once and
returns the same instance on every subsequent call.

Usage (from chat_service.py):
    from financial_chatbot import get_llm
    llm = get_llm()
    output = llm.create_chat_completion(messages=[...])
"""
from __future__ import annotations

import os
import threading


def _resolve_model_path() -> str:
    """Full path to the GGUF file: MODEL_PATH, or LLM_MODEL_PATH relative to this package."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    override = os.environ.get("MODEL_PATH")
    if override:
        return override
    name = (os.environ.get("LLM_MODEL_PATH") or "").strip()
    if not name:
        raise RuntimeError(
            "Set LLM_MODEL_PATH in .env (GGUF filename in chat_model/) "
            "or MODEL_PATH to an absolute path."
        )
    return name if os.path.isabs(name) else os.path.join(base_dir, name)


# ---------------------------------------------------------------------------
# Config — set LLM_MODEL_PATH / MODEL_PATH in .env before importing (see chat_service._load_env)
# ---------------------------------------------------------------------------

MODEL_PATH: str = _resolve_model_path()

# How many layers to offload to GPU (0 = CPU-only).
N_GPU_LAYERS: int = int(os.environ.get("N_GPU_LAYERS", "35"))

# Context window size.
N_CTX: int = int(os.environ.get("N_CTX", "4096"))

# Max tokens in a single response.
MAX_TOKENS: int = int(os.environ.get("MAX_TOKENS", "512"))

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_llm_instance = None
_llm_lock = threading.Lock()


def get_llm():
    """Return the singleton Llama instance, creating it on the first call."""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    with _llm_lock:
        if _llm_instance is not None:          # double-checked locking
            return _llm_instance

        try:
            from llama_cpp import Llama        # imported lazily so import errors are clear
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed in this environment. "
                "Activate chat_model/.venv_llm and run:\n"
                "  pip install llama-cpp-python"
            ) from exc

        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                f"GGUF model not found at: {MODEL_PATH}\n"
                "Check LLM_MODEL_PATH or MODEL_PATH in .env."
            )

        print(f"[chatbot] Loading model from {MODEL_PATH} "
              f"(n_gpu_layers={N_GPU_LAYERS}, n_ctx={N_CTX}) …")

        _llm_instance = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=N_CTX,
            verbose=False,
        )

        print("[chatbot] Model loaded.")
        return _llm_instance
