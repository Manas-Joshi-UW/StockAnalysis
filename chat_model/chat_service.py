"""
chat_service.py
---------------
Lightweight Flask microservice that wraps the local GGUF model.
Runs on http://127.0.0.1:8001 (private to the machine).

Endpoints
---------
GET  /health  ->  {"ok": true}
POST /chat    ->  {"messages": [...]} or {"prompt": "..."}
               <- {"response": "..."}

Start with the LLM venv active:
    python chat_model/chat_service.py
"""
from __future__ import annotations

import os
import sys

# Allow running from the repo root or from inside chat_model/
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

# Load .env from repo root (or chat_model/) before anything else
def _load_env():
    for env_dir in (os.path.dirname(_here), _here):
        env_path = os.path.join(env_dir, ".env")
        if not os.path.isfile(env_path):
            continue
        with open(env_path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip().lstrip("export").strip()
                if key and key not in os.environ:
                    os.environ[key] = val.strip().strip("'\"")
        break

_load_env()

from flask import Flask, jsonify, request as flask_request
from financial_chatbot import get_llm, MAX_TOKENS

app = Flask(__name__)

SYSTEM_PROMPT = (
    "You are a knowledgeable finance assistant. "
    "Answer questions about stocks, companies, and financial markets clearly and concisely. "
    "If you are unsure, say so rather than guessing."
)


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.post("/chat")
def chat():
    body = flask_request.get_json(silent=True) or {}

    # Build the messages list.
    # Accept either {"messages": [...]} (multi-turn) or {"prompt": "..."} (single-turn).
    if "messages" in body and isinstance(body["messages"], list):
        messages = body["messages"]
        # Prepend system prompt if not already present
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    elif "prompt" in body:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(body["prompt"])},
        ]
    else:
        return jsonify({"error": "Request body must contain 'messages' or 'prompt'."}), 400

    try:
        llm = get_llm()
    except (RuntimeError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 503

    try:
        result = llm.create_chat_completion(
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            top_p=0.9,
        )
        reply = result["choices"][0]["message"]["content"].strip()
        return jsonify({"response": reply})
    except Exception as exc:          # noqa: BLE001
        return jsonify({"error": f"Inference error: {exc}"}), 500


if __name__ == "__main__":
    host = os.environ.get("CHAT_SERVICE_HOST", "127.0.0.1")
    port = int(os.environ.get("CHAT_SERVICE_PORT", "8001"))
    print(f"[chat_service] Starting on {host}:{port}")
    # Eagerly load the model so the first request isn't slow
    try:
        get_llm()
    except Exception as exc:
        print(f"[chat_service] WARNING: model pre-load failed: {exc}")
    app.run(host=host, port=port, debug=False, threaded=False)
