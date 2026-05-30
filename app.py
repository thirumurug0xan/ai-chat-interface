"""
app.py — Flask API server for the AI Chat Interface.

Serves the frontend and exposes REST + SSE endpoints for chat inference.
"""

import os
import json
import traceback
import psutil
from flask import Flask, request, Response, jsonify, send_from_directory
from dotenv import load_dotenv
from model_engine import ModelEngine
from system_stats import get_system_stats

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="/static")

# ── Initialize the model engine ──────────────────────────────────────────────
engine = ModelEngine()


# ── Static file serving ──────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main chat UI."""
    return send_from_directory(app.static_folder, "index.html")


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": engine.is_loaded(),
    })


@app.route("/api/config", methods=["GET"])
def config():
    """Return model configuration for the UI."""
    return jsonify(engine.get_config())


@app.route("/api/system/stats", methods=["GET"])
def system_stats():
    """
    System resource stats endpoint.

    Returns GPU VRAM (if available) and system RAM usage.
    """
    try:
        stats = get_system_stats(engine._active_device)
        return jsonify(stats)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/device/switch", methods=["POST"])
def device_switch():
    """
    Switch the inference device at runtime.

    Expects JSON body:
        { "device": "GPU" | "CPU" | "AUTO" | "CPU+GPU" }

    Returns:
        { "success": bool, "active_device": str, "requested_device": str,
          "active_device_friendly": str, "message": str }
    """
    data = request.get_json()
    if not data or "device" not in data:
        return jsonify({"error": "Missing 'device' in request body"}), 400

    requested = data["device"].strip().upper()

    # Don't allow switching while generating
    if engine._lock.locked():
        return jsonify({
            "success": False,
            "active_device": engine._active_device,
            "requested_device": requested,
            "message": "Cannot switch device while generation is in progress. Please wait.",
        }), 409

    try:
        result = engine.switch_device(requested)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "active_device": engine._active_device,
            "requested_device": requested,
            "message": f"Device switch failed: {str(e)}",
        }), 500


@app.route("/api/context/count", methods=["POST"])
def context_count():
    """
    Count how many tokens the given message history consumes.

    Expects JSON body:
        { "messages": [{"role": "user", "content": "Hello"}, ...] }

    Returns:
        { "used_tokens": N, "max_tokens": M, "percent": P }
    """
    data = request.get_json()
    if not data or "messages" not in data:
        return jsonify({"error": "Missing 'messages'"}), 400

    if not engine.is_loaded():
        return jsonify({"error": "Model not loaded"}), 503

    used = engine.count_tokens(data["messages"])
    max_tok = engine.max_input_tokens
    pct = round((used / max_tok) * 100, 1) if max_tok > 0 else 0

    return jsonify({
        "used_tokens": used,
        "max_tokens": max_tok,
        "percent": min(pct, 100),
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Chat endpoint with Server-Sent Events (SSE) streaming.

    Expects JSON body:
        { "messages": [{"role": "user", "content": "Hello"}, ...] }

    Returns:
        SSE stream of token chunks, ending with [DONE].
    """
    data = request.get_json()

    if not data or "messages" not in data:
        return jsonify({"error": "Missing 'messages' in request body"}), 400

    messages = data["messages"]

    if not messages:
        return jsonify({"error": "Messages list is empty"}), 400

    # Validate message format
    for msg in messages:
        if "role" not in msg or "content" not in msg:
            return jsonify({"error": "Each message must have 'role' and 'content'"}), 400

    if not engine.is_loaded():
        return jsonify({"error": "Model is not loaded yet. Please wait."}), 503

    def stream():
        try:
            for chunk in engine.generate_stream(messages):
                if isinstance(chunk, dict) and "__meta__" in chunk:
                    # Forward generation metadata as a separate SSE event
                    payload = json.dumps({"meta": chunk["__meta__"]})
                    yield f"data: {payload}\n\n"
                else:
                    # SSE format: data: <json>\n\n
                    payload = json.dumps({"chunk": chunk})
                    yield f"data: {payload}\n\n"

            # Signal that generation is complete
            yield "data: [DONE]\n\n"

        except Exception as e:
            error_payload = json.dumps({"error": str(e)})
            yield f"data: {error_payload}\n\n"
            yield "data: [DONE]\n\n"
            traceback.print_exc()

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.route("/api/chat/sync", methods=["POST"])
def chat_sync():
    """
    Synchronous chat endpoint (non-streaming fallback).

    Expects JSON body:
        { "messages": [{"role": "user", "content": "Hello"}, ...] }

    Returns:
        { "response": "..." }
    """
    data = request.get_json()

    if not data or "messages" not in data:
        return jsonify({"error": "Missing 'messages' in request body"}), 400

    messages = data["messages"]

    if not engine.is_loaded():
        return jsonify({"error": "Model is not loaded yet. Please wait."}), 503

    try:
        response = engine.generate(messages)
        return jsonify({"response": response})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Startup ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load model at startup
    try:
        engine.load()
    except Exception as e:
        print(f"[WARNING] Failed to load model: {e}")
        print("[WARNING] The server will start, but /api/chat will return 503.")
        print("[WARNING] Fix the model path/device in .env and restart.")

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))

    print(f"\n{'='*50}")
    print(f"  AI Chat Interface running at http://{host}:{port}")
    print(f"{'='*50}\n")

    app.run(host=host, port=port, debug=False, threaded=True)
