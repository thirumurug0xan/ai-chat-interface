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
        { "device": "GPU" | "CPU" | "AUTO" | "XPU" }

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


@app.route("/api/config/update", methods=["POST"])
def config_update():
    """
    Update configuration settings dynamically.

    Expects JSON body:
        {
          "device": "GPU" | "CPU" | "AUTO" | "XPU", (optional)
          "max_new_tokens": int, (optional)
          "max_input_tokens": int (optional)
        }

    Returns updated config.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing config data"}), 400

    device_changed = False
    device_status = None

    # Handle device switch if requested
    if "device" in data:
        requested = data["device"].strip().upper()
        if requested != engine._requested_device:
            if engine._lock.locked():
                return jsonify({
                    "success": False,
                    "active_device": engine._active_device,
                    "requested_device": requested,
                    "message": "Cannot switch device while generation is in progress. Please wait.",
                }), 409
            device_status = engine.switch_device(requested)
            device_changed = True

    # Update tokens settings
    if "max_new_tokens" in data:
        try:
            val = int(data["max_new_tokens"])
            if 64 <= val <= 4096:
                engine.max_new_tokens = val
        except ValueError:
            pass

    if "max_input_tokens" in data:
        try:
            val = int(data["max_input_tokens"])
            if 256 <= val <= 4096:
                engine.max_input_tokens = val
        except ValueError:
            pass

    # Fetch fresh config
    res_config = engine.get_config()
    if device_changed and device_status:
        res_config["device_switch_result"] = device_status

    return jsonify(res_config)



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

    # Allow client to override max_new_tokens via request body
    max_tokens = data.get("max_tokens", None)

    def stream():
        try:
            for chunk in engine.generate_stream(messages, max_new_tokens=max_tokens):
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


@app.route("/api/fs/list", methods=["POST"])
def fs_list():
    """
    List contents of a directory on the server filesystem.
    Supports breadcrumb navigation and OpenVINO model folder detection.
    """
    data = request.get_json() or {}
    target_path = data.get("path")

    # Default to current workspace directory if no path specified
    if not target_path:
        target_path = os.getcwd()
    else:
        # Expand ~ to user home
        if target_path.startswith("~"):
            target_path = os.path.expanduser(target_path)

    # Normalize to absolute path
    target_path = os.path.abspath(target_path)

    if not os.path.exists(target_path):
        return jsonify({"error": f"Path '{target_path}' does not exist"}), 400

    if not os.path.isdir(target_path):
        return jsonify({"error": f"Path '{target_path}' is not a directory"}), 400

    try:
        entries = []
        for entry in os.listdir(target_path):
            entry_path = os.path.join(target_path, entry)
            is_dir = os.path.isdir(entry_path)

            # Check if directory looks like an OpenVINO model folder
            is_model = False
            if is_dir:
                try:
                    files = os.listdir(entry_path)
                    xml_files = [f for f in files if f.endswith(".xml") and "openvino" in f.lower()]
                    is_model = len(xml_files) > 0 or "openvino_model.xml" in files
                except Exception:
                    pass

            entries.append({
                "name": entry,
                "path": entry_path,
                "is_dir": is_dir,
                "is_model": is_model
            })

        # Sort: directories first, then files; both alphabetically case-insensitive
        entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))

        return jsonify({
            "current_path": target_path,
            "parent_path": os.path.dirname(target_path),
            "entries": entries,
            "workspace_path": os.getcwd(),
            "home_path": os.path.expanduser("~")
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/model/switch", methods=["POST"])
def model_switch():
    """
    Switch the active model folder at runtime.
    """
    data = request.get_json()
    if not data or "model_path" not in data:
        return jsonify({"error": "Missing 'model_path' in request body"}), 400

    requested_path = data["model_path"].strip()

    # Don't allow switching while generating
    if engine._lock.locked():
        return jsonify({
            "success": False,
            "model_name": engine.model_name,
            "model_path": engine.model_path,
            "message": "Cannot switch model while generation is in progress. Please wait.",
        }), 409

    try:
        result = engine.switch_model(requested_path)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "model_name": engine.model_name,
            "model_path": engine.model_path,
            "message": f"Model switch failed: {str(e)}",
        }), 500


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
