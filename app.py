"""
app.py — Flask API server for the AI Chat Interface.

Serves the frontend and exposes REST + SSE endpoints for chat inference.
"""

import os
import json
import traceback
import psutil
import urllib.request
import urllib.parse
import uuid
import threading
import time
import shutil
import subprocess
from flask import Flask, request, Response, jsonify, send_from_directory
from dotenv import load_dotenv
from model_engine import MultiModelManager
from system_stats import get_system_stats

load_dotenv()

def update_env_file(key: str, value: str):
    """Safely update or insert a key-value pair in the .env file and reload it."""
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".env"))
    if not os.path.exists(env_path):
        # Create empty .env if not exists
        with open(env_path, "w", encoding="utf-8") as f:
            pass

    with open(env_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    key_found = False
    new_lines = []
    for line in lines:
        stripped = line.strip()
        parts = stripped.split("=", 1)
        if len(parts) > 0:
            left = parts[0].strip()
            if left.startswith("#"):
                left = left.lstrip("#").strip()
            if left == key:
                new_lines.append(f"{key}={value}\n")
                key_found = True
                continue
        new_lines.append(line)

    if not key_found:
        new_lines.append(f"{key}={value}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    # Reload updated environment variables into the process
    load_dotenv(override=True)

app = Flask(__name__, static_folder="static", static_url_path="/static")

# ── Initialize the model engine ──────────────────────────────────────────────
engine = MultiModelManager()

# ── RAG configuration and retrieval caching ──────────────────────────────────
import re
from rag_engine import BM25Retriever, retrieve_web_context, scrape_website_text

URL_PATTERN = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')

def detect_and_scrape_url(query):
    """
    Detects if a URL is in the query. If so, fetches the page text directly
    instead of running a web search.
    Returns (scraped_text, url_found) or (None, None).
    """
    match = URL_PATTERN.search(query)
    if match:
        url = match.group(0)
        if url.startswith("www."):
            url = "http://" + url
        print(f"[URL DETECTED] Directly scraping URL: {url}")
        scraped_text = scrape_website_text(url)
        if scraped_text and not scraped_text.startswith("Error loading webpage:"):
            return scraped_text, url
    return None, None

RAG_ENABLED = False
WEB_SEARCH_ENABLED = False
CHAT2_ENABLED = True
CHAT2_RAG_ENABLED = False
THINKING_ENABLED = False
DEEP_RESEARCH_ENABLED = False
DEEP_RESEARCH_MAX_ROUNDS = 3
FILE_GENERATION_ENABLED = os.getenv("FILE_GENERATION_ENABLED", "True").lower() in ("true", "1", "yes")
_retriever_cache = {}

def get_retriever():
    notes_dir = get_notes_dir()
    if notes_dir not in _retriever_cache:
        if len(_retriever_cache) > 5:
            _retriever_cache.clear()
        _retriever_cache[notes_dir] = BM25Retriever(notes_dir)
    return _retriever_cache[notes_dir]

def retrieve_notes_context(query, top_k=3):
    try:
        retriever = get_retriever()
        return retriever.retrieve(query, top_k=top_k)
    except Exception as e:
        print(f"[RAG ERROR] Failed to retrieve context: {e}")
        traceback.print_exc()
        return []

def strip_think_tags(text):
    """
    Remove <think>...</think> blocks (and any content inside them) from model output.
    Handles models like miniCPM that emit thinking blocks by default.
    """
    if not text:
        return text
    # Remove <think>...</think> blocks (greedy, multiline)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Also remove any lone opening <think> tag with trailing content (incomplete blocks)
    cleaned = re.sub(r'<think>[^<]*$', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    # Remove lone <think> or </think> tags
    cleaned = re.sub(r'</?think>', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

def optimize_search_query(query):
    """
    Extract essential search engine keywords from a user query using the loaded model.
    Only runs if query is long (more than 4 words) to save performance.
    """
    if not query:
        return ""
    if len(query.split()) <= 4:
        return query
        
    if not engine.is_loaded():
        return query

    try:
        search_prompt_history = [
            {"role": "system", "content": "You are a search query optimizer. Given the user's request, extract only the essential search engine keywords. Do not output any explanation or extra text. Output only the query words."},
            {"role": "user", "content": f"Generate a short, concise search engine query for: {query}"}
        ]
        optimized = engine.generate(search_prompt_history)
        if optimized:
            # Strip <think>...</think> blocks that some models emit by default
            optimized = strip_think_tags(optimized)
            cleaned = optimized.strip().replace('"', '').replace("'", "").replace("\n", " ").replace("\r", " ")
            if len(cleaned.strip()) > 0:
                print(f"[SEARCH OPTIMIZATION] Optimized query: '{query}' -> '{cleaned.strip()}'")
                return cleaned.strip()
    except Exception as e:
        print(f"Error optimizing search query: {e}")
        
    return query

def deep_research_web_search(query, max_rounds=None):
    """
    Iterative deep research: searches the web, shows results to the model,
    and asks if more data is needed. If the model says it needs more, it
    generates a follow-up query and searches again. Repeats until the model
    says it has enough data or the max rounds are reached.

    Returns:
        (combined_context_block, all_web_sources) — the accumulated context
        string and the list of all web source dicts.
    """
    if max_rounds is None:
        max_rounds = DEEP_RESEARCH_MAX_ROUNDS

    all_web_sources = []
    accumulated_snippets = []
    seen_urls = set()
    current_query = optimize_search_query(query)

    for round_num in range(1, max_rounds + 1):
        print(f"[DEEP RESEARCH] Round {round_num}/{max_rounds} — Searching: '{current_query}'")

        web_results = retrieve_web_context(current_query, top_k=5)
        new_snippets = []
        for w in web_results:
            if w.get("url") not in seen_urls:
                seen_urls.add(w["url"])
                snippet_text = f"[Source URL: {w['url']}]\nTitle: {w['title']}\nSnippet: {w['snippet']}"
                new_snippets.append(snippet_text)
                accumulated_snippets.append(snippet_text)
                all_web_sources.append({"title": w["title"], "url": w["url"]})

        if not new_snippets:
            print(f"[DEEP RESEARCH] Round {round_num}: No new results found. Stopping.")
            break

        # If this is the last allowed round, don't ask — just stop
        if round_num >= max_rounds:
            print(f"[DEEP RESEARCH] Max rounds ({max_rounds}) reached. Stopping.")
            break

        # Ask the model if it has enough context — simple yes/no
        all_context_so_far = "\n\n".join(accumulated_snippets)
        eval_prompt = [
            {"role": "system", "content": (
                "You are a research assistant. Review the search results and the user's question. "
                "Do you have enough information to fully answer the question? "
                "Say yes to stop or say no to continue searching. "
                "Output only yes or no."
            )},
            {"role": "user", "content": (
                f"Question: {query}\n\n"
                f"Search results so far:\n{all_context_so_far}\n\n"
                f"Do you have enough information? Say yes to stop or say no to continue."
            )}
        ]

        try:
            eval_response = engine.generate(eval_prompt)
            eval_response = strip_think_tags(eval_response).strip().lower()
            print(f"[DEEP RESEARCH] Model evaluation: '{eval_response}'")

            # Check if model says yes (sufficient)
            if "yes" in eval_response:
                print(f"[DEEP RESEARCH] Model says data is sufficient after round {round_num}.")
                break
            elif "no" in eval_response:
                # Model wants more data — ask for a follow-up search query
                followup_prompt = [
                    {"role": "system", "content": (
                        "You are a search query generator. The user needs more information. "
                        "Based on the original question and the search results already collected, "
                        "generate a different, more specific search query to find missing information. "
                        "Output only the search query words, nothing else."
                    )},
                    {"role": "user", "content": (
                        f"Original question: {query}\n\n"
                        f"Already searched and found:\n{all_context_so_far}\n\n"
                        f"Generate a new search query to find what's still missing:"
                    )}
                ]
                follow_up = engine.generate(followup_prompt)
                follow_up = strip_think_tags(follow_up).strip().replace('"', '').replace("'", "").replace("\n", " ")
                if follow_up and len(follow_up.strip()) > 0:
                    current_query = follow_up.strip()
                    print(f"[DEEP RESEARCH] Model requests follow-up search: '{current_query}'")
                else:
                    print(f"[DEEP RESEARCH] Empty follow-up query. Stopping.")
                    break
            else:
                # Model didn't clearly say yes or no — treat as sufficient
                print(f"[DEEP RESEARCH] Unclear model response ('{eval_response}'), treating as sufficient.")
                break
        except Exception as e:
            print(f"[DEEP RESEARCH] Error during evaluation: {e}")
            break

    # Build final context block
    if accumulated_snippets:
        context_str = "\n\n".join(accumulated_snippets)
        context_block = (
            f"[Context retrieved from Deep Web Research ({len(accumulated_snippets)} sources)]\n"
            "----------------------------------------\n"
            f"{context_str}\n"
            "----------------------------------------\n\n"
        )
    else:
        context_block = ""

    return context_block, all_web_sources

# Global dictionary to track Hugging Face model export and download processes
active_downloads = {}

def run_export_process(task_id, cmd, output_dir, hf_token=None):
    task = active_downloads.get(task_id)
    if not task:
        return
    
    try:
        # Create output directory parent folders if they don't exist
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        
        task["logs"].append(f"Starting export command: {' '.join(cmd)}\n\n")
        
        # Prepare environment
        process_env = os.environ.copy()
        if hf_token:
            process_env["HF_TOKEN"] = hf_token
            task["logs"].append("[SYSTEM] Authenticating with Hugging Face Token...\n")
        
        # Start subprocess, redirect stderr to stdout so we capture everything
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=process_env
        )
        task["process"] = process
        
        # Read output line by line as it is generated
        for line in iter(process.stdout.readline, ""):
            task["logs"].append(line)
        
        process.wait()
        
        if process.returncode == 0:
            task["status"] = "completed"
            task["logs"].append("\n[COMPLETED] Model export finished successfully.")
        else:
            task["status"] = "failed"
            task["error"] = f"Exit code: {process.returncode}"
            task["logs"].append(f"\n[FAILED] Model export failed with exit code: {process.returncode}")
    except Exception as e:
        task["status"] = "failed"
        task["error"] = str(e)
        task["logs"].append(f"\n[ERROR] Exception occurred during export: {str(e)}")
        traceback.print_exc()
    finally:
        task["process"] = None


def inject_file_generation_instructions(messages: list) -> list:
    """
    Ensures that a system prompt instructing the model on how to format
    generated files is present. Appends instructions to any existing system prompt.
    """
    if not FILE_GENERATION_ENABLED:
        return messages

    file_instruction = (
        "\n\nIf the user requests you to generate a file (such as a script, a code file, data file, CSV, HTML, CSS, text, etc.), "
        "you MUST specify the filename in the code block info string like this:\n"
        "```python filename=\"main.py\"\n"
        "# code/content here\n"
        "```\n"
        "Or write the file name/path clearly as a comment in the first line of the code block (e.g. `# script.py` or `// main.js`). "
        "This allows the system to present a direct download link/card for the user."
    )
    
    # Check if there is an existing system message
    has_system = False
    for msg in messages:
        if msg.get("role") == "system":
            has_system = True
            content = msg.get("content", "")
            if file_instruction not in content:
                msg["content"] = content + file_instruction
            break
            
    if not has_system:
        # If thinking is enabled, we'll combine it in the new system prompt
        system_content = "You are a helpful assistant."
        if THINKING_ENABLED:
            system_content = (
                "Before answering, you must write out your step-by-step thinking process inside <think>...</think> tags. "
                "In your thinking process, analyze the request, outline what steps/actions are needed, and verify assumptions. "
                "After the </think> tag, output your final response."
            )
        system_content += file_instruction
        messages.insert(0, {
            "role": "system",
            "content": system_content
        })
        
    return messages


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
    cfg = engine.get_config()
    cfg["rag_enabled"] = RAG_ENABLED
    cfg["web_search_enabled"] = WEB_SEARCH_ENABLED
    cfg["chat2_enabled"] = CHAT2_ENABLED
    cfg["chat2_rag_enabled"] = CHAT2_RAG_ENABLED
    cfg["thinking_enabled"] = THINKING_ENABLED
    cfg["deep_research_enabled"] = DEEP_RESEARCH_ENABLED
    cfg["deep_research_max_rounds"] = DEEP_RESEARCH_MAX_ROUNDS
    cfg["file_generation_enabled"] = FILE_GENERATION_ENABLED
    return jsonify(cfg)


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
          "max_input_tokens": int, (optional)
          "kv_cache_precision": "u8" | "f16" | "f32" | "default" (optional)
          "rag_enabled": bool (optional)
        }

    Returns updated config.
    """
    global RAG_ENABLED, WEB_SEARCH_ENABLED, CHAT2_ENABLED, CHAT2_RAG_ENABLED, THINKING_ENABLED, DEEP_RESEARCH_ENABLED, DEEP_RESEARCH_MAX_ROUNDS, FILE_GENERATION_ENABLED
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing config data"}), 400

    device_changed = False
    precision_changed = False
    device_status = None
    reload_status = None

    # Determine if device changed
    if "device" in data:
        requested_device = data["device"].strip().upper()
        if requested_device != engine._requested_device:
            device_changed = True

    # Determine if precision changed
    if "kv_cache_precision" in data:
        requested_prec = data["kv_cache_precision"].strip().lower()
        active_prec = engine.kv_cache_precision.strip().lower() if engine.kv_cache_precision else "default"
        if requested_prec != active_prec:
            precision_changed = True

    # If compilation changes are requested, ensure no active lock (not currently generating)
    if device_changed or precision_changed:
        if engine._lock.locked():
            return jsonify({
                "success": False,
                "active_device": engine._active_device,
                "requested_device": engine._requested_device,
                "message": "Cannot switch device or KV cache precision while generation is in progress. Please wait.",
            }), 409

    # Update precision in env/memory and keep ov_config in sync
    if "kv_cache_precision" in data:
        val = data["kv_cache_precision"].strip().lower()
        if val in ("u8", "f16", "f32", "default"):
            engine.kv_cache_precision = val
            # Sync ov_config so switch_device/reload uses the correct precision
            eng = engine.active_engine
            if eng:
                if val != "default":
                    eng.ov_config["KV_CACHE_PRECISION"] = val
                else:
                    eng.ov_config.pop("KV_CACHE_PRECISION", None)
            update_env_file("OV_KV_CACHE_PRECISION", val)

    # Update device in env/memory
    if "device" in data:
        val = data["device"].strip().upper()
        if val in engine.AVAILABLE_DEVICES:
            update_env_file("DEVICE", val)

    # Update tokens settings
    if "max_new_tokens" in data:
        try:
            val = int(data["max_new_tokens"])
            if 64 <= val <= 8192:
                engine.max_new_tokens = val
                update_env_file("MAX_NEW_TOKENS", str(val))
        except ValueError:
            pass

    if "max_input_tokens" in data:
        try:
            val = int(data["max_input_tokens"])
            if 256 <= val <= 128000:
                engine.max_input_tokens = val
                update_env_file("MAX_INPUT_TOKENS", str(val))
        except ValueError:
            pass

    # Trigger device switch or reload
    if device_changed or precision_changed:
        if device_changed:
            requested_device = data["device"].strip().upper()
            device_status = engine.switch_device(requested_device)
        else:
            reload_status = engine.reload()

    if "rag_enabled" in data:
        RAG_ENABLED = bool(data["rag_enabled"])

    if "web_search_enabled" in data:
        WEB_SEARCH_ENABLED = bool(data["web_search_enabled"])

    if "chat2_enabled" in data:
        CHAT2_ENABLED = bool(data["chat2_enabled"])

    if "chat2_rag_enabled" in data:
        CHAT2_RAG_ENABLED = bool(data["chat2_rag_enabled"])

    if "thinking_enabled" in data:
        THINKING_ENABLED = bool(data["thinking_enabled"])

    if "deep_research_enabled" in data:
        DEEP_RESEARCH_ENABLED = bool(data["deep_research_enabled"])

    if "deep_research_max_rounds" in data:
        try:
            val = int(data["deep_research_max_rounds"])
            if 1 <= val <= 10:
                DEEP_RESEARCH_MAX_ROUNDS = val
        except (ValueError, TypeError):
            pass

    if "file_generation_enabled" in data:
        FILE_GENERATION_ENABLED = bool(data["file_generation_enabled"])
        update_env_file("FILE_GENERATION_ENABLED", str(FILE_GENERATION_ENABLED))

    # Fetch fresh config
    res_config = engine.get_config()
    res_config["rag_enabled"] = RAG_ENABLED
    res_config["web_search_enabled"] = WEB_SEARCH_ENABLED
    res_config["chat2_enabled"] = CHAT2_ENABLED
    res_config["chat2_rag_enabled"] = CHAT2_RAG_ENABLED
    res_config["thinking_enabled"] = THINKING_ENABLED
    res_config["deep_research_enabled"] = DEEP_RESEARCH_ENABLED
    res_config["deep_research_max_rounds"] = DEEP_RESEARCH_MAX_ROUNDS
    res_config["file_generation_enabled"] = FILE_GENERATION_ENABLED
    
    if device_changed and device_status:
        res_config["device_switch_result"] = device_status
    elif precision_changed and reload_status:
        res_config["device_switch_result"] = reload_status

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
    max_tok = engine.effective_max_input_tokens
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

    # Validate message format and copy
    messages_copy = []
    for msg in messages:
        if "role" not in msg or "content" not in msg:
            return jsonify({"error": "Each message must have 'role' and 'content'"}), 400
        messages_copy.append(dict(msg))
    messages = messages_copy

    # Ensure system prompt guides the model to think and supports file generation
    messages = inject_file_generation_instructions(messages)

    if not engine.is_loaded():
        return jsonify({"error": "Model is not loaded yet. Please wait."}), 503

    # Allow client to override max_new_tokens via request body
    max_tokens = data.get("max_tokens", None)

    # Check if RAG/Web Search is enabled for this request
    is_rag = data.get("rag_enabled", RAG_ENABLED)
    is_web = data.get("web_search_enabled", WEB_SEARCH_ENABLED)
    sources = []
    web_sources = []

    if (is_rag or is_web) and messages:
        latest_msg = messages[-1]
        if latest_msg["role"] == "user":
            query = latest_msg["content"]
            context_block = ""
            
            if is_rag:
                results = retrieve_notes_context(query)
                if results:
                    context_lines = []
                    for r in results:
                        context_lines.append(f"[File: {r['filename']}]\n{r['content']}")
                    context_str = "\n\n".join(context_lines)
                    context_block += (
                        "[Context retrieved from Mousepad Notes]\n"
                        "----------------------------------------\n"
                        f"{context_str}\n"
                        "----------------------------------------\n\n"
                    )
                    sources = [{"filename": r["filename"], "score": r["score"]} for r in results]
            
            if is_web:
                scraped_text, scraped_url = detect_and_scrape_url(query)
                if scraped_text:
                    context_block += (
                        f"[Context retrieved from directly visiting URL: {scraped_url}]\n"
                        "----------------------------------------\n"
                        f"{scraped_text}\n"
                        "----------------------------------------\n\n"
                    )
                    web_sources = [{"title": scraped_url, "url": scraped_url}]
                elif DEEP_RESEARCH_ENABLED:
                    # Use iterative deep research mode
                    deep_block, deep_sources = deep_research_web_search(query)
                    if deep_block:
                        context_block += deep_block
                        web_sources = deep_sources
                else:
                    search_query = optimize_search_query(query)
                    web_results = retrieve_web_context(search_query)
                    if web_results:
                        context_lines = []
                        for w in web_results:
                            context_lines.append(f"[Source URL: {w['url']}]\nTitle: {w['title']}\nSnippet: {w['snippet']}")
                        context_str = "\n\n".join(context_lines)
                        context_block += (
                            "[Context retrieved from Web Search]\n"
                            "----------------------------------------\n"
                            f"{context_str}\n"
                            "----------------------------------------\n\n"
                        )
                        web_sources = [{"title": w["title"], "url": w["url"]} for w in web_results]

            if context_block:
                context_block += (
                    "Based on the context retrieved above, please answer the user's request.\n"
                    "User request: "
                )
                latest_msg["content"] = context_block + query

    def stream():
        try:
            # Yield sources as the first SSE event so frontend can show them immediately
            if (is_rag and sources) or (is_web and web_sources):
                meta_payload = {}
                if is_rag and sources:
                    meta_payload["sources"] = sources
                if is_web and web_sources:
                    meta_payload["web_sources"] = web_sources
                yield f"data: {json.dumps({'meta': meta_payload})}\n\n"

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
    if not messages:
        return jsonify({"error": "Messages list is empty"}), 400

    messages = [dict(msg) for msg in messages]
    # Ensure system prompt guides the model to think and supports file generation
    messages = inject_file_generation_instructions(messages)

    if not engine.is_loaded():
        return jsonify({"error": "Model is not loaded yet. Please wait."}), 503

    # Check if RAG/Web Search is enabled for this request
    is_rag = data.get("rag_enabled", RAG_ENABLED)
    is_web = data.get("web_search_enabled", WEB_SEARCH_ENABLED)
    sources = []
    web_sources = []

    if (is_rag or is_web) and messages:
        latest_msg = messages[-1]
        if latest_msg["role"] == "user":
            query = latest_msg["content"]
            context_block = ""
            
            if is_rag:
                results = retrieve_notes_context(query)
                if results:
                    context_lines = []
                    for r in results:
                        context_lines.append(f"[File: {r['filename']}]\n{r['content']}")
                    context_str = "\n\n".join(context_lines)
                    context_block += (
                        "[Context retrieved from Mousepad Notes]\n"
                        "----------------------------------------\n"
                        f"{context_str}\n"
                        "----------------------------------------\n\n"
                    )
                    sources = [{"filename": r["filename"], "score": r["score"]} for r in results]
            
            if is_web:
                scraped_text, scraped_url = detect_and_scrape_url(query)
                if scraped_text:
                    context_block += (
                        f"[Context retrieved from directly visiting URL: {scraped_url}]\n"
                        "----------------------------------------\n"
                        f"{scraped_text}\n"
                        "----------------------------------------\n\n"
                    )
                    web_sources = [{"title": scraped_url, "url": scraped_url}]
                elif DEEP_RESEARCH_ENABLED:
                    deep_block, deep_sources = deep_research_web_search(query)
                    if deep_block:
                        context_block += deep_block
                        web_sources = deep_sources
                else:
                    search_query = optimize_search_query(query)
                    web_results = retrieve_web_context(search_query)
                    if web_results:
                        context_lines = []
                        for w in web_results:
                            context_lines.append(f"[Source URL: {w['url']}]\nTitle: {w['title']}\nSnippet: {w['snippet']}")
                        context_str = "\n\n".join(context_lines)
                        context_block += (
                            "[Context retrieved from Web Search]\n"
                            "----------------------------------------\n"
                            f"{context_str}\n"
                            "----------------------------------------\n\n"
                        )
                        web_sources = [{"title": w["title"], "url": w["url"]} for w in web_results]

            if context_block:
                context_block += (
                    "Based on the context retrieved above, please answer the user's request.\n"
                    "User request: "
                )
                latest_msg["content"] = context_block + query

    try:
        response = engine.generate(messages)
        ret_val = {"response": response}
        if (is_rag and sources) or (is_web and web_sources):
            meta_payload = {}
            if is_rag and sources:
                meta_payload["sources"] = sources
            if is_web and web_sources:
                meta_payload["web_sources"] = web_sources
            ret_val["meta"] = meta_payload
        return jsonify(ret_val)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/mcp/chat", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
def mcp_chat():
    """
    OpenAI-compatible chat completions endpoint.
    Supports streaming and non-streaming modes.
    Enables integration with VS Code extensions (e.g. Continue, Cline),
    Claude Desktop MCP servers, and other local dev tools.
    """
    if not engine.is_loaded():
        return jsonify({"error": "Model is not loaded. Please wait."}), 503

    data = request.get_json() or {}
    messages = data.get("messages")
    if not messages:
        return jsonify({"error": "Missing 'messages' in request body"}), 400

    # Extract parameters
    stream_requested = bool(data.get("stream", False))
    max_tokens = data.get("max_tokens") or data.get("max_new_tokens")
    model_name = engine.model_name
    chat_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())

    # Format messages to role/content pairs
    formatted_messages = []
    for msg in messages:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            content = msg["content"]
            if isinstance(content, list):
                # Extract text parts from block lists (e.g. Cline multimodal/structured format)
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text" and "text" in part:
                            text_parts.append(part["text"])
                        elif "text" in part:
                            text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = "\n".join(text_parts)
            elif content is None:
                content = ""
            else:
                content = str(content)

            formatted_messages.append({
                "role": msg["role"],
                "content": content
            })

    if not formatted_messages:
        return jsonify({"error": "Invalid messages format"}), 400

    if stream_requested:
        def sse_generator():
            try:
                for chunk in engine.generate_stream(formatted_messages, max_new_tokens=max_tokens):
                    # Skip meta dicts
                    if isinstance(chunk, dict) and "__meta__" in chunk:
                        continue
                    
                    payload = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "content": chunk
                            },
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

                # Send final stop chunk
                final_payload = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final_payload)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                err_payload = {
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": 500
                    }
                }
                yield f"data: {json.dumps(err_payload)}\n\n"
                yield "data: [DONE]\n\n"
                traceback.print_exc()

        return Response(
            sse_generator(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
            }
        )
    else:
        try:
            response_text = engine.generate(formatted_messages)
            completion_tokens = engine.count_tokens([{"role": "assistant", "content": response_text}])
            prompt_tokens = engine.count_tokens(formatted_messages)
            
            payload = {
                "id": chat_id,
                "object": "chat.completion",
                "created": created_time,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
            return jsonify(payload)
        except Exception as e:
            traceback.print_exc()
            return jsonify({
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": 500
                }
            }), 500


@app.route("/api/chat2", methods=["GET", "POST"])
def chat2():
    """
    Simple curl-friendly endpoint.
    Accepts GET /api/chat2?quirie=... or POST with 'quirie' param/JSON.
    Returns plain text response (optionally streamed).
    """
    if not CHAT2_ENABLED:
        return "Error: Endpoint /api/chat2 is disabled.\n", 403

    # Check query param 'quirie' or 'query'
    query = request.args.get("quirie") or request.args.get("query")
    
    # If not in query params, check POST body
    if not query and request.method == "POST":
        if request.is_json:
            data = request.get_json() or {}
            query = data.get("quirie") or data.get("query")
        else:
            query = request.form.get("quirie") or request.form.get("query")

    if not query:
        return "Error: Missing parameter 'quirie'\n", 400

    if not engine.is_loaded():
        return "Error: Model is not loaded yet. Please wait.\n", 503

    # Check if RAG/Web Search is enabled for chat2
    rag_val = request.args.get("rag")
    web_val = request.args.get("web")
    if rag_val is None and request.method == "POST":
        if request.is_json:
            data = request.get_json() or {}
            rag_val = data.get("rag")
            web_val = data.get("web")
        else:
            rag_val = request.form.get("rag")
            web_val = request.form.get("web")
            
    is_rag = CHAT2_RAG_ENABLED
    if rag_val is not None:
        if isinstance(rag_val, bool):
            is_rag = rag_val
        else:
            is_rag = str(rag_val).lower() not in ("false", "0", "no")

    is_web = WEB_SEARCH_ENABLED
    if web_val is not None:
        if isinstance(web_val, bool):
            is_web = web_val
        else:
            is_web = str(web_val).lower() not in ("false", "0", "no")

    sources = []
    web_sources = []
    context_block = ""
    
    if is_rag:
        results = retrieve_notes_context(query)
        if results:
            context_lines = []
            for r in results:
                context_lines.append(f"[File: {r['filename']}]\n{r['content']}")
            context_str = "\n\n".join(context_lines)
            context_block += (
                "[Context retrieved from Mousepad Notes]\n"
                "----------------------------------------\n"
                f"{context_str}\n"
                "----------------------------------------\n\n"
            )
            sources = [{"filename": r["filename"], "score": r["score"]} for r in results]

    if is_web:
        scraped_text, scraped_url = detect_and_scrape_url(query)
        if scraped_text:
            context_block += (
                f"[Context retrieved from directly visiting URL: {scraped_url}]\n"
                "----------------------------------------\n"
                f"{scraped_text}\n"
                "----------------------------------------\n\n"
            )
            web_sources = [{"title": scraped_url, "url": scraped_url}]
        elif DEEP_RESEARCH_ENABLED:
            deep_block, deep_sources = deep_research_web_search(query)
            if deep_block:
                context_block += deep_block
                web_sources = deep_sources
        else:
            search_query = optimize_search_query(query)
            web_results = retrieve_web_context(search_query)
            if web_results:
                context_lines = []
                for w in web_results:
                    context_lines.append(f"[Source URL: {w['url']}]\nTitle: {w['title']}\nSnippet: {w['snippet']}")
                context_str = "\n\n".join(context_lines)
                context_block += (
                    "[Context retrieved from Web Search]\n"
                    "----------------------------------------\n"
                    f"{context_str}\n"
                    "----------------------------------------\n\n"
                )
                web_sources = [{"title": w["title"], "url": w["url"]} for w in web_results]

    if context_block:
        context_block += (
            "Based on the context retrieved above, please answer the user's request.\n"
            "User request: "
        )
        query = context_block + query

    messages = []
    messages.append({"role": "user", "content": query})
    messages = inject_file_generation_instructions(messages)

    # Determine if streaming is requested (enabled by default)
    stream_val = request.args.get("stream")
    if stream_val is None and request.method == "POST":
        if request.is_json:
            data = request.get_json() or {}
            stream_val = data.get("stream")
        else:
            stream_val = request.form.get("stream")
            
    if stream_val is None:
        stream_param = True
    else:
        # Convert stream_val to boolean or equivalent string evaluation
        if isinstance(stream_val, bool):
            stream_param = stream_val
        else:
            stream_param = str(stream_val).lower() not in ("false", "0", "no")

    if stream_param:
        def stream_generator():
            try:
                # For plain text stream, print RAG sources header if matched
                if (is_rag and sources) or (is_web and web_sources):
                    headers = []
                    if is_rag and sources:
                        headers.append(f"Notes: {', '.join([s['filename'] for s in sources])}")
                    if is_web and web_sources:
                        headers.append(f"Web: {', '.join([s['title'] for s in web_sources])}")
                    yield f"[RAG Sources | {'; '.join(headers)}]\n\n"

                for chunk in engine.generate_stream(messages):
                    if isinstance(chunk, dict) and "__meta__" in chunk:
                        continue
                    yield chunk
            except Exception as e:
                yield f"\nError during generation: {str(e)}\n"
                traceback.print_exc()

        return Response(
            stream_generator(),
            mimetype="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )
    else:
        try:
            response = engine.generate(messages)
            if (is_rag and sources) or (is_web and web_sources):
                headers = []
                if is_rag and sources:
                    headers.append(f"Notes: {', '.join([s['filename'] for s in sources])}")
                if is_web and web_sources:
                    headers.append(f"Web: {', '.join([s['title'] for s in web_sources])}")
                sources_header = f"[RAG Sources | {'; '.join(headers)}]\n\n"
                response = sources_header + response
            return Response(response, mimetype="text/plain")
        except Exception as e:
            traceback.print_exc()
            return Response(f"Error: {str(e)}\n", mimetype="text/plain", status=500)


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

            # Check if directory looks like an OpenVINO model folder or cache folder
            is_model = False
            if is_dir:
                try:
                    files = os.listdir(entry_path)
                    files_lower = [f.lower() for f in files]
                    has_config = "config.json" in files_lower
                    xml_files = [f for f in files if f.endswith(".xml")]
                    has_cache = any(f.endswith(".cl_cache") or f.endswith(".blob") or "onednn" in f for f in files_lower)
                    is_model = (has_config and (len(xml_files) > 0 or "openvino_model.xml" in files_lower)) or has_cache
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
    Loading a new model is safe during generation because each model
    gets its own independent ModelEngine with its own lock.
    """
    data = request.get_json()
    if not data or "model_path" not in data:
        return jsonify({"error": "Missing 'model_path' in request body"}), 400

    requested_path = data["model_path"].strip()

    try:
        # Extract optional configuration options from the request body
        use_cache = data.get("use_cache")
        model_file = data.get("model_file")
        trust_remote_code = data.get("trust_remote_code")
        fix_mistral_regex = data.get("fix_mistral_regex")
        ov_performance_hint = data.get("ov_performance_hint")

        result = engine.load_model(
            requested_path,
            use_cache=use_cache,
            model_file=model_file,
            trust_remote_code=trust_remote_code,
            fix_mistral_regex=fix_mistral_regex,
            ov_performance_hint=ov_performance_hint,
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "model_name": engine.model_name,
            "model_path": engine.model_path,
            "message": f"Model switch failed: {str(e)}",
        }), 500


@app.route("/api/models/loaded", methods=["GET"])
def models_loaded():
    """Return list of all currently loaded models."""
    return jsonify(engine.get_loaded_models())

@app.route("/api/models/activate", methods=["POST"])
def models_activate():
    """Switch the active model to an already-loaded one."""
    data = request.get_json()
    if not data or "model_path" not in data:
        return jsonify({"error": "Missing 'model_path'"}), 400
    result = engine.activate_model(data["model_path"])
    return jsonify(result)

@app.route("/api/models/unload", methods=["POST"])
def models_unload():
    """Unload a specific model."""
    data = request.get_json()
    if not data or "model_path" not in data:
        return jsonify({"error": "Missing 'model_path'"}), 400
    result = engine.unload_model(data["model_path"])
    return jsonify(result)


# ── Model Downloader Endpoints ───────────────────────────────────────────────

@app.route("/api/models/search", methods=["GET"])
def models_search():
    """
    Search Hugging Face models for text-generation.
    """
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])
    
    url = f"https://huggingface.co/api/models?search={urllib.parse.quote(query)}&filter=text-generation&limit=10&sort=downloads&direction=-1"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            results = []
            for item in data:
                model_id = item.get("id", "")
                downloads = item.get("downloads", 0)
                likes = item.get("likes", 0)
                author = item.get("author", "")
                if model_id:
                    results.append({
                        "model_id": model_id,
                        "downloads": downloads,
                        "likes": likes,
                        "author": author
                    })
            return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Failed to search Hugging Face: {str(e)}"}), 500


@app.route("/api/models/download", methods=["POST"])
def models_download():
    """
    Start downloading and exporting a Hugging Face model to OpenVINO.
    """
    data = request.get_json() or {}
    model_id = data.get("model_id", "").strip()
    weight_format = data.get("weight_format", "int8").strip()
    task = data.get("task", "text-generation-with-past").strip()
    output_dir_name = data.get("output_dir", "").strip()
    hf_token = data.get("hf_token", "").strip()
    
    if not model_id:
        return jsonify({"error": "Missing 'model_id' in request body"}), 400
    if not output_dir_name:
        return jsonify({"error": "Missing 'output_dir' in request body"}), 400
        
    # Prevent directory traversal by sanitizing the directory name
    workspace_path = os.path.abspath(os.getcwd())
    output_dir_name = os.path.basename(output_dir_name.strip())
    output_path = os.path.abspath(os.path.join(workspace_path, output_dir_name))
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())[:8]
    
    # Check if a task is already exporting to the same directory or has the same model ID
    for tid, t in active_downloads.items():
        if t["status"] == "running" and (t["output_dir"] == output_path or t["model_id"] == model_id):
            return jsonify({"error": "An export job for this model or output directory is already running"}), 409

    # Build optimum-cli export command
    optimum_executable = shutil.which("optimum-cli")
    if not optimum_executable:
        # Check local venv/bin
        local_optimum = os.path.join(workspace_path, "venv", "bin", "optimum-cli")
        if os.path.exists(local_optimum):
            optimum_executable = local_optimum
        else:
            optimum_executable = "optimum-cli"
            
    cmd = [optimum_executable, "export", "openvino", "--model", model_id]
    if task:
        cmd.extend(["--task", task])
    if weight_format:
        cmd.extend(["--weight-format", weight_format])
    cmd.append(output_path)
    
    # Initialize task state
    active_downloads[task_id] = {
        "task_id": task_id,
        "model_id": model_id,
        "weight_format": weight_format,
        "task": task,
        "output_dir": output_path,
        "output_dir_name": output_dir_name,
        "status": "running",
        "logs": [],
        "process": None,
        "error": None,
        "start_time": time.time()
    }
    
    # Start thread
    thread = threading.Thread(
        target=run_export_process,
        args=(task_id, cmd, output_path, hf_token),
        daemon=True
    )
    thread.start()
    
    return jsonify({
        "success": True,
        "task_id": task_id,
        "message": f"Export process started for {model_id}."
    })


@app.route("/api/models/download/status", methods=["GET"])
def models_download_status():
    """
    Get status of all download/export tasks.
    """
    tasks_summary = {}
    for tid, t in active_downloads.items():
        elapsed = time.time() - t["start_time"] if t["status"] == "running" else None
        tasks_summary[tid] = {
            "task_id": tid,
            "model_id": t["model_id"],
            "weight_format": t["weight_format"],
            "task": t["task"],
            "output_dir_name": t["output_dir_name"],
            "status": t["status"],
            "error": t["error"],
            "elapsed_sec": round(elapsed, 1) if elapsed else None
        }
    return jsonify(tasks_summary)


@app.route("/api/models/download/stream/<task_id>", methods=["GET"])
def models_download_stream(task_id):
    """
    Stream export logs via SSE.
    """
    task = active_downloads.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
        
    def generate():
        idx = 0
        while True:
            # Yield any unsent logs
            while idx < len(task["logs"]):
                yield f"data: {json.dumps({'log': task['logs'][idx], 'status': task['status']})}\n\n"
                idx += 1
                
            if task["status"] in ["completed", "failed"]:
                # Send final status and end SSE
                yield f"data: [DONE]\n\n"
                break
                
            time.sleep(0.5)
            
    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.route("/api/models/download/cancel/<task_id>", methods=["POST"])
def models_download_cancel(task_id):
    """
    Cancel a running export job.
    """
    task = active_downloads.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
        
    if task["status"] != "running":
        return jsonify({"error": "Task is not running"}), 400
        
    process = task.get("process")
    if process:
        try:
            process.terminate()
            task["status"] = "failed"
            task["error"] = "Cancelled by user"
            task["logs"].append("\n[TERMINATED] Export task cancelled by user.")
            return jsonify({"success": True, "message": "Export task cancelled successfully."})
        except Exception as e:
            return jsonify({"success": False, "message": f"Failed to cancel task: {str(e)}"}), 500
            
    return jsonify({"success": False, "message": "No active process found for task."})


# ── Mousepad Notes Endpoints ─────────────────────────────────────────────────

NOTES_CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "notes_config.json"))

def get_notes_dir():
    """
    Get current active notes directory from config file or default path.
    """
    default_dir = os.path.abspath(os.path.join(os.getcwd(), "notes"))
    if os.path.exists(NOTES_CONFIG_PATH):
        try:
            with open(NOTES_CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
                custom_path = config.get("notes_dir")
                if custom_path:
                    # Expand user path (e.g. ~)
                    expanded_path = os.path.abspath(os.path.expanduser(custom_path))
                    os.makedirs(expanded_path, exist_ok=True)
                    return expanded_path
        except Exception:
            pass
    os.makedirs(default_dir, exist_ok=True)
    return default_dir

def set_notes_dir(new_path):
    """
    Save custom notes directory to config file.
    """
    expanded_path = os.path.abspath(os.path.expanduser(new_path))
    os.makedirs(expanded_path, exist_ok=True)
    with open(NOTES_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({"notes_dir": expanded_path}, f, indent=2)
    return expanded_path

def safe_notes_path(filename):
    """
    Sanitize filename and ensure it resolves inside active notes directory.
    """
    if not filename:
        raise ValueError("Filename is empty")
    if "/" in filename or "\\" in filename:
        raise ValueError("Directory traversal/subdirectories not allowed")
    # Strip any directory separators to prevent directory traversal
    filename = os.path.basename(filename)
    if filename in (".", "..") or not filename:
        raise ValueError("Invalid filename")
    
    notes_dir = get_notes_dir()
    # Secure double check
    full_path = os.path.abspath(os.path.join(notes_dir, filename))
    if not full_path.startswith(notes_dir):
        raise ValueError("Directory traversal attempt detected")
    return full_path

@app.route("/api/notes/list", methods=["GET"])
def notes_list():
    """
    List all text/markdown notes in the active notes directory.
    """
    try:
        notes_dir = get_notes_dir()
        notes = []
        for filename in os.listdir(notes_dir):
            file_path = os.path.join(notes_dir, filename)
            # Only list files (not folders) and only txt/md files
            if os.path.isfile(file_path) and (filename.endswith(".txt") or filename.endswith(".md")):
                stat = os.stat(file_path)
                # Load a preview of the note content (first 100 characters)
                preview = ""
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        preview = f.read(100)
                except Exception:
                    pass
                notes.append({
                    "name": filename,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "preview": preview
                })
        
        # Sort notes by modification time, newest first
        notes.sort(key=lambda n: n["mtime"], reverse=True)
        return jsonify(notes)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/notes/get", methods=["GET"])
def notes_get():
    """
    Retrieve the content of a specific note.
    """
    filename = request.args.get("filename", "").strip()
    try:
        file_path = safe_notes_path(filename)
        if not os.path.exists(file_path):
            return jsonify({"error": f"Note '{filename}' does not exist"}), 404
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        return jsonify({
            "filename": filename,
            "content": content
        })
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/notes/save", methods=["POST"])
def notes_save():
    """
    Create or update a note in the filesystem.
    """
    data = request.get_json() or {}
    filename = data.get("filename", "").strip()
    content = data.get("content", "")
    
    if not filename:
        return jsonify({"error": "Missing filename"}), 400
        
    # Append default extension if missing
    if not (filename.endswith(".txt") or filename.endswith(".md")):
        filename += ".txt"
        
    try:
        file_path = safe_notes_path(filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return jsonify({
            "success": True,
            "filename": filename,
            "message": "Note saved successfully"
        })
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/notes/delete", methods=["POST"])
def notes_delete():
    """
    Delete a note from the filesystem.
    """
    data = request.get_json() or {}
    filename = data.get("filename", "").strip()
    
    if not filename:
        return jsonify({"error": "Missing filename"}), 400
        
    try:
        file_path = safe_notes_path(filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({
                "success": True,
                "message": f"Note '{filename}' deleted successfully"
            })
        else:
            return jsonify({"error": f"Note '{filename}' does not exist"}), 404
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/notes/rename", methods=["POST"])
def notes_rename():
    """
    Rename a note in the filesystem.
    """
    data = request.get_json() or {}
    old_filename = data.get("old_filename", "").strip()
    new_filename = data.get("new_filename", "").strip()
    
    if not old_filename or not new_filename:
        return jsonify({"error": "Missing old_filename or new_filename"}), 400
        
    # Append default extension if missing
    if not (new_filename.endswith(".txt") or new_filename.endswith(".md")):
        new_filename += ".txt"
        
    try:
        old_path = safe_notes_path(old_filename)
        new_path = safe_notes_path(new_filename)
        
        if not os.path.exists(old_path):
            return jsonify({"error": f"Note '{old_filename}' does not exist"}), 404
            
        if os.path.exists(new_path) and old_path != new_path:
            return jsonify({"error": f"Note '{new_filename}' already exists"}), 400
            
        os.rename(old_path, new_path)
        return jsonify({
            "success": True,
            "filename": new_filename,
            "message": "Note renamed successfully"
        })
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/notes/get_directory", methods=["GET"])
def notes_get_directory():
    """
    Get current active notes directory path.
    """
    try:
        return jsonify({
            "path": get_notes_dir()
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/notes/set_directory", methods=["POST"])
def notes_set_directory():
    """
    Set a custom notes directory path.
    """
    data = request.get_json() or {}
    new_path = data.get("path", "").strip()
    
    if not new_path:
        return jsonify({"error": "Path is empty"}), 400
        
    try:
        updated_path = set_notes_dir(new_path)
        return jsonify({
            "success": True,
            "path": updated_path,
            "message": f"Notes directory changed to {updated_path}"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Failed to set directory: {str(e)}"}), 500


# ── Startup ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load initial model at startup
    model_path = os.getenv("MODEL_PATH", "./qwen-0.5b-ov")
    model_path = os.path.expanduser(model_path.strip()) if model_path else "./qwen-0.5b-ov"
    try:
        result = engine.load_model(model_path)
        if not result["success"]:
            print(f"[WARNING] Failed to load model: {result['message']}")
            print("[WARNING] The server will start, but /api/chat will return 503.")
            print("[WARNING] Fix the model path/device in .env and restart.")
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
