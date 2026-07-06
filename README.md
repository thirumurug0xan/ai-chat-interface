# 🧠 local-ai-workstation (AI Chat Interface for Intel OpenVINO)

A premium, lightweight, local AI workstation and chat interface designed specifically for Intel hardware. Powered by [Intel OpenVINO](https://docs.openvino.ai/) and Hugging Face's `optimum-intel`, this repository provides a complete web workspace to download, export, dynamically configure, and chat with local LLMs, while managing server notes and monitoring hardware resources.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![OpenVINO](https://img.shields.io/badge/OpenVINO-Optimized-purple)
![Intel GPU](https://img.shields.io/badge/Intel%20GPU-Accelerated-blue)

---

## ✨ Key Pillars

### 1. 💬 Optimized Chat Interface
- **Server-Sent Events (SSE)**: Real-time, token-by-token streaming responses.
- **Hardware Acceleration**: Built-in support for Intel GPU, CPU, and XPU inference, with automatic CPU fallback.
- **Context Management**: Auto-trims conversational history to respect input context limitations and prevent out-of-memory (OOM) failures.

### 2. 📁 Filesystem Explorer & Dynamic GUI Loader
- **Local Browser**: Browse the server's directory tree directly from the web GUI to locate model folders.
- **On-the-Fly Configuration**: Select a folder and configure performance parameters *without* editing `.env` files:
  - **Performance Hint**: Choose between `LATENCY` (default), `THROUGHPUT`, or `CUMULATIVE_THROUGHPUT`.
  - **Compilation Caching**: Enable compiled graph storage (`ov_cache`) to speed up subsequent load requests.
  - **Custom Model file**: Choose which `.xml` file to compile from the selected directory (useful for loaded subgraphs or multiple quantized layouts).
  - **Parameters**: Suppress regex warnings, toggle KV-cache, and toggle remote code execution on startup.

### 3. 📥 Background Model Downloader & Exporter
- **Hugging Face Search**: Query the Hugging Face Hub directly inside the browser.
- **Dynamic Conversion**: Run `optimum-cli` conversion pipelines in a background thread to export models to OpenVINO IR format (FP16, INT8, INT4 quantization).
- **Log Streaming**: Monitor model download and compilation logs live via SSE stream in the UI.

### 4. 📝 Mousepad (Workspace Notes)
- **Document Workspace**: Access, edit, rename, and organize Markdown (`.md`) or text (`.txt`) notes on the server alongside your chat window.
- **Directory Selection**: Securely map custom server note paths from the UI explorer.

### 5. 📊 Live System Resource Monitor
- **GPU Stats**: Real-time monitoring of Intel GPU utilization, memory usage (VRAM), and active device status.
- **NVIDIA Fallback**: Supports NVIDIA GPU monitoring via `nvidia-smi` where available.
- **System Memory**: CPU cores, CPU utilization, and System RAM metrics.

---

## 📋 Prerequisites

- **Python 3.10+**
- **Intel GPU Drivers** (if targeting GPU acceleration; CPU is supported out of the box)
- **OpenVINO Runtime** (installed automatically via `optimum[openvino]`)

---

## 🚀 Quick Start

### 1. Clone & Navigate
```bash
git clone https://github.com/thirumurug0xan/ai-chat-interface.git
cd ai-chat-interface
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Template
Copy `.env.example` to `.env` and configure your default paths:
```env
MODEL_PATH=./qwen-0.5b-ov    # Path to default local model
DEVICE=AUTO                    # AUTO, GPU, CPU, or XPU
USE_CACHE=True                # Enable KV caching
OV_CACHE_DIR=./ov_cache       # Compile cache folder
PORT=5000
```

### 5. Start Workspace
```bash
python app.py
```
Open **http://localhost:5000** in your browser.

---

## 🔌 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the main workstation UI |
| `/api/config` | GET | Returns active model configuration |
| `/api/chat` | POST | SSE chat inference stream |
| `/api/chat2` | GET/POST | Simple curl-friendly endpoint (streaming/sync, optional RAG) |
| `/api/mcp/chat` | POST | OpenAI-compatible completions endpoint (for MCP) |
| `/v1/chat/completions` | POST | OpenAI-compatible completions endpoint (for VS Code) |
| `/api/fs/list` | POST | Directory tree filesystem lister |
| `/api/model/switch` | POST | Switches model path with custom overrides |
| `/api/models/download` | POST | Triggers HF optimum-cli export background task |
| `/api/models/download/status` | GET | Fetches all model export tasks |
| `/api/notes/list` | GET | Lists files inside the note directory |

---

## 🎛️ New Workspace Capabilities

### 1. 📚 Dual RAG Framework & Direct URL Web Scraper
* **Local Notes RAG**: Queries your active server notes directory dynamically using a native BM25 search algorithm.
* **Web RAG**: Performs duckduckgo search query keyword optimization (using the loaded model) to fetch real-time search context.
* **Direct URL Scraper**: If a URL is detected inside the user prompt, the system directly scrapes the webpage text content, bypassing searches.

### 2. 🔌 Simple Chat API `/api/chat2`
A terminal-friendly chat entry point designed for curl scripting:
* Supports both `GET` (query-string parameters) and `POST` (JSON/Form-urlencoded).
* Configurable settings via UI to toggle endpoint access and default RAG behavior.
```bash
curl -G "http://localhost:5000/api/chat2" --data-urlencode "quirie=Explain local RAG"
```

### 3. 🧠 Collapsible Model Thinking Process (Chain-of-Thought)
* Prompts models (like Phi-4 or Qwen-2.5) to output step-by-step reasoning steps inside `<think>...</think>` containers, parsed by the UI into clean collapsible accordions.
* **Thinking Mode Toggle**: Toggle this setting on/off in the Inference Settings panel. *Thinking mode is disabled by default to optimize local response speeds.*

### 4. 🔌 OpenAI & MCP Compatibility Layer (VS Code & IDE Extensions)
Exposes standard OpenAI-compatible endpoints to power external development extensions (like **Continue**, **Cline**, **Roo Code**) and custom Model Context Protocol (MCP) clients locally.
* **Compatibility URL**: `http://localhost:5000/v1`
* **Direct Integration**: Refer to the [mcp_and_extension_integration.md](file:///home/aitest/ai-chat-interface/docs/mcp_and_extension_integration.md) guide for full config templates for Claude Desktop, Continue, and Cline.

---

## 🗂️ Project Structure

```
local-ai-workstation/
├── .env                  # Configuration details
├── app.py                # API routing & background workers
├── model_engine.py       # OpenVINO graph loading & session control
├── system_stats.py       # Host resources telemetry
├── requirements.txt      # Python dependencies
└── static/
    ├── index.html        # Glassmorphic client app
    ├── css/style.css     # CSS variable tokens & UI themes
    └── js/app.js         # AJAX request logic & DOM state
```

---

## 📄 License

MIT — Use freely for personal and commercial projects.

