# 🧠 AI Chat Interface

A premium web-based chat interface for local AI models powered by [Intel OpenVINO](https://docs.openvino.ai/). Features streaming responses, conversation history, and a sleek dark theme.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![OpenVINO](https://img.shields.io/badge/OpenVINO-Optimized-purple)

---

## ✨ Features

- **🚀 Streaming Responses** — Token-by-token output via Server-Sent Events (SSE)
- **💬 Conversation History** — Multiple conversations stored in browser localStorage
- **🎨 Premium Dark UI** — Glassmorphism design with smooth animations
- **⚡ OpenVINO Accelerated** — Runs on Intel GPU or CPU
- **📋 Code Blocks** — Syntax display with one-click copy
- **📱 Responsive** — Works on desktop and mobile
- **⚙️ Configurable** — Customize model, device, and token limits via `.env`

---

## 📋 Prerequisites

- **Python 3.10+**
- **Intel GPU drivers** (if using GPU inference)
- **OpenVINO Runtime** (installed via `optimum[openvino]`)
- **A compatible model** in OpenVINO format (e.g., `qwen-0.5b-ov`)

---

## 🚀 Quick Start

### 1. Clone / Navigate to the project

```bash
cd ai-chat-interface
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your model

Edit the `.env` file:

```env
MODEL_PATH=./qwen-0.5b-ov    # Path to your OpenVINO model
DEVICE=GPU                     # GPU, CPU, or AUTO
MAX_NEW_TOKENS=512             # Max tokens per response
MAX_HISTORY=20                 # Max messages to keep in context
HOST=0.0.0.0                  # Server host
PORT=5000                     # Server port
```

### 5. Run the server

```bash
python app.py
```

### 6. Open the chat

Navigate to **http://localhost:5000** in your browser.

---

## 🔌 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the chat UI |
| `/api/health` | GET | Health check — `{"status": "ok", "model_loaded": true}` |
| `/api/config` | GET | Returns model configuration |
| `/api/chat` | POST | Streaming chat via SSE |
| `/api/chat/sync` | POST | Non-streaming chat (fallback) |

### POST `/api/chat`

**Request:**
```json
{
    "messages": [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
}
```

**Response:** Server-Sent Events stream
```
data: {"chunk": "I'm"}
data: {"chunk": " doing"}
data: {"chunk": " great!"}
data: [DONE]
```

---

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Enter` | Send message |
| `Shift + Enter` | New line |
| `Ctrl + N` | New conversation |

---

## 🗂️ Project Structure

```
ai-chat-interface/
├── .env                  # Your configuration
├── .env.example          # Configuration template
├── app.py                # Flask server & API endpoints
├── model_engine.py       # OpenVINO model wrapper
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── static/
    ├── index.html        # Chat UI (single page)
    ├── css/
    │   └── style.css     # Dark theme styles
    └── js/
        └── app.js        # Frontend application logic
```

---

## 🛠️ Troubleshooting

### Model fails to load
- Verify `MODEL_PATH` in `.env` points to a valid OpenVINO model directory
- Ensure the model directory contains `openvino_model.xml` and `openvino_model.bin`
- Try setting `DEVICE=CPU` if GPU drivers aren't configured

### Server starts but shows "Loading model..."
- Large models take time to compile for GPU. Wait 30–60 seconds.
- Check the terminal for loading progress.

### "Disconnected" status in the UI
- Ensure the Flask server is running
- Check that you're connecting to the correct host/port

---

## 📄 License

MIT — Use freely for personal and commercial projects.
