# Connecting VS Code Extensions & MCP Clients to Your Local Model

The AI Chat Interface exposes an OpenAI-compatible endpoint that allows you to use your locally loaded OpenVINO models inside popular IDE extensions and Model Context Protocol (MCP) clients.

---

## 1. Endpoints

* **Base URL**: `http://localhost:5000/v1`
* **Chat Completions**: `http://localhost:5000/v1/chat/completions` (OpenAI format)
* **Custom MCP Path**: `http://localhost:5000/api/mcp/chat`

---

## 2. VS Code Extensions

### Continue
[Continue](https://www.continue.dev/) is an open-source autopilot for VS Code and JetBrains.

To connect Continue to your local model, add the following to your `config.json` configuration file (usually located at `~/.continue/config.json`):

```json
{
  "models": [
    {
      "title": "Local OpenVINO Model",
      "provider": "openai",
      "model": "qwen2.5-7b",
      "apiBase": "http://localhost:5000/v1"
    }
  ]
}
```

---

### Cline / Roo Code
[Cline](https://github.com/cline/cline) (formerly Roo Cline/Roo Code) is an agentic coding assistant that can run commands, edit files, and use MCP servers.

To connect Cline to your local server:
1. Open Cline Settings in VS Code.
2. Select **OpenAI Compatible** as the API Provider.
3. Set the **Base URL** to `http://localhost:5000/v1`.
4. Enter any dummy value for the **API Key** (e.g. `local-key`).
5. Enter the model ID (e.g. `qwen2.5-7b-instruct`).
6. Set the model's **Context Window** limit (e.g. `4096` or `8192` matching your setting).

---

## 3. Claude Desktop App (MCP Integration)

The Claude Desktop app supports custom MCP servers. To configure a custom server that leverages your local OpenVINO model, modify the Claude Desktop configuration file:

* **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
* **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Add a custom MCP server definition that interfaces with your local API:

```json
{
  "mcpServers": {
    "local-ov-chat": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-everything"
      ],
      "env": {
        "OPENAI_API_KEY": "local-key",
        "OPENAI_BASE_URL": "http://localhost:5000/v1"
      }
    }
  }
}
```

*(Note: Replace `@modelcontextprotocol/server-everything` with the specific MCP server tool package you wish to run).*

---

## 4. Querying the API via Command Line

### A. Non-Streaming Request
```bash
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain KV Cache in one sentence."}
    ],
    "stream": false
  }'
```

### B. Streaming Request (Server-Sent Events)
```bash
curl -N -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Count from 1 to 5."}
    ],
    "stream": true
  }'
```
