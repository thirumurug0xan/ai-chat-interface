/**
 * app.js — AI Chat Interface Frontend Application
 *
 * Handles chat messaging, SSE streaming, conversation management,
 * markdown rendering, and local storage persistence.
 */

// ── State ────────────────────────────────────────────────────────────────────
const state = {
    conversations: [],          // Array of { id, title, messages: [{role, content, timestamp}] }
    activeConversationId: null,
    isGenerating: false,
    abortController: null,
    modelConfig: null,
    maxTokens: 512,
};

// ── DOM References ───────────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dom = {
    chatArea: $("#chat-area"),
    chatContainer: $("#chat-container"),
    welcomeScreen: $("#welcome-screen"),
    chatInput: $("#chat-input"),
    btnSend: $("#btn-send"),
    btnNewChat: $("#btn-new-chat"),
    btnToggleSidebar: $("#btn-toggle-sidebar"),
    btnScrollBottom: $("#btn-scroll-bottom"),
    sidebar: $("#sidebar"),
    sidebarOverlay: $("#sidebar-overlay"),
    conversationList: $("#conversation-list"),
    emptyConversations: $("#empty-conversations"),
    headerTitle: $("#header-title"),
    statusDot: $("#status-dot"),
    statusText: $("#status-text"),
    modelName: $("#model-name"),
    modelDevice: $("#model-device"),
    deviceBadge: $("#device-badge"),
    modelStatus: $("#model-status"),
    maxTokensSlider: $("#max-tokens-slider"),
    maxTokensDisplay: $("#max-tokens-display"),
    charCount: $("#char-count"),
    toastContainer: $("#toast-container"),
    // Memory monitor
    memoryMonitor: $("#memory-monitor"),
    memoryRefresh: $("#memory-refresh"),
    gpuMemorySection: $("#gpu-memory-section"),
    gpuMemoryLabel: $("#gpu-memory-label"),
    gpuMemoryValues: $("#gpu-memory-values"),
    gpuMemoryBar: $("#gpu-memory-bar"),
    gpuMemoryUsed: $("#gpu-memory-used"),
    gpuMemoryFree: $("#gpu-memory-free"),
    gpuMemoryNote: $("#gpu-memory-note"),
    ramMemoryValues: $("#ram-memory-values"),
    ramMemoryBar: $("#ram-memory-bar"),
    ramMemoryUsed: $("#ram-memory-used"),
    ramMemoryFree: $("#ram-memory-free"),
};

// ── Initialization ───────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    loadState();
    initEventListeners();
    fetchConfig();
    renderConversationList();
    renderActiveConversation();
    initMemoryMonitor();
});

// ── Event Listeners ──────────────────────────────────────────────────────────
function initEventListeners() {
    // Send message
    dom.btnSend.addEventListener("click", handleSend);

    dom.chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    // Auto-resize textarea
    dom.chatInput.addEventListener("input", () => {
        autoResizeTextarea();
        dom.charCount.textContent = dom.chatInput.value.length;
    });

    // New chat
    dom.btnNewChat.addEventListener("click", createNewConversation);

    // Toggle sidebar
    dom.btnToggleSidebar.addEventListener("click", toggleSidebar);
    dom.sidebarOverlay.addEventListener("click", () => {
        dom.sidebar.classList.add("collapsed");
        dom.sidebarOverlay.classList.remove("visible");
    });

    // Scroll-to-bottom
    dom.btnScrollBottom.addEventListener("click", scrollToBottom);
    dom.chatArea.addEventListener("scroll", () => {
        const { scrollTop, scrollHeight, clientHeight } = dom.chatArea;
        const isNearBottom = scrollHeight - scrollTop - clientHeight < 120;
        dom.btnScrollBottom.classList.toggle("visible", !isNearBottom);
    });

    // Suggestion cards
    $$(".suggestion-card").forEach((card) => {
        card.addEventListener("click", () => {
            const prompt = card.dataset.prompt;
            if (prompt) {
                dom.chatInput.value = prompt;
                autoResizeTextarea();
                handleSend();
            }
        });
    });

    // Max tokens slider
    dom.maxTokensSlider.addEventListener("input", (e) => {
        state.maxTokens = parseInt(e.target.value);
        dom.maxTokensDisplay.textContent = state.maxTokens;
        saveState();
    });

    // Keyboard shortcuts
    document.addEventListener("keydown", (e) => {
        // Ctrl+N — new chat
        if ((e.ctrlKey || e.metaKey) && e.key === "n") {
            e.preventDefault();
            createNewConversation();
        }
    });
}

// ── API ──────────────────────────────────────────────────────────────────────
async function fetchConfig() {
    try {
        const res = await fetch("/api/config");
        const data = await res.json();
        state.modelConfig = data;

        dom.modelName.textContent = data.model_name || "Unknown";
        dom.deviceBadge.textContent = data.device || "—";
        dom.deviceBadge.className = `device-badge ${(data.device || "").toLowerCase()}`;

        if (data.loaded) {
            setStatus("ready", "Ready");
            dom.modelStatus.textContent = "Loaded";
        } else {
            setStatus("loading", "Loading model...");
            dom.modelStatus.textContent = "Loading...";
            // Poll until loaded
            pollModelStatus();
        }

        // Sync max tokens from server config
        if (data.max_new_tokens) {
            state.maxTokens = data.max_new_tokens;
            dom.maxTokensSlider.value = data.max_new_tokens;
            dom.maxTokensDisplay.textContent = data.max_new_tokens;
        }
    } catch (err) {
        setStatus("error", "Disconnected");
        dom.modelStatus.textContent = "Error";
        console.error("Failed to fetch config:", err);
    }
}

async function pollModelStatus() {
    const interval = setInterval(async () => {
        try {
            const res = await fetch("/api/health");
            const data = await res.json();
            if (data.model_loaded) {
                clearInterval(interval);
                setStatus("ready", "Ready");
                dom.modelStatus.textContent = "Loaded";
            }
        } catch {
            // Keep polling
        }
    }, 3000);
}

function setStatus(status, text) {
    dom.statusDot.className = `status-dot ${status === "loading" ? "loading" : status === "error" ? "error" : ""}`;
    dom.statusText.textContent = text;
}

// ── Chat Handling ────────────────────────────────────────────────────────────
async function handleSend() {
    const text = dom.chatInput.value.trim();
    if (!text || state.isGenerating) return;

    // Ensure active conversation
    if (!state.activeConversationId) {
        createNewConversation(false);
    }

    const conv = getActiveConversation();
    if (!conv) return;

    // Add user message
    const userMsg = { role: "user", content: text, timestamp: Date.now() };
    conv.messages.push(userMsg);

    // Auto-set conversation title from first message
    if (conv.messages.length === 1) {
        conv.title = text.slice(0, 50) + (text.length > 50 ? "..." : "");
        renderConversationList();
    }

    // Clear input
    dom.chatInput.value = "";
    dom.charCount.textContent = "0";
    autoResizeTextarea();

    // Render
    hideWelcome();
    appendMessageToDOM(userMsg);
    scrollToBottom();
    saveState();

    // Start generating
    await generateResponse(conv);
}

async function generateResponse(conv) {
    state.isGenerating = true;
    state.abortController = new AbortController();
    dom.btnSend.classList.add("generating");
    dom.btnSend.innerHTML = "■";
    dom.btnSend.title = "Stop generating";

    // Prepare messages for API BEFORE adding assistant placeholder
    // Only include messages that have actual content
    const apiMessages = conv.messages
        .filter(m => m.content && m.content.trim())
        .map(m => ({ role: m.role, content: m.content }));

    // Create assistant message placeholder
    const assistantMsg = { role: "assistant", content: "", timestamp: Date.now() };
    conv.messages.push(assistantMsg);

    const msgEl = appendMessageToDOM(assistantMsg);
    const contentEl = msgEl.querySelector(".message-content");

    // Show typing indicator
    contentEl.innerHTML = `<div class="typing-indicator"><span></span><span></span><span></span></div>`;
    scrollToBottom();

    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ messages: apiMessages }),
            signal: state.abortController.signal,
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.error || `HTTP ${res.status}`);
        }

        // Read SSE stream
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let fullResponse = "";
        let firstChunk = true;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Process complete SSE messages
            const lines = buffer.split("\n");
            buffer = lines.pop(); // Keep incomplete line in buffer

            for (const line of lines) {
                if (!line.startsWith("data: ")) continue;
                const data = line.slice(6).trim();

                if (data === "[DONE]") continue;

                try {
                    const parsed = JSON.parse(data);

                    if (parsed.error) {
                        throw new Error(parsed.error);
                    }

                    if (parsed.chunk) {
                        if (firstChunk) {
                            contentEl.innerHTML = "";
                            firstChunk = false;
                        }
                        fullResponse += parsed.chunk;
                        contentEl.innerHTML = renderMarkdown(fullResponse);
                        scrollToBottom();
                    }
                } catch (parseErr) {
                    if (parseErr.message && !parseErr.message.includes("JSON")) {
                        throw parseErr;
                    }
                }
            }
        }

        // Update message content
        assistantMsg.content = fullResponse;

        // Handle empty response
        if (!fullResponse || !fullResponse.trim()) {
            assistantMsg.content = "(No response generated — the model returned empty output.)";
            contentEl.innerHTML = `<p style="color: #a0a0a0; font-style: italic;">No response generated. Try rephrasing your message.</p>`;
        } else {
            // Final render with proper markdown
            contentEl.innerHTML = renderMarkdown(fullResponse);
            attachCodeCopyButtons(contentEl);
        }

    } catch (err) {
        // Handle user-initiated abort gracefully
        if (err.name === "AbortError") {
            assistantMsg.content = fullResponse || "(Generation stopped)";
            contentEl.innerHTML = fullResponse
                ? renderMarkdown(fullResponse) + `<p style="color: #a0a0a0; font-style: italic;">⏹ Generation stopped.</p>`
                : `<p style="color: #a0a0a0; font-style: italic;">⏹ Generation stopped.</p>`;
        } else {
            console.error("Generation error:", err);
            assistantMsg.content = `⚠️ Error: ${err.message}`;
            contentEl.innerHTML = `<p style="color: #ff6b6b;">⚠️ ${escapeHtml(err.message)}</p>`;
            showToast(err.message, "error");
        }
    } finally {
        state.isGenerating = false;
        state.abortController = null;
        dom.btnSend.classList.remove("generating");
        dom.btnSend.innerHTML = "➤";
        dom.btnSend.title = "Send message (Enter)";
        saveState();
        scrollToBottom();
    }
}

// ── Stop generating (wired to the send button during generation) ─────────
dom.btnSend.addEventListener("click", () => {
    if (state.isGenerating && state.abortController) {
        state.abortController.abort();
    }
});

// ── Conversation Management ──────────────────────────────────────────────────
function createNewConversation(switchTo = true) {
    const conv = {
        id: generateId(),
        title: "New Chat",
        messages: [],
        createdAt: Date.now(),
    };
    state.conversations.unshift(conv);
    state.activeConversationId = conv.id;
    saveState();
    renderConversationList();

    if (switchTo) {
        renderActiveConversation();
        dom.chatInput.focus();
    }
}

function switchConversation(id) {
    state.activeConversationId = id;
    saveState();
    renderConversationList();
    renderActiveConversation();

    // Close sidebar on mobile
    if (window.innerWidth <= 768) {
        dom.sidebar.classList.add("collapsed");
        dom.sidebarOverlay.classList.remove("visible");
    }
}

function deleteConversation(id, e) {
    e.stopPropagation();
    state.conversations = state.conversations.filter((c) => c.id !== id);

    if (state.activeConversationId === id) {
        state.activeConversationId = state.conversations[0]?.id || null;
    }

    saveState();
    renderConversationList();
    renderActiveConversation();
}

function getActiveConversation() {
    return state.conversations.find((c) => c.id === state.activeConversationId);
}

// ── Rendering ────────────────────────────────────────────────────────────────
function renderConversationList() {
    const list = dom.conversationList;
    list.innerHTML = "";

    if (state.conversations.length === 0) {
        dom.emptyConversations.style.display = "block";
        return;
    }

    dom.emptyConversations.style.display = "none";

    state.conversations.forEach((conv) => {
        const li = document.createElement("li");
        li.className = `conversation-item ${conv.id === state.activeConversationId ? "active" : ""}`;
        li.innerHTML = `
            <span class="conv-icon">💬</span>
            <span class="conv-title">${escapeHtml(conv.title)}</span>
            <button class="conv-delete" title="Delete conversation">✕</button>
        `;
        li.addEventListener("click", () => switchConversation(conv.id));
        li.querySelector(".conv-delete").addEventListener("click", (e) => deleteConversation(conv.id, e));
        list.appendChild(li);
    });
}

function renderActiveConversation() {
    const conv = getActiveConversation();
    const container = dom.chatContainer;

    // Clear all messages but keep welcome screen
    container.querySelectorAll(".message").forEach((el) => el.remove());

    if (!conv || conv.messages.length === 0) {
        showWelcome();
        dom.headerTitle.textContent = "New Chat";
        return;
    }

    hideWelcome();
    dom.headerTitle.textContent = conv.title;

    conv.messages.forEach((msg) => {
        appendMessageToDOM(msg, false);
    });

    // Attach copy buttons to all code blocks
    container.querySelectorAll(".message.assistant .message-content").forEach(attachCodeCopyButtons);

    scrollToBottom(false);
}

function appendMessageToDOM(msg, animate = true) {
    hideWelcome();

    const div = document.createElement("div");
    div.className = `message ${msg.role}`;
    if (!animate) div.style.animation = "none";

    const time = new Date(msg.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const avatarEmoji = msg.role === "user" ? "👤" : "🧠";
    const senderName = msg.role === "user" ? "You" : (state.modelConfig?.model_name || "Assistant");

    div.innerHTML = `
        <div class="message-header">
            <div class="message-avatar">${avatarEmoji}</div>
            <span class="message-sender">${escapeHtml(senderName)}</span>
            <span class="message-time">${time}</span>
        </div>
        <div class="message-content">${msg.content ? renderMarkdown(msg.content) : ""}</div>
        <div class="message-actions">
            <button class="btn-message-action btn-copy-message" title="Copy message">📋 Copy</button>
        </div>
    `;

    // Copy button
    div.querySelector(".btn-copy-message")?.addEventListener("click", () => {
        navigator.clipboard.writeText(msg.content).then(() => {
            showToast("Message copied!", "success");
        });
    });

    dom.chatContainer.appendChild(div);
    return div;
}

function showWelcome() {
    if (dom.welcomeScreen) dom.welcomeScreen.style.display = "flex";
}

function hideWelcome() {
    if (dom.welcomeScreen) dom.welcomeScreen.style.display = "none";
}

// ── Markdown Rendering ───────────────────────────────────────────────────────
function renderMarkdown(text) {
    if (!text) return "";

    let html = escapeHtml(text);

    // Code blocks: ```lang\n...\n```
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
        const langLabel = lang || "code";
        return `<div class="code-block-wrapper">
            <div class="code-block-header">
                <span class="code-lang">${escapeHtml(langLabel)}</span>
                <button class="btn-copy-code" data-code="${encodeURIComponent(code.trim())}">📋 Copy</button>
            </div>
            <pre><code>${code.trim()}</code></pre>
        </div>`;
    });

    // Inline code: `code`
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Bold: **text**
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

    // Italic: *text*
    html = html.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, "<em>$1</em>");

    // Unordered lists: - item or * item
    html = html.replace(/^(?:[-*])\s+(.+)$/gm, "<li>$1</li>");
    html = html.replace(/(<li>.*<\/li>\n?)+/g, "<ul>$&</ul>");

    // Ordered lists: 1. item
    html = html.replace(/^\d+\.\s+(.+)$/gm, "<li>$1</li>");

    // Headers: ### text
    html = html.replace(/^### (.+)$/gm, "<h4>$1</h4>");
    html = html.replace(/^## (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^# (.+)$/gm, "<h2>$1</h2>");

    // Links: [text](url)
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

    // Paragraphs: double newline
    html = html.replace(/\n\n/g, "</p><p>");

    // Single newlines (outside of code blocks)
    html = html.replace(/\n/g, "<br>");

    // Wrap in paragraph
    html = `<p>${html}</p>`;

    // Clean up empty paragraphs
    html = html.replace(/<p><\/p>/g, "");

    return html;
}

function attachCodeCopyButtons(container) {
    container.querySelectorAll(".btn-copy-code").forEach((btn) => {
        btn.addEventListener("click", () => {
            const code = decodeURIComponent(btn.dataset.code);
            navigator.clipboard.writeText(code).then(() => {
                btn.textContent = "✓ Copied";
                btn.classList.add("copied");
                setTimeout(() => {
                    btn.textContent = "📋 Copy";
                    btn.classList.remove("copied");
                }, 2000);
            });
        });
    });
}

// ── UI Helpers ───────────────────────────────────────────────────────────────
function autoResizeTextarea() {
    const el = dom.chatInput;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
}

function scrollToBottom(smooth = true) {
    requestAnimationFrame(() => {
        dom.chatArea.scrollTo({
            top: dom.chatArea.scrollHeight,
            behavior: smooth ? "smooth" : "instant",
        });
    });
}

function toggleSidebar() {
    const isCollapsed = dom.sidebar.classList.toggle("collapsed");
    dom.sidebarOverlay.classList.toggle("visible", !isCollapsed);
}

function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    dom.toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateX(40px)";
        toast.style.transition = "all 0.3s ease";
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ── Persistence ──────────────────────────────────────────────────────────────
function saveState() {
    try {
        const data = {
            conversations: state.conversations,
            activeConversationId: state.activeConversationId,
            maxTokens: state.maxTokens,
        };
        localStorage.setItem("ai-chat-state", JSON.stringify(data));
    } catch (err) {
        console.warn("Failed to save state:", err);
    }
}

function loadState() {
    try {
        const saved = localStorage.getItem("ai-chat-state");
        if (saved) {
            const data = JSON.parse(saved);
            state.conversations = data.conversations || [];
            state.activeConversationId = data.activeConversationId || null;
            state.maxTokens = data.maxTokens || 512;
            dom.maxTokensSlider.value = state.maxTokens;
            dom.maxTokensDisplay.textContent = state.maxTokens;
        }
    } catch (err) {
        console.warn("Failed to load state:", err);
    }
}

// ── Utilities ────────────────────────────────────────────────────────────────
function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function escapeHtml(text) {
    const el = document.createElement("div");
    el.textContent = text;
    return el.innerHTML;
}

// ── Memory Monitor ───────────────────────────────────────────────────────────
let memoryPollInterval = null;

function initMemoryMonitor() {
    // Fetch initial stats
    fetchSystemStats();

    // Poll every 5 seconds
    memoryPollInterval = setInterval(fetchSystemStats, 5000);

    // Manual refresh button
    if (dom.memoryRefresh) {
        dom.memoryRefresh.addEventListener("click", () => {
            dom.memoryRefresh.classList.add("spinning");
            fetchSystemStats();
            setTimeout(() => dom.memoryRefresh.classList.remove("spinning"), 600);
        });
    }
}

async function fetchSystemStats() {
    try {
        const res = await fetch("/api/system/stats");
        if (!res.ok) return;

        const stats = await res.json();
        updateMemoryUI(stats);
    } catch (err) {
        // Silently fail — don't spam errors for monitoring
        console.debug("Memory stats fetch failed:", err);
    }
}

function updateMemoryUI(stats) {
    // ── Update RAM section ──
    if (stats.ram) {
        const ram = stats.ram;
        const ramPercent = ram.percent || 0;

        dom.ramMemoryValues.textContent = `${ramPercent}%`;
        dom.ramMemoryBar.style.width = `${ramPercent}%`;
        dom.ramMemoryUsed.textContent = `Used: ${ram.used_display}`;
        dom.ramMemoryFree.textContent = `Free: ${ram.free_display}`;

        // Color coding based on usage
        dom.ramMemoryBar.classList.remove("warning", "critical");
        if (ramPercent >= 90) {
            dom.ramMemoryBar.classList.add("critical");
        } else if (ramPercent >= 75) {
            dom.ramMemoryBar.classList.add("warning");
        }
    }

    // ── Update GPU section ──
    if (stats.gpu) {
        const gpu = stats.gpu;
        dom.gpuMemorySection.style.display = "block";

        // Set GPU name label
        dom.gpuMemoryLabel.textContent = gpu.name || "GPU VRAM";

        if (gpu.percent !== null && gpu.percent !== undefined) {
            // We have actual usage data
            const gpuPercent = gpu.percent;

            dom.gpuMemoryValues.textContent = `${gpuPercent}%`;
            dom.gpuMemoryBar.style.width = `${gpuPercent}%`;
            dom.gpuMemoryUsed.textContent = `Used: ${gpu.used_display}`;
            dom.gpuMemoryFree.textContent = `Free: ${gpu.free_display}`;

            // Color coding
            dom.gpuMemoryBar.classList.remove("warning", "critical");
            if (gpuPercent >= 90) {
                dom.gpuMemoryBar.classList.add("critical");
            } else if (gpuPercent >= 75) {
                dom.gpuMemoryBar.classList.add("warning");
            }
        } else {
            // Shared memory — can't measure exact GPU usage
            dom.gpuMemoryValues.textContent = gpu.total_display || "—";
            dom.gpuMemoryBar.style.width = "0%";
            dom.gpuMemoryUsed.textContent = `Total: ${gpu.total_display}`;
            dom.gpuMemoryFree.textContent = gpu.max_alloc_display
                ? `Max alloc: ${gpu.max_alloc_display}`
                : "";
        }

        // Show note if available
        if (gpu.note) {
            dom.gpuMemoryNote.textContent = gpu.note;
            dom.gpuMemoryNote.style.display = "block";
        } else {
            dom.gpuMemoryNote.style.display = "none";
        }
    } else {
        // No GPU detected — hide GPU section
        dom.gpuMemorySection.style.display = "none";
    }
}
