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
    maxInputTokens: 1024,       // context window size from config
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
    ramMemoryLabel: $("#ram-memory-label"),
    ramMemoryValues: $("#ram-memory-values"),
    ramMemoryBar: $("#ram-memory-bar"),
    ramMemoryUsed: $("#ram-memory-used"),
    ramMemoryFree: $("#ram-memory-free"),
    ramMemoryNote: $("#ram-memory-note"),
    // Utilization graphs
    cpuGraphCanvas: $("#cpu-graph-canvas"),
    gpuGraphCanvas: $("#gpu-graph-canvas"),
    ramGraphCanvas: $("#ram-graph-canvas"),
    cpuUtilValue: $("#cpu-util-value"),
    gpuUtilValue: $("#gpu-util-value"),
    ramUtilValue: $("#ram-util-value"),
    gpuGraphSection: $("#gpu-graph-section"),
    // Context window
    contextWindow: $("#context-window"),
    contextBarFill: $("#context-bar-fill"),
    contextLabel: $("#context-label"),
    // Context popup
    contextPopup: $("#context-popup"),
    contextPopupClose: $("#context-popup-close"),
    gaugePercent: $("#gauge-percent"),
    gaugeFill: $("#gauge-fill"),
    ctxTokensUsed: $("#ctx-tokens-used"),
    ctxTokensMax: $("#ctx-tokens-max"),
    ctxTokensRemaining: $("#ctx-tokens-remaining"),
    ctxMessageCount: $("#ctx-message-count"),
    contextHealth: $("#context-health"),
    contextMsgBreakdown: $("#context-msg-breakdown"),
    contextBreakdownList: $("#context-breakdown-list"),
    // Help modal
    btnHelp: $("#btn-help"),
    helpModalOverlay: $("#help-modal-overlay"),
    helpModalClose: $("#help-modal-close"),
    // Device selector
    deviceSelector: $("#device-selector"),
    deviceSelectorBtn: $("#device-selector-btn"),
    deviceDropdown: $("#device-dropdown"),
    autoResolvedLabel: $("#auto-resolved-label"),
    deviceSwitchingOverlay: $("#device-switching-overlay"),
    deviceSwitchingText: $("#device-switching-text"),
};

// ── Initialization ───────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    loadState();
    initEventListeners();
    fetchConfig();
    renderConversationList();
    renderActiveConversation();
    initMemoryMonitor();
    initSVGGradient();
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

        // Esc — close popups and modals
        if (e.key === "Escape") {
            closeContextPopup();
            closeHelpModal();
            closeDeviceDropdown();
        }

        // ? — open help (only when not typing in input)
        if (e.key === "?" && document.activeElement !== dom.chatInput) {
            e.preventDefault();
            openHelpModal();
        }
    });

    // Context window click
    if (dom.contextWindow) {
        dom.contextWindow.addEventListener("click", (e) => {
            e.stopPropagation();
            toggleContextPopup();
        });
    }

    // Context popup close
    if (dom.contextPopupClose) {
        dom.contextPopupClose.addEventListener("click", closeContextPopup);
    }

    // Help button
    if (dom.btnHelp) {
        dom.btnHelp.addEventListener("click", openHelpModal);
    }

    // Help modal close
    if (dom.helpModalClose) {
        dom.helpModalClose.addEventListener("click", closeHelpModal);
    }

    if (dom.helpModalOverlay) {
        dom.helpModalOverlay.addEventListener("click", (e) => {
            if (e.target === dom.helpModalOverlay) closeHelpModal();
        });
    }

    // Close context popup on outside click
    document.addEventListener("click", (e) => {
        if (dom.contextPopup && dom.contextPopup.classList.contains("visible")) {
            if (!dom.contextPopup.contains(e.target) && !dom.contextWindow.contains(e.target)) {
                closeContextPopup();
            }
        }
        // Close device dropdown on outside click
        if (dom.deviceDropdown && dom.deviceDropdown.classList.contains("visible")) {
            if (!dom.deviceSelector.contains(e.target)) {
                closeDeviceDropdown();
            }
        }
    });

    // Device selector
    if (dom.deviceSelectorBtn) {
        dom.deviceSelectorBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            toggleDeviceDropdown();
        });
    }

    // Device option clicks
    $$(".device-option").forEach((opt) => {
        opt.addEventListener("click", (e) => {
            e.stopPropagation();
            const device = opt.dataset.device;
            if (device) {
                closeDeviceDropdown();
                switchDevice(device);
            }
        });
    });
}

// ── API ──────────────────────────────────────────────────────────────────────
async function fetchConfig() {
    try {
        const res = await fetch("/api/config");
        const data = await res.json();
        state.modelConfig = data;

        dom.modelName.textContent = data.model_name || "Unknown";

        // Update device badge with friendly name
        const friendlyDevice = data.device_friendly || data.device || "—";
        const requestedDevice = (data.requested_device || "").toUpperCase();
        updateDeviceBadge(friendlyDevice, requestedDevice);

        // Update device selector active state
        updateDeviceSelectorUI(requestedDevice, friendlyDevice);

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

        // Store context window size
        if (data.max_input_tokens) {
            state.maxInputTokens = data.max_input_tokens;
            updateContextWindowUI(0, data.max_input_tokens);
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

    let generationMeta = null;

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

                    // Capture generation metadata (tokens/sec etc.)
                    if (parsed.meta) {
                        generationMeta = parsed.meta;
                        continue;
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

        // Store meta on the message for later display
        if (generationMeta) {
            assistantMsg.meta = generationMeta;
        }

        // Handle empty response
        if (!fullResponse || !fullResponse.trim()) {
            assistantMsg.content = "(No response generated — the model returned empty output.)";
            contentEl.innerHTML = `<p style="color: #a0a0a0; font-style: italic;">No response generated. Try rephrasing your message.</p>`;
        } else {
            // Final render with proper markdown
            contentEl.innerHTML = renderMarkdown(fullResponse);
            attachCodeCopyButtons(contentEl);
        }

        // Render generation stats footer
        if (generationMeta) {
            renderMessageStats(msgEl, generationMeta);
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

        // Update context window after generation
        fetchContextUsage(conv);
    }
}

// ── Render generation stats (tok/s) below a message ──────────────────────
function renderMessageStats(msgEl, meta) {
    // Remove any existing stats
    const existing = msgEl.querySelector(".message-stats");
    if (existing) existing.remove();

    const statsDiv = document.createElement("div");
    statsDiv.className = "message-stats";

    const tps = meta.tokens_per_sec || 0;
    const tokens = meta.tokens || 0;
    const elapsed = meta.elapsed_sec || 0;

    statsDiv.innerHTML = `
        <span class="stats-item" title="Tokens per second">⚡ ${tps} tok/s</span>
        <span class="stats-separator">·</span>
        <span class="stats-item" title="Total tokens generated">${tokens} tokens</span>
        <span class="stats-separator">·</span>
        <span class="stats-item" title="Generation time">${elapsed}s</span>
    `;

    // Insert after message-content, before message-actions
    const actionsEl = msgEl.querySelector(".message-actions");
    if (actionsEl) {
        msgEl.insertBefore(statsDiv, actionsEl);
    } else {
        msgEl.appendChild(statsDiv);
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

    // Update context window for the loaded conversation
    fetchContextUsage(conv);
}

function appendMessageToDOM(msg, animate = true) {
    hideWelcome();

    const div = document.createElement("div");
    div.className = `message ${msg.role}`;
    if (!animate) div.style.animation = "none";

    const time = new Date(msg.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const avatarEmoji = msg.role === "user" ? "👤" : "🧠";
    const senderName = msg.role === "user" ? "You" : (state.modelConfig?.model_name || "Assistant");

    // Build action buttons — assistant messages get a retry button
    let actionsHtml = `<button class="btn-message-action btn-copy-message" title="Copy message">📋 Copy</button>`;
    if (msg.role === "assistant") {
        actionsHtml += `<button class="btn-message-action btn-retry" title="Regenerate this response">🔄 Retry</button>`;
    }

    div.innerHTML = `
        <div class="message-header">
            <div class="message-avatar">${avatarEmoji}</div>
            <span class="message-sender">${escapeHtml(senderName)}</span>
            <span class="message-time">${time}</span>
        </div>
        <div class="message-content">${msg.content ? renderMarkdown(msg.content) : ""}</div>
        <div class="message-actions">
            ${actionsHtml}
        </div>
    `;

    // Copy button
    div.querySelector(".btn-copy-message")?.addEventListener("click", () => {
        navigator.clipboard.writeText(msg.content).then(() => {
            showToast("Message copied!", "success");
        });
    });

    // Retry button (assistant messages only)
    const retryBtn = div.querySelector(".btn-retry");
    if (retryBtn) {
        retryBtn.addEventListener("click", () => handleRetry(msg, div));
    }

    // Render stored meta stats (for messages loaded from localStorage)
    if (msg.meta) {
        renderMessageStats(div, msg.meta);
    }

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

// Graph history buffers (60 points = 5 min at 5s intervals)
const GRAPH_MAX_POINTS = 60;
const cpuHistory = [];
const gpuHistory = [];
const ramHistory = [];

function pushGraphData(history, value) {
    history.push(value);
    if (history.length > GRAPH_MAX_POINTS) {
        history.shift();
    }
}

function drawGraph(canvas, data, lineColor, fillColor) {
    if (!canvas || data.length < 2) return;

    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    // Set canvas resolution for crisp rendering
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    // Clear
    ctx.clearRect(0, 0, w, h);

    // Draw subtle grid lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
    ctx.lineWidth = 1;
    for (let i = 1; i <= 3; i++) {
        const y = (h / 4) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }

    // Compute points
    const padding = 2;
    const graphW = w - padding * 2;
    const graphH = h - padding * 2;
    const step = graphW / (GRAPH_MAX_POINTS - 1);
    const offset = GRAPH_MAX_POINTS - data.length;

    const points = data.map((val, i) => ({
        x: padding + (offset + i) * step,
        y: padding + graphH - (val / 100) * graphH,
    }));

    // Draw filled area
    ctx.beginPath();
    ctx.moveTo(points[0].x, h);
    ctx.lineTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
        const prev = points[i - 1];
        const curr = points[i];
        const cpx = (prev.x + curr.x) / 2;
        ctx.bezierCurveTo(cpx, prev.y, cpx, curr.y, curr.x, curr.y);
    }
    ctx.lineTo(points[points.length - 1].x, h);
    ctx.closePath();

    const gradient = ctx.createLinearGradient(0, 0, 0, h);
    gradient.addColorStop(0, fillColor);
    gradient.addColorStop(1, "rgba(0, 0, 0, 0)");
    ctx.fillStyle = gradient;
    ctx.fill();

    // Draw line
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
        const prev = points[i - 1];
        const curr = points[i];
        const cpx = (prev.x + curr.x) / 2;
        ctx.bezierCurveTo(cpx, prev.y, cpx, curr.y, curr.x, curr.y);
    }
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Draw current value dot
    const last = points[points.length - 1];
    ctx.beginPath();
    ctx.arc(last.x, last.y, 3, 0, Math.PI * 2);
    ctx.fillStyle = lineColor;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(last.x, last.y, 5, 0, Math.PI * 2);
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.3;
    ctx.stroke();
    ctx.globalAlpha = 1;
}

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

        // WSL2-aware label
        if (ram.wsl2 && dom.ramMemoryLabel) {
            dom.ramMemoryLabel.textContent = ram.label || "WSL2 Memory";
        }

        dom.ramMemoryValues.textContent = `${ramPercent}%`;
        dom.ramMemoryBar.style.width = `${ramPercent}%`;
        dom.ramMemoryUsed.textContent = `Used: ${ram.used_display}`;
        dom.ramMemoryFree.textContent = `Free: ${ram.free_display}`;

        // WSL2 note
        if (ram.note && dom.ramMemoryNote) {
            dom.ramMemoryNote.textContent = ram.note;
            dom.ramMemoryNote.style.display = "block";
        } else if (dom.ramMemoryNote) {
            dom.ramMemoryNote.style.display = "none";
        }

        // Color coding based on usage
        dom.ramMemoryBar.classList.remove("warning", "critical");
        if (ramPercent >= 90) {
            dom.ramMemoryBar.classList.add("critical");
        } else if (ramPercent >= 75) {
            dom.ramMemoryBar.classList.add("warning");
        }

        // Push to graph history
        pushGraphData(ramHistory, ramPercent);
        if (dom.ramUtilValue) dom.ramUtilValue.textContent = `${Math.round(ramPercent)}%`;
        drawGraph(dom.ramGraphCanvas, ramHistory, "#6c5ce7", "rgba(108, 92, 231, 0.15)");
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

    // ── Update CPU graph ──
    if (stats.cpu) {
        const cpuPct = stats.cpu.percent || 0;
        pushGraphData(cpuHistory, cpuPct);
        if (dom.cpuUtilValue) dom.cpuUtilValue.textContent = `${Math.round(cpuPct)}%`;
        drawGraph(dom.cpuGraphCanvas, cpuHistory, "#ffa502", "rgba(255, 165, 2, 0.12)");
    }

    // ── Update GPU graph ──
    if (stats.gpu && stats.gpu.percent !== null && stats.gpu.percent !== undefined) {
        const gpuPct = stats.gpu.percent;
        pushGraphData(gpuHistory, gpuPct);
        if (dom.gpuUtilValue) dom.gpuUtilValue.textContent = `${Math.round(gpuPct)}%`;
        drawGraph(dom.gpuGraphCanvas, gpuHistory, "#00cec9", "rgba(0, 206, 201, 0.12)");
        if (dom.gpuGraphSection) dom.gpuGraphSection.style.display = "block";
    } else {
        if (dom.gpuGraphSection) dom.gpuGraphSection.style.display = "none";
    }
}

// ── Retry Handling ───────────────────────────────────────────────────────────
async function handleRetry(msg, msgEl) {
    if (state.isGenerating) {
        showToast("Please wait for the current generation to finish.", "error");
        return;
    }

    const conv = getActiveConversation();
    if (!conv) return;

    // Find the index of this assistant message
    const msgIndex = conv.messages.indexOf(msg);
    if (msgIndex === -1) return;

    // Remove this assistant message and everything after it
    const removedMessages = conv.messages.splice(msgIndex);

    // Remove corresponding DOM elements
    const allMsgEls = [...dom.chatContainer.querySelectorAll(".message")];
    const domIndex = allMsgEls.indexOf(msgEl);
    if (domIndex !== -1) {
        // Remove this element and all following message elements
        for (let i = allMsgEls.length - 1; i >= domIndex; i--) {
            allMsgEls[i].remove();
        }
    }

    saveState();

    // Re-generate
    await generateResponse(conv);
}

// ── Context Window Tracking ──────────────────────────────────────────────────
async function fetchContextUsage(conv) {
    if (!conv || conv.messages.length === 0) {
        updateContextWindowUI(0, state.maxInputTokens);
        return;
    }

    const apiMessages = conv.messages
        .filter(m => m.content && m.content.trim())
        .map(m => ({ role: m.role, content: m.content }));

    try {
        const res = await fetch("/api/context/count", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ messages: apiMessages }),
        });

        if (!res.ok) return;
        const data = await res.json();

        state.maxInputTokens = data.max_tokens || state.maxInputTokens;
        updateContextWindowUI(data.used_tokens, data.max_tokens);
    } catch (err) {
        console.debug("Context count failed:", err);
    }
}

function updateContextWindowUI(used, max) {
    const pct = max > 0 ? Math.min((used / max) * 100, 100) : 0;
    const roundedPct = Math.round(pct);

    if (dom.contextBarFill) {
        dom.contextBarFill.style.width = `${pct}%`;

        // Color coding
        dom.contextBarFill.classList.remove("warning", "critical");
        if (pct >= 90) {
            dom.contextBarFill.classList.add("critical");
        } else if (pct >= 75) {
            dom.contextBarFill.classList.add("warning");
        }
    }

    if (dom.contextLabel) {
        dom.contextLabel.textContent = `${used} / ${max}`;
    }

    if (dom.contextWindow) {
        dom.contextWindow.title = `Context window: ${used} of ${max} tokens used (${roundedPct}%) — Click for details`;
    }

    // ── Update popup details ──
    if (dom.gaugePercent) {
        dom.gaugePercent.textContent = `${roundedPct}%`;
    }

    // Animate SVG gauge
    if (dom.gaugeFill) {
        const circumference = 2 * Math.PI * 52; // ~326.7
        const offset = circumference - (pct / 100) * circumference;
        dom.gaugeFill.style.strokeDashoffset = offset;

        dom.gaugeFill.classList.remove("warning", "critical");
        if (pct >= 90) {
            dom.gaugeFill.classList.add("critical");
        } else if (pct >= 75) {
            dom.gaugeFill.classList.add("warning");
        }
    }

    // Stats cards
    if (dom.ctxTokensUsed) dom.ctxTokensUsed.textContent = used.toLocaleString();
    if (dom.ctxTokensMax) dom.ctxTokensMax.textContent = max.toLocaleString();
    if (dom.ctxTokensRemaining) dom.ctxTokensRemaining.textContent = Math.max(0, max - used).toLocaleString();

    // Message count
    const conv = getActiveConversation();
    const msgCount = conv ? conv.messages.filter(m => m.content && m.content.trim()).length : 0;
    if (dom.ctxMessageCount) dom.ctxMessageCount.textContent = msgCount;

    // Health indicator
    if (dom.contextHealth) {
        dom.contextHealth.classList.remove("warning", "critical");
        const healthLabel = dom.contextHealth.querySelector(".context-health-label");

        if (pct >= 90) {
            dom.contextHealth.classList.add("critical");
            if (healthLabel) healthLabel.textContent = "Critical — Responses may be truncated";
        } else if (pct >= 75) {
            dom.contextHealth.classList.add("warning");
            if (healthLabel) healthLabel.textContent = "Warning — Consider starting a new chat";
        } else {
            if (healthLabel) healthLabel.textContent = "Healthy — Plenty of room";
        }
    }

    // Message breakdown
    if (dom.contextBreakdownList && conv && conv.messages.length > 0) {
        dom.contextMsgBreakdown.classList.add("visible");
        dom.contextBreakdownList.innerHTML = "";

        conv.messages
            .filter(m => m.content && m.content.trim())
            .forEach((m, i) => {
                const emoji = m.role === "user" ? "👤" : "🧠";
                const preview = m.content.slice(0, 60) + (m.content.length > 60 ? "..." : "");
                // Rough token estimate: ~4 chars per token
                const estTokens = Math.ceil(m.content.length / 4);

                const item = document.createElement("div");
                item.className = "context-breakdown-item";
                item.innerHTML = `
                    <span class="context-breakdown-role">${emoji}</span>
                    <span class="context-breakdown-text">${escapeHtml(preview)}</span>
                    <span class="context-breakdown-tokens">~${estTokens} tok</span>
                `;
                dom.contextBreakdownList.appendChild(item);
            });
    } else if (dom.contextMsgBreakdown) {
        dom.contextMsgBreakdown.classList.remove("visible");
    }
}

// ── Context Popup ─────────────────────────────────────────────────────────
function toggleContextPopup() {
    if (dom.contextPopup.classList.contains("visible")) {
        closeContextPopup();
    } else {
        openContextPopup();
    }
}

function openContextPopup() {
    dom.contextPopup.classList.add("visible");
    dom.contextWindow.classList.add("active");
}

function closeContextPopup() {
    dom.contextPopup.classList.remove("visible");
    dom.contextWindow.classList.remove("active");
}

// ── Help Modal ────────────────────────────────────────────────────────────
function openHelpModal() {
    dom.helpModalOverlay.classList.add("visible");
}

function closeHelpModal() {
    dom.helpModalOverlay.classList.remove("visible");
}

// ── SVG Gradient for Gauge ───────────────────────────────────────────────
function initSVGGradient() {
    const svg = document.querySelector(".gauge-svg");
    if (!svg) return;

    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    const gradient = document.createElementNS("http://www.w3.org/2000/svg", "linearGradient");
    gradient.setAttribute("id", "gauge-gradient");
    gradient.setAttribute("x1", "0%");
    gradient.setAttribute("y1", "0%");
    gradient.setAttribute("x2", "100%");
    gradient.setAttribute("y2", "0%");

    const stop1 = document.createElementNS("http://www.w3.org/2000/svg", "stop");
    stop1.setAttribute("offset", "0%");
    stop1.setAttribute("stop-color", "#6c5ce7");

    const stop2 = document.createElementNS("http://www.w3.org/2000/svg", "stop");
    stop2.setAttribute("offset", "100%");
    stop2.setAttribute("stop-color", "#00cec9");

    gradient.appendChild(stop1);
    gradient.appendChild(stop2);
    defs.appendChild(gradient);
    svg.insertBefore(defs, svg.firstChild);
}

// ── Device Switching ─────────────────────────────────────────────────────
async function switchDevice(device) {
    if (state.isGenerating) {
        showToast("Cannot switch device while generating. Please wait.", "error");
        return;
    }

    const currentRequested = (state.modelConfig?.requested_device || "").toUpperCase();
    if (device.toUpperCase() === currentRequested) {
        showToast(`Already using ${device}`, "info");
        return;
    }

    // Show switching overlay
    if (dom.deviceSwitchingOverlay) {
        dom.deviceSwitchingText.textContent = `Switching to ${device}...`;
        dom.deviceSwitchingOverlay.classList.add("visible");
    }

    // Update badge to switching state
    dom.deviceBadge.textContent = device;
    dom.deviceBadge.className = "device-badge switching";

    // Disable send button during switch
    dom.btnSend.disabled = true;
    setStatus("loading", "Switching device...");
    dom.modelStatus.textContent = "Switching...";

    try {
        const res = await fetch("/api/device/switch", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ device: device }),
        });

        const data = await res.json();

        if (data.success) {
            // Successful switch
            const friendly = data.active_device_friendly || data.active_device || device;
            updateDeviceBadge(friendly, data.requested_device || device);
            updateDeviceSelectorUI(data.requested_device || device, friendly);

            setStatus("ready", "Ready");
            dom.modelStatus.textContent = "Loaded";
            showToast(data.message || `Switched to ${friendly}`, "success");

            // Refresh full config
            await fetchConfig();
        } else {
            // Failed — show fallback info
            const fallbackFriendly = data.active_device_friendly || data.active_device || "Unknown";
            updateDeviceBadge(fallbackFriendly, data.requested_device || currentRequested);
            updateDeviceSelectorUI(
                state.modelConfig?.requested_device || currentRequested,
                fallbackFriendly
            );

            if (data.active_device) {
                setStatus("ready", "Ready");
                dom.modelStatus.textContent = "Loaded";
            } else {
                setStatus("error", "Model offline");
                dom.modelStatus.textContent = "Error";
            }

            showToast(data.message || `Failed to switch to ${device}`, "error");

            // Refresh config to get accurate state
            await fetchConfig();
        }
    } catch (err) {
        console.error("Device switch error:", err);
        showToast(`Failed to switch to ${device}: ${err.message}`, "error");

        // Restore previous state
        setStatus("error", "Error");
        dom.modelStatus.textContent = "Error";
        await fetchConfig();
    } finally {
        // Hide switching overlay
        if (dom.deviceSwitchingOverlay) {
            dom.deviceSwitchingOverlay.classList.remove("visible");
        }
        // Re-enable send button
        dom.btnSend.disabled = false;
    }
}

function updateDeviceBadge(friendlyDevice, requestedDevice) {
    if (!dom.deviceBadge) return;

    const requested = (requestedDevice || "").toUpperCase();
    const friendly = (friendlyDevice || "").toUpperCase();

    // Determine badge CSS class
    let badgeClass = "device-badge ";
    if (friendly.includes("HETERO") || friendly === "CPU+GPU") {
        badgeClass += "hetero";
    } else if (friendly === "GPU") {
        badgeClass += "gpu";
    } else if (friendly === "CPU") {
        badgeClass += "cpu";
    } else {
        badgeClass += friendly.toLowerCase();
    }
    dom.deviceBadge.className = badgeClass;

    // Determine badge text
    if (requested === "AUTO") {
        dom.deviceBadge.textContent = `AUTO (${friendly})`;
    } else {
        dom.deviceBadge.textContent = friendly;
    }
}

function updateDeviceSelectorUI(requestedDevice, activeFriendly) {
    const requested = (requestedDevice || "").toUpperCase();
    const friendly = (activeFriendly || "").toUpperCase();

    // Update active state on options
    $$(".device-option").forEach((opt) => {
        const optDevice = (opt.dataset.device || "").toUpperCase();
        if (optDevice === requested) {
            opt.classList.add("active");
        } else {
            opt.classList.remove("active");
        }
    });

    // Update AUTO resolved label
    if (dom.autoResolvedLabel) {
        if (requested === "AUTO") {
            dom.autoResolvedLabel.textContent = `Using ${friendly}`;
        } else {
            dom.autoResolvedLabel.textContent = "Best available";
        }
    }
}

// ── Device Dropdown ──────────────────────────────────────────────────────
function toggleDeviceDropdown() {
    if (dom.deviceDropdown.classList.contains("visible")) {
        closeDeviceDropdown();
    } else {
        openDeviceDropdown();
    }
}

function openDeviceDropdown() {
    dom.deviceDropdown.classList.add("visible");
    dom.deviceSelector.classList.add("open");
}

function closeDeviceDropdown() {
    dom.deviceDropdown.classList.remove("visible");
    dom.deviceSelector.classList.remove("open");
}
