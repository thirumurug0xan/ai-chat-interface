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
    maxTokens: 2048,
    maxInputTokens: 1024,       // context window size from config
};

const fsState = {
    currentPath: "",
    parentPath: "",
    workspacePath: "",
    homePath: "",
    selectedPath: "",
    expandedPaths: new Set(),
    treeDataCache: new Map(),
};

const downloaderState = {
    activeTaskId: null,
    jobStatus: "idle", // idle, running, completed, failed
    eventSource: null,
};

const notesState = {
    notes: [],
    activeNoteName: null,
    currentContent: "",
    currentFilename: "",
    originalContent: "",
    originalFilename: "",
    isUnsaved: false,
    syntaxHighlightOn: localStorage.getItem("notes_syntax_highlight") !== "false",
    tabSize: parseInt(localStorage.getItem("notes_tab_size")) || 4
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
    contextPopupMinimize: $("#context-popup-minimize"),
    contextPopupFullscreen: $("#context-popup-fullscreen"),
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
    helpModal: $("#help-modal"),
    helpModalMinimize: $("#help-modal-minimize"),
    helpModalFullscreen: $("#help-modal-fullscreen"),
    // Theme toggle
    btnTheme: $("#btn-theme"),
    // Settings modal
    btnSettings: $("#btn-settings"),
    settingsModalOverlay: $("#settings-modal-overlay"),
    settingsModalClose: $("#settings-modal-close"),
    settingsModal: $("#settings-modal"),
    settingsModalMinimize: $("#settings-modal-minimize"),
    settingsModalFullscreen: $("#settings-modal-fullscreen"),
    settingDevice: $("#setting-device"),
    settingMaxNewTokens: $("#setting-max-new-tokens"),
    settingMaxNewTokensVal: $("#setting-max-new-tokens-val"),
    settingMaxInputTokens: $("#setting-max-input-tokens"),
    settingMaxInputTokensVal: $("#setting-max-input-tokens-val"),
    btnSaveSettings: $("#btn-save-settings"),
    // Device switching
    deviceSwitchingOverlay: $("#device-switching-overlay"),
    deviceSwitchingText: $("#device-switching-text"),
    // Export
    btnExport: $("#btn-export"),
    // Model Filesystem Explorer Modal
    btnSwitchModel: $("#btn-switch-model"),
    modelBrowserOverlay: $("#model-browser-modal-overlay"),
    modelBrowserClose: $("#model-browser-modal-close"),
    modelBrowserModal: $("#model-browser-modal"),
    modelBrowserMinimize: $("#model-browser-modal-minimize"),
    modelBrowserFullscreen: $("#model-browser-modal-fullscreen"),
    btnFsUp: $("#btn-fs-up"),
    btnFsHome: $("#btn-fs-home"),
    btnFsWorkspace: $("#btn-fs-workspace"),
    btnFsRoot: $("#btn-fs-root"),
    fsPathInput: $("#fs-path-input"),
    btnFsGo: $("#btn-fs-go"),
    fsExplorerTree: $("#fs-explorer-tree"),
    fsItemDetails: $("#fs-item-details"),
    fsSelectedPath: $("#fs-selected-path"),
    btnFsCancel: $("#btn-fs-cancel"),
    btnFsConfirm: $("#btn-fs-confirm"),
    modelSwitchingOverlay: $("#model-switching-overlay"),
    modelSwitchingTitle: $("#model-switching-title"),
    // Model Downloader Modal
    btnDownloadModel: $("#btn-download-model"),
    modelDownloaderOverlay: $("#model-downloader-modal-overlay"),
    modelDownloaderClose: $("#model-downloader-modal-close"),
    modelDownloaderModal: $("#model-downloader-modal"),
    modelDownloaderMinimize: $("#model-downloader-modal-minimize"),
    modelDownloaderFullscreen: $("#model-downloader-modal-fullscreen"),
    downloaderSearchInput: $("#downloader-search-input"),
    btnDownloaderSearch: $("#btn-downloader-search"),
    downloaderResultsContainer: $("#downloader-results-container"),
    downloaderModelId: $("#downloader-model-id"),
    downloaderWeightFormat: $("#downloader-weight-format"),
    downloaderTask: $("#downloader-task"),
    downloaderOutputDir: $("#downloader-output-dir"),
    downloaderHfToken: $("#downloader-hf-token"),
    btnDownloaderHfTokenToggle: $("#btn-downloader-hf-token-toggle"),
    downloaderJobDot: $("#downloader-job-dot"),
    downloaderJobStatusText: $("#downloader-job-status-text"),
    downloaderConsoleOutput: $("#downloader-console-output"),
    downloaderProgressWrap: $("#downloader-progress-wrap"),
    downloaderProgressFill: $("#downloader-progress-fill"),
    downloaderProgressText: $("#downloader-progress-text"),
    btnDownloaderCancel: $("#btn-downloader-cancel"),
    btnDownloaderClose: $("#btn-downloader-close"),
    btnDownloaderStart: $("#btn-downloader-start"),
    // Sidebar search
    sidebarSearch: $("#sidebar-search"),
    sidebarSearchClear: $("#sidebar-search-clear"),
    // Command palette
    paletteOverlay: $("#palette-overlay"),
    paletteSearchInput: $("#palette-search-input"),
    paletteResults: $("#palette-results"),
    // Mousepad Notes
    btnNotes: $("#btn-notes"),
    notesOverlay: $("#notes-modal-overlay"),
    notesClose: $("#notes-modal-close"),
    notesModal: $("#notes-modal"),
    notesMinimize: $("#notes-modal-minimize"),
    notesFullscreen: $("#notes-modal-fullscreen"),
    btnNewNote: $("#btn-new-note"),
    notesSearch: $("#notes-search"),
    notesSearchClear: $("#notes-search-clear"),
    notesList: $("#notes-list"),
    emptyNotes: $("#empty-notes"),
    notesFilename: $("#notes-filename"),
    notesTextarea: $("#notes-textarea"),
    notesStatusBadge: $("#notes-status-badge"),
    notesWordCount: $("#notes-word-count"),
    notesCharCount: $("#notes-char-count"),
    btnNotesDownload: $("#btn-notes-download") || $("#menu-file-download"),
    btnNotesSave: $("#btn-notes-save") || $("#menu-file-save"),
    notesDirInput: $("#notes-dir-input"),
    btnNotesDirSave: $("#btn-notes-dir-save"),
    btnNotesDirBrowse: $("#btn-notes-dir-browse"),
    btnNotesTabAdd: $("#btn-notes-tab-add"),
    notesTabsBar: $("#notes-tabs-bar"),
    // New Mousepad toolbar and settings
    btnTbNew: $("#btn-tb-new"),
    btnTbOpen: $("#btn-tb-open"),
    btnTbSettings: $("#btn-tb-settings"),
    notesSettingsDropdown: $("#notes-settings-dropdown"),
    chkSyntaxHighlight: $("#chk-syntax-highlight"),
    selTabSize: $("#sel-tab-size"),
    notesHighlightPre: $("#notes-highlight-pre"),
    notesCurrentDirDisplay: $("#notes-current-dir-display"),
    notesLineNumbers: $("#notes-line-numbers"),
    notesWindowTitle: $("#notes-window-title"),
    notesFileType: $("#notes-file-type"),
    notesLnCol: $("#notes-ln-col"),
    findReplaceBar: $("#find-replace-bar"),
    frFindInput: $("#fr-find-input"),
    frReplaceInput: $("#fr-replace-input"),
    btnFrNext: $("#btn-fr-next"),
    btnFrReplace: $("#btn-fr-replace"),
    btnFrReplaceAll: $("#btn-fr-replace-all"),
    btnFrClose: $("#btn-fr-close"),
    chkMenuLinenumbers: $("#chk-menu-linenumbers"),
    chkMenuWordwrap: $("#chk-menu-wordwrap"),
    chkMenuHighlight: $("#chk-menu-highlight"),
};

// ── Window Stacking Management ───────────────────────────────────────────────
let highestZIndex = 900;
const openOverlaysStack = [];

function bringToFront(overlay) {
    if (!overlay) return;
    highestZIndex++;
    overlay.style.zIndex = highestZIndex;
}

function openOverlay(overlayEl) {
    if (!overlayEl) return;
    const idx = openOverlaysStack.indexOf(overlayEl);
    if (idx !== -1) {
        openOverlaysStack.splice(idx, 1);
    }
    overlayEl.classList.add("visible");
    bringToFront(overlayEl);
    openOverlaysStack.push(overlayEl);
}

function closeOverlay(overlayEl) {
    if (!overlayEl) return;
    overlayEl.classList.remove("visible");
    const idx = openOverlaysStack.indexOf(overlayEl);
    if (idx !== -1) {
        openOverlaysStack.splice(idx, 1);
    }
}

// ── Initialization ───────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    loadState();
    initEventListeners();
    fetchConfig();
    renderConversationList();
    renderActiveConversation();
    initMemoryMonitor();
    initSVGGradient();
    initPanelToggles();
    initWindowDraggingAndResizing();
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

    // Keyboard shortcuts
    document.addEventListener("keydown", (e) => {
        // Ctrl+K or Cmd+K — Command Palette
        if ((e.ctrlKey || e.metaKey) && e.key === "k") {
            e.preventDefault();
            togglePalette();
            return;
        }

        // Ctrl+N — new chat or new note (if notes modal is visible)
        if ((e.ctrlKey || e.metaKey) && e.key === "n") {
            e.preventDefault();
            if (dom.notesOverlay && dom.notesOverlay.classList.contains("visible")) {
                createNewNote();
            } else {
                createNewConversation();
            }
        }

        // Ctrl+M — Mousepad Notes
        if ((e.ctrlKey || e.metaKey) && e.key === "m") {
            e.preventDefault();
            toggleNotesModal();
        }

        // Ctrl+S / Ctrl+Shift+S — save note (if notes modal is visible)
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
            if (dom.notesOverlay && dom.notesOverlay.classList.contains("visible")) {
                e.preventDefault();
                if (e.shiftKey) {
                    saveNoteAs();
                } else {
                    saveActiveNote();
                }
            }
        }

        // Ctrl+O — open folder browser (if notes modal is visible)
        if ((e.ctrlKey || e.metaKey) && e.key === "o") {
            if (dom.notesOverlay && dom.notesOverlay.classList.contains("visible")) {
                e.preventDefault();
                openNotesFolderBrowser();
            }
        }

        // F2 — Rename note (if notes modal is visible)
        if (e.key === "F2") {
            if (dom.notesOverlay && dom.notesOverlay.classList.contains("visible")) {
                e.preventDefault();
                renameActiveNote();
            }
        }

        // Ctrl+F — find & replace (if notes modal is visible)
        if ((e.ctrlKey || e.metaKey) && e.key === "f") {
            if (dom.notesOverlay && dom.notesOverlay.classList.contains("visible")) {
                e.preventDefault();
                toggleFindReplaceBar(dom.findReplaceBar ? dom.findReplaceBar.style.display === "none" : true);
            }
        }

        // Esc — close popups, modals and command palette
        if (e.key === "Escape") {
            if (dom.findReplaceBar && dom.findReplaceBar.style.display === "flex") {
                toggleFindReplaceBar(false);
                return;
            }
            if (paletteState.isOpen) {
                closePalette();
                return;
            }
            if (openOverlaysStack.length > 0) {
                const topOverlay = openOverlaysStack[openOverlaysStack.length - 1];
                if (topOverlay === dom.helpModalOverlay) closeHelpModal();
                else if (topOverlay === dom.settingsModalOverlay) closeSettingsModal();
                else if (topOverlay === dom.modelBrowserOverlay) closeModelBrowserModal();
                else if (topOverlay === dom.modelDownloaderOverlay) closeModelDownloaderModal();
                else if (topOverlay === dom.notesOverlay) closeNotesModal();
                return;
            }
            closeContextPopup();
        }

        // ? — open help (only when not typing in inputs)
        if (e.key === "?" && document.activeElement !== dom.chatInput && document.activeElement !== dom.sidebarSearch && document.activeElement !== dom.paletteSearchInput) {
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

    // Settings button
    if (dom.btnSettings) {
        dom.btnSettings.addEventListener("click", openSettingsModal);
    }

    // Settings modal close
    if (dom.settingsModalClose) {
        dom.settingsModalClose.addEventListener("click", closeSettingsModal);
    }

    if (dom.settingsModalOverlay) {
        dom.settingsModalOverlay.addEventListener("click", (e) => {
            if (e.target === dom.settingsModalOverlay) closeSettingsModal();
        });
    }

    // Settings sliders dynamic badges
    if (dom.settingMaxNewTokens) {
        dom.settingMaxNewTokens.addEventListener("input", (e) => {
            dom.settingMaxNewTokensVal.textContent = e.target.value;
        });
    }

    if (dom.settingMaxInputTokens) {
        dom.settingMaxInputTokens.addEventListener("input", (e) => {
            dom.settingMaxInputTokensVal.textContent = e.target.value;
        });
    }

    // Settings save button
    if (dom.btnSaveSettings) {
        dom.btnSaveSettings.addEventListener("click", saveSettings);
    }

    // Close context popup on outside click
    document.addEventListener("click", (e) => {
        if (dom.contextPopup && dom.contextPopup.classList.contains("visible")) {
            if (!dom.contextPopup.contains(e.target) && !dom.contextWindow.contains(e.target)) {
                closeContextPopup();
            }
        }
    });

    // Export button
    if (dom.btnExport) {
        dom.btnExport.addEventListener("click", handleExport);
    }

    // Switch Model button
    if (dom.btnSwitchModel) {
        dom.btnSwitchModel.addEventListener("click", openModelBrowserModal);
    }

    // Model Browser Modal Close & Cancel
    if (dom.modelBrowserClose) {
        dom.modelBrowserClose.addEventListener("click", closeModelBrowserModal);
    }
    if (dom.btnFsCancel) {
        dom.btnFsCancel.addEventListener("click", closeModelBrowserModal);
    }
    if (dom.modelBrowserOverlay) {
        dom.modelBrowserOverlay.addEventListener("click", (e) => {
            if (e.target === dom.modelBrowserOverlay) closeModelBrowserModal();
        });
    }

    // File Explorer Path Navigation Shortcuts
    if (dom.btnFsUp) {
        dom.btnFsUp.addEventListener("click", () => {
            if (fsState.parentPath && fsState.parentPath !== fsState.currentPath) {
                fsState.currentPath = fsState.parentPath;
                renderRootTree();
            }
        });
    }
    if (dom.btnFsHome) {
        dom.btnFsHome.addEventListener("click", () => {
            if (fsState.homePath) {
                fsState.currentPath = fsState.homePath;
                renderRootTree();
            }
        });
    }
    if (dom.btnFsWorkspace) {
        dom.btnFsWorkspace.addEventListener("click", () => {
            if (fsState.workspacePath) {
                fsState.currentPath = fsState.workspacePath;
                renderRootTree();
            }
        });
    }
    if (dom.btnFsRoot) {
        dom.btnFsRoot.addEventListener("click", () => {
            fsState.currentPath = "/";
            renderRootTree();
        });
    }

    // Path Input execution
    if (dom.btnFsGo) {
        dom.btnFsGo.addEventListener("click", () => {
            const val = dom.fsPathInput.value.trim();
            if (val) {
                fsState.currentPath = val;
                renderRootTree();
            }
        });
    }
    if (dom.fsPathInput) {
        dom.fsPathInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                dom.btnFsGo.click();
            }
        });
    }

    // Confirm selection (Load Model or select notes folder)
    if (dom.btnFsConfirm) {
        dom.btnFsConfirm.addEventListener("click", handleFsConfirm);
    }

    // Model Downloader Open
    if (dom.btnDownloadModel) {
        dom.btnDownloadModel.addEventListener("click", openModelDownloaderModal);
    }
    // Model Downloader Close
    if (dom.modelDownloaderClose) {
        dom.modelDownloaderClose.addEventListener("click", closeModelDownloaderModal);
    }
    if (dom.btnDownloaderClose) {
        dom.btnDownloaderClose.addEventListener("click", closeModelDownloaderModal);
    }
    if (dom.modelDownloaderOverlay) {
        dom.modelDownloaderOverlay.addEventListener("click", (e) => {
            if (e.target === dom.modelDownloaderOverlay) closeModelDownloaderModal();
        });
    }
    // Model Downloader Search
    if (dom.btnDownloaderSearch) {
        dom.btnDownloaderSearch.addEventListener("click", handleDownloaderSearch);
    }
    if (dom.downloaderSearchInput) {
        dom.downloaderSearchInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                handleDownloaderSearch();
            }
        });
    }
    // Input validation monitor
    const validateDownloaderInputs = () => {
        const modelId = dom.downloaderModelId.value.trim();
        const outputDir = dom.downloaderOutputDir.value.trim();
        if (modelId && outputDir && (downloaderState.jobStatus !== "running")) {
            dom.btnDownloaderStart.disabled = false;
        } else {
            dom.btnDownloaderStart.disabled = true;
        }
    };
    if (dom.downloaderModelId) {
        dom.downloaderModelId.addEventListener("input", validateDownloaderInputs);
    }
    if (dom.downloaderOutputDir) {
        dom.downloaderOutputDir.addEventListener("input", validateDownloaderInputs);
    }
    if (dom.btnDownloaderHfTokenToggle && dom.downloaderHfToken) {
        dom.btnDownloaderHfTokenToggle.addEventListener("click", () => {
            if (dom.downloaderHfToken.type === "password") {
                dom.downloaderHfToken.type = "text";
                dom.btnDownloaderHfTokenToggle.textContent = "🙈";
                dom.btnDownloaderHfTokenToggle.title = "Hide Token";
            } else {
                dom.downloaderHfToken.type = "password";
                dom.btnDownloaderHfTokenToggle.textContent = "👁️";
                dom.btnDownloaderHfTokenToggle.title = "Show Token";
            }
        });
    }
    // Start/Load Export Action
    if (dom.btnDownloaderStart) {
        dom.btnDownloaderStart.addEventListener("click", handleStartExport);
    }
    // Cancel Export Action
    if (dom.btnDownloaderCancel) {
        dom.btnDownloaderCancel.addEventListener("click", handleCancelExport);
    }

    // Modal & Popup Minimize and Fullscreen Event Listeners
    setupWindowControlsListeners();

    // Theme toggle listener
    if (dom.btnTheme) {
        dom.btnTheme.addEventListener("click", toggleTheme);
    }

    // Sidebar search listeners
    if (dom.sidebarSearch) {
        dom.sidebarSearch.addEventListener("input", (e) => {
            const query = e.target.value.trim();
            if (dom.sidebarSearchClear) {
                dom.sidebarSearchClear.style.display = query ? "block" : "none";
            }
            renderConversationList();
        });
    }

    if (dom.sidebarSearchClear) {
        dom.sidebarSearchClear.addEventListener("click", () => {
            if (dom.sidebarSearch) {
                dom.sidebarSearch.value = "";
            }
            dom.sidebarSearchClear.style.display = "none";
            renderConversationList();
        });
    }

    // Command palette overlay click to close
    if (dom.paletteOverlay) {
        dom.paletteOverlay.addEventListener("click", (e) => {
            if (e.target === dom.paletteOverlay) {
                closePalette();
            }
        });
    }

    // Command palette search input event
    if (dom.paletteSearchInput) {
        dom.paletteSearchInput.addEventListener("input", () => {
            renderPaletteResults();
        });

        dom.paletteSearchInput.addEventListener("keydown", (e) => {
            if (e.key === "ArrowDown") {
                e.preventDefault();
                navigatePalette(1);
            } else if (e.key === "ArrowUp") {
                e.preventDefault();
                navigatePalette(-1);
            } else if (e.key === "Enter") {
                e.preventDefault();
                selectActivePaletteItem();
            } else if (e.key === "Escape") {
                e.preventDefault();
                closePalette();
            }
        });
    }

    // Mousepad Notes Event Listeners
    if (dom.btnNotes) {
        dom.btnNotes.addEventListener("click", openNotesModal);
    }
    if (dom.notesClose) {
        dom.notesClose.addEventListener("click", closeNotesModal);
    }
    if (dom.notesOverlay) {
        dom.notesOverlay.addEventListener("click", (e) => {
            if (e.target === dom.notesOverlay) closeNotesModal();
        });
    }
    if (dom.btnNewNote) {
        dom.btnNewNote.addEventListener("click", createNewNote);
    }
    if (dom.btnNotesSave) {
        dom.btnNotesSave.addEventListener("click", saveActiveNote);
    }
    if (dom.btnNotesDownload) {
        dom.btnNotesDownload.addEventListener("click", downloadActiveNote);
    }
    if (dom.btnNotesTabAdd) {
        dom.btnNotesTabAdd.addEventListener("click", createNewNoteTab);
    }
    if (dom.notesTextarea) {
        dom.notesTextarea.addEventListener("input", (e) => {
            notesState.currentContent = e.target.value;
            updateNotesCounters();
            checkUnsavedChanges();
            syncHighlight();
        });
    }

    // Notes Toolbar & Settings listeners
    if (dom.btnTbNew) {
        dom.btnTbNew.addEventListener("click", createNewNote);
    }
    if (dom.btnTbOpen) {
        dom.btnTbOpen.addEventListener("click", () => {
            fsState.selectorMode = "notes";
            const val = dom.notesDirInput ? dom.notesDirInput.value.trim() : "";
            if (val) {
                fsState.currentPath = val;
            } else {
                fsState.currentPath = "";
            }
            openModelBrowserModal();
        });
    }
    if (dom.btnTbSettings) {
        dom.btnTbSettings.addEventListener("click", (e) => {
            e.stopPropagation();
            if (dom.notesSettingsDropdown) {
                dom.notesSettingsDropdown.classList.toggle("show");
            }
        });
    }
    // Close settings dropdown when clicking outside
    document.addEventListener("click", (e) => {
        if (dom.notesSettingsDropdown && dom.notesSettingsDropdown.classList.contains("show")) {
            if (!e.target.closest(".notes-tb-dropdown-container")) {
                dom.notesSettingsDropdown.classList.remove("show");
            }
        }
    });
    if (dom.chkSyntaxHighlight) {
        dom.chkSyntaxHighlight.checked = notesState.syntaxHighlightOn;
        dom.chkSyntaxHighlight.addEventListener("change", (e) => {
            notesState.syntaxHighlightOn = e.target.checked;
            localStorage.setItem("notes_syntax_highlight", notesState.syntaxHighlightOn);
            updateHighlightView();
            syncHighlight();
        });
    }
    if (dom.selTabSize) {
        dom.selTabSize.value = notesState.tabSize;
        dom.selTabSize.addEventListener("change", (e) => {
            notesState.tabSize = parseInt(e.target.value) || 4;
            localStorage.setItem("notes_tab_size", notesState.tabSize);
        });
    }
    if (dom.notesTextarea) {
        // Sync scroll
        dom.notesTextarea.addEventListener("scroll", () => {
            if (dom.notesHighlightPre) {
                dom.notesHighlightPre.scrollTop = dom.notesTextarea.scrollTop;
                dom.notesHighlightPre.scrollLeft = dom.notesTextarea.scrollLeft;
            }
            if (dom.notesLineNumbers) {
                dom.notesLineNumbers.scrollTop = dom.notesTextarea.scrollTop;
            }
        });
        // Caret/selection position listener
        dom.notesTextarea.addEventListener("keyup", updateCursorPosition);
        dom.notesTextarea.addEventListener("click", updateCursorPosition);
        dom.notesTextarea.addEventListener("focus", updateCursorPosition);
        dom.notesTextarea.addEventListener("select", updateCursorPosition);
        // Intercept Tab key
        dom.notesTextarea.addEventListener("keydown", (e) => {
            if (e.key === "Tab") {
                e.preventDefault();
                const start = dom.notesTextarea.selectionStart;
                const end = dom.notesTextarea.selectionEnd;
                const val = dom.notesTextarea.value;
                const tabSpaces = " ".repeat(notesState.tabSize);
                
                dom.notesTextarea.value = val.substring(0, start) + tabSpaces + val.substring(end);
                dom.notesTextarea.selectionStart = dom.notesTextarea.selectionEnd = start + notesState.tabSize;
                
                notesState.currentContent = dom.notesTextarea.value;
                updateNotesCounters();
                checkUnsavedChanges();
                syncHighlight();
            }
        });
    }
    if (dom.notesSearch) {
        dom.notesSearch.addEventListener("input", (e) => {
            const query = e.target.value.toLowerCase().trim();
            if (dom.notesSearchClear) {
                dom.notesSearchClear.style.display = query ? "block" : "none";
            }
            renderNotesList(query);
        });
    }
    if (dom.notesSearchClear) {
        dom.notesSearchClear.addEventListener("click", () => {
            if (dom.notesSearch) {
                dom.notesSearch.value = "";
            }
            dom.notesSearchClear.style.display = "none";
            renderNotesList();
        });
    }
    if (dom.btnNotesDirSave) {
        dom.btnNotesDirSave.addEventListener("click", saveNotesDirectory);
    }
    if (dom.notesDirInput) {
        dom.notesDirInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                saveNotesDirectory();
            }
        });
    }
    if (dom.btnNotesDirBrowse) {
        dom.btnNotesDirBrowse.addEventListener("click", () => {
            fsState.selectorMode = "notes";
            // Initialize path browser to the input path
            const val = dom.notesDirInput ? dom.notesDirInput.value.trim() : "";
            if (val) {
                fsState.currentPath = val;
            }
            openModelBrowserModal();
        });
    }

    // Find & Replace Buttons
    if (dom.btnFrNext) dom.btnFrNext.addEventListener("click", handleFindNext);
    if (dom.btnFrReplace) dom.btnFrReplace.addEventListener("click", handleReplace);
    if (dom.btnFrReplaceAll) dom.btnFrReplaceAll.addEventListener("click", handleReplaceAll);
    if (dom.btnFrClose) dom.btnFrClose.addEventListener("click", () => toggleFindReplaceBar(false));
    if (dom.frFindInput) {
        dom.frFindInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                handleFindNext();
            }
        });
    }

    // Initialize GTK Menu Bar
    initGtkMenuBar();

    // Bring modal overlay to front on mousedown/click
    $$(".modal-overlay").forEach((overlay) => {
        overlay.addEventListener("mousedown", () => {
            bringToFront(overlay);
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
        updateDeviceSelectorUI(requestedDevice, friendlyDevice, data.available_devices);

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
            if (dom.settingMaxNewTokens) {
                dom.settingMaxNewTokens.value = data.max_new_tokens;
                dom.settingMaxNewTokensVal.textContent = data.max_new_tokens;
            }
        }

        // Store context window size
        if (data.max_input_tokens) {
            state.maxInputTokens = data.max_input_tokens;
            if (dom.settingMaxInputTokens) {
                dom.settingMaxInputTokens.value = data.max_input_tokens;
                dom.settingMaxInputTokensVal.textContent = data.max_input_tokens;
            }
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
            body: JSON.stringify({ messages: apiMessages, max_tokens: state.maxTokens }),
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

    const searchQuery = dom.sidebarSearch ? dom.sidebarSearch.value.toLowerCase().trim() : "";

    const filteredConversations = state.conversations.filter((conv) => {
        if (!searchQuery) return true;
        // Check title
        if (conv.title.toLowerCase().includes(searchQuery)) return true;
        // Check message contents
        return conv.messages.some((msg) => msg.content.toLowerCase().includes(searchQuery));
    });

    if (filteredConversations.length === 0) {
        if (state.conversations.length === 0) {
            dom.emptyConversations.querySelector(".empty-icon").textContent = "💬";
            dom.emptyConversations.querySelector("div:last-child").textContent = "No conversations yet";
        } else {
            dom.emptyConversations.querySelector(".empty-icon").textContent = "🔍";
            dom.emptyConversations.querySelector("div:last-child").textContent = "No matches found";
        }
        dom.emptyConversations.style.display = "block";
        return;
    }

    dom.emptyConversations.style.display = "none";

    filteredConversations.forEach((conv) => {
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
    let actionsHtml = `
        <button class="btn-message-action btn-copy-message" title="Copy message">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="action-icon">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
            Copy
        </button>
    `;
    if (msg.role === "assistant") {
        actionsHtml += `
            <button class="btn-message-action btn-retry" title="Regenerate this response">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="action-icon">
                    <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path>
                    <path d="M3 3v5h5"></path>
                    <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"></path>
                    <path d="M16 16h5v5"></path>
                </svg>
                Retry
            </button>
        `;
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
            state.maxTokens = data.maxTokens || 2048;
            if (dom.settingMaxNewTokens) {
                dom.settingMaxNewTokens.value = state.maxTokens;
                dom.settingMaxNewTokensVal.textContent = state.maxTokens;
            }
        }
        
        if (dom.downloaderHfToken) {
            dom.downloaderHfToken.value = localStorage.getItem("hf_token") || "";
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
    if (dom.contextWindow && dom.contextWindow.classList.contains("minimized-active")) {
        openContextPopup();
    } else if (dom.contextPopup.classList.contains("visible")) {
        closeContextPopup();
    } else {
        openContextPopup();
    }
}

function openContextPopup() {
    if (dom.contextWindow) dom.contextWindow.classList.remove("minimized-active");
    dom.contextPopup.classList.add("visible");
    dom.contextWindow.classList.add("active");
}

function closeContextPopup() {
    dom.contextPopup.classList.remove("visible");
    dom.contextWindow.classList.remove("active");
    if (dom.contextWindow) dom.contextWindow.classList.remove("minimized-active");
    if (dom.contextPopup) dom.contextPopup.classList.remove("fullscreen");
}

// ── Help Modal ────────────────────────────────────────────────────────────
function openHelpModal() {
    if (dom.btnHelp) dom.btnHelp.classList.remove("minimized-active");
    openOverlay(dom.helpModalOverlay);
}

function closeHelpModal() {
    closeOverlay(dom.helpModalOverlay);
    if (dom.btnHelp) dom.btnHelp.classList.remove("minimized-active");
    if (dom.helpModal) dom.helpModal.classList.remove("fullscreen");
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

// ── Settings Modal ────────────────────────────────────────────────────────
function openSettingsModal() {
    if (dom.btnSettings) dom.btnSettings.classList.remove("minimized-active");
    if (state.modelConfig) {
        if (dom.settingDevice) {
            dom.settingDevice.value = state.modelConfig.requested_device || "AUTO";
        }
        if (dom.settingMaxNewTokens) {
            dom.settingMaxNewTokens.value = state.modelConfig.max_new_tokens || 2048;
            dom.settingMaxNewTokensVal.textContent = dom.settingMaxNewTokens.value;
        }
        if (dom.settingMaxInputTokens) {
            dom.settingMaxInputTokens.value = state.modelConfig.max_input_tokens || 1024;
            dom.settingMaxInputTokensVal.textContent = dom.settingMaxInputTokens.value;
        }
        // update select option disabled states
        const friendly = state.modelConfig.device_friendly || state.modelConfig.device || "—";
        updateDeviceSelectorUI(state.modelConfig.requested_device || "AUTO", friendly, state.modelConfig.available_devices);
    }
    openOverlay(dom.settingsModalOverlay);
}

function closeSettingsModal() {
    closeOverlay(dom.settingsModalOverlay);
    if (dom.btnSettings) dom.btnSettings.classList.remove("minimized-active");
    if (dom.settingsModal) dom.settingsModal.classList.remove("fullscreen");
}

async function saveSettings() {
    if (state.isGenerating) {
        showToast("Cannot save settings while generating. Please wait.", "error");
        return;
    }

    const newDevice = dom.settingDevice.value.toUpperCase();
    const newMaxNewTokens = parseInt(dom.settingMaxNewTokens.value, 10);
    const newMaxInputTokens = parseInt(dom.settingMaxInputTokens.value, 10);

    const currentDevice = (state.modelConfig?.requested_device || "").toUpperCase();
    const deviceChanged = newDevice !== currentDevice;

    // Show switching overlay if device changed
    if (deviceChanged) {
        if (dom.deviceSwitchingOverlay) {
            dom.deviceSwitchingText.textContent = `Switching device to ${newDevice}...`;
            dom.deviceSwitchingOverlay.classList.add("visible");
        }
        dom.deviceBadge.textContent = newDevice;
        dom.deviceBadge.className = "device-badge switching";
        dom.btnSend.disabled = true;
        setStatus("loading", "Switching device...");
        dom.modelStatus.textContent = "Switching...";
    }

    try {
        const res = await fetch("/api/config/update", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                device: newDevice,
                max_new_tokens: newMaxNewTokens,
                max_input_tokens: newMaxInputTokens
            }),
        });

        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.message || errData.error || "Failed to update configuration");
        }

        const data = await res.json();

        if (deviceChanged) {
            // Check device switch result
            const switchResult = data.device_switch_result || { success: data.loaded };
            if (switchResult.success) {
                showToast(switchResult.message || `Switched to ${newDevice}`, "success");
            } else {
                showToast(switchResult.message || `Failed to switch to ${newDevice}`, "error");
            }
        } else {
            showToast("Settings saved successfully", "success");
        }

        // Apply updated config
        state.modelConfig = data;
        state.maxTokens = data.max_new_tokens;
        state.maxInputTokens = data.max_input_tokens;
        saveState();

        const friendly = data.device_friendly || data.device || newDevice;
        updateDeviceBadge(friendly, data.requested_device || newDevice);
        updateDeviceSelectorUI(data.requested_device || newDevice, friendly, data.available_devices);

        // Update read-only Model Name
        if (dom.modelName && data.model_name) {
            dom.modelName.textContent = data.model_name;
        }

        if (data.loaded) {
            setStatus("ready", "Ready");
            dom.modelStatus.textContent = "Loaded";
        } else {
            setStatus("error", "Model offline");
            dom.modelStatus.textContent = "Error";
        }

        // Update context window UI max limit
        updateContextWindowUI(0, data.max_input_tokens);
        
        // Close modal
        closeSettingsModal();
    } catch (err) {
        console.error("Save settings error:", err);
        showToast(`Failed to save settings: ${err.message}`, "error");
        
        // Restore previous status
        setStatus("error", "Error");
        dom.modelStatus.textContent = "Error";
        await fetchConfig();
    } finally {
        if (dom.deviceSwitchingOverlay) {
            dom.deviceSwitchingOverlay.classList.remove("visible");
        }
        dom.btnSend.disabled = false;
    }
}

function updateDeviceBadge(friendlyDevice, requestedDevice) {
    if (!dom.deviceBadge) return;

    const requested = (requestedDevice || "").toUpperCase();
    const friendly = (friendlyDevice || "").toUpperCase();

    // Determine badge CSS class
    let badgeClass = "device-badge ";
    if (friendly.includes("HETERO") || friendly === "CPU+GPU" || friendly === "XPU") {
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

function updateDeviceSelectorUI(requestedDevice, activeFriendly, availableDevices) {
    const requested = (requestedDevice || "").toUpperCase();
    const available = (availableDevices || []).map(d => d.toUpperCase());

    if (dom.settingDevice) {
        dom.settingDevice.value = requested;
        
        // Enable/disable select options based on availability
        [...dom.settingDevice.options].forEach(opt => {
            const optVal = opt.value;
            // AUTO and CPU are always available, others checked dynamically
            if (optVal !== "AUTO" && optVal !== "CPU" && available.length > 0 && !available.includes(optVal)) {
                opt.disabled = true;
                opt.text = `${optVal} (Unavailable)`;
            } else {
                opt.disabled = false;
                // restore text
                if (optVal === "AUTO") opt.text = "AUTO (Best available)";
                if (optVal === "GPU") opt.text = "GPU (Intel integrated/discrete)";
                if (optVal === "CPU") opt.text = "CPU (Universal compatibility)";
                if (optVal === "XPU") opt.text = "XPU (Combined CPU + GPU)";
            }
        });
    }
}

// ── Collapsible Panel Toggles ────────────────────────────────────────────
function initPanelToggles() {
    // Restore saved panel states
    const savedPanelState = JSON.parse(localStorage.getItem("ai-chat-panels") || "{}");

    $$(".sidebar-panel-toggle").forEach((toggleBtn) => {
        const panelId = toggleBtn.dataset.panel;
        const panelBody = document.getElementById(panelId);
        if (!panelBody) return;

        // Restore saved state (default: collapsed)
        const isOpen = savedPanelState[panelId] === true;
        if (isOpen) {
            panelBody.classList.remove("collapsed");
            toggleBtn.classList.add("open");
        } else {
            panelBody.classList.add("collapsed");
            toggleBtn.classList.remove("open");
        }

        toggleBtn.addEventListener("click", () => {
            const isNowCollapsed = !panelBody.classList.contains("collapsed");
            panelBody.classList.toggle("collapsed", isNowCollapsed);
            toggleBtn.classList.toggle("open", !isNowCollapsed);

            // Save state
            const state = JSON.parse(localStorage.getItem("ai-chat-panels") || "{}");
            state[panelId] = !isNowCollapsed;
            localStorage.setItem("ai-chat-panels", JSON.stringify(state));
        });
    });
}

// ── Export Conversation ──────────────────────────────────────────────────
function handleExport() {
    const conv = getActiveConversation();
    if (!conv || conv.messages.length === 0) {
        showToast("No conversation to export.", "error");
        return;
    }

    // Build export data
    const exportData = {
        id: conv.id,
        title: conv.title,
        createdAt: conv.createdAt ? new Date(conv.createdAt).toISOString() : null,
        exportedAt: new Date().toISOString(),
        model: state.modelConfig?.model_name || "Unknown",
        device: state.modelConfig?.device_friendly || state.modelConfig?.device || "Unknown",
        messageCount: conv.messages.length,
        messages: conv.messages.map((msg) => ({
            role: msg.role,
            content: msg.content,
            timestamp: msg.timestamp ? new Date(msg.timestamp).toISOString() : null,
            ...(msg.meta ? { meta: msg.meta } : {}),
        })),
    };

    // Create and download the JSON file
    const jsonStr = JSON.stringify(exportData, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    // Sanitize title for filename
    const safeTitle = conv.title
        .replace(/[^a-zA-Z0-9_\-\s]/g, "")
        .replace(/\s+/g, "_")
        .slice(0, 40)
        || "conversation";
    a.href = url;
    a.download = `${safeTitle}_${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showToast("Conversation exported!", "success");
}

// ── Model Filesystem Explorer Modal ──────────────────────────────────────

function openModelBrowserModal() {
    if (dom.btnSwitchModel) dom.btnSwitchModel.classList.remove("minimized-active");
    
    if (fsState.selectorMode === "notes") {
        const titleEl = document.getElementById("model-browser-title-text");
        if (titleEl) titleEl.textContent = "Select Notes Folder / File";
        if (dom.btnFsConfirm) dom.btnFsConfirm.textContent = "Select Folder";
        
        if (dom.notesDirInput && dom.notesDirInput.value.trim()) {
            fsState.currentPath = dom.notesDirInput.value.trim();
        } else {
            fsState.currentPath = "";
        }
    } else {
        const titleEl = document.getElementById("model-browser-title-text");
        if (titleEl) titleEl.textContent = "Select Local OpenVINO Model";
        if (dom.btnFsConfirm) dom.btnFsConfirm.textContent = "Load Model";
        
        if (state.modelConfig && state.modelConfig.model_path) {
            fsState.currentPath = state.modelConfig.model_path;
        } else {
            fsState.currentPath = "";
        }
    }

    fsState.selectedPath = "";
    fsState.expandedPaths.clear();
    fsState.treeDataCache.clear();

    if (dom.fsSelectedPath) dom.fsSelectedPath.textContent = "—";
    if (dom.btnFsConfirm) dom.btnFsConfirm.disabled = true;

    if (dom.fsItemDetails) {
        dom.fsItemDetails.innerHTML = `
            <div class="fs-details-empty">
                <span class="fs-details-empty-icon">📁</span>
                <div>Select a folder in the explorer tree on the left to see details.</div>
            </div>
        `;
    }

    if (dom.modelBrowserOverlay) {
        openOverlay(dom.modelBrowserOverlay);
    }
    renderRootTree();
}

function closeModelBrowserModal() {
    if (dom.modelBrowserOverlay) {
        closeOverlay(dom.modelBrowserOverlay);
    }
    if (dom.btnSwitchModel) dom.btnSwitchModel.classList.remove("minimized-active");
    if (dom.modelBrowserModal) dom.modelBrowserModal.classList.remove("fullscreen");
    
    // Restore default texts
    const titleEl = document.getElementById("model-browser-title-text");
    if (titleEl) titleEl.textContent = "Select Local OpenVINO Model";
    if (dom.btnFsConfirm) dom.btnFsConfirm.textContent = "Load Model";
    
    // Reset mode
    fsState.selectorMode = "model";
}

async function renderRootTree() {
    if (!dom.fsExplorerTree) return;
    
    dom.fsExplorerTree.innerHTML = `
        <div class="fs-tree-loading">
            <div class="device-switching-spinner" style="margin: 0 auto 10px;"></div>
            Loading file system...
        </div>
    `;

    try {
        const res = await fetch("/api/fs/list", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path: fsState.currentPath })
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.error || `HTTP ${res.status}`);
        }

        const data = await res.json();
        fsState.currentPath = data.current_path;
        fsState.parentPath = data.parent_path;
        fsState.workspacePath = data.workspace_path;
        fsState.homePath = data.home_path;

        if (dom.fsPathInput) {
            dom.fsPathInput.value = data.current_path;
        }

        dom.fsExplorerTree.innerHTML = "";

        if (data.entries.length === 0) {
            dom.fsExplorerTree.innerHTML = `<div class="fs-tree-empty">Empty directory</div>`;
            return;
        }

        fsState.treeDataCache.set(data.current_path, data.entries);

        data.entries.forEach(entry => {
            const node = renderTreeNode(entry, 0);
            dom.fsExplorerTree.appendChild(node);
        });
    } catch (err) {
        console.error(err);
        dom.fsExplorerTree.innerHTML = `
            <div class="fs-tree-empty" style="color: #ff6b6b; text-align: center; padding: 20px;">
                ⚠️ Error: ${escapeHtml(err.message)}
            </div>
        `;
    }
}

function renderTreeNode(entry, level) {
    const isExpanded = fsState.expandedPaths.has(entry.path);
    const hasChildren = entry.is_dir;

    const nodeWrapper = document.createElement("div");
    nodeWrapper.className = "fs-tree-node-wrapper";
    nodeWrapper.dataset.path = entry.path;

    let chevronHtml = "";
    if (hasChildren) {
        chevronHtml = `<span class="fs-chevron ${isExpanded ? 'expanded' : ''}">▶</span>`;
    } else {
        chevronHtml = `<span class="fs-chevron hidden">▶</span>`;
    }

    let icon = "📄";
    if (entry.is_dir) {
        icon = entry.is_model ? "🧠" : "📁";
    }

    const nodeHeader = document.createElement("div");
    nodeHeader.className = `fs-tree-node ${fsState.selectedPath === entry.path ? 'selected' : ''}`;
    nodeHeader.style.paddingLeft = `${level * 12 + 6}px`;
    nodeHeader.innerHTML = `
        ${chevronHtml}
        <span class="fs-icon">${icon}</span>
        <span class="fs-node-name">${escapeHtml(entry.name)}</span>
    `;

    nodeWrapper.appendChild(nodeHeader);

    nodeHeader.addEventListener("click", (e) => {
        e.stopPropagation();
        selectFsItem(entry);

        const clickedChevron = e.target.classList.contains("fs-chevron");
        if (entry.is_dir && (!clickedChevron || isExpanded)) {
            toggleFolderExpand(entry.path, nodeWrapper, level + 1);
        }
    });

    if (isExpanded && hasChildren) {
        const childrenDiv = document.createElement("div");
        childrenDiv.className = "fs-tree-children";
        nodeWrapper.appendChild(childrenDiv);
        loadAndRenderChildren(entry.path, childrenDiv, level + 1);
    }

    return nodeWrapper;
}

async function toggleFolderExpand(path, nodeWrapper, nextLevel) {
    const isExpanded = fsState.expandedPaths.has(path);
    const nodeHeader = nodeWrapper.querySelector(".fs-tree-node");
    const chevron = nodeHeader ? nodeHeader.querySelector(".fs-chevron") : null;

    if (isExpanded) {
        fsState.expandedPaths.delete(path);
        if (chevron) chevron.classList.remove("expanded");
        const childrenDiv = nodeWrapper.querySelector(".fs-tree-children");
        if (childrenDiv) childrenDiv.remove();
    } else {
        fsState.expandedPaths.add(path);
        if (chevron) chevron.classList.add("expanded");

        const childrenDiv = document.createElement("div");
        childrenDiv.className = "fs-tree-children";
        nodeWrapper.appendChild(childrenDiv);

        await loadAndRenderChildren(path, childrenDiv, nextLevel);
    }
}

async function loadAndRenderChildren(path, containerEl, level) {
    containerEl.innerHTML = `<div class="fs-tree-loading" style="padding: 6px 12px; font-size: 11px; text-align: left;">Loading...</div>`;

    try {
        let entries = fsState.treeDataCache.get(path);
        if (!entries) {
            const res = await fetch("/api/fs/list", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ path: path })
            });
            if (!res.ok) throw new Error("Failed to load sub-directory");
            const data = await res.json();
            entries = data.entries;
            fsState.treeDataCache.set(path, entries);
        }

        containerEl.innerHTML = "";
        if (entries.length === 0) {
            containerEl.innerHTML = `<div class="fs-tree-empty" style="padding-left: ${level * 12 + 6}px;">(empty folder)</div>`;
            return;
        }

        entries.forEach(entry => {
            const childNode = renderTreeNode(entry, level);
            containerEl.appendChild(childNode);
        });
    } catch (err) {
        console.error(err);
        containerEl.innerHTML = `<div class="fs-tree-empty" style="color: #ff6b6b; padding-left: ${level * 12 + 6}px;">⚠️ Error loading</div>`;
    }
}

function selectFsItem(entry) {
    fsState.selectedPath = entry.path;
    fsState.selectedPathIsDir = entry.is_dir;
    if (dom.fsSelectedPath) {
        dom.fsSelectedPath.textContent = entry.path;
    }

    if (dom.fsExplorerTree) {
        const allNodes = dom.fsExplorerTree.querySelectorAll(".fs-tree-node");
        allNodes.forEach(node => {
            const wrapper = node.closest(".fs-tree-node-wrapper");
            if (wrapper && wrapper.dataset.path === entry.path) {
                node.classList.add("selected");
            } else {
                node.classList.remove("selected");
            }
        });
    }

    if (dom.btnFsConfirm) {
        if (fsState.selectorMode === "notes") {
            dom.btnFsConfirm.textContent = entry.is_dir ? "Select Folder" : "Open File";
        } else {
            dom.btnFsConfirm.textContent = "Load Model";
        }
    }

    renderDetailsPane(entry);
}

async function renderDetailsPane(entry) {
    const pane = dom.fsItemDetails;
    if (!pane) return;

    pane.innerHTML = `<div class="fs-tree-loading" style="padding: 10px;">Reading details...</div>`;

    if (!entry.is_dir) {
        if (fsState.selectorMode === "notes") {
            pane.innerHTML = `
                <div class="fs-details-card">
                    <div class="fs-details-title">${escapeHtml(entry.name)}</div>
                    <div class="fs-details-path">${escapeHtml(entry.path)}</div>
                    <div class="fs-model-badge valid" style="background: rgba(46, 213, 115, 0.15); color: #2ed573;">📄 Note File</div>
                    <div style="font-size: 12px; color: var(--text-tertiary); line-height: 1.6; margin-top: 10px;">
                        Select this file to open it in Mousepad. The notes directory will be set to its parent folder.
                    </div>
                </div>
            `;
            if (dom.btnFsConfirm) dom.btnFsConfirm.disabled = false;
        } else {
            pane.innerHTML = `
                <div class="fs-details-card">
                    <div class="fs-details-title">${escapeHtml(entry.name)}</div>
                    <div class="fs-details-path">${escapeHtml(entry.path)}</div>
                    <div class="fs-model-badge invalid">📄 File</div>
                    <div style="font-size: 12px; color: var(--text-tertiary); line-height: 1.6; margin-top: 10px;">
                        This is a file. OpenVINO models are loaded from directories. Please select the folder containing the model files.
                    </div>
                </div>
            `;
            if (dom.btnFsConfirm) dom.btnFsConfirm.disabled = true;
        }
        return;
    }

    try {
        let entries = fsState.treeDataCache.get(entry.path);
        if (!entries) {
            const res = await fetch("/api/fs/list", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ path: entry.path })
            });
            if (res.ok) {
                const data = await res.json();
                entries = data.entries;
                fsState.treeDataCache.set(entry.path, entries);
            }
        }

        const filesList = entries || [];
        
        let badgeHtml = "";
        let warningHtml = "";
        if (fsState.selectorMode === "notes") {
            badgeHtml = `<div class="fs-model-badge valid" style="background: rgba(46, 213, 115, 0.15); color: #2ed573;">📁 Notes Folder</div>`;
            warningHtml = `
                <div style="font-size: 11.5px; color: var(--text-secondary); line-height: 1.5; margin-top: 10px; padding: 10px; background: rgba(255, 255, 255, 0.02); border: 1px solid var(--border-subtle); border-radius: 6px;">
                    Select this directory to load its text/markdown notes into Mousepad.
                </div>
            `;
            if (dom.btnFsConfirm) dom.btnFsConfirm.disabled = false;
        } else {
            const xmlFiles = filesList.filter(f => !f.is_dir && f.name.endsWith(".xml"));
            const hasXmlModel = filesList.some(f => !f.is_dir && f.name.toLowerCase() === "openvino_model.xml");
            const isValidModel = xmlFiles.length > 0 || hasXmlModel;
            if (isValidModel) {
                badgeHtml = `<div class="fs-model-badge valid">🧠 OpenVINO Model Folder</div>`;
                if (dom.btnFsConfirm) dom.btnFsConfirm.disabled = false;
            } else {
                badgeHtml = `<div class="fs-model-badge invalid">⚠️ Regular Folder</div>`;
                warningHtml = `
                    <div style="font-size: 11.5px; color: #ffa502; line-height: 1.5; margin-top: 10px; padding: 10px; background: rgba(255, 165, 2, 0.05); border: 1px solid rgba(255, 165, 2, 0.15); border-radius: 6px;">
                        <strong>Note:</strong> No model XML files detected in this folder. Loading it as a model path might fail unless it contains valid config/model files.
                    </div>
                `;
                if (dom.btnFsConfirm) dom.btnFsConfirm.disabled = false;
            }

            let xmlOptionsHtml = '<option value="">Default (openvino_model.xml)</option>';
            xmlFiles.forEach(x => {
                if (x.name.toLowerCase() !== "openvino_model.xml") {
                    xmlOptionsHtml += `<option value="${escapeHtml(x.name)}">${escapeHtml(x.name)}</option>`;
                }
            });

            const configFormHtml = `
                <div class="model-config-form" style="margin-top: 15px; padding: 12px; background: rgba(255, 255, 255, 0.02); border: 1px solid var(--border-subtle); border-radius: 8px;">
                    <div style="font-weight: 600; font-size: 12px; margin-bottom: 8px; color: var(--text-primary); border-bottom: 1px solid var(--border-subtle); padding-bottom: 4px; display: flex; align-items: center; gap: 5px;">
                        <span>⚙️</span> OpenVINO Settings
                    </div>
                    
                    <div style="margin-bottom: 8px;">
                        <label style="display: block; font-size: 10px; color: var(--text-secondary); margin-bottom: 2px; text-transform: uppercase; letter-spacing: 0.5px;">Performance Hint</label>
                        <select id="model-config-perf" style="width: 100%; padding: 5px; background: var(--bg-secondary, #1e1e1e); border: 1px solid var(--border-subtle); color: var(--text-primary); border-radius: 4px; font-size: 11.5px; outline: none;">
                            <option value="LATENCY" selected>LATENCY (Recommended)</option>
                            <option value="THROUGHPUT">THROUGHPUT</option>
                            <option value="CUMULATIVE_THROUGHPUT">CUMULATIVE_THROUGHPUT</option>
                            <option value="NONE">None</option>
                        </select>
                    </div>
                    
                    <div style="margin-bottom: 8px;">
                        <label style="display: block; font-size: 10px; color: var(--text-secondary); margin-bottom: 2px; text-transform: uppercase; letter-spacing: 0.5px;">Cache Directory</label>
                        <input type="text" id="model-config-cache" value="./ov_cache" placeholder="e.g. ./ov_cache" style="width: 100%; padding: 5px; background: var(--bg-secondary, #1e1e1e); border: 1px solid var(--border-subtle); color: var(--text-primary); border-radius: 4px; font-size: 11.5px; box-sizing: border-box; outline: none;">
                    </div>
                    
                    <div style="margin-bottom: 8px;">
                        <label style="display: block; font-size: 10px; color: var(--text-secondary); margin-bottom: 2px; text-transform: uppercase; letter-spacing: 0.5px;">Model XML File</label>
                        <select id="model-config-file" style="width: 100%; padding: 5px; background: var(--bg-secondary, #1e1e1e); border: 1px solid var(--border-subtle); color: var(--text-primary); border-radius: 4px; font-size: 11.5px; outline: none;">
                            ${xmlOptionsHtml}
                        </select>
                    </div>

                    <div style="display: flex; flex-direction: column; gap: 6px; margin-top: 10px;">
                        <label style="display: flex; align-items: center; font-size: 11px; color: var(--text-secondary); cursor: pointer; user-select: none;">
                            <input type="checkbox" id="model-config-use-cache" checked style="margin-right: 6px; accent-color: var(--accent-primary, #007acc);"> Use Key-Value (KV) Cache
                        </label>
                        <label style="display: flex; align-items: center; font-size: 11px; color: var(--text-secondary); cursor: pointer; user-select: none;">
                            <input type="checkbox" id="model-config-trust-remote" checked style="margin-right: 6px; accent-color: var(--accent-primary, #007acc);"> Trust Remote Code
                        </label>
                        <label style="display: flex; align-items: center; font-size: 11px; color: var(--text-secondary); cursor: pointer; user-select: none;">
                            <input type="checkbox" id="model-config-fix-regex" checked style="margin-right: 6px; accent-color: var(--accent-primary, #007acc);"> Fix Mistral/Qwen Regex
                        </label>
                    </div>
                </div>
            `;
            warningHtml += configFormHtml;
        }

        let filesHtml = "";
        if (filesList.length > 0) {
            filesHtml = `
                <div style="margin-top: 15px;">
                    <div class="fs-files-list-title">Folder Contents (${filesList.length} items):</div>
                    <div class="fs-files-list">
            `;
            filesList.forEach(f => {
                const isModelCore = !f.is_dir && (f.name.endsWith(".xml") || f.name.endsWith(".bin") || f.name.endsWith(".json"));
                const icon = f.is_dir ? "📁" : "📄";
                filesHtml += `
                    <div class="fs-file-item ${isModelCore ? 'model-core' : ''}">
                        <span>${icon} ${escapeHtml(f.name)}</span>
                        <span>${f.is_dir ? 'dir' : 'file'}</span>
                    </div>
                `;
            });
            filesHtml += `</div></div>`;
        } else {
            filesHtml = `<div style="font-size: 12px; color: var(--text-tertiary); font-style: italic; margin-top: 12px;">Folder is empty.</div>`;
        }

        pane.innerHTML = `
            <div class="fs-details-card">
                <div class="fs-details-title">${escapeHtml(entry.name)}</div>
                <div class="fs-details-path">${escapeHtml(entry.path)}</div>
                ${badgeHtml}
                ${warningHtml}
                ${filesHtml}
            </div>
        `;
    } catch (err) {
        console.error(err);
        pane.innerHTML = `
            <div class="fs-details-card">
                <div class="fs-details-title">${escapeHtml(entry.name)}</div>
                <div class="fs-details-path">${escapeHtml(entry.path)}</div>
                <div class="fs-model-badge invalid">⚠️ Error Loading Details</div>
                <div style="font-size: 12px; color: #ff6b6b; margin-top: 10px;">
                    Could not list directory contents.
                </div>
            </div>
        `;
        if (dom.btnFsConfirm) dom.btnFsConfirm.disabled = false;
    }
}

async function handleFsConfirm() {
    if (!fsState.selectedPath) return;

    if (fsState.selectorMode === "notes") {
        let folderPath = fsState.selectedPath;
        let filenameToLoad = null;

        if (!fsState.selectedPathIsDir) {
            // It's a file!
            const lastSlash = fsState.selectedPath.lastIndexOf("/");
            if (lastSlash !== -1) {
                folderPath = fsState.selectedPath.substring(0, lastSlash);
                filenameToLoad = fsState.selectedPath.substring(lastSlash + 1);
            }
        }

        if (dom.notesDirInput) {
            dom.notesDirInput.value = folderPath;
        }
        closeModelBrowserModal();
        await saveNotesDirectory(folderPath, filenameToLoad);
    } else {
        await loadModelFromPath();
    }
}

async function loadModelFromPath() {
    if (!fsState.selectedPath) return;

    if (state.isGenerating) {
        showToast("Cannot switch model while generation is in progress. Please wait.", "error");
        return;
    }

    const targetModelPath = fsState.selectedPath;

    // Retrieve form settings from DOM
    const perfHintEl = document.getElementById("model-config-perf");
    const cacheDirEl = document.getElementById("model-config-cache");
    const modelFileEl = document.getElementById("model-config-file");
    const useCacheEl = document.getElementById("model-config-use-cache");
    const trustRemoteEl = document.getElementById("model-config-trust-remote");
    const fixRegexEl = document.getElementById("model-config-fix-regex");

    const payload = { model_path: targetModelPath };
    if (perfHintEl) payload.ov_performance_hint = perfHintEl.value;
    if (cacheDirEl) payload.ov_cache_dir = cacheDirEl.value.trim();
    if (modelFileEl) payload.model_file = modelFileEl.value;
    if (useCacheEl) payload.use_cache = useCacheEl.checked;
    if (trustRemoteEl) payload.trust_remote_code = trustRemoteEl.checked;
    if (fixRegexEl) payload.fix_mistral_regex = fixRegexEl.checked;

    if (dom.modelSwitchingTitle) {
        dom.modelSwitchingTitle.textContent = `Loading OpenVINO Model...`;
    }
    if (dom.modelSwitchingOverlay) {
        dom.modelSwitchingOverlay.classList.add("visible");
    }
    if (dom.btnFsConfirm) dom.btnFsConfirm.disabled = true;
    if (dom.btnFsCancel) dom.btnFsCancel.disabled = true;

    setStatus("loading", "Loading model...");
    if (dom.modelStatus) dom.modelStatus.textContent = "Loading...";

    try {
        const res = await fetch("/api/model/switch", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.message || errData.error || "Failed to switch model");
        }

        const data = await res.json();

        if (data.success) {
            showToast(data.message || `Loaded model: ${data.model_name}`, "success");
            // Show any warnings from the load process (e.g. use_cache=False fallback)
            if (data.warnings && data.warnings.length > 0) {
                data.warnings.forEach(w => showToast(w, "warning"));
            }
        } else {
            showToast(data.message || `Failed to load model`, "error");
        }

        await fetchConfig();
        closeModelBrowserModal();
    } catch (err) {
        console.error("Switch model error:", err);
        showToast(`Model load failed: ${err.message}`, "error");
        await fetchConfig();
    } finally {
        if (dom.modelSwitchingOverlay) {
            dom.modelSwitchingOverlay.classList.remove("visible");
        }
        if (dom.btnFsConfirm) dom.btnFsConfirm.disabled = false;
        if (dom.btnFsCancel) dom.btnFsCancel.disabled = false;
    }
}

// ── Model Downloader Functions ──────────────────────────────────────────────

function openModelDownloaderModal() {
    if (dom.btnDownloadModel) dom.btnDownloadModel.classList.remove("minimized-active");
    openOverlay(dom.modelDownloaderOverlay);
    // Check if there is an active running download on the server to resume monitoring
    checkActiveDownloadsAndResume();
}

function closeModelDownloaderModal() {
    closeOverlay(dom.modelDownloaderOverlay);
    if (dom.btnDownloadModel) dom.btnDownloadModel.classList.remove("minimized-active");
    if (dom.modelDownloaderModal) dom.modelDownloaderModal.classList.remove("fullscreen");
}

async function checkActiveDownloadsAndResume() {
    try {
        const res = await fetch("/api/models/download/status");
        const tasks = await res.json();
        
        // Find if there is any running task
        const runningTask = Object.values(tasks).find(t => t.status === "running");
        if (runningTask) {
            downloaderState.activeTaskId = runningTask.task_id;
            downloaderState.jobStatus = "running";
            resumeLogStreaming(runningTask.task_id);
        } else {
            // Update UI status if no task is running
            if (downloaderState.jobStatus === "idle") {
                updateDownloaderJobUI("idle", "No active job");
            }
        }
    } catch (err) {
        console.error("Failed to check active downloads:", err);
    }
}

async function handleDownloaderSearch() {
    const q = dom.downloaderSearchInput.value.trim();
    if (!q) return;
    
    dom.downloaderResultsContainer.innerHTML = `
        <div class="downloader-results-empty">
            <div class="device-switching-spinner" style="margin: 0 auto 10px;"></div>
            Searching Hugging Face...
        </div>
    `;
    
    try {
        const res = await fetch(`/api/models/search?q=${encodeURIComponent(q)}`);
        const data = await res.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        renderDownloaderSearchResults(data);
    } catch (err) {
        console.error(err);
        dom.downloaderResultsContainer.innerHTML = `
            <div class="downloader-results-empty" style="color: #ff6b6b;">
                Failed to search: ${err.message}
            </div>
        `;
    }
}

function renderDownloaderSearchResults(results) {
    const container = dom.downloaderResultsContainer;
    container.innerHTML = "";
    
    if (results.length === 0) {
        container.innerHTML = `
            <div class="downloader-results-empty">
                No matching models found. Try another search.
            </div>
        `;
        return;
    }
    
    results.forEach(model => {
        const card = document.createElement("div");
        card.className = "downloader-result-card";
        
        const downloadsK = model.downloads >= 1000 ? (model.downloads / 1000).toFixed(1) + "k" : model.downloads;
        
        card.innerHTML = `
            <div class="downloader-result-title">${escapeHtml(model.model_id)}</div>
            <div class="downloader-result-meta">
                <span class="downloader-meta-item">📥 ${downloadsK} downloads</span>
                <span class="downloader-meta-item">❤️ ${model.likes} likes</span>
                ${model.author ? `<span class="downloader-meta-item">👤 ${escapeHtml(model.author)}</span>` : ""}
            </div>
        `;
        
        card.addEventListener("click", () => {
            // Select card
            $$(".downloader-result-card").forEach(c => c.classList.remove("selected"));
            card.classList.add("selected");
            
            // Populate fields
            dom.downloaderModelId.value = model.model_id;
            
            // Suggest output directory name (slugify)
            const slug = model.model_id.toLowerCase()
                .replace(/[^a-z0-9]/g, "-")
                .replace(/-+/g, "-")
                .replace(/^-|-$/g, "");
            dom.downloaderOutputDir.value = slug + "-ov";
            
            // Trigger input validation
            dom.downloaderModelId.dispatchEvent(new Event("input"));
        });
        
        container.appendChild(card);
    });
}

function updateDownloaderJobUI(status, text) {
    downloaderState.jobStatus = status;
    
    const dot = dom.downloaderJobDot;
    const label = dom.downloaderJobStatusText;
    
    dot.style.display = status === "idle" ? "none" : "inline-block";
    label.textContent = text;
    
    if (status === "running") {
        dot.className = "status-dot loading";
        dom.downloaderProgressWrap.style.display = "block";
        dom.downloaderProgressFill.className = "downloader-progress-fill running";
        dom.downloaderProgressFill.style.width = "100%";
        dom.downloaderProgressText.textContent = "Exporting model...";
        dom.btnDownloaderCancel.disabled = false;
        dom.btnDownloaderStart.disabled = true;
        dom.btnDownloaderStart.textContent = "Exporting...";
    } else if (status === "completed") {
        dot.className = "status-dot";
        dot.style.background = "#44b700";
        dot.style.boxShadow = "0 0 8px rgba(68,183,0,0.5)";
        dom.downloaderProgressWrap.style.display = "block";
        dom.downloaderProgressFill.className = "downloader-progress-fill";
        dom.downloaderProgressFill.style.width = "100%";
        dom.downloaderProgressText.textContent = "Success! Export completed.";
        dom.btnDownloaderCancel.disabled = true;
        dom.btnDownloaderStart.disabled = false;
        dom.btnDownloaderStart.textContent = "Load Model";
    } else if (status === "failed") {
        dot.className = "status-dot error";
        dom.downloaderProgressWrap.style.display = "block";
        dom.downloaderProgressFill.className = "downloader-progress-fill";
        dom.downloaderProgressFill.style.width = "0%";
        dom.downloaderProgressText.textContent = "Export failed.";
        dom.btnDownloaderCancel.disabled = true;
        dom.btnDownloaderStart.disabled = false;
        dom.btnDownloaderStart.textContent = "Start Export";
    } else {
        // idle
        dom.downloaderProgressWrap.style.display = "none";
        dom.btnDownloaderCancel.disabled = true;
        dom.btnDownloaderStart.disabled = !dom.downloaderModelId.value.trim() || !dom.downloaderOutputDir.value.trim();
        dom.btnDownloaderStart.textContent = "Start Export";
    }
}

async function handleStartExport() {
    // If job was completed, click loads the model immediately
    if (downloaderState.jobStatus === "completed") {
        const outputDir = dom.downloaderOutputDir.value.trim();
        if (outputDir) {
            closeModelDownloaderModal();
            
            // Show load loader overlay
            if (dom.modelSwitchingOverlay) {
                dom.modelSwitchingOverlay.classList.add("visible");
                dom.modelSwitchingTitle.textContent = `Loading OpenVINO Model...`;
            }
            
            try {
                // Fetch workspace path
                const fsRes = await fetch("/api/fs/list", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ path: "" })
                });
                const fsData = await fsRes.json();
                const workspacePath = fsData.workspace_path || ".";
                const fullModelPath = `${workspacePath}/${outputDir}`;
                
                const res = await fetch("/api/model/switch", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ model_path: fullModelPath })
                });
                const data = await res.json();
                if (data.success) {
                    showToast(`Successfully loaded downloaded model!`, "success");
                    // Show any warnings from the load process
                    if (data.warnings && data.warnings.length > 0) {
                        data.warnings.forEach(w => showToast(w, "warning"));
                    }
                    await fetchConfig();
                } else {
                    showToast(`Failed to load model: ${data.message}`, "error");
                }
            } catch (err) {
                showToast(`Failed to switch: ${err.message}`, "error");
            } finally {
                if (dom.modelSwitchingOverlay) {
                    dom.modelSwitchingOverlay.classList.remove("visible");
                }
            }
        }
        return;
    }

    const modelId = dom.downloaderModelId.value.trim();
    const weightFormat = dom.downloaderWeightFormat.value;
    const task = dom.downloaderTask.value;
    const outputDir = dom.downloaderOutputDir.value.trim();
    const hfToken = dom.downloaderHfToken ? dom.downloaderHfToken.value.trim() : "";
    
    // Persist HF token
    localStorage.setItem("hf_token", hfToken);
    
    if (!modelId || !outputDir) {
        showToast("Please fill in all model details.", "error");
        return;
    }
    
    updateDownloaderJobUI("running", "Starting...");
    dom.downloaderConsoleOutput.innerHTML = `<div style="color: var(--accent-2)">[SYSTEM] Requesting export for ${modelId}...</div>\n`;
    
    try {
        const res = await fetch("/api/models/download", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model_id: modelId,
                weight_format: weightFormat,
                task: task,
                output_dir: outputDir,
                hf_token: hfToken
            })
        });
        
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.error || "Failed to start export");
        }
        
        downloaderState.activeTaskId = data.task_id;
        showToast("Export process started in background", "success");
        resumeLogStreaming(data.task_id);
    } catch (err) {
        console.error(err);
        showToast(err.message, "error");
        dom.downloaderConsoleOutput.innerHTML += `<div style="color: #ff6b6b;">[SYSTEM ERROR] Failed to start job: ${err.message}</div>\n`;
        updateDownloaderJobUI("failed", "Failed to start");
    }
}

function resumeLogStreaming(taskId) {
    if (downloaderState.eventSource) {
        downloaderState.eventSource.close();
    }
    
    const consoleEl = dom.downloaderConsoleOutput;
    consoleEl.innerHTML = `<div style="color: var(--accent-2)">[SYSTEM] Connected to export process logs...</div>\n`;
    
    updateDownloaderJobUI("running", "Exporting...");
    
    const ev = new EventSource(`/api/models/download/stream/${taskId}`);
    downloaderState.eventSource = ev;
    
    ev.onmessage = (event) => {
        if (event.data === "[DONE]") {
            ev.close();
            downloaderState.eventSource = null;
            checkTaskStatus(taskId);
            return;
        }
        
        try {
            const data = JSON.parse(event.data);
            
            const line = document.createElement("div");
            line.textContent = data.log;
            
            if (data.log.toLowerCase().includes("error") || data.log.toLowerCase().includes("failed")) {
                line.style.color = "#ff6b6b";
            } else if (data.log.toLowerCase().includes("successfully") || data.log.toLowerCase().includes("completed")) {
                line.style.color = "#2ed573";
            } else if (data.log.startsWith("Starting export command")) {
                line.style.color = "var(--accent-3)";
                line.style.fontWeight = "bold";
            }
            
            consoleEl.appendChild(line);
            consoleEl.scrollTop = consoleEl.scrollHeight;
            
            if (data.status && data.status !== downloaderState.jobStatus) {
                updateDownloaderJobUI(data.status, data.status === "completed" ? "Completed" : data.status === "failed" ? "Failed" : "Exporting...");
            }
        } catch (err) {
            console.error("Error parsing SSE log stream data:", err);
        }
    };
    
    ev.onerror = () => {
        console.error("SSE connection error for task log streaming");
        ev.close();
        downloaderState.eventSource = null;
        checkTaskStatus(taskId);
    };
}

async function checkTaskStatus(taskId) {
    try {
        const res = await fetch("/api/models/download/status");
        const tasks = await res.json();
        const task = tasks[taskId];
        if (task) {
            updateDownloaderJobUI(task.status, task.status === "completed" ? "Completed" : task.status === "failed" ? "Failed" : "Exporting...");
            if (task.error) {
                dom.downloaderConsoleOutput.innerHTML += `<div style="color: #ff6b6b;">[SYSTEM ERROR] ${escapeHtml(task.error)}</div>\n`;
                dom.downloaderConsoleOutput.scrollTop = dom.downloaderConsoleOutput.scrollHeight;
            }
        }
    } catch (err) {
        console.error("Failed to check task status:", err);
    }
}

async function handleCancelExport() {
    const taskId = downloaderState.activeTaskId;
    if (!taskId) return;
    
    dom.downloaderConsoleOutput.innerHTML += `<div style="color: #ffa502;">[SYSTEM] Requesting cancellation...</div>\n`;
    dom.downloaderConsoleOutput.scrollTop = dom.downloaderConsoleOutput.scrollHeight;
    
    try {
        const res = await fetch(`/api/models/download/cancel/${taskId}`, {
            method: "POST"
        });
        const data = await res.json();
        
        if (data.success) {
            showToast("Export process cancelled", "info");
            if (downloaderState.eventSource) {
                downloaderState.eventSource.close();
                downloaderState.eventSource = null;
            }
            updateDownloaderJobUI("failed", "Cancelled");
        } else {
            showToast(data.message || "Failed to cancel process", "error");
        }
    } catch (err) {
        console.error(err);
        showToast(`Failed to cancel: ${err.message}`, "error");
    }
}

// ── Window Controls Helpers (macOS style Minimize & Fullscreen) ──────────────────

function setupWindowControlsListeners() {
    // Settings Modal
    if (dom.settingsModalMinimize) {
        dom.settingsModalMinimize.addEventListener("click", () => minimizeWindow("settings"));
    }
    if (dom.settingsModalFullscreen) {
        dom.settingsModalFullscreen.addEventListener("click", () => toggleFullscreen("settings"));
    }

    // Help Modal
    if (dom.helpModalMinimize) {
        dom.helpModalMinimize.addEventListener("click", () => minimizeWindow("help"));
    }
    if (dom.helpModalFullscreen) {
        dom.helpModalFullscreen.addEventListener("click", () => toggleFullscreen("help"));
    }

    // Model Browser Modal
    if (dom.modelBrowserMinimize) {
        dom.modelBrowserMinimize.addEventListener("click", () => minimizeWindow("model-browser"));
    }
    if (dom.modelBrowserFullscreen) {
        dom.modelBrowserFullscreen.addEventListener("click", () => toggleFullscreen("model-browser"));
    }

    // Model Downloader Modal
    if (dom.modelDownloaderMinimize) {
        dom.modelDownloaderMinimize.addEventListener("click", () => minimizeWindow("model-downloader"));
    }
    if (dom.modelDownloaderFullscreen) {
        dom.modelDownloaderFullscreen.addEventListener("click", () => toggleFullscreen("model-downloader"));
    }

    // Context Popup
    if (dom.contextPopupMinimize) {
        dom.contextPopupMinimize.addEventListener("click", () => minimizeWindow("context"));
    }
    if (dom.contextPopupFullscreen) {
        dom.contextPopupFullscreen.addEventListener("click", () => toggleFullscreen("context"));
    }

    // Notes Modal Window Controls
    if (dom.notesMinimize) {
        dom.notesMinimize.addEventListener("click", () => minimizeWindow("notes"));
    }
    if (dom.notesFullscreen) {
        dom.notesFullscreen.addEventListener("click", () => toggleFullscreen("notes"));
    }
}

function minimizeWindow(type) {
    let modalEl = null;
    let overlayEl = null;
    let triggerBtn = null;
    let closeFunc = null;

    if (type === "settings") {
        modalEl = dom.settingsModal;
        overlayEl = dom.settingsModalOverlay;
        triggerBtn = dom.btnSettings;
        closeFunc = () => {
            closeOverlay(overlayEl);
            if (modalEl) modalEl.classList.remove("fullscreen");
        };
    } else if (type === "help") {
        modalEl = dom.helpModal;
        overlayEl = dom.helpModalOverlay;
        triggerBtn = dom.btnHelp;
        closeFunc = () => {
            closeOverlay(overlayEl);
            if (modalEl) modalEl.classList.remove("fullscreen");
        };
    } else if (type === "model-browser") {
        modalEl = dom.modelBrowserModal;
        overlayEl = dom.modelBrowserOverlay;
        triggerBtn = dom.btnSwitchModel;
        closeFunc = () => {
            closeOverlay(overlayEl);
            if (modalEl) modalEl.classList.remove("fullscreen");
            
            // Restore default texts
            const titleEl = document.getElementById("model-browser-title-text");
            if (titleEl) titleEl.textContent = "Select Local OpenVINO Model";
            if (dom.btnFsConfirm) dom.btnFsConfirm.textContent = "Load Model";
            fsState.selectorMode = "model";
        };
    } else if (type === "model-downloader") {
        modalEl = dom.modelDownloaderModal;
        overlayEl = dom.modelDownloaderOverlay;
        triggerBtn = dom.btnDownloadModel;
        closeFunc = () => {
            closeOverlay(overlayEl);
            if (modalEl) modalEl.classList.remove("fullscreen");
        };
    } else if (type === "context") {
        modalEl = dom.contextPopup;
        triggerBtn = dom.contextWindow;
        closeFunc = () => {
            if (modalEl) {
                modalEl.classList.remove("visible");
                modalEl.classList.remove("fullscreen");
            }
        };
    } else if (type === "notes") {
        modalEl = dom.notesModal;
        overlayEl = dom.notesOverlay;
        triggerBtn = dom.btnNotes;
        closeFunc = () => {
            closeOverlay(overlayEl);
            if (modalEl) modalEl.classList.remove("fullscreen");
        };
    }

    if (!modalEl) return;

    modalEl.classList.add("minimizing");
    if (triggerBtn) triggerBtn.classList.add("minimized-active");

    setTimeout(() => {
        closeFunc();
        modalEl.classList.remove("minimizing");
    }, 400);
}

function toggleFullscreen(type) {
    let el = null;
    if (type === "settings") el = dom.settingsModal;
    else if (type === "help") el = dom.helpModal;
    else if (type === "model-browser") el = dom.modelBrowserModal;
    else if (type === "model-downloader") el = dom.modelDownloaderModal;
    else if (type === "context") el = dom.contextPopup;
    else if (type === "notes") el = dom.notesModal;

    if (!el) return;

    const isEntering = !el.classList.contains("fullscreen");
    const isDragged = !!(el.style.left && el.style.top);

    if (isEntering) {
        if (!isDragged) {
            const rect = el.getBoundingClientRect();
            el.style.position = "fixed";
            el.style.margin = "0";
            el.style.transform = "none";
            el.style.left = `${rect.left}px`;
            el.style.top = `${rect.top}px`;
            el.style.width = `${rect.width}px`;
            el.style.height = `${rect.height}px`;
            el.dataset.wasAutoFrozen = "true";
        }
        
        // Force reflow
        el.offsetHeight;
        
        el.classList.add("fullscreen");
    } else {
        el.classList.remove("fullscreen");
        
        const wasAutoFrozen = el.dataset.wasAutoFrozen === "true";
        if (wasAutoFrozen) {
            const handleTransitionEnd = (e) => {
                if (e.propertyName === "width" || e.propertyName === "height") {
                    el.removeEventListener("transitionend", handleTransitionEnd);
                    if (!el.classList.contains("fullscreen") && el.dataset.wasAutoFrozen === "true") {
                        el.style.position = "";
                        el.style.margin = "";
                        el.style.transform = "";
                        el.style.left = "";
                        el.style.top = "";
                        el.style.width = "";
                        el.style.height = "";
                        delete el.dataset.wasAutoFrozen;
                    }
                }
            };
            el.addEventListener("transitionend", handleTransitionEnd);
        }
    }
}

// ── Theme Management (Light / Dark mode) ───────────────────────────────────

function initTheme() {
    const savedTheme = localStorage.getItem("theme") || "dark";
    if (savedTheme === "light") {
        document.body.classList.add("light-theme");
        updateThemeToggleUI("light");
    } else {
        document.body.classList.remove("light-theme");
        updateThemeToggleUI("dark");
    }
}

function toggleTheme() {
    if (document.body.classList.contains("light-theme")) {
        document.body.classList.remove("light-theme");
        localStorage.setItem("theme", "dark");
        updateThemeToggleUI("dark");
        showToast("Switched to dark mode", "info");
    } else {
        document.body.classList.add("light-theme");
        localStorage.setItem("theme", "light");
        updateThemeToggleUI("light");
        showToast("Switched to light mode", "info");
    }
}

function updateThemeToggleUI(theme) {
    const btn = document.getElementById("btn-theme");
    if (!btn) return;
    if (theme === "light") {
        btn.querySelector(".header-action-icon").textContent = "☀️";
        btn.title = "Switch to Dark Mode";
    } else {
        btn.querySelector(".header-action-icon").textContent = "🌙";
        btn.title = "Switch to Light Mode";
    }
}

// ── Command Palette Management ──────────────────────────────────────────────

const paletteState = {
    isOpen: false,
    selectedIndex: 0,
    items: [],
};

const COMMANDS = [
    { id: "new-chat", title: "New Chat", icon: "💬", shortcut: "Ctrl+N", action: () => createNewConversation() },
    { id: "open-settings", title: "Open Settings", icon: "⚙️", shortcut: "", action: () => openSettingsModal() },
    { id: "open-help", title: "Open Help & Features", icon: "📖", shortcut: "?", action: () => openHelpModal() },
    { id: "switch-model", title: "Switch Active Model", icon: "📁", shortcut: "", action: () => openModelBrowserModal() },
    { id: "download-model", title: "Download Model", icon: "📥", shortcut: "", action: () => openModelDownloaderModal() },
    { id: "toggle-theme", title: "Toggle Theme (Light / Dark)", icon: "🌓", shortcut: "", action: () => toggleTheme() },
    { id: "clear-chat", title: "Clear current conversation", icon: "🧹", shortcut: "", action: () => clearCurrentChat() },
];

function openPalette() {
    if (dom.paletteOverlay) {
        dom.paletteOverlay.classList.add("visible");
    }
    paletteState.isOpen = true;
    paletteState.selectedIndex = 0;
    if (dom.paletteSearchInput) {
        dom.paletteSearchInput.value = "";
    }
    renderPaletteResults();
    setTimeout(() => {
        if (dom.paletteSearchInput) {
            dom.paletteSearchInput.focus();
        }
    }, 50);
}

function closePalette() {
    if (dom.paletteOverlay) {
        dom.paletteOverlay.classList.remove("visible");
    }
    paletteState.isOpen = false;
}

function togglePalette() {
    if (paletteState.isOpen) {
        closePalette();
    } else {
        openPalette();
    }
}

function renderPaletteResults() {
    const query = dom.paletteSearchInput ? dom.paletteSearchInput.value.toLowerCase().trim() : "";
    const container = dom.paletteResults;
    if (!container) return;
    container.innerHTML = "";

    paletteState.items = [];

    // Filter commands
    const matchingCommands = COMMANDS.filter(cmd => cmd.title.toLowerCase().includes(query));
    matchingCommands.forEach(cmd => {
        paletteState.items.push({ type: "command", data: cmd });
    });

    // Filter conversations
    const matchingConversations = state.conversations.filter(conv => conv.title.toLowerCase().includes(query));
    matchingConversations.forEach(conv => {
        paletteState.items.push({ type: "conversation", data: conv });
    });

    if (paletteState.items.length === 0) {
        container.innerHTML = `
            <div class="palette-results-empty">
                <span class="palette-empty-icon">🔍</span>
                <div>No results found for "${escapeHtml(query)}"</div>
            </div>
        `;
        return;
    }

    // Keep selected index in bounds
    if (paletteState.selectedIndex >= paletteState.items.length) {
        paletteState.selectedIndex = 0;
    } else if (paletteState.selectedIndex < 0) {
        paletteState.selectedIndex = paletteState.items.length - 1;
    }

    // Render results
    let currentIdx = 0;

    if (matchingCommands.length > 0) {
        const header = document.createElement("div");
        header.className = "palette-section-header";
        header.textContent = "Commands";
        container.appendChild(header);

        matchingCommands.forEach(cmd => {
            const el = document.createElement("div");
            const isSelected = currentIdx === paletteState.selectedIndex;
            el.className = `palette-item ${isSelected ? "active" : ""}`;
            el.innerHTML = `
                <span class="palette-item-icon">${cmd.icon}</span>
                <span class="palette-item-text">${escapeHtml(cmd.title)}</span>
                ${cmd.shortcut ? `<span class="palette-item-shortcut">${cmd.shortcut}</span>` : ""}
            `;
            const thisIdx = currentIdx;
            el.addEventListener("click", () => {
                paletteState.selectedIndex = thisIdx;
                selectActivePaletteItem();
            });
            container.appendChild(el);
            currentIdx++;
        });
    }

    if (matchingConversations.length > 0) {
        const header = document.createElement("div");
        header.className = "palette-section-header";
        header.textContent = "Conversations";
        container.appendChild(header);

        matchingConversations.forEach(conv => {
            const el = document.createElement("div");
            const isSelected = currentIdx === paletteState.selectedIndex;
            el.className = `palette-item ${isSelected ? "active" : ""}`;
            el.innerHTML = `
                <span class="palette-item-icon">💬</span>
                <span class="palette-item-text">${escapeHtml(conv.title)}</span>
            `;
            const thisIdx = currentIdx;
            el.addEventListener("click", () => {
                paletteState.selectedIndex = thisIdx;
                selectActivePaletteItem();
            });
            container.appendChild(el);
            currentIdx++;
        });
    }
}

function updateActivePaletteItem() {
    if (!dom.paletteResults) return;
    const items = dom.paletteResults.querySelectorAll(".palette-item");
    items.forEach((el, idx) => {
        if (idx === paletteState.selectedIndex) {
            el.classList.add("active");
            el.scrollIntoView({ block: "nearest" });
        } else {
            el.classList.remove("active");
        }
    });
}

function navigatePalette(dir) {
    if (paletteState.items.length === 0) return;
    paletteState.selectedIndex = (paletteState.selectedIndex + dir + paletteState.items.length) % paletteState.items.length;
    updateActivePaletteItem();
}

function selectActivePaletteItem() {
    const activeItem = paletteState.items[paletteState.selectedIndex];
    if (activeItem) {
        triggerPaletteItem(activeItem);
    }
}

function triggerPaletteItem(item) {
    closePalette();
    if (item.type === "command") {
        item.data.action();
    } else if (item.type === "conversation") {
        switchConversation(item.data.id);
    }
}

function clearCurrentChat() {
    const conv = getActiveConversation();
    if (conv) {
        conv.messages = [];
        saveState();
        renderActiveConversation();
        updateContextWindowUI(0, state.maxInputTokens);
        showToast("Conversation cleared", "info");
    } else {
        showToast("No active conversation to clear", "warning");
    }
}


// ── Mousepad Notes Handlers ──────────────────────────────────────────────────

// ── Mousepad Notes Handlers & Tab State ──────────────────────────────────────

const notesTabState = {
    openTabs: [], // Array of { filename, content, originalContent, isUnsaved }
    activeFilename: null
};

async function openNotesModal() {
    // If already visible, just bring to front
    if (dom.notesOverlay && dom.notesOverlay.classList.contains("visible")) {
        bringToFront(dom.notesOverlay);
        return;
    }

    if (dom.btnNotes) {
        if (dom.btnNotes.classList.contains("minimized-active")) {
            dom.btnNotes.classList.remove("minimized-active");
            if (dom.notesOverlay) {
                openOverlay(dom.notesOverlay);
            }
            return;
        }
        dom.btnNotes.classList.remove("minimized-active");
    }
    
    // Clear tabs first and reset
    notesTabState.openTabs = [];
    notesTabState.activeFilename = null;
    
    if (dom.notesSearch) dom.notesSearch.value = "";
    if (dom.notesSearchClear) dom.notesSearchClear.style.display = "none";
    if (dom.notesDirInput) dom.notesDirInput.value = "";
    
    updateNotesCounters();
    
    if (dom.notesOverlay) {
        openOverlay(dom.notesOverlay);
    }
    
    // Fetch current notes directory path
    try {
        const res = await fetch("/api/notes/get_directory");
        if (res.ok) {
            const data = await res.json();
            if (dom.notesDirInput) {
                dom.notesDirInput.value = data.path;
            }
            if (dom.notesCurrentDirDisplay) {
                dom.notesCurrentDirDisplay.textContent = data.path;
            }
        }
    } catch (err) {
        console.error("Failed to load notes directory:", err);
    }
    
    await fetchNotesList();
    
    // Select first note if available, else new tab
    if (notesState.notes.length > 0) {
        await openNoteInTab(notesState.notes[0].name);
    } else {
        createNewNoteTab();
    }

    if (dom.chkSyntaxHighlight) dom.chkSyntaxHighlight.checked = notesState.syntaxHighlightOn;
    if (dom.selTabSize) dom.selTabSize.value = notesState.tabSize;
    updateHighlightView();
    syncHighlight();
}

function closeNotesModal() {
    const hasAnyUnsaved = notesTabState.openTabs.some(t => t.isUnsaved);
    if (hasAnyUnsaved) {
        const confirmClose = confirm("You have unsaved changes in one or more tabs. Are you sure you want to close and discard changes?");
        if (!confirmClose) return;
    }
    
    if (dom.notesOverlay) {
        closeOverlay(dom.notesOverlay);
    }
    if (dom.btnNotes) dom.btnNotes.classList.remove("minimized-active");
    if (dom.notesModal) dom.notesModal.classList.remove("fullscreen");
}

function toggleNotesModal() {
    if (dom.notesOverlay && dom.notesOverlay.classList.contains("visible")) {
        closeNotesModal();
    } else {
        openNotesModal();
    }
}

async function fetchNotesList() {
    try {
        const res = await fetch("/api/notes/list");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        
        const data = await res.json();
        notesState.notes = data;
        renderNotesList();
    } catch (err) {
        console.error("Failed to fetch notes list:", err);
        showToast("Failed to load notes from server", "error");
    }
}

function renderNotesList(filterQuery = "") {
    // Hidden sidebar placeholder to prevent reference errors
}

async function loadNote(filename) {
    await openNoteInTab(filename);
}

async function openNoteInTab(filename) {
    // Check if already open
    const existing = notesTabState.openTabs.find(t => t.filename === filename);
    if (existing) {
        switchTab(filename);
        return;
    }
    
    try {
        const res = await fetch(`/api/notes/get?filename=${encodeURIComponent(filename)}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        
        const data = await res.json();
        
        // Add to open tabs
        const newTab = {
            filename: data.filename,
            content: data.content,
            originalContent: data.content,
            isUnsaved: false
        };
        notesTabState.openTabs.push(newTab);
        switchTab(data.filename);
    } catch (err) {
        console.error("Failed to load note content:", err);
        showToast(`Failed to load note '${filename}'`, "error");
    }
}

function switchTab(filename) {
    if (notesTabState.activeFilename === filename) return;
    
    // Save current textarea changes to current active tab state before switching
    if (notesTabState.activeFilename) {
        const activeTab = notesTabState.openTabs.find(t => t.filename === notesTabState.activeFilename);
        if (activeTab) {
            activeTab.content = dom.notesTextarea ? dom.notesTextarea.value : "";
        }
    }
    
    notesTabState.activeFilename = filename;
    const targetTab = notesTabState.openTabs.find(t => t.filename === filename);
    if (targetTab) {
        notesState.activeNoteName = targetTab.filename;
        notesState.currentFilename = targetTab.filename;
        notesState.currentContent = targetTab.content;
        notesState.originalFilename = targetTab.filename;
        notesState.originalContent = targetTab.originalContent;
        notesState.isUnsaved = targetTab.isUnsaved;
        
        if (dom.notesTextarea) dom.notesTextarea.value = targetTab.content;
        
        updateNotesCounters();
        checkUnsavedChanges();
        syncHighlight();
        renderTabs();
    }
}

function createNewNoteTab() {
    let baseNum = 1;
    let newFilename = "untitled.txt";
    while (notesTabState.openTabs.some(t => t.filename === newFilename) || notesState.notes.some(n => n.name === newFilename)) {
        newFilename = `untitled_${baseNum}.txt`;
        baseNum++;
    }
    
    const newTab = {
        filename: newFilename,
        content: "",
        originalContent: "",
        isUnsaved: true
    };
    notesTabState.openTabs.push(newTab);
    switchTab(newFilename);
}

function createNewNote() {
    createNewNoteTab();
}

function closeTab(filename) {
    const tabIndex = notesTabState.openTabs.findIndex(t => t.filename === filename);
    if (tabIndex === -1) return;
    
    const tab = notesTabState.openTabs[tabIndex];
    if (tab.isUnsaved) {
        const proceed = confirm(`Discard unsaved changes in '${filename}'?`);
        if (!proceed) return;
    }
    
    notesTabState.openTabs.splice(tabIndex, 1);
    
    // If closed the active tab, switch to another
    if (notesTabState.activeFilename === filename) {
        if (notesTabState.openTabs.length > 0) {
            const nextActiveIndex = Math.min(tabIndex, notesTabState.openTabs.length - 1);
            const nextActiveFilename = notesTabState.openTabs[nextActiveIndex].filename;
            notesTabState.activeFilename = null;
            switchTab(nextActiveFilename);
        } else {
            notesTabState.activeFilename = null;
            createNewNoteTab();
        }
    } else {
        renderTabs();
    }
}

function renderTabs() {
    const tabContainer = dom.notesTabsBar;
    if (!tabContainer) return;
    
    // Clear all children except the add button
    const tabs = tabContainer.querySelectorAll(".notes-tab");
    tabs.forEach(t => t.remove());
    
    const addBtn = dom.btnNotesTabAdd;
    
    notesTabState.openTabs.forEach(tab => {
        const isActive = tab.filename === notesTabState.activeFilename;
        const tabEl = document.createElement("div");
        tabEl.className = `notes-tab ${isActive ? 'active' : ''} ${tab.isUnsaved ? 'modified' : ''}`;
        tabEl.dataset.filename = tab.filename;
        
        tabEl.innerHTML = `
            <span class="notes-tab-title" title="${escapeHtml(tab.filename)}">${escapeHtml(tab.filename)}</span>
            <span class="notes-tab-close" title="Close Tab">✕</span>
        `;
        
        tabEl.addEventListener("click", (e) => {
            if (e.target.classList.contains("notes-tab-close")) {
                e.stopPropagation();
                closeTab(tab.filename);
            } else {
                switchTab(tab.filename);
            }
        });
        
        tabEl.addEventListener("dblclick", (e) => {
            e.stopPropagation();
            renameActiveNote();
        });
        
        tabContainer.insertBefore(tabEl, addBtn);
    });
}

async function saveActiveNote() {
    let filename = notesTabState.activeFilename || "untitled.txt";
    if (filename.startsWith("untitled")) {
        const newFilename = prompt("Save Note - Enter filename:", filename);
        if (newFilename === null) return; // Cancelled
        filename = newFilename.trim();
        if (!filename) {
            showToast("Filename cannot be empty", "warning");
            return;
        }
        if (!filename.endsWith(".txt") && !filename.endsWith(".md")) {
            filename += ".txt";
        }
    }
    
    const content = dom.notesTextarea ? dom.notesTextarea.value : "";
    
    if (filename.includes("/") || filename.includes("\\") || filename.includes("..")) {
        showToast("Invalid filename. Subdirectories are not allowed.", "error");
        return;
    }
    
    try {
        const res = await fetch("/api/notes/save", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename, content })
        });
        
        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.error || `HTTP ${res.status}`);
        }
        
        const data = await res.json();
        showToast(data.message || "Note saved successfully!", "success");
        
        const activeTab = notesTabState.openTabs.find(t => t.filename === notesTabState.activeFilename);
        if (activeTab) {
            activeTab.filename = data.filename;
            activeTab.content = content;
            activeTab.originalContent = content;
            activeTab.isUnsaved = false;
        }
        
        notesTabState.activeFilename = data.filename;
        notesState.activeNoteName = data.filename;
        notesState.currentFilename = data.filename;
        notesState.currentContent = content;
        notesState.originalFilename = data.filename;
        notesState.originalContent = content;
        notesState.isUnsaved = false;
        
        checkUnsavedChanges();
        syncHighlight();
        
        await fetchNotesList();
        renderTabs();
    } catch (err) {
        console.error("Failed to save note:", err);
        showToast(`Failed to save note: ${err.message}`, "error");
    }
}

async function saveNoteAs() {
    const currentName = notesTabState.activeFilename || "untitled.txt";
    let newFilename = prompt("Save As - Enter new filename:", currentName);
    if (newFilename === null) return; // User cancelled
    
    newFilename = newFilename.trim();
    if (!newFilename) {
        showToast("Filename cannot be empty", "warning");
        return;
    }
    
    if (!newFilename.endsWith(".txt") && !newFilename.endsWith(".md")) {
        newFilename += ".txt";
    }
    
    const content = dom.notesTextarea ? dom.notesTextarea.value : "";
    
    try {
        const res = await fetch("/api/notes/save", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename: newFilename, content })
        });
        
        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.error || `HTTP ${res.status}`);
        }
        
        const data = await res.json();
        showToast(data.message || "Note saved as " + data.filename, "success");
        
        const activeTab = notesTabState.openTabs.find(t => t.filename === notesTabState.activeFilename);
        if (activeTab) {
            activeTab.filename = data.filename;
            activeTab.content = content;
            activeTab.originalContent = content;
            activeTab.isUnsaved = false;
        }
        
        notesTabState.activeFilename = data.filename;
        notesState.activeNoteName = data.filename;
        notesState.currentFilename = data.filename;
        notesState.currentContent = content;
        notesState.originalFilename = data.filename;
        notesState.originalContent = content;
        notesState.isUnsaved = false;
        
        checkUnsavedChanges();
        syncHighlight();
        
        await fetchNotesList();
        renderTabs();
    } catch (err) {
        console.error("Save As failed:", err);
        showToast(`Save As failed: ${err.message}`, "error");
    }
}

async function renameActiveNote() {
    const currentName = notesTabState.activeFilename;
    if (!currentName) {
        showToast("No active note to rename", "warning");
        return;
    }
    
    let newFilename = prompt("Rename Note - Enter new filename:", currentName);
    if (newFilename === null) return; // User cancelled
    
    newFilename = newFilename.trim();
    if (!newFilename) {
        showToast("Filename cannot be empty", "warning");
        return;
    }
    
    // Add extension if not present
    if (!newFilename.endsWith(".txt") && !newFilename.endsWith(".md")) {
        newFilename += ".txt";
    }
    
    if (newFilename === currentName) return; // No change
    
    if (newFilename.includes("/") || newFilename.includes("\\") || newFilename.includes("..")) {
        showToast("Invalid filename. Subdirectories are not allowed.", "error");
        return;
    }
    
    // Check if new filename already exists in current tabs
    const existsInTabs = notesTabState.openTabs.some(t => t.filename.toLowerCase() === newFilename.toLowerCase());
    if (existsInTabs) {
        showToast(`A note named '${newFilename}' is already open in tabs`, "error");
        return;
    }

    // Check if it's a completely local unsaved/untitled note that doesn't exist on disk yet
    const existsOnServer = notesState.notes.some(n => n.name.toLowerCase() === newFilename.toLowerCase());
    const isLocalOnly = !notesState.notes.some(n => n.name === currentName);
    
    if (isLocalOnly) {
        if (existsOnServer) {
            showToast(`A note named '${newFilename}' already exists in notes folder`, "error");
            return;
        }
        
        // Update local tab filename
        const activeTab = notesTabState.openTabs.find(t => t.filename === currentName);
        if (activeTab) {
            activeTab.filename = newFilename;
        }
        notesTabState.activeFilename = newFilename;
        notesState.activeNoteName = newFilename;
        notesState.currentFilename = newFilename;
        
        checkUnsavedChanges();
        renderTabs();
        showToast("Note renamed locally", "success");
        return;
    }
    
    // Otherwise it exists on the server, call the api
    try {
        const res = await fetch("/api/notes/rename", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ old_filename: currentName, new_filename: newFilename })
        });
        
        if (!res.ok) {
            const data = await res.json();
            throw new Error(data.error || `HTTP ${res.status}`);
        }
        
        const data = await res.json();
        showToast(data.message || "Note renamed successfully", "success");
        
        const activeTab = notesTabState.openTabs.find(t => t.filename === currentName);
        if (activeTab) {
            activeTab.filename = data.filename;
        }
        notesTabState.activeFilename = data.filename;
        notesState.activeNoteName = data.filename;
        notesState.currentFilename = data.filename;
        
        checkUnsavedChanges();
        await fetchNotesList();
        renderTabs();
    } catch (err) {
        console.error("Failed to rename note:", err);
        showToast(`Rename failed: ${err.message}`, "error");
    }
}

function openNotesFolderBrowser() {
    fsState.selectorMode = "notes";
    const val = dom.notesDirInput ? dom.notesDirInput.value.trim() : "";
    if (val) {
        fsState.currentPath = val;
    } else {
        fsState.currentPath = "";
    }
    openModelBrowserModal();
}

async function deleteNote(filename) {
    const proceed = confirm(`Are you sure you want to delete note '${filename}'? This cannot be undone.`);
    if (!proceed) return;
    
    try {
        const res = await fetch("/api/notes/delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename })
        });
        
        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.error || `HTTP ${res.status}`);
        }
        
        showToast(`Deleted note '${filename}'`, "info");
        
        // Find if open in tabs and close it
        const tabIndex = notesTabState.openTabs.findIndex(t => t.filename === filename);
        if (tabIndex !== -1) {
            notesTabState.openTabs.splice(tabIndex, 1);
            if (notesTabState.activeFilename === filename) {
                if (notesTabState.openTabs.length > 0) {
                    const nextActiveIndex = Math.min(tabIndex, notesTabState.openTabs.length - 1);
                    switchTab(notesTabState.openTabs[nextActiveIndex].filename);
                } else {
                    notesTabState.activeFilename = null;
                    createNewNoteTab();
                }
            } else {
                renderTabs();
            }
        }
        
        await fetchNotesList();
    } catch (err) {
        console.error("Failed to delete note:", err);
        showToast(`Failed to delete note: ${err.message}`, "error");
    }
}

function deleteCurrentNote() {
    if (notesTabState.activeFilename) {
        deleteNote(notesTabState.activeFilename);
    }
}

function downloadActiveNote() {
    const filename = notesTabState.activeFilename || "note.txt";
    const content = dom.notesTextarea ? dom.notesTextarea.value : "";
    
    try {
        const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        showToast("Note downloaded locally!", "success");
    } catch (err) {
        console.error("Download failed:", err);
        showToast("Failed to download note locally", "error");
    }
}

function updateNotesCounters() {
    const content = dom.notesTextarea ? dom.notesTextarea.value : "";
    
    const charCount = content.length;
    if (dom.notesCharCount) dom.notesCharCount.textContent = `${charCount} chars`;
    
    const words = content.trim().split(/\s+/).filter(w => w.length > 0);
    const wordCount = words.length;
    if (dom.notesWordCount) dom.notesWordCount.textContent = `${wordCount} words`;
}

function checkUnsavedChanges() {
    const filenameInput = notesTabState.activeFilename || "";
    const contentInput = dom.notesTextarea ? dom.notesTextarea.value : "";
    
    const activeTab = notesTabState.openTabs.find(t => t.filename === notesTabState.activeFilename);
    if (activeTab) {
        activeTab.content = contentInput;
        const contentChanged = contentInput !== activeTab.originalContent;
        activeTab.isUnsaved = contentChanged;
        notesState.isUnsaved = activeTab.isUnsaved;
    }
    
    const badge = dom.notesStatusBadge;
    if (badge) {
        if (notesState.isUnsaved) {
            badge.textContent = "Modified";
            badge.className = "notes-status-badge unsaved";
        } else {
            badge.textContent = "Saved";
            badge.className = "notes-status-badge";
        }
    }

    const filename = filenameInput || "Untitled";
    if (dom.notesWindowTitle) {
        dom.notesWindowTitle.textContent = `${notesState.isUnsaved ? '*' : ''}${filename} - Mousepad`;
    }

    const activeTabEl = document.querySelector(`.notes-tab[data-filename="${CSS.escape(filenameInput)}"]`);
    if (activeTabEl) {
        activeTabEl.classList.toggle("modified", notesState.isUnsaved);
    }

    updateLanguageBadge(filename);
    updateLineNumbers();
    updateCursorPosition();
}

async function saveNotesDirectory(customPath = null, filenameToLoad = null) {
    const inputPath = customPath || (dom.notesDirInput ? dom.notesDirInput.value.trim() : "");
    if (!inputPath) {
        showToast("Please enter a folder path", "warning");
        return;
    }
    
    try {
        const res = await fetch("/api/notes/set_directory", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path: inputPath })
        });
        
        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.error || `HTTP ${res.status}`);
        }
        
        const data = await res.json();
        showToast(data.message || "Storage directory updated successfully", "success");
        if (dom.notesDirInput) {
            dom.notesDirInput.value = data.path;
        }
        if (dom.notesCurrentDirDisplay) {
            dom.notesCurrentDirDisplay.textContent = data.path;
        }
        
        // Reload notes list for the new directory
        await fetchNotesList();
        
        // Select first note in the new folder if available, else new note
        if (filenameToLoad) {
            await loadNote(filenameToLoad);
        } else if (notesState.notes.length > 0) {
            await loadNote(notesState.notes[0].name);
        } else {
            notesState.activeNoteName = null;
            notesState.currentFilename = "untitled.txt";
            notesState.currentContent = "";
            notesState.originalFilename = "";
            notesState.originalContent = "";
            notesState.isUnsaved = true;
            if (dom.notesFilename) dom.notesFilename.value = "untitled.txt";
            if (dom.notesTextarea) dom.notesTextarea.value = "";
            updateNotesCounters();
            checkUnsavedChanges();
            syncHighlight();
        }
    } catch (err) {
        console.error("Failed to change notes directory:", err);
        showToast(`Failed to change folder: ${err.message}`, "error");
    }
}

// ── Syntax Highlighting Engine ───────────────────────────────────────────────

const pyRules = {
    comment: /#[^\n]*/,
    string: /"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/,
    decorator: /@[a-zA-Z_][a-zA-Z0-9_]*/,
    keyword: /\b(?:def|class|return|if|else|elif|for|while|import|from|as|in|is|not|and|or|try|except|finally|with|lambda|global|nonlocal|pass|break|continue|None|True|False)\b/,
    builtin: /\b(?:print|len|range|str|int|float|dict|list|set|tuple|open|sum|min|max|abs|type)\b/,
    function: /\b[a-zA-Z_][a-zA-Z0-9_]*(?=\s*\()/,
    number: /\b\d+(?:\.\d+)?\b/
};

const jsRules = {
    comment: /\/\/.*|\/\*[\s\S]*?\*\//,
    string: /"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`/,
    keyword: /\b(?:const|let|var|function|return|if|else|for|while|do|switch|case|break|continue|import|export|from|default|class|extends|new|this|typeof|instanceof|try|catch|finally|throw|async|await|in|of|null|undefined|true|false)\b/,
    builtin: /\b(?:console|window|document|process|require|module|exports|Object|Array|String|Number|Boolean|Function|Promise|Map|Set|JSON|Math|Error)\b/,
    function: /\b[a-zA-Z_][a-zA-Z0-9_]*(?=\s*\()/,
    number: /\b\d+(?:\.\d+)?\b/
};

const cssRules = {
    comment: /\/\*[\s\S]*?\*\//,
    string: /"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/,
    selector: /[a-zA-Z0-9\-\.\#\_\*:\s,>+~]+(?=\s*\{)/,
    property: /[a-zA-Z\-]+(?=\s*:)/,
    number: /\b\d+(?:px|em|rem|%|vh|vw|s|ms|deg)?\b/
};

const htmlRules = {
    comment: /&lt;!--[\s\S]*?--&gt;/,
    doctype: /&lt;![dD][oO][cC][tT][yY][pP][eE][\s\S]*?&gt;/,
    tag: /&lt;\/?[a-zA-Z0-9\-]+/,
    attr: /\b[a-zA-Z\-]+(?=\s*=\s*["'])/,
    string: /"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/
};

const mdRules = {
    codeblock: /```[\s\S]*?```/,
    inlinecode: /`[^`]+`/,
    header: /^#{1,6}\s+.+$/m,
    bold: /\*\*[^\*]+\*\*/,
    italic: /\*[^\*]+\*/,
    link: /\[[^\]]+\]\([^\)]+\)/,
    blockquote: /^\s*&gt;\s+.+$/m,
    list: /^\s*(?:[\*\-\+]|\d+\.)\s+.+$/m
};

function buildHighlightRegex(rules) {
    const parts = Object.entries(rules).map(([name, regex]) => {
        return `(?<${name}>${regex.source})`;
    });
    return new RegExp(parts.join('|'), 'g');
}

const pyRegex = buildHighlightRegex(pyRules);
const jsRegex = buildHighlightRegex(jsRules);
const cssRegex = buildHighlightRegex(cssRules);
const htmlRegex = buildHighlightRegex(htmlRules);
const mdRegex = buildHighlightRegex(mdRules);

function escapeHtmlCode(text) {
    if (!text) return "";
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

function highlightCode(text, filename) {
    if (!text) return "";
    const escaped = escapeHtmlCode(text);
    
    const ext = filename ? filename.split('.').pop().toLowerCase() : '';
    let regex = null;
    
    if (ext === 'py') regex = pyRegex;
    else if (['js', 'ts', 'json'].includes(ext)) regex = jsRegex;
    else if (ext === 'css') regex = cssRegex;
    else if (['html', 'xml'].includes(ext)) regex = htmlRegex;
    else if (ext === 'md') regex = mdRegex;
    
    if (!regex) return escaped;
    
    return escaped.replace(regex, (match, ...args) => {
        const groups = args[args.length - 1];
        if (typeof groups === 'object' && groups !== null) {
            for (const [key, val] of Object.entries(groups)) {
                if (val !== undefined) {
                    return `<span class="hl-${key}">${val}</span>`;
                }
            }
        }
        return match;
    });
}

function syncHighlight() {
    if (!dom.notesTextarea || !dom.notesHighlightPre) return;
    
    const text = dom.notesTextarea.value;
    const filename = dom.notesFilename ? dom.notesFilename.value.trim() : "";
    
    if (notesState.syntaxHighlightOn) {
        dom.notesHighlightPre.innerHTML = highlightCode(text, filename) + "\n";
    } else {
        dom.notesHighlightPre.textContent = text;
    }
    
    dom.notesHighlightPre.scrollTop = dom.notesTextarea.scrollTop;
    dom.notesHighlightPre.scrollLeft = dom.notesTextarea.scrollLeft;
}

function updateHighlightView() {
    if (!dom.notesTextarea) return;
    if (notesState.syntaxHighlightOn) {
        dom.notesTextarea.classList.remove("no-highlight");
        if (dom.notesHighlightPre) dom.notesHighlightPre.style.display = "block";
    } else {
        dom.notesTextarea.classList.add("no-highlight");
        if (dom.notesHighlightPre) dom.notesHighlightPre.style.display = "none";
    }
}

// ── Window Draggable & Resizable Utilities ────────────────────────────────────

function initWindowDraggingAndResizing() {
    // 1. Settings Modal
    if (dom.settingsModal) {
        const header = dom.settingsModal.querySelector(".modal-header");
        if (header) makeWindowDraggable(dom.settingsModal, header);
        makeWindowResizable(dom.settingsModal);
    }
    
    // 2. Help Modal
    if (dom.helpModal) {
        const header = dom.helpModal.querySelector(".modal-header");
        if (header) makeWindowDraggable(dom.helpModal, header);
        makeWindowResizable(dom.helpModal);
    }
    
    // 3. Model Browser Modal
    if (dom.modelBrowserModal) {
        const header = dom.modelBrowserModal.querySelector(".modal-header");
        if (header) makeWindowDraggable(dom.modelBrowserModal, header);
        makeWindowResizable(dom.modelBrowserModal);
    }
    
    // 4. Model Downloader Modal
    if (dom.modelDownloaderModal) {
        const header = dom.modelDownloaderModal.querySelector(".modal-header");
        if (header) makeWindowDraggable(dom.modelDownloaderModal, header);
        makeWindowResizable(dom.modelDownloaderModal);
    }
    
    // 5. Context Popup
    if (dom.contextPopup) {
        const header = dom.contextPopup.querySelector(".context-popup-header");
        if (header) makeWindowDraggable(dom.contextPopup, header);
        makeWindowResizable(dom.contextPopup);
    }
    
    // 6. Notes Modal (Mousepad)
    if (dom.notesModal) {
        const header = dom.notesModal.querySelector(".modal-header");
        if (header) makeWindowDraggable(dom.notesModal, header);
        makeWindowResizable(dom.notesModal);
    }
    
    // Reset positions whenever modals are opened to re-center them initially
    const resetWindowStyle = (win) => {
        if (!win) return;
        
        let isMinimized = false;
        if (win === dom.settingsModal && dom.btnSettings && dom.btnSettings.classList.contains("minimized-active")) isMinimized = true;
        else if (win === dom.helpModal && dom.btnHelp && dom.btnHelp.classList.contains("minimized-active")) isMinimized = true;
        else if (win === dom.modelBrowserModal && dom.btnSwitchModel && dom.btnSwitchModel.classList.contains("minimized-active")) isMinimized = true;
        else if (win === dom.modelDownloaderModal && dom.btnDownloadModel && dom.btnDownloadModel.classList.contains("minimized-active")) isMinimized = true;
        else if (win === dom.notesModal && dom.btnNotes && dom.btnNotes.classList.contains("minimized-active")) isMinimized = true;
        
        if (isMinimized) return;

        win.style.position = "";
        win.style.margin = "";
        win.style.transform = "";
        win.style.left = "";
        win.style.top = "";
        win.style.width = "";
        win.style.height = "";
        win.style.maxWidth = "";
        win.style.maxHeight = "";
    };
    
    // Listen to toggles/clicks to reset positioning
    if (dom.btnSettings) dom.btnSettings.addEventListener("click", () => resetWindowStyle(dom.settingsModal));
    if (dom.btnHelp) dom.btnHelp.addEventListener("click", () => resetWindowStyle(dom.helpModal));
    if (dom.btnSwitchModel) dom.btnSwitchModel.addEventListener("click", () => resetWindowStyle(dom.modelBrowserModal));
    if (dom.btnDownloadModel) dom.btnDownloadModel.addEventListener("click", () => resetWindowStyle(dom.modelDownloaderModal));
    if (dom.btnNotes) dom.btnNotes.addEventListener("click", () => resetWindowStyle(dom.notesModal));
}

function makeWindowDraggable(win, header) {
    let mouseX = 0, mouseY = 0;
    
    header.addEventListener("mousedown", dragStart);
    header.addEventListener("touchstart", dragStart, { passive: false });
    
    function dragStart(e) {
        // Exclude interactive elements in header
        if (e.target.closest("button") || e.target.closest(".window-controls")) {
            return;
        }
        if (win.classList.contains("fullscreen")) {
            return;
        }
        
        e.preventDefault();
        
        // Handle touch vs mouse
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        
        mouseX = clientX;
        mouseY = clientY;
        
        // Disable CSS transitions during drag
        win.style.transition = "none";
        
        if (e.touches) {
            document.addEventListener("touchmove", dragMove, { passive: false });
            document.addEventListener("touchend", dragEnd);
        } else {
            document.addEventListener("mousemove", dragMove);
            document.addEventListener("mouseup", dragEnd);
        }
    }
    
    function dragMove(e) {
        if (win.classList.contains("fullscreen")) return;
        e.preventDefault();
        
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        
        const dx = clientX - mouseX;
        const dy = clientY - mouseY;
        
        mouseX = clientX;
        mouseY = clientY;
        
        let left = parseFloat(win.style.left) || 0;
        let top = parseFloat(win.style.top) || 0;
        
        // On first drag, change layout structure from flex center to absolute fixed
        if (!win.style.left || !win.style.top) {
            const rect = win.getBoundingClientRect();
            left = rect.left;
            top = rect.top;
            
            win.style.position = "fixed";
            win.style.margin = "0";
            win.style.transform = "none";
            
            // Set explicit width/height to prevent sudden size changes when changing position type
            win.style.width = `${rect.width}px`;
            win.style.height = `${rect.height}px`;
        }
        
        win.style.left = `${left + dx}px`;
        win.style.top = `${top + dy}px`;
    }
    
    function dragEnd(e) {
        document.removeEventListener("mousemove", dragMove);
        document.removeEventListener("mouseup", dragEnd);
        document.removeEventListener("touchmove", dragMove);
        document.removeEventListener("touchend", dragEnd);
        
        win.style.transition = ""; // Restore transitions
    }
}

function makeWindowResizable(win) {
    const handle = document.createElement("div");
    handle.className = "resize-handle";
    win.appendChild(handle);
    
    handle.addEventListener("mousedown", resizeStart);
    handle.addEventListener("touchstart", resizeStart, { passive: false });
    
    function resizeStart(e) {
        if (win.classList.contains("fullscreen")) return;
        e.preventDefault();
        
        win.style.transition = "none";
        
        if (e.touches) {
            document.addEventListener("touchmove", resizeMove, { passive: false });
            document.addEventListener("touchend", resizeEnd);
        } else {
            document.addEventListener("mousemove", resizeMove);
            document.addEventListener("mouseup", resizeEnd);
        }
    }
    
    function resizeMove(e) {
        if (win.classList.contains("fullscreen")) return;
        e.preventDefault();
        
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        
        const rect = win.getBoundingClientRect();
        
        // Calculate new dimensions relative to the top-left of the window
        const width = clientX - rect.left;
        const height = clientY - rect.top;
        
        // Constraints
        const minWidth = 350;
        const minHeight = 220;
        
        // On first resize, lock position from center to fixed position
        if (!win.style.left || !win.style.top) {
            win.style.position = "fixed";
            win.style.margin = "0";
            win.style.transform = "none";
            win.style.left = `${rect.left}px`;
            win.style.top = `${rect.top}px`;
        }
        
        if (width >= minWidth) {
            win.style.width = `${width}px`;
            win.style.maxWidth = "none";
        }
        if (height >= minHeight) {
            win.style.height = `${height}px`;
            win.style.maxHeight = "none";
        }
    }
    
    function resizeEnd() {
        document.removeEventListener("mousemove", resizeMove);
        document.removeEventListener("mouseup", resizeEnd);
        document.removeEventListener("touchmove", resizeMove);
        document.removeEventListener("touchend", resizeEnd);
        
        win.style.transition = ""; // Restore transitions
    }
}

// ── XFCE Mousepad Helper Functions ──────────────────────────────────────────

function updateLineNumbers() {
    if (!dom.notesLineNumbers || !dom.notesTextarea) return;
    
    // Check if line numbers view is enabled
    const showLines = dom.chkMenuLinenumbers ? dom.chkMenuLinenumbers.checked : true;
    if (!showLines) {
        dom.notesLineNumbers.style.display = "none";
        return;
    }
    dom.notesLineNumbers.style.display = "block";
    
    const text = dom.notesTextarea.value;
    const lines = text.split('\n');
    const lineCount = lines.length;
    
    let numbersHtml = "";
    for (let i = 1; i <= lineCount; i++) {
        numbersHtml += `<div>${i}</div>`;
    }
    
    dom.notesLineNumbers.innerHTML = numbersHtml;
    // Align scroll
    dom.notesLineNumbers.scrollTop = dom.notesTextarea.scrollTop;
}

function updateCursorPosition() {
    if (!dom.notesTextarea) return;
    
    const text = dom.notesTextarea.value;
    const selStart = dom.notesTextarea.selectionStart;
    
    // Split text up to selection start to count lines
    const textUpToCursor = text.substring(0, selStart);
    const lines = textUpToCursor.split('\n');
    
    const lineNum = lines.length;
    const colNum = lines[lines.length - 1].length + 1;
    
    if (dom.notesLnCol) {
        dom.notesLnCol.textContent = `Ln ${lineNum}, Col ${colNum}`;
    }
}

function updateLanguageBadge(filename) {
    if (!dom.notesFileType) return;
    
    if (!filename || !filename.includes('.')) {
        dom.notesFileType.textContent = "Plain Text";
        return;
    }
    
    const ext = filename.split('.').pop().toLowerCase();
    let lang = "Plain Text";
    switch (ext) {
        case "js": lang = "JavaScript"; break;
        case "py": lang = "Python"; break;
        case "html": lang = "HTML"; break;
        case "css": lang = "CSS"; break;
        case "json": lang = "JSON"; break;
        case "md": lang = "Markdown"; break;
        case "sh": lang = "Shell Script"; break;
        case "cpp": case "cc": lang = "C++"; break;
        case "c": lang = "C"; break;
        case "java": lang = "Java"; break;
        case "ts": lang = "TypeScript"; break;
    }
    dom.notesFileType.textContent = lang;
}

function deleteCurrentNote() {
    const filename = dom.notesFilename ? dom.notesFilename.value.trim() : "";
    if (!filename) {
        showToast("No note is currently open to delete", "warning");
        return;
    }
    deleteNote(filename);
}

// Find & Replace Functions
let lastFindIndex = -1;

function toggleFindReplaceBar(show) {
    if (!dom.findReplaceBar) return;
    if (show) {
        dom.findReplaceBar.style.display = "flex";
        if (dom.frFindInput) {
            dom.frFindInput.focus();
            dom.frFindInput.select();
        }
    } else {
        dom.findReplaceBar.style.display = "none";
        lastFindIndex = -1;
    }
}

function handleFindNext() {
    if (!dom.notesTextarea || !dom.frFindInput) return;
    const query = dom.frFindInput.value;
    if (!query) {
        showToast("Enter text to find", "warning");
        return;
    }

    const text = dom.notesTextarea.value;
    const startIndex = lastFindIndex === -1 ? 0 : lastFindIndex + 1;
    
    // Find next case-insensitive match
    const index = text.toLowerCase().indexOf(query.toLowerCase(), startIndex);
    
    if (index !== -1) {
        dom.notesTextarea.focus();
        dom.notesTextarea.setSelectionRange(index, index + query.length);
        
        // Scroll selection into view
        const numLines = text.substring(0, index).split('\n').length;
        const lineHeight = 13 * 1.6;
        dom.notesTextarea.scrollTop = (numLines - 4) * lineHeight;
        
        lastFindIndex = index;
    } else {
        if (startIndex > 0) {
            // Wrap around
            showToast("Search wrapped around", "info");
            lastFindIndex = -1;
            handleFindNext();
        } else {
            showToast("Search string not found", "info");
        }
    }
}

function handleReplace() {
    if (!dom.notesTextarea || !dom.frFindInput || !dom.frReplaceInput) return;
    const findText = dom.frFindInput.value;
    const replaceText = dom.frReplaceInput.value;
    
    if (!findText) return;

    const textarea = dom.notesTextarea;
    const text = textarea.value;
    
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const selectedText = text.substring(start, end);
    
    if (selectedText.toLowerCase() === findText.toLowerCase()) {
        textarea.value = text.substring(0, start) + replaceText + text.substring(end);
        textarea.setSelectionRange(start, start + replaceText.length);
        
        notesState.currentContent = textarea.value;
        checkUnsavedChanges();
        syncHighlight();
        
        lastFindIndex = start + replaceText.length - 1;
        handleFindNext();
    } else {
        handleFindNext();
    }
}

function handleReplaceAll() {
    if (!dom.notesTextarea || !dom.frFindInput || !dom.frReplaceInput) return;
    const findText = dom.frFindInput.value;
    const replaceText = dom.frReplaceInput.value;
    
    if (!findText) return;

    const textarea = dom.notesTextarea;
    const text = textarea.value;
    
    const escapedFind = findText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const regex = new RegExp(escapedFind, 'gi');
    
    const matches = text.match(regex);
    if (matches && matches.length > 0) {
        textarea.value = text.replace(regex, replaceText);
        notesState.currentContent = textarea.value;
        checkUnsavedChanges();
        syncHighlight();
        showToast(`Replaced ${matches.length} occurrence(s)`, "success");
    } else {
        showToast("Search string not found", "info");
    }
}

function initGtkMenuBar() {
    const menuItems = $$(".gtk-menu-item");
    
    document.addEventListener("click", (e) => {
        if (!e.target.closest(".gtk-menubar")) {
            menuItems.forEach(item => item.classList.remove("active"));
        }
    });

    menuItems.forEach((menuItem) => {
        const label = menuItem.querySelector(".gtk-menu-label");
        if (!label) return;
        
        label.addEventListener("click", (e) => {
            e.stopPropagation();
            const wasActive = menuItem.classList.contains("active");
            menuItems.forEach(item => item.classList.remove("active"));
            if (!wasActive) {
                menuItem.classList.add("active");
            }
        });

        menuItem.addEventListener("mouseenter", () => {
            const anyActive = Array.from(menuItems).some(item => item.classList.contains("active"));
            if (anyActive) {
                menuItems.forEach(item => item.classList.remove("active"));
                menuItem.classList.add("active");
            }
        });
    });

    // File Menu Actions
    const fileNew = $("#menu-file-new");
    if (fileNew) fileNew.addEventListener("click", () => { createNewNote(); closeGtkMenus(); });
    
    const fileOpen = $("#menu-file-open");
    if (fileOpen) fileOpen.addEventListener("click", () => {
        openNotesFolderBrowser();
        closeGtkMenus();
    });
    
    const fileSave = $("#menu-file-save");
    if (fileSave) fileSave.addEventListener("click", () => { saveActiveNote(); closeGtkMenus(); });
    
    const fileSaveAs = $("#menu-file-saveas");
    if (fileSaveAs) fileSaveAs.addEventListener("click", () => { saveNoteAs(); closeGtkMenus(); });
    
    const fileRename = $("#menu-file-rename");
    if (fileRename) fileRename.addEventListener("click", () => { renameActiveNote(); closeGtkMenus(); });
    
    const fileDownload = $("#menu-file-download");
    if (fileDownload) fileDownload.addEventListener("click", () => { downloadActiveNote(); closeGtkMenus(); });
    
    const fileClose = $("#menu-file-close");
    if (fileClose) fileClose.addEventListener("click", () => { closeNotesModal(); closeGtkMenus(); });

    // Edit Menu Actions
    const editUndo = $("#menu-edit-undo");
    if (editUndo) editUndo.addEventListener("click", () => { document.execCommand("undo"); closeGtkMenus(); });
    
    const editRedo = $("#menu-edit-redo");
    if (editRedo) editRedo.addEventListener("click", () => { document.execCommand("redo"); closeGtkMenus(); });
    
    const editSelectAll = $("#menu-edit-selectall");
    if (editSelectAll) editSelectAll.addEventListener("click", () => {
        if (dom.notesTextarea) {
            dom.notesTextarea.focus();
            dom.notesTextarea.select();
        }
        closeGtkMenus();
    });
    
    const editDelete = $("#menu-edit-delete");
    if (editDelete) editDelete.addEventListener("click", () => { deleteCurrentNote(); closeGtkMenus(); });

    // Search Menu Actions
    const searchFind = $("#menu-search-find");
    if (searchFind) searchFind.addEventListener("click", () => { toggleFindReplaceBar(true); closeGtkMenus(); });

    // View Menu Toggles
    const chkMenuLinenumbers = $("#chk-menu-linenumbers");
    if (chkMenuLinenumbers) {
        chkMenuLinenumbers.addEventListener("change", (e) => {
            if (dom.notesLineNumbers) {
                dom.notesLineNumbers.style.display = e.target.checked ? "block" : "none";
            }
        });
    }

    const chkMenuWordwrap = $("#chk-menu-wordwrap");
    if (chkMenuWordwrap) {
        chkMenuWordwrap.addEventListener("change", (e) => {
            const wrap = e.target.checked;
            if (dom.notesTextarea && dom.notesHighlightPre) {
                if (wrap) {
                    dom.notesTextarea.style.whiteSpace = "pre-wrap";
                    dom.notesTextarea.style.wordWrap = "break-word";
                    dom.notesHighlightPre.style.whiteSpace = "pre-wrap";
                    dom.notesHighlightPre.style.wordWrap = "break-word";
                } else {
                    dom.notesTextarea.style.whiteSpace = "pre";
                    dom.notesTextarea.style.wordWrap = "normal";
                    dom.notesHighlightPre.style.whiteSpace = "pre";
                    dom.notesHighlightPre.style.wordWrap = "normal";
                }
            }
        });
    }

    const chkMenuHighlight = $("#chk-menu-highlight");
    if (chkMenuHighlight) {
        chkMenuHighlight.addEventListener("change", (e) => {
            notesState.syntaxHighlightOn = e.target.checked;
            if (dom.chkSyntaxHighlight) dom.chkSyntaxHighlight.checked = notesState.syntaxHighlightOn;
            updateHighlightView();
            syncHighlight();
        });
    }

    // Help Menu Actions
    const helpAbout = $("#menu-help-about");
    if (helpAbout) helpAbout.addEventListener("click", () => {
        alert("About Mousepad\n\nA simple, lightweight text editor clone inspired by XFCE Mousepad. Built for the AI Chat Interface.");
        closeGtkMenus();
    });
}

function closeGtkMenus() {
    $$(".gtk-menu-item").forEach(item => item.classList.remove("active"));
}
