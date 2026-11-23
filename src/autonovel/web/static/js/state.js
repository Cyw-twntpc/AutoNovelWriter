// --- State & Utilities Module ---
// Frontend controller for the 3-Stage AutoNovelWriter Web UI (Zero console.log)

let genreCatalog = {};
let currentView = "WIZARD";
let currentQuestion = null;
let preflightNewOrDupPaths = [];
let activeEventSource = null;
let currentReaderFontSize = 1.06;
let activeReaderChapter = null;
const collapsedVolumes = new Set();


let lastUserInteractionLocalTs = Date.now();
function recordUserFrontendActivity() {
    lastUserInteractionLocalTs = Date.now();
}

function initTheme() {
    const savedTheme = localStorage.getItem("autonovel_theme") || "light";
    applyTheme(savedTheme);
}

function applyTheme(theme) {
    const isDark = theme === "dark";
    document.documentElement.setAttribute("data-theme", isDark ? "dark" : "light");
    document.body.classList.toggle("theme-dark", isDark);
    const btn = document.getElementById("btn-toggle-theme");
    if (btn) {
        btn.textContent = isDark ? "☀️ 淺色模式" : "🌙 深色模式";
    }
    localStorage.setItem("autonovel_theme", isDark ? "dark" : "light");
}

function toggleDarkTheme() {
    const current = document.documentElement.getAttribute("data-theme") || "light";
    applyTheme(current === "dark" ? "light" : "dark");
}

// Initialize theme as early as script loads
initTheme();

function initUserActivityHeartbeat() {
    ['mousedown', 'keydown', 'touchstart', 'scroll', 'click'].forEach(evt => {
        window.addEventListener(evt, recordUserFrontendActivity, { passive: true });
    });
    fetch("/api/system/heartbeat", { method: "POST" }).catch(() => {});
    setInterval(async () => {
        if (Date.now() - lastUserInteractionLocalTs < 60 * 1000) {
            try {
                await fetch("/api/system/heartbeat", { method: "POST" });
            } catch (_) {}
        }
    }, 30 * 1000);
}

function sanitizeErrorMessage(msg) {
    if (!msg || typeof msg !== "string") return "遭遇未預期的錯誤，詳細資訊請參閱伺服器日誌。";
    const lower = msg.toLowerCase();
    // Intercept raw HTTP error dumps, request IDs, python exceptions, and technical URLs
    if (
        msg.includes("Client Error") ||
        msg.includes("Request ID:") ||
        msg.includes("Traceback") ||
        msg.includes("https://") ||
        msg.includes("http://") ||
        lower.includes("entry not found")
    ) {
        if (msg.includes("404") || lower.includes("entry not found") || lower.includes("not found")) {
            return "遠端未找到指定的模型檔案，請檢查模型名稱或版本設定。";
        }
        if (msg.includes("401") || msg.includes("403") || lower.includes("unauthorized")) {
            return "遠端存取權限不足，該模型需要 Hugging Face 存取權杖。";
        }
        if (lower.includes("timeout") || lower.includes("connection")) {
            return "網路連線逾時或中斷，請確認網路環境狀態。";
        }
        return "模型作業失敗，詳細錯誤已記錄於後端控制台日誌。";
    }
    return msg;
}

// Global button loading and anti-double-click wrapper
async function withButtonLoading(btnOrSelector, asyncFn, loadingText = "處理中...") {
    const btn = typeof btnOrSelector === "string" ? document.querySelector(btnOrSelector) : btnOrSelector;
    if (!btn) {
        return await asyncFn();
    }
    if (btn.classList.contains("btn-loading") || btn.disabled) {
        return; // Guard against concurrent double execution
    }
    const origHtml = btn.innerHTML;
    btn.classList.add("btn-loading");
    btn.disabled = true;
    btn.innerHTML = `<span class="btn-spinner"></span>${escapeHtml(loadingText)}`;
    try {
        return await asyncFn();
    } finally {
        btn.classList.remove("btn-loading");
        btn.disabled = false;
        btn.innerHTML = origHtml;
    }
}

function showToast(message) {
    const el = document.getElementById("global-toast");
    if (!el) return;
    el.textContent = sanitizeErrorMessage(message);
    el.classList.remove("hidden");
    setTimeout(() => el.classList.add("hidden"), 3500);
}

function escapeHtml(raw) {
    return String(raw == null ? "" : raw)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function setStageView(stageName) {
    const normalized = stageName === "RUNNING" || stageName === "PAUSED" || stageName === "COMPLETED" ? "WARROOM" : stageName;
    currentView = normalized;
    document.getElementById("view-wizard").classList.toggle("hidden", normalized !== "WIZARD");
    document.getElementById("view-review").classList.toggle("hidden", normalized !== "REVIEW_GATE");
    document.getElementById("view-warroom").classList.toggle("hidden", normalized !== "WARROOM");
    document.getElementById("step-pill-wizard").classList.toggle("active", normalized === "WIZARD");
    document.getElementById("step-pill-review").classList.toggle("active", normalized === "REVIEW_GATE");
    document.getElementById("step-pill-warroom").classList.toggle("active", normalized === "WARROOM");
}

function selectStageDirect(stageName) {
    if (currentProjectsCount === 0) {
        showToast("作品庫目前尚無任何小說，請先建立新小說。");
        switchMainView("projects");
        return;
    }
    if (stageName === "WIZARD") {
        const badgeEl = document.getElementById("top-pipeline-badge");
        const curState = badgeEl?.dataset?.state;
        if (curState === "RUNNING") {
            showToast("小說目前正在連載創作中，禁止進入創世頁面。請先暫停連載！");
            return;
        }
    }
    if (stageName === "REVIEW_GATE") {
        loadReviewGateState();
    } else if (stageName === "RUNNING" || stageName === "WARROOM") {
        refreshWarroomDashboard();
    }
    setStageView(stageName);
}

// Format word count: under 10000 -> plain number, >= 10000 -> unit 萬 with 3 digits (e.g. 1202 -> 2.31萬 -> 26.3萬 -> 370萬), >= 1000萬 -> 4 digits (e.g. 1000萬)
function formatWordCount(words) {
    const w = Math.max(0, parseInt(words, 10) || 0);
    if (w < 10000) {
        return `${w}`;
    }
    const wan = w / 10000;
    if (wan < 10) {
        return `${wan.toFixed(2)}萬`;
    } else if (wan < 100) {
        return `${wan.toFixed(1)}萬`;
    } else {
        return `${Math.round(wan)}萬`;
    }
}

function handleAiModelError(data) {
    if (!data || typeof data !== "object") return false;
    if (data.needs_env_setup || data.error === "AI_MODEL_UNAVAILABLE") {
        updateEngineModeBadge("OFFLINE");
        showToast(`⚠ AI 模型未就緒：${data.detail || "本地推論服務離線"}，已自動打開環境設置！`);
        openEnvModal();
        return true;
    }
    return false;
}

function updateEngineModeBadge(aiMode) {
    const badge = document.getElementById("qa-engine-mode-badge");
    if (!badge) return;
    badge.style.cursor = "pointer";
    if (aiMode === "LLM") {
        badge.textContent = "● 本地 AI 即時生成";
        badge.className = "engine-mode-pill mode-llm";
        badge.style.display = "inline-flex";
        badge.title = "已連線至本地 llama-server，所有問題與選項均由神經網路即時生成。點擊檢視環境狀態。";
        badge.onclick = () => openEnvModal();
    } else {
        badge.textContent = "● AI 離線 (點擊打開環境設置)";
        badge.className = "engine-mode-pill";
        badge.style.display = "inline-flex";
        badge.style.background = "rgba(239, 68, 68, 0.12)";
        badge.style.color = "#dc2626";
        badge.style.borderColor = "rgba(239, 68, 68, 0.3)";
        badge.title = "未偵測到本地 AI 模型或推論服務離線。已停用虛擬模板兜底流程。點擊以開啟環境檢查與下載模型。";
        badge.onclick = () => openEnvModal();
    }
}

function closeModal(id) {
    const el = document.getElementById(id);
    if (el) {
        el.classList.add("hidden");
        el.style.display = "none";
    }
    if (id === "modal-reference" && refModalPollTimer) {
        clearInterval(refModalPollTimer);
        refModalPollTimer = null;
    }
}
