// --- Main Bootstrap Module ---

let sseReconnectTimer = null;
let tokenBuffer = "";
let tokenRafId = null;

function flushTokenBuffer() {
    tokenRafId = null;
    if (!tokenBuffer) return;
    const pane = document.getElementById("chapter-content-viewer");
    if (pane) {
        pane.textContent += tokenBuffer;
        pane.scrollTop = pane.scrollHeight;
    }
    tokenBuffer = "";
}

function connectSSEStream() {
    if (sseReconnectTimer) {
        clearTimeout(sseReconnectTimer);
        sseReconnectTimer = null;
    }
    if (activeEventSource) activeEventSource.close();
    activeEventSource = new EventSource("/api/warroom/stream");
    activeEventSource.onmessage = (ev) => {
        const msg = JSON.parse(ev.data || "{}");
        if (msg.event === "chapter_start") {
            tokenBuffer = "";
            if (tokenRafId) {
                cancelAnimationFrame(tokenRafId);
                tokenRafId = null;
            }
            const titleEl = document.getElementById("reader-chapter-title");
            if (titleEl) titleEl.textContent = `正在即時寫作：第 ${msg.data.chapter_num} 章...`;
            const pane = document.getElementById("chapter-content-viewer");
            if (pane) pane.textContent = "";
        } else if (msg.event === "chapter_status") {
            const dot = document.getElementById(`dot-ch-${msg.data.chapter_num}`);
            if (dot) {
                dot.className = `status-dot dot-${msg.data.status}`;
            }
            if (msg.data.status === "REVISING") {
                const titleEl = document.getElementById("reader-chapter-title");
                if (titleEl) titleEl.textContent = `第 ${msg.data.chapter_num} 章：自審修訂中 (Gate B)...`;
            }
        } else if (msg.event === "token") {
            tokenBuffer += msg.data.token || "";
            if (!tokenRafId) {
                tokenRafId = requestAnimationFrame(flushTokenBuffer);
            }
        } else if (msg.event === "ref_pacing_progress") {
            const pacingData = msg.data || {};
            updatePacingProgressMonitor(pacingData);
        } else if (msg.event === "ref_progress") {
            const refData = msg.data || {};
            updateReferenceProgressPill(refData);
            updateReferenceProgressMonitor(refData);
            if (!refData.active) {
                refreshWizardGenreCatalog();
                loadReferenceBooks();
                if (document.getElementById("ref-tab-btn-entries")?.classList.contains("active")) {
                    loadReferenceEntries(refEntriesCurrentPage);
                    loadReferenceHotTags();
                }
            }
        } else if (msg.event === "env_download") {
            const st = msg.data || {};
            const modalEnv = document.getElementById("modal-env");
            if (modalEnv && !modalEnv.classList.contains("hidden")) {
                refreshEnvModalStatus();
            }
            if (!st.active) {
                showToast(st.message || "模型下載作業已結束");
                refreshEnvModalStatus();
            }
        } else if (msg.event === "chapter_complete" || msg.event === "chapter_rewritten") {
            flushTokenBuffer();
            refreshWarroomDashboard();
            if (msg.data && msg.data.chapter_num) {
                loadChapterReader(msg.data.chapter_num);
            }
        } else if (msg.event === "state_change") {
            updateTopBar(msg.data || {});
        } else if (msg.event === "volume_planned") {
            refreshWarroomDashboard();
            showToast(`已自動滾動展開下一卷：${(msg.data || {}).title || ""}`);
        } else if (msg.event === "deep_organize") {
            showToast("已完成每 10 章深度整理與邊緣角色歸檔。");
        } else if (msg.event === "env_status_changed") {
            refreshEnvModalStatus();
            updateEngineModeBadge(msg.data?.llama_server_healthy ? 'LOCAL_LLAMA' : 'OFFLINE');
        }
    };
    activeEventSource.onerror = () => {
        if (activeEventSource) {
            activeEventSource.close();
            activeEventSource = null;
        }
        // onerror can fire again for the same failed connection; without this guard
        // each retry stacked another timer and another EventSource.
        if (!sseReconnectTimer) {
            sseReconnectTimer = setTimeout(() => {
                sseReconnectTimer = null;
                connectSSEStream();
            }, 3000);
        }
    };
}

window.addEventListener("DOMContentLoaded", async () => {
    if (!document.querySelector(".qa-fullwidth-title")) {
        window.location.replace(`/?v=${Date.now()}`);
        return;
    }
    await loadProjectsList();
    if (currentProjectsCount > 0) {
        await initWizardState();
        await refreshWarroomDashboard();
    } else {
        await refreshWarroomDashboard();
        switchMainView("projects");
    }
    connectSSEStream();

    // Smart polling: every 5s in War Room; 15s in background views to save network/CPU
    let dashboardPollCounter = 0;
    setInterval(() => {
        dashboardPollCounter++;
        const isWarroom = (typeof currentMainView !== "undefined" && currentMainView === "studio") &&
                          (typeof currentView !== "undefined" && currentView === "WARROOM");
        if (isWarroom || dashboardPollCounter % 3 === 0) {
            refreshWarroomDashboard();
        }
    }, 5000);

    initUserActivityHeartbeat();
});
