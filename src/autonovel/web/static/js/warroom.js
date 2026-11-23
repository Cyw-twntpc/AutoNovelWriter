// --- War Room Module ---
function adjustReaderFontSize(delta) {
    currentReaderFontSize = Math.min(1.45, Math.max(0.86, +(currentReaderFontSize + delta).toFixed(2)));
    document.documentElement.style.setProperty("--reader-font-size", `${currentReaderFontSize}rem`);
}

function toggleReaderPaperMode() {
    const pane = document.getElementById("chapter-content-viewer");
    const btn = document.getElementById("btn-paper-mode");
    const isWarm = pane.classList.toggle("paper-warm");
    btn.textContent = isWarm ? "💎 晶透玻璃" : "📜 宣紙護眼";
}

function cycleStageView() {
    const order = ["WIZARD", "REVIEW_GATE", "WARROOM"];
    const next = order[(order.indexOf(currentView) + 1) % order.length];
    if (next === "REVIEW_GATE") {
        loadReviewGateState();
    }
    setStageView(next);
}
// --- Stage 3: 3-Column Integrated War Room ---
let _dashboardRefreshing = false;
async function refreshWarroomDashboard() {
    // Skip when a previous poll is still in flight: slow responses would otherwise
    // stack up parallel requests and surface unhandled rejections.
    if (_dashboardRefreshing) return;
    _dashboardRefreshing = true;
    try {
        const resp = await fetch("/api/warroom/dashboard");
        if (!resp.ok) return;
        const data = await resp.json();
        if (data.has_project === false) {
            updateTopBar({ title: "", pipeline_state: "IDLE_NO_PROJECT" }, {});
            renderOutlineTree([], []);
            renderHardwareAndStats(data.hardware || {}, {}, data.reference_status || {});
            renderTensionECGChart([]);
            renderAnomalyHistory([]);
            return;
        }
        updateTopBar(data.project_config || {}, data.summary_stats || {});
        renderOutlineTree(data.volumes || [], data.chapters || []);
        renderHardwareAndStats(data.hardware || {}, data.summary_stats || {}, data.reference_status || {});
        renderTensionECGChart(data.metrics_30 || []);
        renderAnomalyHistory(data.anomalies || []);
    } catch (_err) {
        // Background poll: a transient network error must not raise an unhandled rejection.
    } finally {
        _dashboardRefreshing = false;
    }
}

const PIPELINE_STATE_LABELS = {
    IDLE_NO_PROJECT: "目前狀態：尚未選擇或建立作品",
    WIZARD: "目前狀態：創世問答設定中",
    REVIEW_GATE: "目前狀態：藍圖審核待鎖定",
    RUNNING: "目前狀態：背景全自動連載中",
    PAUSED: "目前狀態：連載已安全暫停",
    COMPLETED: "目前狀態：全書已完結",
};

function syncPipelineButtonsState(state, stats) {
    const isNoProject = (state === "IDLE_NO_PROJECT");
    const isWizard = (state === "WIZARD" || isNoProject);
    const isRunning = (state === "RUNNING");
    const hasCompletedChapters = Boolean(!isNoProject && stats && stats.completed_chapters > 0);

    const pillWizard = document.getElementById("step-pill-wizard");
    if (pillWizard) {
        pillWizard.classList.toggle("subtab-disabled", isRunning);
        if (isRunning) {
            pillWizard.setAttribute("title", "目前正在連載創作中，禁止進入創世頁面（請先暫停連載）");
        } else {
            pillWizard.removeAttribute("title");
        }
    }

    const snapBtn = document.getElementById("btn-manual-snapshot");
    if (snapBtn) {
        snapBtn.disabled = isWizard;
        snapBtn.style.opacity = isWizard ? "0.45" : "";
        snapBtn.style.cursor = isWizard ? "not-allowed" : "";
        snapBtn.title = isNoProject
            ? "目前無作品，無法建立快照。"
            : isWizard
            ? "目前處於創世問答階段，尚未生成小說藍圖，無法建立快照。"
            : "建立當前進度之永久手動快照";
    }

    const stepBtn = document.getElementById("btn-step-once");
    if (stepBtn) {
        const canStep = !isWizard;
        stepBtn.disabled = !canStep;
        stepBtn.style.opacity = canStep ? "" : "0.45";
        stepBtn.style.cursor = canStep ? "" : "not-allowed";
        stepBtn.title = isWizard ? "目前處於創世問答階段，請先完成問答並生成大綱藍圖。" : "生成並推進下一章";
    }

    const rewriteBtn = document.getElementById("btn-warroom-rewrite");
    if (rewriteBtn) {
        const canRewrite = !isWizard && hasCompletedChapters;
        rewriteBtn.disabled = !canRewrite;
        rewriteBtn.style.opacity = canRewrite ? "" : "0.45";
        rewriteBtn.style.cursor = canRewrite ? "" : "not-allowed";
        rewriteBtn.title = isWizard
            ? "目前處於創世問答階段，尚無已完成章節可供重寫。"
            : !hasCompletedChapters
            ? "尚無已完成章節可供重寫。"
            : "重寫最新完成章節";
    }

    const exportBtn = document.getElementById("btn-top-export");
    if (exportBtn) {
        const canExport = !isWizard && hasCompletedChapters;
        exportBtn.disabled = !canExport;
        exportBtn.style.opacity = canExport ? "" : "0.45";
        exportBtn.style.cursor = canExport ? "" : "not-allowed";
        exportBtn.title = isWizard
            ? "目前處於創世問答階段，尚無已完成章節可供匯出。"
            : !hasCompletedChapters
            ? "尚無已完成章節可供匯出。"
            : "多格式匯出小說章節";
    }
}

function updateTopBar(cfg, stats) {
    const titleEl = document.getElementById("top-novel-title");
    if (titleEl) {
        if (!cfg.title && (cfg.pipeline_state === "IDLE_NO_PROJECT" || !cfg.pipeline_state)) {
            titleEl.textContent = "尚未選擇小說";
        } else {
            titleEl.textContent = cfg.title ? `《${cfg.title}》` : "未命名長篇小說";
        }
    }
    const st = cfg.pipeline_state || "WIZARD";
    const badgeEl = document.getElementById("top-pipeline-badge");
    if (badgeEl) {
        badgeEl.dataset.state = st;
        badgeEl.textContent = PIPELINE_STATE_LABELS[st] || `目前狀態：${st}`;
    }
    const btn = document.getElementById("btn-pause-resume");
    if (btn) {
        btn.textContent = st === "RUNNING" ? "⏸ 暫停連載" : "▶ 啟動連載";
        btn.classList.toggle("hidden", currentMainView === "projects" || st === "IDLE_NO_PROJECT");
    }
    syncPipelineButtonsState(st, stats);
}

function toggleVolumeNode(volNum) {
    if (collapsedVolumes.has(volNum)) {
        collapsedVolumes.delete(volNum);
    } else {
        collapsedVolumes.add(volNum);
    }
    const ul = document.getElementById(`vol-ch-list-${volNum}`);
    if (ul) {
        ul.classList.toggle("hidden", collapsedVolumes.has(volNum));
    }
}

const CHAPTER_STATUS_LABELS = {
    UNWRITTEN: "未寫",
    WRITING: "寫作中",
    STAGING: "暫存檢核中",
    REVISING: "自審修訂中",
    COMPLETED: "已完成",
    REVISED: "修訂過",
};

function renderOutlineTree(volumes, chapters) {
    const container = document.getElementById("outline-tree-container");
    if (!volumes || volumes.length === 0) {
        container.innerHTML = `<div class="muted-text" style="padding: 1.5rem 1rem; text-align: center; line-height: 1.6;">尚未建立分卷大綱<br/><small style="color:#94a3b8;">請先於第 1 步完成創世問答並生成藍圖</small></div>`;
        return;
    }
    container.innerHTML = volumes
        .map((vol) => {
            const vNum = vol.volume_num;
            const isCollapsed = collapsedVolumes.has(vNum);
            const vChs = chapters.filter((c) => c.volume_num === vNum);
            const volStateTag =
                vol.status === "COMPLETED" || vol.progress_percent >= 100
                    ? "已結卷"
                    : vol.status === "WRITING"
                    ? "連載中"
                    : "已規劃";
            const chItems = vChs
                .map(
                    (c) =>
                        `<li id="ch-row-${c.chapter_num}" class="ch-item ${activeReaderChapter === c.chapter_num ? "active-chapter" : ""}" onclick="loadChapterReader(${c.chapter_num})">
                            <span id="dot-ch-${c.chapter_num}" class="status-dot dot-${c.status}"></span>
                            <span>${escapeHtml(c.title)}</span>
                            <span class="muted-text" style="margin-left:auto;font-size:0.75rem;">【${CHAPTER_STATUS_LABELS[c.status] || c.status}】</span>
                        </li>`
                )
                .join("");
            return `<div class="vol-node">
                <div class="vol-header" onclick="toggleVolumeNode(${vNum})">
                    <span>▾ ${escapeHtml(vol.title)}</span>
                    <span>[${vol.completed_count}/${vol.total_count}章 ${vol.progress_percent}% ${volStateTag}]</span>
                </div>
                <div class="vol-progress-bar">
                    <div class="vol-progress-fill" style="width:${Math.min(100, vol.progress_percent || 0)}%;"></div>
                </div>
                <ul id="vol-ch-list-${vNum}" class="ch-list ${isCollapsed ? "hidden" : ""}">${chItems}</ul>
            </div>`;
        })
        .join("");
}

async function loadChapterReader(chapterNum) {
    activeReaderChapter = chapterNum;
    document.querySelectorAll(".ch-item").forEach((el) => el.classList.remove("active-chapter"));
    const activeRow = document.getElementById(`ch-row-${chapterNum}`);
    if (activeRow) activeRow.classList.add("active-chapter");
    const resp = await fetch(`/api/warroom/chapter/${chapterNum}`);
    if (!resp.ok) return;
    const data = await resp.json();
    const ch = data.chapter;
    const m = data.metrics;
    const zhStatus = CHAPTER_STATUS_LABELS[ch.status] || ch.status;
    document.getElementById("reader-chapter-title").textContent = `${ch.title}（狀態：${zhStatus}）`;
    document.getElementById("chapter-content-viewer").textContent =
        ch.final_content || ch.staging_content || "（本章尚未生成內容）";
    const card = document.getElementById("rubric-score-card");
    if (m) {
        card.classList.remove("hidden");
        const rubric = m.literary_rubric || {};
        const itemsHtml = (rubric.rubric_items || [])
            .map(
                (it) =>
                    `<span>[${escapeHtml(it.item_id)}:${escapeHtml(it.dimension)}] <strong>${escapeHtml(it.verdict)}</strong></span>`
            )
            .join(" ｜ ");
        const anchorHtml = rubric.best_paragraph_text
            ? `<div style="margin-top:0.35rem;" class="muted-text"><strong>動態語感錨點段落（${rubric.best_paragraph_score || 0}分）：</strong>${escapeHtml(rubric.best_paragraph_text)}</div>`
            : "";
        card.innerHTML = `
            <div class="rubric-chips-row">
                <span class="rubric-chip chip-gold"><strong>文學自審總分</strong><span>${m.literary_score} 分</span></span>
                <span class="rubric-chip"><strong>字數</strong><span>${m.word_count} 字</span></span>
                <span class="rubric-chip"><strong>衝突強度</strong><span>${m.tension_intensity}/5</span></span>
                <span class="rubric-chip"><strong>目標推進</strong><span>${m.goal_progress}/5</span></span>
                <span class="rubric-chip"><strong>情緒爽感</strong><span>${m.emotional_payoff}/5</span></span>
                <span class="rubric-chip"><strong>修訂紀錄</strong><span>微重寫 ${m.micro_rewrite_count} 次 · 潤色 ${m.revision_count} 輪</span></span>
            </div>
            <details class="rubric-details-fold">
                <summary>展開 10 項評分細項與動態語感錨點</summary>
                <div class="rubric-items-grid">${itemsHtml || "無細項資料"}</div>
                ${anchorHtml}
            </details>
        `;
    } else {
        card.classList.add("hidden");
    }
}

function renderHardwareAndStats(hw, stats, refStatus) {
    const hasChapters = Boolean(stats && stats.completed_chapters > 0);
    const hasGpu = Boolean(hw && hw.vram_total_mb > 0);
    const vramPct = hasGpu ? Math.min(100, Math.max(0, hw.vram_percent || 0)) : 0;
    const tempC = (hw && typeof hw.gpu_temp_c === "number") ? hw.gpu_temp_c : 0;
    const vramUsedGb = hasGpu ? ((hw.vram_used_mb || 0) / 1024).toFixed(2) : "--";
    const vramTotalGb = hasGpu ? ((hw.vram_total_mb || 0) / 1024).toFixed(2) : "--";

    const deviceLabel = document.getElementById("hw-device-label");
    if (deviceLabel) {
        if (hw && hw.gpu_name) {
            deviceLabel.textContent = `${hw.gpu_name} 實時監控`;
        } else {
            deviceLabel.textContent = "GPU 實時硬體監控";
        }
    }

    document.getElementById("total-words-badge").textContent = `${stats.total_words || 0} 字`;
    document.getElementById("hw-vram").textContent = hasGpu ? `${vramUsedGb} / ${vramTotalGb} GB` : "-- / -- GB";
    document.getElementById("hw-temp").textContent = tempC > 0 ? `${tempC} °C` : "-- °C";
    document.getElementById("hw-tps").textContent = hasChapters ? `${stats.tokens_per_sec || 0} 字/秒` : "-- 字/秒";
    document.getElementById("hw-duration").textContent = hasChapters ? `${stats.last_chapter_duration_sec || 0} 秒` : "-- 秒";
    document.getElementById("hw-avg-score").textContent = hasChapters ? `${stats.avg_literary_score || 0} 分` : "-- 分";
    document.getElementById("hw-sleep-lock").textContent = hw.sleep_lock_active ? "已鎖定防睡眠" : "待命";
    const vramBar = document.getElementById("hw-vram-bar");
    if (vramBar) {
        vramBar.style.width = `${vramPct}%`;
    }
    const tempBar = document.getElementById("hw-temp-bar");
    if (tempBar) {
        tempBar.style.width = `${tempC > 0 ? Math.min(100, tempC) : 0}%`;
        tempBar.className = `mini-progress-fill ${tempC >= 80 ? "fill-warn" : "fill-green"}`;
    }
    updateReferenceProgressPill(refStatus);
}

function updateReferenceProgressPill(refStatus) {
    const pill = document.getElementById("ref-progress-pill");
    pill.classList.remove("hidden");
    if (refStatus.active) {
        pill.textContent = `書庫入庫中 ${refStatus.processed_chunks}/${refStatus.total_chunks} 塊`;
    } else {
        pill.textContent = `書庫條目：${refStatus.entries || 0} 則（標籤 ${refStatus.tags || 0}）`;
    }
}

function renderTensionECGChart(metrics) {
    const canvas = document.getElementById("tension-ecg-canvas");
    if (!canvas || canvas.offsetParent === null) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
    ctx.lineWidth = 1;
    for (let lv = 1; lv <= 5; lv += 1) {
        const yGrid = h - (lv / 5) * (h - 26) - 13;
        ctx.beginPath();
        ctx.moveTo(12, yGrid);
        ctx.lineTo(w - 12, yGrid);
        ctx.stroke();
    }
    ctx.strokeStyle = "rgba(255, 59, 48, 0.38)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    const yWarnHigh = h - (4 / 5) * (h - 26) - 13;
    const yWarnLow = h - (1 / 5) * (h - 26) - 13;
    ctx.moveTo(12, yWarnHigh);
    ctx.lineTo(w - 12, yWarnHigh);
    ctx.moveTo(12, yWarnLow);
    ctx.lineTo(w - 12, yWarnLow);
    ctx.stroke();
    ctx.setLineDash([]);
    if (!metrics.length) {
        ctx.fillStyle = "rgba(148, 163, 184, 0.65)";
        ctx.font = "12px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("尚無章節張力數據（請先啟動寫作）", w / 2, h / 2 + 4);
        return;
    }
    const drawSeries = (key, color) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        metrics.forEach((m, i) => {
            const x = 18 + (i / Math.max(1, metrics.length - 1)) * (w - 36);
            const val = m[key] || 3;
            const y = h - (val / 5) * (h - 26) - 13;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
    };
    drawSeries("tension_intensity", "#e11d48");
    drawSeries("goal_progress", "#007aff");
    drawSeries("emotional_payoff", "#16a34a");
}

function renderAnomalyHistory(anomalies) {
    const box = document.getElementById("anomaly-log-list");
    if (!box) return;
    box.replaceChildren();
    if (!anomalies.length) {
        const empty = document.createElement("div");
        empty.className = "muted-text";
        empty.textContent = "目前尚無異常告警紀錄。";
        box.appendChild(empty);
        return;
    }
    for (const a of anomalies) {
        const details = document.createElement("details");
        details.className = "anomaly-item";
        const summary = document.createElement("summary");
        // textContent keeps model output and exception messages inert.
        summary.textContent = `[第${a.chapter_num}章｜${a.anomaly_type}] ${a.resolution_method}`;
        const body = document.createElement("div");
        body.className = "muted-text";
        body.textContent = `時間：${a.timestamp}｜受影響章節：${a.affected_chapters}`;
        const br = document.createElement("br");
        const detail = document.createTextNode(`詳情：${a.detail_diff_json}`);
        body.append(br, detail);
        details.append(summary, body);
        box.appendChild(details);
    }
}

// --- Toolbar Controls & Modals ---
async function togglePauseResume() {
    const badgeEl = document.getElementById("top-pipeline-badge");
    const badge = badgeEl.dataset.state || badgeEl.textContent;
    if (badge === "RUNNING") {
        await fetch("/api/warroom/pause", { method: "POST" });
        showToast("已發送暫停連載指令，待當前章完成後安全掛起。");
    } else {
        const resp = await fetch("/api/warroom/resume", { method: "POST" });
        const data = await resp.json();
        if (handleAiModelError(data)) return;
        if (!resp.ok) {
            showToast(data.detail || data.error || "啟動連載失敗，請重試。");
            return;
        }
        setStageView("WARROOM");
        showToast("已啟動全自動連載！");
    }
    await refreshWarroomDashboard();
}

async function triggerSingleStep() {
    await withButtonLoading("#btn-step-once", async () => {
        showToast("正在生成並自審下一章...");
        const resp = await fetch("/api/warroom/step", { method: "POST" });
        const data = await resp.json();
        if (handleAiModelError(data)) return;
        if (!resp.ok) {
            showToast(data.detail || data.error || "單步推進失敗，請重試。");
            return;
        }
        await refreshWarroomDashboard();
        if (data.result && data.result.chapter_num) {
            await loadChapterReader(data.result.chapter_num);
        }
    }, "推進中...");
}

function openRewriteModal() {
    const btn = document.getElementById("btn-warroom-rewrite");
    if (btn && btn.disabled) {
        showToast("目前處於創世問答階段，尚無已完成章節可供重寫。");
        return;
    }
    const el = document.getElementById("modal-rewrite");
    if (el) {
        el.classList.remove("hidden");
        el.style.display = "flex";
    }
}


async function confirmRewriteLatest() {
    closeModal("modal-rewrite");
    await withButtonLoading("#btn-warroom-rewrite", async () => {
        showToast("正在重抽隨機種子並重寫最新完成章節...");
        const resp = await fetch("/api/warroom/rewrite_latest", { method: "POST" });
        const res = await resp.json();
        if (handleAiModelError(res)) return;
        if (!resp.ok) {
            showToast(res.detail || res.error || "重寫章節失敗，請重試。");
            return;
        }
        await refreshWarroomDashboard();
        await loadChapterReader(res.chapter_num);
        showToast(`第 ${res.chapter_num} 章已完成重寫（得分：${res.literary_score}）。`);
    }, "重寫中...");
}

async function createManualSnapshot() {
    const btn = document.getElementById("btn-manual-snapshot");
    if (btn && btn.disabled) {
        showToast("目前處於創世問答階段，尚未生成小說藍圖，無法建立快照。");
        return;
    }
    try {
        const resp = await fetch("/api/warroom/snapshot/create", { method: "POST" });
        const snap = await resp.json();
        if (!resp.ok) {
            showToast(snap.detail || snap.error || "目前無法建立快照。");
            return;
        }
        showToast(`已建立永久手動快照：${snap.snapshot_id}`);
    } catch (e) {
        showToast("建立快照時發生異常，請重試。");
    }
}

async function openSnapshotModal() {
    const el = document.getElementById("modal-snapshots");
    if (el) {
        el.classList.remove("hidden");
        el.style.display = "flex";
    }
    const resp = await fetch("/api/warroom/snapshot/list");
    const data = await resp.json();
    const box = document.getElementById("snapshot-list-box");
    box.innerHTML = (data.snapshots || [])
        .map(
            (s) =>
                `<div class="qa-tree-item" style="display:flex;justify-content:space-between;align-items:center;">
                    <div><strong>${s.snapshot_id}</strong><br/>類型：${s.kind}｜章數：${s.completed_chapters}｜字數：${s.total_words}</div>
                    <button class="btn btn-warn btn-sm" onclick="restoreSnapshot('${s.snapshot_id}')">還原此快照</button>
                </div>`
        )
        .join("") || "尚無快照檔案。";
}

async function restoreSnapshot(snapshotId) {
    const resp = await fetch("/api/warroom/snapshot/restore", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ snapshot_id: snapshotId }),
    });
    const data = await resp.json();
    if (!resp.ok) {
        showToast(data.detail || "還原失敗");
        return;
    }
    closeModal("modal-snapshots");
    await refreshWarroomDashboard();
    showToast(`已成功回滾還原至快照：${snapshotId}`);
}

function openExportModal() {
    const btn = document.getElementById("btn-top-export");
    if (btn && btn.disabled) {
        showToast("目前處於創世問答階段，尚無已完成章節可供匯出。");
        return;
    }
    const el = document.getElementById("modal-export");
    if (el) {
        el.classList.remove("hidden");
        el.style.display = "flex";
    }
}

function onExportScopeChange() {
    const scope = document.getElementById("exp-scope").value;
    document.getElementById("exp-vol-row").classList.toggle("hidden", scope !== "VOLUME");
    document.getElementById("exp-range-row").classList.toggle("hidden", scope !== "RANGE");
}

async function submitNovelExport() {
    const formats = Array.from(document.querySelectorAll(".exp-fmt:checked")).map((c) => c.value);
    const payload = {
        scope_type: document.getElementById("exp-scope").value,
        volume_num: parseInt(document.getElementById("exp-vol-num").value, 10) || 1,
        start_chapter: parseInt(document.getElementById("exp-start-ch").value, 10) || 1,
        end_chapter: parseInt(document.getElementById("exp-end-ch").value, 10) || 10,
        formats: formats,
        merge_mode: document.getElementById("exp-merge-mode").value,
    };
    const resp = await fetch("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    const data = await resp.json();
    const box = document.getElementById("exp-result-box");
    if (!resp.ok) {
        box.textContent = data.detail || "匯出失敗";
        return;
    }
    box.innerHTML = `已成功匯出 ${data.chapter_count} 章至：<br/><code>${data.export_dir}</code>`;
    showToast("全系列格式匯出完成！");
}

