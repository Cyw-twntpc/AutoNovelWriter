// --- Projects Module ---
// --- Novel Project Hub & Workspace Navigation ---
let currentMainView = "projects";
let currentProjectsCount = 0;
let currentActiveProjectId = "";

function switchMainView(mainView) {
    if (mainView === "studio" && currentProjectsCount === 0) {
        showToast("作品庫目前尚無任何小說，請先建立新小說。");
        mainView = "projects";
    }
    currentMainView = mainView;
    const isProjects = mainView === "projects";
    const vProj = document.getElementById("view-projects");
    const vStudio = document.getElementById("view-studio");
    const btnProj = document.getElementById("nav-btn-projects");
    const btnStudio = document.getElementById("nav-btn-studio");
    const btnPauseResume = document.getElementById("btn-pause-resume");

    if (vProj) vProj.classList.toggle("hidden", !isProjects);
    if (vStudio) vStudio.classList.toggle("hidden", isProjects);
    if (btnProj) btnProj.classList.toggle("active", isProjects);
    if (btnStudio) btnStudio.classList.toggle("active", !isProjects);
    if (btnPauseResume) btnPauseResume.classList.toggle("hidden", isProjects || currentProjectsCount === 0);

    if (isProjects) {
        loadProjectsList();
    } else {
        refreshWarroomDashboard();
    }
}

async function loadProjectsList() {
    try {
        const resp = await fetch("/api/projects");
        if (!resp.ok) return;
        const data = await resp.json();
        const projects = data.projects || [];
        const activeId = data.active_project_id || "";
        currentProjectsCount = projects.length;
        currentActiveProjectId = activeId;

        const totalProjectsEl = document.getElementById("kpi-total-projects");
        const totalWordsEl = document.getElementById("kpi-total-words");
        const activeTitleEl = document.getElementById("kpi-active-title");

        if (totalProjectsEl) totalProjectsEl.textContent = `${projects.length} 本`;

        const totalWords = projects.reduce((acc, p) => acc + (p.current_words || p.total_words || 0), 0);
        if (totalWordsEl) totalWordsEl.textContent = `${formatWordCount(totalWords)} 字`;

        const activeProj = projects.find((p) => p.project_id === activeId);
        if (activeTitleEl) {
            activeTitleEl.textContent = activeProj ? `《${activeProj.title}》` : "無";
        }
        const topTitleEl = document.getElementById("top-novel-title");
        if (topTitleEl) {
            topTitleEl.textContent = activeProj ? `《${activeProj.title}》` : "尚未選擇小說";
        }

        renderProjectsGrid(projects, activeId);
    } catch (_err) {
        // Silent error catch for recurring background poll
    }
}

function formatNormalizedDateTime(val) {
    if (!val || val === "--") return "--";
    try {
        const str = String(val).trim();
        if (/^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}$/.test(str)) {
            return str;
        }
        let d = new Date(str);
        if (isNaN(d.getTime()) && str.includes(" ")) {
            d = new Date(str.replace(" ", "T") + "Z");
        }
        if (isNaN(d.getTime())) {
            if (str.length >= 19 && str[4] === "-" && str[7] === "-") {
                return str.substring(0, 19).replace("T", " ");
            }
            return str;
        }
        const pad = (n) => String(n).padStart(2, "0");
        return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
    } catch (_) {
        return String(val);
    }
}

function renderProjectsGrid(projects, activeId) {
    const container = document.getElementById("projects-grid-container");
    if (!container) return;
    container.onclick = handleProjectsGridClick;

    if (!projects || projects.length === 0) {
        container.innerHTML = `
            <div class="card muted-text" style="grid-column: 1 / -1; text-align: center; padding: 4rem 2rem; border: 2px dashed rgba(255, 255, 255, 0.15); border-radius: 12px; background: rgba(255, 255, 255, 0.02);">
                <div style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.8;">📚</div>
                <h3 style="font-size: 1.3rem; margin-bottom: 0.5rem; color: #fff;">作品庫目前沒有任何小說</h3>
                <p style="font-size: 0.95rem; margin-bottom: 1.5rem; color: #a0aec0;">
                    全庫所有小說均已清空。您可以點擊下方按鈕創建全新長篇小說，立即開始創世旅程！
                </p>
                <button class="btn btn-primary" style="padding: 0.75rem 1.8rem; font-size: 1rem;" onclick="handleCreateNovelDirect()">
                    ＋ 創建全新長篇小說
                </button>
            </div>
        `;
        return;
    }

    container.innerHTML = projects
        .map((p) => {
            const isActive = p.project_id === activeId || p.is_active;
            const currentStage = p.current_stage || p.pipeline_state || "WIZARD";
            const stageLabel = PIPELINE_STATE_LABELS[currentStage] || currentStage || "創世問答";
            const progressPct = Math.min(100, Math.max(0, p.progress_percent || 0));
            const wordsNum = p.current_words ?? p.total_words ?? 0;
            const formattedWords = formatWordCount(wordsNum);
            const totalChs = p.total_chapters ?? p.chapters_count ?? 0;
            const completedChs = p.completed_chapters ?? 0;
            const chProgressText = totalChs > 0 ? `${completedChs} / ${totalChs} 章` : `${completedChs} 章`;
            const volsText = `(${p.volumes_count || 0} 卷)`;

            return `
            <div class="project-card ${isActive ? 'active-project' : ''}" data-project-id="${escapeHtml(p.project_id)}">
                <div class="project-card-header">
                    <div class="project-card-title-row">
                        <h3 class="project-card-title">《${escapeHtml(p.title || "未命名小說")}》</h3>
                        <button class="project-card-btn-rename" title="重命名小說" data-action="rename" data-project-id="${escapeHtml(p.project_id)}" data-title="${escapeHtml(p.title || "")}">✏️</button>
                    </div>
                    ${isActive ? '<span class="active-novel-pill">● 當前寫作中</span>' : ''}
                </div>

                <div class="project-card-tags">
                    <span class="project-tag-genre">${escapeHtml(p.genre_primary || "通用")}</span>
                    <span class="project-tag-genre">${escapeHtml(p.genre_sub || "長篇")}</span>
                    <span class="project-state-pill" data-state="${escapeHtml(currentStage)}">${escapeHtml(stageLabel)}</span>
                </div>

                <div class="project-card-metrics">
                    <div class="project-metric-row">
                        <span>創作字數</span>
                        <strong>${formattedWords}</strong>
                    </div>
                    <div class="project-metric-row">
                        <span>章節進度</span>
                        <strong>${chProgressText} <small>${volsText}</small></strong>
                    </div>
                    <div class="project-metric-row">
                        <span>預估完成度</span>
                        <strong>${progressPct}%</strong>
                    </div>
                    <div class="project-progress-track">
                        <div class="project-progress-fill" style="width: ${progressPct}%;"></div>
                    </div>
                </div>

                <div class="project-card-footer">
                    <span>修改時間：${escapeHtml(formatNormalizedDateTime(p.updated_at || p.created_at))}</span>
                </div>

                <div class="project-card-actions">
                    <button class="btn ${isActive ? 'btn-primary' : 'btn-accent'} btn-sm" data-action="switch" data-project-id="${escapeHtml(p.project_id)}">
                        🖋️ ${isActive ? '進入工作室' : '續寫作品'}
                    </button>
                    <button class="btn btn-glass btn-sm" data-action="export" data-project-id="${escapeHtml(p.project_id)}">
                        📤 匯出
                    </button>
                    <button class="btn btn-glass btn-sm text-red" data-action="delete" data-project-id="${escapeHtml(p.project_id)}" data-title="${escapeHtml(p.title || "")}" data-only-one="${projects.length === 1 ? "1" : "0"}" title="永久刪除這部小說">
                        🗑️ 刪除
                    </button>
                </div>
            </div>
            `;
        })
        .join("");
}

/**
 * Delegated click handler for the project grid.
 * Using data-* attributes avoids interpolating book titles into inline JS,
 * where an apostrophe would break out of the string literal entirely.
 */
function handleProjectsGridClick(event) {
    const btn = event.target.closest("[data-action]");
    if (!btn) return;
    const action = btn.dataset.action;
    const projectId = btn.dataset.projectId || "";
    const title = btn.dataset.title || "";
    if (action === "rename") {
        openRenameProjectModal(projectId, title);
    } else if (action === "switch") {
        switchActiveProject(projectId, true);
    } else if (action === "export") {
        openProjectExportModal(projectId);
    } else if (action === "delete") {
        deleteProjectConfirm(projectId, title, btn.dataset.onlyOne === "1");
    }
}

async function handleCreateNovelDirect() {
    const badgeEl = document.getElementById("top-pipeline-badge");
    const isRunning = (badgeEl?.dataset?.state === "RUNNING");
    if (isRunning) {
        try {
            await fetch("/api/warroom/pause", { method: "POST" });
            showToast("已自動暫停當前作品連載，正在建立新小說...");
        } catch (_e) {
            // ignore network pause error
        }
    } else {
        showToast("正在建立新小說專屬檔案沙盒...");
    }

    try {
        const resp = await fetch("/api/projects/create", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                title: "未命名長篇小說",
                genre_primary: "仙俠修真",
                genre_sub: "古典修仙",
                min_word_count: 5000000,
            }),
        });
        const data = await resp.json();
        if (!resp.ok) {
            showToast(data.detail || data.error || "建立新小說失敗。");
            return;
        }

        await loadProjectsList();
        await initWizardState();
        await refreshWarroomDashboard();
        setStageView("WIZARD");
        switchMainView("studio");
        showToast("已成功建立全新小說，進入創世問答！");
    } catch (_err) {
        showToast("建立小說時發生連線錯誤。");
    }
}

function openNewProjectModal() {
    handleCreateNovelDirect();
}

async function switchActiveProject(projectId, navigateToStudio = false) {
    try {
        showToast("正在切換小說專案...");
        const resp = await fetch("/api/projects/switch", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ project_id: projectId }),
        });
        const data = await resp.json();
        if (!resp.ok) {
            showToast(data.detail || data.error || "切換專案失敗。");
            return;
        }

        const activeMeta = data.active_project || {};
        showToast(`已切換至《${activeMeta.title || projectId}》！`);

        const topTitle = document.getElementById("top-novel-title");
        if (topTitle) topTitle.textContent = `《${activeMeta.title || "未命名小說"}》`;

        const st = data.stage || activeMeta.current_stage || "WIZARD";
        if (st === "WIZARD") {
            await initWizardState();
            setStageView("WIZARD");
        } else if (st === "REVIEW_GATE") {
            await loadReviewGateState();
            setStageView("REVIEW_GATE");
        } else {
            setStageView("WARROOM");
        }

        await refreshWarroomDashboard();
        await loadProjectsList();

        if (navigateToStudio) {
            switchMainView("studio");
        }
    } catch (_e) {
        showToast("切換專案時發生連線錯誤。");
    }
}

function openRenameProjectModal(projectId, currentTitle) {
    const modal = document.getElementById("modal-rename-project");
    const idInput = document.getElementById("rename-project-id");
    const titleInput = document.getElementById("rename-project-title");
    if (idInput) idInput.value = projectId;
    if (titleInput) {
        titleInput.value = currentTitle;
        setTimeout(() => titleInput.focus(), 50);
    }
    if (modal) {
        modal.classList.remove("hidden");
        modal.style.display = "flex";
    }
}

async function submitRenameProject() {
    const projectId = document.getElementById("rename-project-id")?.value;
    const newTitle = (document.getElementById("rename-project-title")?.value || "").trim();
    if (!newTitle) {
        showToast("小說名稱不能為空。");
        return;
    }
    closeModal("modal-rename-project");

    try {
        const resp = await fetch("/api/projects/rename", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ project_id: projectId, new_title: newTitle }),
        });
        const data = await resp.json();
        if (!resp.ok) {
            showToast(data.detail || data.error || "重新命名失敗。");
            return;
        }
        showToast(`已成功重新命名為《${newTitle}》！`);
        await loadProjectsList();
        await refreshWarroomDashboard();
    } catch (_e) {
        showToast("重新命名時發生連線錯誤。");
    }
}

async function deleteProjectConfirm(projectId, title, isOnlyOne = false) {
    const warningMsg = isOnlyOne
        ? `⚠️ 注意：這是作品庫中「最後一部小說」！\n\n確定要永久刪除《${title}》嗎？\n刪除後作品庫將清空為空狀態，所有大綱與章節資料庫均無法復原！`
        : `確定要永久刪除小說《${title}》及其所有大綱與章節資料庫嗎？\n\n此操作無法復原！`;
    if (!window.confirm(warningMsg)) {
        return;
    }
    try {
        const resp = await fetch(`/api/projects/${encodeURIComponent(projectId)}`, {
            method: "DELETE",
        });
        const data = await resp.json();
        if (!resp.ok) {
            showToast(data.detail || data.error || "刪除專案失敗。");
            return;
        }
        showToast(`已永久刪除小說《${title}》。`);
        await loadProjectsList();
        await refreshWarroomDashboard();
        if (currentProjectsCount === 0) {
            switchMainView("projects");
        }
    } catch (_e) {
        showToast("刪除專案時發生連線錯誤。");
    }
}

async function openProjectExportModal(projectId) {
    try {
        const pmResp = await fetch("/api/projects");
        const pmData = await pmResp.json();
        const p = (pmData.projects || []).find((item) => item.project_id === projectId);
        if (p && (!p.chapters_count || p.chapters_count === 0)) {
            showToast(`《${p.title}》尚處於初期籌備階段，無已完成章節可供匯出。`);
            return;
        }
        if (pmData.active_project_id !== projectId) {
            await switchActiveProject(projectId, false);
        }
        const el = document.getElementById("modal-export");
        if (el) {
            el.classList.remove("hidden");
            el.style.display = "flex";
        }
    } catch (_e) {
        openExportModal();
    }
}

