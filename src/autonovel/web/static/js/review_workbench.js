// --- Review Workbench Module ---
// --- Stage 2: Review Gate Structured Workbench ---
async function loadReviewGateState() {
    const resp = await fetch("/api/review/state");
    const data = await resp.json();
    populateReviewGateUI(data);
}

function buildHierarchyRowHtml(h) {
    return `<div class="edit-grid-row rev-hl-row" data-cat="${escapeHtml(h.category || "CULTIVATION")}" style="grid-template-columns:75px 180px 1fr 64px;">
        <input type="number" class="hl-num" value="${h.level_num || 1}" min="1" />
        <input type="text" class="hl-name" value="${escapeHtml(h.level_name || "")}" placeholder="境界名稱" />
        <input type="text" class="hl-desc" value="${escapeHtml(h.description || "")}" placeholder="境界描述" />
        <button class="btn btn-sm" onclick="this.parentElement.remove()">刪除</button>
    </div>`;
}

function addHierarchyRow() {
    const count = document.querySelectorAll(".rev-hl-row").length + 1;
    const container = document.getElementById("rev-hierarchy-table");
    container.insertAdjacentHTML(
        "beforeend",
        buildHierarchyRowHtml({ level_num: count, level_name: `第${count}階`, description: "" })
    );
}

function buildCharacterCardHtml(c) {
    let aliasesStr = "";
    if (Array.isArray(c.aliases)) {
        aliasesStr = c.aliases.join(", ");
    } else if (typeof c.aliases === "string") {
        try {
            const parsed = JSON.parse(c.aliases);
            aliasesStr = Array.isArray(parsed) ? parsed.join(", ") : c.aliases;
        } catch {
            aliasesStr = c.aliases;
        }
    }
    return `<div class="char-edit-card rev-char-card">
        <div class="char-card-row1">
            <input type="text" class="ch-name" value="${escapeHtml(c.name || "")}" placeholder="角色姓名" style="font-weight:700;" />
            <input type="text" class="ch-role" value="${escapeHtml(c.role_type || "MAJOR")}" placeholder="定位(PROTAGONIST/MAJOR)" />
            <input type="number" class="ch-rank" value="${c.rank_level || 1}" min="1" title="境界等級" style="width:75px;" />
            <input type="text" class="ch-fac" value="${escapeHtml(c.faction_name || "")}" placeholder="所屬勢力" />
            <button class="btn btn-sm btn-glass text-red" onclick="this.closest('.rev-char-card').remove()" title="刪除角色">✕ 刪除</button>
        </div>
        <div class="char-card-row2">
            <input type="text" class="ch-aliases" value="${escapeHtml(aliasesStr)}" placeholder="角色別名(逗號分隔)" style="flex:1;" />
            <input type="text" class="ch-loc" value="${escapeHtml(c.location_name || "")}" placeholder="初始地點" style="flex:1;" />
        </div>
        <textarea class="ch-desc" rows="2" placeholder="人物性格、外貌特徵與核心動機設定">${escapeHtml(c.description || "")}</textarea>
    </div>`;
}

function addCharacterCard() {
    const container = document.getElementById("rev-characters-list");
    container.insertAdjacentHTML(
        "beforeend",
        buildCharacterCardHtml({ name: "", aliases: [], role_type: "SUPPORTING", rank_level: 1 })
    );
}

function buildStageGateCardHtml(g, idx) {
    const stageIdx = g.stage_index || (idx + 1);
    const title = g.stage_title || `第 ${stageIdx} 階段`;
    const words = g.target_word_quota || 1000000;
    const maxRank = g.max_allowed_rank_level || stageIdx * 2;
    const mystery = g.core_mystery_quota || "";
    const milestone = g.milestone_goal || "";

    return `
    <div class="rev-stage-card" data-stage-index="${stageIdx}">
        <div class="rev-stage-header">
            <span class="rev-stage-badge">STAGE ${stageIdx}</span>
            <input type="text" class="stage-title-input" value="${escapeHtml(title)}" style="font-weight:700; width:150px; padding:2px 6px;" placeholder="階段名稱" oninput="syncStageGatesToJson()" />
            <button class="btn btn-sm btn-glass text-red" onclick="this.closest('.rev-stage-card').remove(); syncStageGatesToJson();" title="刪除階段">✕</button>
        </div>
        <div class="form-row" style="margin:0; gap:6px;">
            <label style="flex:1;">目標字數
                <input type="number" class="stage-words-input" value="${words}" min="10000" step="100000" oninput="syncStageGatesToJson()" />
            </label>
            <label style="width:90px;">階梯上限
                <input type="number" class="stage-rank-input" value="${maxRank}" min="1" max="50" oninput="syncStageGatesToJson()" />
            </label>
        </div>
        <label style="margin-top:2px;">核心伏筆謎團
            <input type="text" class="stage-mystery-input" value="${escapeHtml(mystery)}" placeholder="揭曉的世界法則與核心伏筆" oninput="syncStageGatesToJson()" />
        </label>
        <label style="margin-top:2px;">里程碑破局目標
            <input type="text" class="stage-milestone-input" value="${escapeHtml(milestone)}" placeholder="本階段主線破局目標" oninput="syncStageGatesToJson()" />
        </label>
    </div>
    `;
}

function addStageGateCard() {
    const container = document.getElementById("rev-stage-gates-container");
    if (!container) return;
    const currentCount = container.querySelectorAll(".rev-stage-card").length + 1;
    container.insertAdjacentHTML(
        "beforeend",
        buildStageGateCardHtml(
            {
                stage_index: currentCount,
                stage_title: `第 ${currentCount} 階段`,
                target_word_quota: 1000000,
                max_allowed_rank_level: currentCount * 2,
                core_mystery_quota: "揭示世界底層真相",
                milestone_goal: "完成本階段主線破局",
            },
            currentCount - 1
        )
    );
    syncStageGatesToJson();
}

function syncStageGatesToJson() {
    const cards = Array.from(document.querySelectorAll(".rev-stage-card"));
    const stageGates = cards.map((card, idx) => ({
        stage_index: parseInt(card.getAttribute("data-stage-index"), 10) || (idx + 1),
        stage_title: card.querySelector(".stage-title-input")?.value.trim() || `第 ${idx + 1} 階段`,
        target_word_quota: parseInt(card.querySelector(".stage-words-input")?.value, 10) || 1000000,
        max_allowed_rank_level: parseInt(card.querySelector(".stage-rank-input")?.value, 10) || (idx + 1) * 2,
        core_mystery_quota: card.querySelector(".stage-mystery-input")?.value.trim() || "",
        milestone_goal: card.querySelector(".stage-milestone-input")?.value.trim() || "",
    }));
    const hiddenJsonEl = document.getElementById("rev-stage-gates-json");
    if (hiddenJsonEl) {
        hiddenJsonEl.value = JSON.stringify(stageGates, null, 2);
    }
}

function buildChapterOutlineRowHtml(ch) {
    let beats = [];
    try {
        beats = typeof ch.beats_json === "string" ? JSON.parse(ch.beats_json || "[]") : (ch.beats_json || []);
    } catch {
        beats = [];
    }
    const beatsHtml = (beats || []).map((b, bIdx) => `
        <div class="rev-beat-item" data-beat-index="${b.beat_index || (bIdx + 1)}">
            <div class="rev-beat-header">
                <span class="rev-beat-pill">Beat ${b.beat_index || (bIdx + 1)}</span>
                <input type="text" class="beat-tag-input" value="${escapeHtml(b.scene_tag || "")}" placeholder="分鏡標籤（如：日常密謀）" style="width:130px; font-weight:700; padding:2px 6px;" />
                <input type="text" class="beat-pacing-input" value="${escapeHtml(b.pacing_instruction || "")}" placeholder="節奏指令" style="flex:1; padding:2px 6px;" />
                <button class="btn btn-sm btn-glass text-red" onclick="this.closest('.rev-beat-item').remove()" title="刪除分鏡">✕</button>
            </div>
            <textarea class="beat-objective-input" rows="1" placeholder="本鏡情節推進目標">${escapeHtml(b.plot_objective || "")}</textarea>
        </div>
    `).join("");

    return `
    <div class="qa-tree-item rev-ch-row" data-chnum="${ch.chapter_num}" style="margin-bottom:0.75rem; border-left:3px solid var(--accent-blue);">
        <div class="form-row" style="margin-bottom:4px;">
            <strong style="color:var(--accent-blue);">第 ${ch.chapter_num} 章</strong>
            <input type="text" class="rev-ch-title" value="${escapeHtml(ch.title || "")}" style="flex:1; max-width:360px; font-weight:700;" placeholder="章節標題" />
            <button class="btn btn-sm btn-glass" onclick="addBeatToChapter(this)">＋ 新增分鏡</button>
        </div>
        <div class="rev-chapter-beats-list">
            ${beatsHtml}
        </div>
    </div>
    `;
}

function addBeatToChapter(btn) {
    const row = btn.closest(".rev-ch-row");
    if (!row) return;
    const beatsList = row.querySelector(".rev-chapter-beats-list");
    if (!beatsList) return;
    const count = beatsList.querySelectorAll(".rev-beat-item").length + 1;
    const newBeatHtml = `
        <div class="rev-beat-item" data-beat-index="${count}">
            <div class="rev-beat-header">
                <span class="rev-beat-pill">Beat ${count}</span>
                <input type="text" class="beat-tag-input" value="交鋒轉折" placeholder="分鏡標籤" style="width:130px; font-weight:700; padding:2px 6px;" />
                <input type="text" class="beat-pacing-input" value="短句疾進，衝突升溫" placeholder="節奏指令" style="flex:1; padding:2px 6px;" />
                <button class="btn btn-sm btn-glass text-red" onclick="this.closest('.rev-beat-item').remove()" title="刪除分鏡">✕</button>
            </div>
            <textarea class="beat-objective-input" rows="1" placeholder="本鏡情節推進目標">推進情節衝突與危機</textarea>
        </div>
    `;
    beatsList.insertAdjacentHTML("beforeend", newBeatHtml);
}

function populateReviewGateUI(data) {
    const cfg = data.project_config || {};
    updateTopBar(cfg);
    document.getElementById("rev-world-rules").value = (data.world_rules || [])
        .map((r) => r.rule_text)
        .join("\n");
    document.getElementById("rev-hierarchy-table").innerHTML = (data.hierarchy_levels || [])
        .map((h) => buildHierarchyRowHtml(h))
        .join("");
    document.getElementById("rev-characters-list").innerHTML = (data.characters || [])
        .map((c) => buildCharacterCardHtml(c))
        .join("");
    document.getElementById("rev-factions-list").value = (data.factions || []).map((f) => f.name).join("\n");
    document.getElementById("rev-locations-list").value = (data.locations || []).map((l) => l.name).join("\n");
    const gOutline = JSON.parse(cfg.global_outline_json || "{}");
    document.getElementById("rev-opening-plan").value = gOutline.opening_plan || "";
    document.getElementById("rev-ending-draft").value = gOutline.ending_draft || "";
    
    // Render visual Stage Gates Bento cards
    const stageGates = gOutline.stage_gates || [];
    const stageContainer = document.getElementById("rev-stage-gates-container");
    if (stageContainer) {
        stageContainer.innerHTML = stageGates.map((g, idx) => buildStageGateCardHtml(g, idx)).join("");
    }
    const hiddenJsonEl = document.getElementById("rev-stage-gates-json");
    if (hiddenJsonEl) {
        hiddenJsonEl.value = JSON.stringify(stageGates, null, 2);
    }

    // Render visual Chapter Outline Scene Beats
    const outlineBox = document.getElementById("rev-outline-box");
    if (outlineBox) {
        outlineBox.innerHTML = (data.chapters || []).map((ch) => buildChapterOutlineRowHtml(ch)).join("");
    }

    document.getElementById("rev-style-contract").value = cfg.style_contract_text || "";
    document.getElementById("rev-static-anchor").value = cfg.static_anchor_text || "";
    const bl = JSON.parse(cfg.blacklist_json || "{}");
    document.getElementById("rev-bl-tier1").value = (bl.tier1_global_cliches || []).join("\n");
    document.getElementById("rev-bl-tier2").value = (bl.tier2_genre_anachronisms || []).join("\n");
    document.getElementById("rev-bl-tier3").value = (bl.tier3_user_custom || []).join("\n");
}

function switchReviewTab(tabId) {
    document.querySelectorAll(".tab-pane").forEach((el) => el.classList.add("hidden"));
    document.getElementById(tabId).classList.remove("hidden");
    document.querySelectorAll(".tab-btn").forEach((btn) => {
        btn.classList.toggle("active", btn.getAttribute("onclick").includes(tabId));
    });
}

function collectReviewPayload() {
    const splitLines = (id) =>
        document.getElementById(id).value.split("\n").map((s) => s.trim()).filter(Boolean);
    const hierarchyLevels = Array.from(document.querySelectorAll(".rev-hl-row")).map((row) => ({
        category: row.getAttribute("data-cat") || "CULTIVATION",
        level_num: parseInt(row.querySelector(".hl-num").value, 10) || 1,
        level_name: row.querySelector(".hl-name").value.trim(),
        description: row.querySelector(".hl-desc").value.trim(),
    }));
    const characters = Array.from(document.querySelectorAll(".rev-char-card")).map((card) => ({
        name: card.querySelector(".ch-name").value.trim(),
        aliases: card
            .querySelector(".ch-aliases")
            .value.split(",")
            .map((a) => a.trim())
            .filter(Boolean),
        role_type: card.querySelector(".ch-role").value.trim(),
        rank_level: parseInt(card.querySelector(".ch-rank").value, 10) || 1,
        faction_name: card.querySelector(".ch-fac").value.trim(),
        location_name: card.querySelector(".ch-loc").value.trim(),
        description: card.querySelector(".ch-desc").value.trim(),
    }));

    // Extract Stage Gates from visual cards or fallback to JSON
    const stageCards = Array.from(document.querySelectorAll(".rev-stage-card"));
    let stageGates = [];
    if (stageCards.length > 0) {
        stageGates = stageCards.map((card, idx) => ({
            stage_index: parseInt(card.getAttribute("data-stage-index"), 10) || (idx + 1),
            stage_title: card.querySelector(".stage-title-input")?.value.trim() || `第 ${idx + 1} 階段`,
            target_word_quota: parseInt(card.querySelector(".stage-words-input")?.value, 10) || 1000000,
            max_allowed_rank_level: parseInt(card.querySelector(".stage-rank-input")?.value, 10) || (idx + 1) * 2,
            core_mystery_quota: card.querySelector(".stage-mystery-input")?.value.trim() || "",
            milestone_goal: card.querySelector(".stage-milestone-input")?.value.trim() || "",
        }));
    } else {
        try {
            stageGates = JSON.parse(document.getElementById("rev-stage-gates-json")?.value || "[]");
        } catch {
            stageGates = [];
        }
    }

    // Extract Chapter beats from visual cards or fallback
    const chapters = Array.from(document.querySelectorAll(".rev-ch-row")).map((row) => {
        const beatItems = Array.from(row.querySelectorAll(".rev-beat-item"));
        let beatsJson = "[]";
        if (beatItems.length > 0) {
            const beats = beatItems.map((b, idx) => ({
                beat_index: parseInt(b.getAttribute("data-beat-index"), 10) || (idx + 1),
                scene_tag: b.querySelector(".beat-tag-input")?.value.trim() || "日常密謀",
                pacing_instruction: b.querySelector(".beat-pacing-input")?.value.trim() || "沉穩鋪墊",
                plot_objective: b.querySelector(".beat-objective-input")?.value.trim() || "",
            }));
            beatsJson = JSON.stringify(beats);
        } else if (row.querySelector(".rev-ch-beats")) {
            beatsJson = row.querySelector(".rev-ch-beats").value.trim();
        }
        return {
            chapter_num: parseInt(row.getAttribute("data-chnum"), 10) || 1,
            title: row.querySelector(".rev-ch-title").value.trim(),
            beats_json: beatsJson,
        };
    });

    return {
        world_rules: splitLines("rev-world-rules"),
        hierarchy_levels: hierarchyLevels,
        characters: characters,
        factions: splitLines("rev-factions-list"),
        locations: splitLines("rev-locations-list"),
        global_outline: {
            opening_plan: document.getElementById("rev-opening-plan").value.trim(),
            ending_draft: document.getElementById("rev-ending-draft").value.trim(),
            stage_gates: stageGates,
        },
        chapters: chapters,
        style_contract_text: document.getElementById("rev-style-contract").value,
        static_anchor_text: document.getElementById("rev-static-anchor").value,
        blacklist: {
            tier1_global_cliches: splitLines("rev-bl-tier1"),
            tier2_genre_anachronisms: splitLines("rev-bl-tier2"),
            tier3_user_custom: splitLines("rev-bl-tier3"),
        },
    };
}

async function saveReviewChanges() {
    const payload = collectReviewPayload();
    let resp;
    try {
        resp = await fetch("/api/review/update", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
    } catch (_err) {
        showToast("儲存失敗：無法連線至伺服器。");
        return false;
    }
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
        // Never repopulate the workbench from an error payload: doing so wiped the
        // user's unsaved edits while the toast still claimed the save succeeded.
        showToast(data.detail || data.error || "儲存失敗，請重試。");
        return false;
    }
    populateReviewGateUI(data);
    showToast("審核關卡五大分頁修改已全部儲存至 SQLite。");
    return true;
}

async function regenerateReviewSection(sectionName) {
    const btn = event?.target;
    await withButtonLoading(btn, async () => {
        showToast(`正在局部重抽分頁：${sectionName}...`);
        const resp = await fetch("/api/review/regenerate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ section: sectionName }),
        });
        const data = await resp.json();
        if (handleAiModelError(data)) return;
        if (!resp.ok) {
            showToast(data.detail || data.error || "局部重抽失敗，請重試。");
            return;
        }
        populateReviewGateUI(data);
        showToast(`分頁 [${sectionName}] 已完成局部重抽並更新！`);
    }, "重抽中...");
}

async function lockReviewAndStart() {
    await withButtonLoading("#btn-review-lock-start", async () => {
        if (!(await saveReviewChanges())) return;
        const resp = await fetch("/api/review/lock_and_start", { method: "POST" });
        const data = await resp.json();
        if (handleAiModelError(data)) return;
        if (!resp.ok) {
            showToast(data.detail || data.error || "鎖定並啟動連載失敗，請重試。");
            return;
        }
        setStageView("WARROOM");
        await refreshWarroomDashboard();
        showToast("已鎖定《風格契約書》並啟動後台全自動連載！");
    }, "鎖定並啟動中...");
}

