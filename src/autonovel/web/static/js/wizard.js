// --- Wizard Module ---

// --- Stage 1: Story Setup (Step 1) -> Sequential AI Infinite-Depth Q&A (Step 2) ---
function showWizardStep(stepNum) {
    document.getElementById("wizard-step1-panel").classList.toggle("hidden", stepNum !== 1);
    document.getElementById("wizard-step2-panel").classList.toggle("hidden", stepNum !== 2);
}

function updateStep2SummaryBanner(cfg) {
    const title = cfg.title || "未命名長篇小說";
    const pri = cfg.genre_primary || "仙俠";
    const sub = cfg.genre_sub || "古典修仙";
    const words = Number(cfg.min_word_count || 5000000).toLocaleString();
    document.getElementById("step2-locked-summary").textContent =
        `《${title}》｜主題材：${pri}｜子題材（複選）：${sub}｜保底字數：${words} 字`;
}

async function refreshWizardGenreCatalog() {
    const resp = await fetch("/api/wizard/genres");
    if (!resp.ok) return;
    const data = await resp.json();
    genreCatalog = data.genres || {};
    const primarySel = document.getElementById("wiz-genre-primary");
    const prevPrimary = primarySel.value;
    const prevSubs = Array.from(document.querySelectorAll(".sub-genre-chk:checked")).map((el) => el.value);
    primarySel.innerHTML = Object.keys(genreCatalog)
        .map((g) => `<option value="${escapeHtml(g)}">${escapeHtml(g)}</option>`)
        .join("");
    if (prevPrimary && genreCatalog[prevPrimary]) {
        primarySel.value = prevPrimary;
    }
    onPrimaryGenreChange(prevSubs);
}


async function initWizardState() {
    const resp = await fetch("/api/wizard/genres");
    const data = await resp.json();
    genreCatalog = data.genres || {};
    const cfg = data.project_config || {};
    const history = data.qa_history || [];
    const primarySel = document.getElementById("wiz-genre-primary");
    primarySel.innerHTML = Object.keys(genreCatalog)
        .map((g) => `<option value="${escapeHtml(g)}">${escapeHtml(g)}</option>`)
        .join("");
    if (cfg.genre_primary && genreCatalog[cfg.genre_primary]) {
        primarySel.value = cfg.genre_primary;
    }
    const savedSubs = history.length > 0
        ? (cfg.genre_sub || "")
              .split(/[、,，/]+/)
              .map((s) => s.trim())
              .filter(Boolean)
        : [];
    onPrimaryGenreChange(savedSubs);
    const knownSubs = new Set(genreCatalog[primarySel.value] || []);
    const customParts = savedSubs.filter((s) => !knownSubs.has(s));
    document.getElementById("wiz-genre-custom").value = customParts.join("、");
    if (cfg.title && cfg.title !== "未命名小說" && cfg.title !== "未命名長篇小說") {
        document.getElementById("wiz-title").value = cfg.title;
    } else {
        document.getElementById("wiz-title").value = "";
    }
    if (cfg.min_word_count) {
        document.getElementById("wiz-min-words").value = cfg.min_word_count;
    }
    updateStep2SummaryBanner(cfg);
    updateEngineModeBadge(data.ai_mode);
    if (data.qa_threshold) {
        currentQAThreshold = data.qa_threshold;
    } else if (cfg.min_word_count) {
        currentQAThreshold = computeQAThreshold(cfg.min_word_count);
    }
    renderQAHistory(history, currentQAThreshold);
    let nextQuestion = data.next_question || null;
    if (!nextQuestion && history.length > 0) {
        // /wizard/genres is now side-effect free; fetch the pending question explicitly.
        try {
            const qResp = await fetch("/api/wizard/next_question", { method: "POST" });
            if (qResp.ok) {
                const qData = await qResp.json();
                nextQuestion = qData.next_question || null;
            }
        } catch (_err) {
            nextQuestion = null;
        }
    }
    renderWizardQuestion(nextQuestion);
    showWizardStep(history.length > 0 ? 2 : 1);
    const state = cfg.pipeline_state || "WIZARD";
    updateTopBar(cfg);
    if (state === "REVIEW_GATE") {
        await loadReviewGateState();
        setStageView("REVIEW_GATE");
    } else if (state === "RUNNING" || state === "PAUSED" || state === "COMPLETED") {
        setStageView("WARROOM");
    }
}

function onPrimaryGenreChange(preselectedSubs) {
    const p = document.getElementById("wiz-genre-primary").value;
    const subs = genreCatalog[p] || ["自訂子題材"];
    const selectedSet = Array.isArray(preselectedSubs) ? new Set(preselectedSubs) : new Set();
    if (!Array.isArray(preselectedSubs)) {
        const customInput = document.getElementById("wiz-genre-custom");
        if (customInput) customInput.value = "";
    }
    document.getElementById("wiz-genre-sub").innerHTML = subs
        .map(
            (s) =>
                `<label class="option-item sub-genre-bar"><input type="checkbox" class="sub-genre-chk hidden-check" value="${escapeHtml(s)}" ${selectedSet.has(s) ? "checked" : ""}/><span>${escapeHtml(s)}</span></label>`
        )
        .join("");
}

async function submitGenreSetup() {
    await withButtonLoading("#wiz-btn-start", async () => {
        const selectedSubs = Array.from(document.querySelectorAll(".sub-genre-chk:checked")).map((el) => el.value);
        const customSub = document.getElementById("wiz-genre-custom").value.trim();
        let finalSubs = [...selectedSubs];
        if (finalSubs.length === 0 && !customSub) {
            const firstChk = document.querySelector(".sub-genre-chk");
            if (firstChk) {
                finalSubs = [firstChk.value];
            } else {
                finalSubs = ["標準"];
            }
        }
        const payload = {
            title: document.getElementById("wiz-title").value.trim(),
            genre_primary: document.getElementById("wiz-genre-primary").value,
            genre_subs: finalSubs,
            genre_sub: customSub,
            min_word_count: parseInt(document.getElementById("wiz-min-words").value, 10) || 5000000,
        };
        const resp = await fetch("/api/wizard/setup", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await resp.json();
        if (handleAiModelError(data)) {
            return;
        }
        if (!resp.ok) {
            showToast(data.detail || data.error || "設定失敗，請重試。");
            return;
        }
        const cfg = data.project_config || {};
        updateTopBar(cfg);
        updateStep2SummaryBanner(cfg);
        updateEngineModeBadge(data.ai_mode);
        if (data.qa_threshold) currentQAThreshold = data.qa_threshold;
        renderQAHistory([], currentQAThreshold);
        renderWizardQuestion(data.next_question);
        showWizardStep(2);
        showToast("已鎖定題材設定，進入 Step 2 由 AI 衍生專屬文風與設定問答。");
    }, "鎖定並生成首問中...");
}

async function regenerateCurrentQuestion() {
    const btn = document.querySelector("#wizard-step2-panel .qa-studio-top-bar button");
    await withButtonLoading(btn, async () => {
        const resp = await fetch("/api/wizard/regenerate_question", { method: "POST" });
        const data = await resp.json();
        if (handleAiModelError(data)) return;
        if (!resp.ok) {
            showToast(data.detail || data.error || "重新衍生失敗，請重試。");
            return;
        }
        updateEngineModeBadge(data.ai_mode);
        renderWizardQuestion(data.next_question);
        showToast("AI 已根據您的題材與書庫養分重新衍生本層選項！");
    }, "AI 重新衍生中...");
}

function renderWizardQuestion(q) {
    if (!q) {
        currentQuestion = null;
        document.getElementById("qa-question-title").textContent = "需要本地 AI 模型驅動問答";
        document.getElementById("qa-question-aspect").textContent = "推論服務離線";
        const box = document.getElementById("qa-options-box");
        box.innerHTML = `
            <div style="padding: 24px 16px; text-align: center; background: rgba(255,255,255,0.7); border-radius: 14px; border: 1px dashed rgba(239,68,68,0.35); margin: 12px 0;">
                <div style="font-size: 15px; font-weight: 700; color: #dc2626; margin-bottom: 8px;">⚠ 本地 AI 模型推論服務未啟動</div>
                <div style="font-size: 13px; color: var(--text-secondary); margin-bottom: 16px;">
                    本系統已全面停用虛擬模板兜底流程。請先下載 GGUF 權重並啟動本地推論服務，以進行真實 AI 創作。
                </div>
                <button class="btn btn-primary" onclick="openEnvModal()">打開環境設置與模型下載中心</button>
            </div>
        `;
        return;
    }
    currentQuestion = q;
    const cleanTitle = String(q.question_title || "")
        .replace(/^【[^】]*】[^，]*，/, "")
        .replace(/：$/, "")
        .trim();
    const cleanAspect = String(q.question_aspect || "")
        .replace(/^AI\s*動態衍生\s*[·・]\s*/, "")
        .replace(/（.*）$/, "")
        .trim();
    document.getElementById("qa-question-title").textContent = cleanTitle || q.question_title;
    document.getElementById("qa-question-aspect").textContent = cleanAspect || q.question_aspect;
    // Default to multi-select checkboxes for rich combinations
    const isMulti = q.allow_multi_select !== false;
    const inputType = isMulti ? "checkbox" : "radio";
    const box = document.getElementById("qa-options-box");
    box.innerHTML = (q.options || [])
        .map(
            (opt, idx) =>
                `<label class="option-item"><input type="${inputType}" name="qa-opt-group" class="qa-opt-chk hidden-check" value="${escapeHtml(opt)}" ${idx === 0 ? "checked" : ""}/><span>${escapeHtml(opt)}</span></label>`
        )
        .join("");
    document.getElementById("qa-custom-note").value = "";
}

let currentQAThreshold = 10;

function computeQAThreshold(minWords) {
    const w = Number(minWords) || 5000000;
    if (w < 1000000) return 7;
    if (w < 3000000) return 10;
    return 14;
}

function updateQAProgress(count, threshold) {
    const totalGoal = threshold || currentQAThreshold || 10;
    const pct = Math.min(100, Math.round((count / totalGoal) * 100));
    const fillEl = document.getElementById("qa-progress-fill");
    const textEl = document.getElementById("qa-progress-text");
    const hintEl = document.getElementById("qa-progress-hint");
    const unlockArea = document.getElementById("qa-blueprint-unlock-area");
    if (fillEl) fillEl.style.width = `${pct}%`;
    if (textEl) textEl.textContent = `問答累積深度：${count} / ${totalGoal} 題`;
    if (hintEl) {
        if (count >= totalGoal) {
            hintEl.textContent = `✔ 已達標 ${totalGoal} 題！隨時可點擊下方金色按鈕建構全書藍圖，亦可繼續深挖`;
            hintEl.style.color = "#10b981";
        } else {
            hintEl.textContent = `滿 ${totalGoal} 題即可進入下一階段開始建構藍圖`;
            hintEl.style.color = "var(--text-secondary)";
        }
    }
    if (unlockArea) {
        if (count >= totalGoal) {
            unlockArea.classList.remove("hidden");
        } else {
            unlockArea.classList.add("hidden");
        }
    }
}

function renderQAHistory(history, threshold) {
    const list = history || [];
    if (typeof threshold === "number") {
        currentQAThreshold = threshold;
    }
    updateQAProgress(list.length, currentQAThreshold);
    const ul = document.getElementById("qa-history-list");
    if (!ul) return;
    ul.innerHTML = list
        .map(
            (item) =>
                `<li class="qa-tree-item">
                    <div><strong>第 ${item.step_index} 層</strong>${item.aspect ? ` [${escapeHtml(item.aspect)}]` : ""}：${escapeHtml(item.answer)}</div>
                    <button class="btn btn-sm btn-glass" onclick="rollbackQuestion(${item.step_index})">回退修改此層</button>
                </li>`
        )
        .join("");
}

async function submitWizardAnswer() {
    if (!currentQuestion) return;
    const selected = Array.from(document.querySelectorAll(".qa-opt-chk:checked")).map((el) => el.value);
    const customNote = document.getElementById("qa-custom-note").value.trim();
    if (selected.length === 0 && !customNote) {
        showToast("請至少勾選一個選項或填寫補充說明。");
        return;
    }
    await withButtonLoading("#btn-submit-wizard", async () => {
        const resp = await fetch("/api/wizard/answer", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question_title: currentQuestion.question_title,
                question_aspect: currentQuestion.question_aspect || "",
                selected_answers: selected,
                custom_note: customNote,
            }),
        });
        const data = await resp.json();
        if (handleAiModelError(data)) return;
        if (!resp.ok) {
            showToast(data.detail || data.error || "提交問答失敗，請重試。");
            return;
        }
        const banner = document.getElementById("contradiction-banner");
        if (!data.accepted && data.contradiction) {
            banner.textContent = `【設定矛盾警示】${data.contradiction.reason}`;
            banner.classList.remove("hidden");
            return;
        }
        banner.classList.add("hidden");
        updateEngineModeBadge(data.ai_mode);
        if (data.qa_threshold) currentQAThreshold = data.qa_threshold;
        renderQAHistory(data.history || [], currentQAThreshold);
        renderWizardQuestion(data.next_question);
    }, "AI 衍生下一問中...");
}

let isAutoQARunning = false;

async function triggerAutoQALoop(totalSteps = 7) {
    if (isAutoQARunning) return;
    if (!currentQuestion) {
        showToast("目前沒有可推演的問題。");
        return;
    }
    isAutoQARunning = true;
    const modal = document.getElementById("modal-blueprint-generating");
    const modalTitle = document.getElementById("loading-modal-title");
    const modalDesc = document.getElementById("loading-modal-desc");
    const submitBtn = document.getElementById("btn-submit-wizard");
    const autoBtn = document.getElementById("btn-autocomplete-qa");
    const finishBtn = document.getElementById("btn-finish-wizard");

    if (submitBtn) submitBtn.disabled = true;
    if (autoBtn) autoBtn.disabled = true;
    if (finishBtn) finishBtn.disabled = true;

    if (modal) {
        if (modalTitle) modalTitle.textContent = "AI 正在即時進行交替問答推演...";
        modal.classList.remove("hidden");
    }

    try {
        let currentQ = currentQuestion;
        for (let i = 1; i <= totalSteps; i++) {
            if (modalDesc) {
                modalDesc.textContent = `正在進行第 ${i} / ${totalSteps} 輪交替推演：特約顧問正在針對「${currentQ.question_title}」權衡題材決策...`;
            }
            const resp = await fetch("/api/wizard/autocomplete_step", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    question_title: currentQ.question_title,
                    question_aspect: currentQ.question_aspect || "",
                    options: currentQ.options || [],
                }),
            });
            const data = await resp.json();
            if (handleAiModelError(data)) {
                return;
            }
            if (!resp.ok) {
                showToast(data.detail || data.error || `第 ${i} 步推演失敗，中斷執行。`);
                return;
            }
            updateEngineModeBadge(data.ai_mode);
            if (data.qa_threshold) currentQAThreshold = data.qa_threshold;
            renderQAHistory(data.history || [], currentQAThreshold);
            renderWizardQuestion(data.next_question);
            currentQ = data.next_question;
        }
        showToast(`AI 已成功完成 7 輪交替推演，問答歷史已同步更新！`);
    } catch (_err) {
        showToast("交替推演時發生連線中斷，請重試。");
    } finally {
        isAutoQARunning = false;
        if (modal) {
            modal.classList.add("hidden");
            if (modalTitle) modalTitle.textContent = "AI 正在衍生小說宏觀藍圖...";
            if (modalDesc) {
                modalDesc.textContent = "正在深度解析問答歷史，推演世界底層法則、勢力網絡與 500 萬字階段配速大綱，請稍候（約需 20～40 秒）...";
            }
        }
        if (submitBtn) submitBtn.disabled = false;
        if (autoBtn) autoBtn.disabled = false;
        if (finishBtn) finishBtn.disabled = false;
    }
}

async function rollbackQuestion(stepIndex) {
    const resp = await fetch("/api/wizard/rollback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ step_index: stepIndex }),
    });
    const data = await resp.json();
    if (handleAiModelError(data)) return;
    if (!resp.ok) {
        showToast(data.detail || data.error || "回溯失敗，請重試。");
        return;
    }
    if (data.qa_threshold) currentQAThreshold = data.qa_threshold;
    renderQAHistory(data.history || [], currentQAThreshold);
    renderWizardQuestion(data.next_question);
    showToast(`已回溯至第 ${stepIndex} 層問答。`);
}

async function finishWizardAndBuildBlueprint() {
    const modal = document.getElementById("modal-blueprint-generating");
    const modalTitle = document.getElementById("loading-modal-title");
    const modalDesc = document.getElementById("loading-modal-desc");
    if (modal) {
        if (modalTitle) modalTitle.textContent = "AI 正在衍生小說宏觀藍圖與風格契約...";
        if (modalDesc) {
            modalDesc.textContent = "正在深度解析問答歷史，推演世界底層法則、勢力網絡與 500 萬字階段配速大綱，請稍候（約需 20～40 秒）...";
        }
        modal.classList.remove("hidden");
    }
    const finishBtn = document.getElementById("btn-finish-wizard");
    if (finishBtn) {
        finishBtn.disabled = true;
    }
    try {
        const resp = await fetch("/api/wizard/complete", { method: "POST" });
        const data = await resp.json();
        if (handleAiModelError(data)) return;
        if (!resp.ok) {
            showToast(data.detail || data.error || "生成風格契約書失敗，請重試。");
            return;
        }
        populateReviewGateUI(data);
        setStageView("REVIEW_GATE");
        showToast("AI 藍圖建構完畢，已自動進入審核關卡！");
    } catch (_e) {
        showToast("生成藍圖時發生連線錯誤，請重試。");
    } finally {
        if (modal) {
            modal.classList.add("hidden");
        }
        if (finishBtn) {
            finishBtn.disabled = false;
        }
    }
}

