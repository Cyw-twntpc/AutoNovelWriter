// --- Environment Module ---
async function openEnvModal() {
    const el = document.getElementById("modal-env");
    if (el) {
        el.classList.remove("hidden");
        el.style.display = "flex";
    }
    await refreshEnvModalStatus();
}

async function refreshEnvModalStatus() {
    const resp = await fetch("/api/system/env_status");
    if (!resp.ok) return;
    const st = await resp.json();
    renderEnvStatusBox(st);
}

let envPollInterval = null;

function renderEnvStatusBox(st) {
    const dl = st.download_state || {};
    const isDlActive = Boolean(dl.active);
    const dlTarget = dl.target || "";

    // Keep legacy banner and progress containers hidden; all progress is now cleanly displayed inside per-component cards
    const banner = document.getElementById("env-download-banner");
    if (banner) {
        banner.classList.add("hidden");
        banner.textContent = "";
    }

    const progWrap = document.getElementById("env-progress-container");
    if (progWrap) {
        progWrap.classList.add("hidden");
    }

    const box = document.getElementById("env-status-box");
    if (!box) return;

    function renderItemProgress(itemTarget) {
        if (isDlActive && dlTarget === itemTarget) {
            const pct = dl.percent != null && dl.percent >= 0 ? dl.percent : 0;
            const fillWidth = Math.min(100, Math.max(2, Math.round(pct)));
            const msg = sanitizeErrorMessage(dl.message || "下載處理中...");
            return `
                <div class="env-item-progress-wrap">
                    <div class="env-item-progress-meta">
                        <span class="env-item-progress-msg">${escapeHtml(msg)}</span>
                        <span class="env-item-progress-pct">${Math.round(pct)}%</span>
                    </div>
                    <div class="env-item-progress-bar-bg">
                        <div class="env-item-progress-bar-fill" style="width: ${fillWidth}%;"></div>
                    </div>
                </div>
            `;
        }
        if (dl.error && dlTarget === itemTarget) {
            return `
                <div class="env-item-progress-wrap">
                    <div class="env-item-progress-meta">
                        <span class="env-item-progress-msg" style="color:#ff6b6b; font-weight:600;">⚠️ ${escapeHtml(sanitizeErrorMessage(dl.message || "下載中斷"))}</span>
                        <span class="env-item-progress-pct" style="color:#ff6b6b;">已暫停</span>
                    </div>
                    <div class="env-item-progress-bar-bg">
                        <div class="env-item-progress-bar-fill" style="width: 100%; background: #ef4444;"></div>
                    </div>
                </div>
            `;
        }
        return "";
    }

    // Component 1: Embedding Model
    let bgeLed = "led-yellow";
    let bgeText = "待部署";
    let bgeVersionHtml = "";
    if (st.bge_onnx_ready) {
        bgeLed = "led-green";
        bgeText = "已就緒";
        bgeVersionHtml = `<div class="env-status-version">${escapeHtml(st.bge_version || "BGE-Small-ZH v1.5 (ONNX CPU)")}</div>`;
    } else if (isDlActive && dlTarget === "bge") {
        bgeLed = "led-blue pulse";
        bgeText = "下載部署中...";
    }

    // Component 2: Core Large Language Model (LLM)
    let ggufLed = "led-yellow";
    let ggufText = "待部署";
    let ggufVersionHtml = "";
    if (st.gguf_ready) {
        ggufLed = "led-green";
        ggufText = "已就緒";
        const sizeStr = st.gguf_size_mb ? `${Number(st.gguf_size_mb).toLocaleString()} MB` : "";
        const verTitle = st.gguf_filename || "Qwen2.5-7B-Instruct-Q5_K_M.gguf";
        const fullVer = sizeStr ? `${verTitle} (${sizeStr})` : verTitle;
        ggufVersionHtml = `<div class="env-status-version">${escapeHtml(st.gguf_version || fullVer)}</div>`;
    } else if (isDlActive && dlTarget === "gguf") {
        ggufLed = "led-blue pulse";
        ggufText = "下載部署中...";
    }

    // Component 3: Local Inference Engine
    let llamaLed = "led-red";
    let llamaText = "未安裝";
    let llamaVersionHtml = "";
    if (st.llama_server_healthy) {
        llamaLed = "led-green";
        llamaText = "在線服務中";
        llamaVersionHtml = `<div class="env-status-version">${escapeHtml(st.llama_version || "llama.cpp (支援 CUDA / CPU 推論)")}</div>`;
    } else if (isDlActive && dlTarget === "llama") {
        llamaLed = "led-blue pulse";
        llamaText = st.llama_server_found ? "啟動服務中..." : "下載安裝中...";
        if (st.llama_version) {
            llamaVersionHtml = `<div class="env-status-version">${escapeHtml(st.llama_version)}</div>`;
        }
    } else if (st.llama_server_found) {
        llamaLed = "led-yellow";
        llamaText = "待啟動";
        llamaVersionHtml = `<div class="env-status-version">${escapeHtml(st.llama_version || "llama.cpp (本地已安裝)")}</div>`;
    }

    box.className = "env-status-list";
    box.innerHTML = `
        <div class="env-status-row">
            <div class="env-status-header">
                <div class="env-status-title-col">
                    <span class="env-status-name">🔤 語意嵌入模型</span>
                    ${bgeVersionHtml}
                </div>
                <div class="env-led-status">
                    <span class="env-led-dot ${bgeLed}"></span>
                    <span>${escapeHtml(bgeText)}</span>
                </div>
            </div>
            ${renderItemProgress("bge")}
        </div>
        <div class="env-status-row">
            <div class="env-status-header">
                <div class="env-status-title-col">
                    <span class="env-status-name">🧠 核心大語言模型 (LLM)</span>
                    ${ggufVersionHtml}
                </div>
                <div class="env-led-status">
                    <span class="env-led-dot ${ggufLed}"></span>
                    <span>${escapeHtml(ggufText)}</span>
                </div>
            </div>
            ${renderItemProgress("gguf")}
        </div>
        <div class="env-status-row">
            <div class="env-status-header">
                <div class="env-status-title-col">
                    <span class="env-status-name">⚡ 本地推論引擎</span>
                    ${llamaVersionHtml}
                </div>
                <div class="env-led-status">
                    <span class="env-led-dot ${llamaLed}"></span>
                    <span>${escapeHtml(llamaText)}</span>
                </div>
            </div>
            ${renderItemProgress("llama")}
        </div>
    `;

    // Update Deploy All button state
    const deployBtn = document.getElementById("btn-deploy-all-env");
    if (deployBtn) {
        if (isDlActive) {
            deployBtn.disabled = true;
            deployBtn.textContent = "⏳ 正在部署環境...";
        } else if (st.bge_onnx_ready && st.gguf_ready && st.llama_server_healthy) {
            deployBtn.disabled = false;
            deployBtn.textContent = "✔ 所有模型環境已就緒 (重新檢查)";
        } else {
            deployBtn.disabled = false;
            deployBtn.textContent = "⚡ 一鍵檢查並部署所有環境";
        }
    }
}

async function triggerDeployAllEnvironments() {
    const deployBtn = document.getElementById("btn-deploy-all-env");
    if (deployBtn) {
        deployBtn.disabled = true;
        deployBtn.textContent = "⏳ 正在啟動一鍵部署...";
    }
    showToast("開始一鍵檢查與排程部署所有模型環境...");

    const repoInput = document.getElementById("env-gguf-repo");
    const fileInput = document.getElementById("env-gguf-file");
    const payload = {
        repo_id: repoInput ? repoInput.value.trim() : "",
        filename: fileInput ? fileInput.value.trim() : "",
        run_in_background: true,
    };

    try {
        const resp = await fetch("/api/system/deploy_all", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await resp.json();
        if (data.env_status) {
            renderEnvStatusBox(data.env_status);
        }
        if (data.message) {
            showToast(data.message);
        }

        // Start polling until deployment is finished
        if (envPollInterval) clearInterval(envPollInterval);
        envPollInterval = setInterval(async () => {
            const pollResp = await fetch("/api/system/env_status");
            if (pollResp.ok) {
                const st = await pollResp.json();
                renderEnvStatusBox(st);
                if (!st.download_state || !st.download_state.active) {
                    clearInterval(envPollInterval);
                    envPollInterval = null;
                }
            }
        }, 1500);
    } catch (err) {
        showToast(`部署請求失敗：${err.message}`);
        if (deployBtn) deployBtn.disabled = false;
    }
}

async function triggerModelDownload(target) {
    const isGguf = target === "gguf";
    const isBge = target === "bge";
    const payload = {
        target: target,
        repo_id: isGguf ? document.getElementById("env-gguf-repo").value.trim() : (isBge ? "Xenova/bge-small-zh-v1.5" : ""),
        filename: isGguf ? document.getElementById("env-gguf-file").value.trim() : "",
        run_in_background: true,
    };
    showToast(`已排入背景下載任務：${target.toUpperCase()}...`);
    const resp = await fetch("/api/system/download_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (data.env_status) {
        renderEnvStatusBox(data.env_status);
    }
    if (!envPollInterval) {
        envPollInterval = setInterval(async () => {
            const pollResp = await fetch("/api/system/env_status");
            if (pollResp.ok) {
                const st = await pollResp.json();
                renderEnvStatusBox(st);
                if (!st.download_state || !st.download_state.active) {
                    clearInterval(envPollInterval);
                    envPollInterval = null;
                }
            }
        }, 1500);
    }
}

async function triggerStartLlama() {
    showToast("正在嘗試啟動並探測本地 llama-server...");
    const resp = await fetch("/api/system/start_llama", { method: "POST" });
    const data = await resp.json();
    if (data.env_status) {
        renderEnvStatusBox(data.env_status);
    }
    showToast(data.started ? "llama-server 已啟動連線！" : "未偵測到可用 llama-server 或 GGUF 模型，請先下載權重。");
}



async function unloadModelAndPauseJobs() {
    const btn = document.getElementById("btn-unload-model-pause"); const topBtn = document.getElementById("btn-top-unload-model"); const refBtn = document.getElementById("btn-ref-unload-pause");
    if (btn) btn.disabled = true; if (topBtn) topBtn.disabled = true; if (refBtn) refBtn.disabled = true;
    showToast("正在卸載模型並安全暫停工作...");
    try {
        const resp = await fetch("/api/system/unload_and_pause", { method: "POST" });
        const data = await resp.json();
        if (resp.ok) {
            showToast("🛑 本地模型已成功卸載（顯存已釋放），正在執行的寫作與萃取工作已安全暫停。");
            await refreshEnvModalStatus();
            await refreshWarroomDashboard();
            await checkReferenceProgressNow();
        } else {
            showToast(data.detail || "卸載模型失敗。");
        }
    } catch (err) {
        showToast("網路異常，無法完成卸載。");
    } finally {
        if (btn) btn.disabled = false; if (topBtn) topBtn.disabled = false; if (refBtn) refBtn.disabled = false;
    }
}
