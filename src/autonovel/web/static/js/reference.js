// --- Reference Module ---
// --- Reference Corpus Hub (Redesigned) State & Controls ---
let refActiveTab = "ingest";
let refEntriesCurrentPage = 1;
let currentSelectedRefTag = "";
let searchEntriesTimer = null;
let refModalPollTimer = null;
let refStagedFilesQueue = [];
let wasReferenceExtractingActive = false;
let wasPacingActive = false;
let pacingCompletedHandled = false;
let pacingHideTimer = null;

async function openReferenceModal() {
    const el = document.getElementById("modal-reference");
    if (el) {
        el.classList.remove("hidden");
        el.style.display = "flex";
    }
    switchRefTab(refActiveTab || "ingest");
    await loadReferenceOverview();
    if (!refModalPollTimer) {
        refModalPollTimer = setInterval(async () => {
            const modal = document.getElementById("modal-reference");
            if (!modal || modal.classList.contains("hidden")) {
                clearInterval(refModalPollTimer);
                refModalPollTimer = null;
                return;
            }
            // Only poll as a fallback if SSE is disconnected
            if (!activeEventSource || activeEventSource.readyState !== EventSource.OPEN) {
                await checkReferenceProgressNow();
            }
        }, 8000);
    }
}

function switchRefTab(tab) {
    refActiveTab = tab;
    const btnIngest = document.getElementById("ref-tab-btn-ingest");
    const btnEntries = document.getElementById("ref-tab-btn-entries");
    const btnAssets = document.getElementById("ref-tab-btn-assets");
    const btnPacing = document.getElementById("ref-tab-btn-pacing");
    const paneIngest = document.getElementById("ref-tab-ingest");
    const paneEntries = document.getElementById("ref-tab-entries");
    const paneAssets = document.getElementById("ref-tab-assets");
    const panePacing = document.getElementById("ref-tab-pacing");

    btnIngest?.classList.toggle("active", tab === "ingest");
    btnEntries?.classList.toggle("active", tab === "entries");
    btnAssets?.classList.toggle("active", tab === "assets");
    btnPacing?.classList.toggle("active", tab === "pacing");

    paneIngest?.classList.toggle("hidden", tab !== "ingest");
    paneEntries?.classList.toggle("hidden", tab !== "entries");
    paneAssets?.classList.toggle("hidden", tab !== "assets");
    panePacing?.classList.toggle("hidden", tab !== "pacing");

    if (tab === "ingest") {
        loadReferenceBooks();
    } else if (tab === "entries") {
        loadReferenceHotTags();
        loadReferenceEntries(1);
    } else if (tab === "assets") {
        loadReferenceTagAssets();
    } else if (tab === "pacing") {
        loadPacingTemplates();
    }
}

async function loadReferenceOverview() {
    await loadReferenceBooks();
    await checkReferenceProgressNow();
}

async function checkReferenceProgressNow() {
    try {
        const resp = await fetch("/api/reference/progress");
        if (resp.ok) {
            const st = await resp.json();
            updateReferenceProgressMonitor(st);
            if (st.pacing_status) {
                updatePacingProgressMonitor(st.pacing_status);
            }
            if (st.sources !== undefined) {
                const bEl = document.getElementById("ref-stat-books");
                if (bEl) bEl.textContent = st.sources || 0;
                const eEl = document.getElementById("ref-stat-entries");
                if (eEl) eEl.textContent = st.entries || 0;
                const tEl = document.getElementById("ref-stat-tags");
                if (tEl) tEl.textContent = st.tags || 0;
            }
            if (st.active && st.processed_chunks % 2 === 0) {
                loadReferenceBooks();
            }
            if (wasReferenceExtractingActive && !st.active) {
                loadReferenceBooks();
                loadPacingTemplates();
                refreshWizardGenreCatalog();
            }
            wasReferenceExtractingActive = Boolean(st.active);
        }
    } catch (e) {
        console.error("Failed to check reference progress:", e);
    }
}

function updateReferenceProgressMonitor(st) {
    const card = document.getElementById("ref-active-progress-card");
    if (!card) return;
    if (st && st.active) {
        card.classList.remove("hidden");
        const fileEl = document.getElementById("ref-prog-file");
        const phasePrefix = st.phase === "TRACK_B_PACING" ? "正在提煉情節與藍圖" : "正在萃取";
        if (fileEl) fileEl.textContent = `${phasePrefix}：《${st.current_file || "參考書籍"}》`;
        const batchEl = document.getElementById("ref-prog-batch");
        if (batchEl) batchEl.textContent = `（當前批次：第 ${st.batch_index || 1} / ${st.batch_total || 1} 本）`;
        const queueBadge = document.getElementById("ref-prog-queue-badge");
        if (queueBadge) {
            const qCount = st.queued_batches || 0;
            queueBadge.classList.toggle("hidden", qCount <= 0);
            if (qCount > 0) queueBadge.textContent = `候選佇列：+${qCount} 批`;
        }
        const pct = typeof st.percent === "number" ? st.percent.toFixed(1) : "0.0";
        const pctEl = document.getElementById("ref-prog-pct");
        if (pctEl) pctEl.textContent = `${pct}%`;
        const barEl = document.getElementById("ref-prog-bar");
        if (barEl) barEl.style.width = `${pct}%`;
        const chunksEl = document.getElementById("ref-prog-chunks");
        if (chunksEl) {
            if (st.phase === "TRACK_B_PACING") {
                chunksEl.textContent = `階段：全書目錄與宏觀情節波浪解構中`;
            } else if (st.phase && st.phase.startsWith("STAGE_1")) {
                chunksEl.textContent = `階段 1/2：全書章節客觀事實與高價值章節篩選中`;
            } else {
                chunksEl.textContent = `階段 2/2 切塊萃取：${st.processed_chunks || 0} / ${st.total_chunks || 0} 塊`;
            }
        }
        const btnPause = document.getElementById("btn-pause-extraction");
        const btnResume = document.getElementById("btn-resume-extraction");
        const btnCancel = document.getElementById("btn-cancel-extraction");
        if (btnCancel) {
            btnCancel.classList.remove("hidden");
            btnCancel.disabled = false;
        }
        if (st.paused) {
            if (btnPause) btnPause.classList.add("hidden");
            if (btnResume) {
                btnResume.classList.remove("hidden");
                btnResume.disabled = false;
            }
            const speedEl = document.getElementById("ref-prog-speed");
            if (speedEl) speedEl.textContent = "【已暫停】點擊「繼續萃取」恢復處理...";
        } else {
            if (btnResume) btnResume.classList.add("hidden");
            if (btnPause) {
                btnPause.classList.remove("hidden");
                btnPause.disabled = false;
            }
            const speedEl = document.getElementById("ref-prog-speed");
            if (speedEl) {
                speedEl.textContent = st.phase === "TRACK_B_PACING"
                    ? "後台神經網路正在提煉全書波浪曲線與事實藍圖..."
                    : "後台神經網路寫作技法分析中...";
            }
        }

        // Update real-time metric counter cards
        if (st.entries !== undefined) {
            const eEl = document.getElementById("ref-stat-entries");
            if (eEl) eEl.textContent = st.entries;
        }
        if (st.tags !== undefined) {
            const tEl = document.getElementById("ref-stat-tags");
            if (tEl) tEl.textContent = st.tags;
        }
        if (st.sources !== undefined) {
            const bEl = document.getElementById("ref-stat-books");
            if (bEl) bEl.textContent = st.sources;
        }
    } else {
        card.classList.add("hidden");
    }
    updateConfirmIngestButtonLabel();
}

async function loadReferenceBooks() {
    const box = document.getElementById("ref-books-list-box");
    try {
        const resp = await fetch("/api/reference/books");
        if (!resp.ok) {
            if (box) box.innerHTML = `<div class="muted-text" style="padding: 12px; text-align: center;">尚無已入庫之參考書籍。請由上方選擇檔案上傳。</div>`;
            return;
        }
        const data = await resp.json();
        const books = data.books || [];
        const stats = data.stats || {};
        const bEl = document.getElementById("ref-stat-books");
        if (bEl) bEl.textContent = stats.sources || books.length;
        const eEl = document.getElementById("ref-stat-entries");
        if (eEl) eEl.textContent = stats.entries || 0;
        const tEl = document.getElementById("ref-stat-tags");
        if (tEl) tEl.textContent = stats.tags || 0;

        // Populate source filter dropdown
        const sourceSelect = document.getElementById("ref-source-filter");
        if (sourceSelect) {
            const currentVal = sourceSelect.value;
            sourceSelect.innerHTML = `<option value="">全部書籍來源</option>` +
                books.map(b => `<option value="${b.id}">${escapeHtml(b.file_name)} (${b.entry_count}條)</option>`).join("");
            sourceSelect.value = currentVal;
        }

        // Render books list
        if (!box) return;
        if (books.length === 0) {
            box.innerHTML = `<div class="muted-text" style="padding: 12px; text-align: center;">尚無已入庫之參考書籍。請由上方選擇檔案上傳。</div>`;
            return;
        }
        box.innerHTML = books.map(b => {
            const skillsBadge = b.entry_count > 0
                ? `<span class="ref-pill" style="font-size:0.72rem; color:#0066ff;">✍️ 寫作技法：${b.entry_count} 則</span>`
                : `<span class="ref-pill" style="font-size:0.72rem; color:#64748b;">✍️ 寫作技法：未萃取</span>`;
            const pacingBadge = b.has_pacing
                ? `<span class="ref-pill" style="font-size:0.72rem; color:#10b981;">📈 情節規劃：已建立範本</span>`
                : `<span class="ref-pill" style="font-size:0.72rem; color:#64748b;">📈 情節規劃：未萃取</span>`;
            const bpBadge = b.has_blueprint
                ? `<span class="ref-pill" style="font-size:0.72rem; color:#8b5cf6;">🧭 事實藍圖：已快取${b.total_chapters ? ` (${b.total_chapters}章)` : ""}</span>`
                : "";
            return `
            <div class="ref-book-item">
                <div>
                    <div style="display:flex; align-items:center; gap:0.45rem; flex-wrap:wrap;">
                        <strong>📖 ${escapeHtml(b.file_name)}</strong>
                        ${skillsBadge}
                        ${pacingBadge}
                        ${bpBadge}
                    </div>
                    <div class="muted-text" style="font-size:0.74rem; margin-top:4px;">
                        入庫時間：${b.imported_at} ｜ 目標切片數：${b.chunk_count} 塊
                    </div>
                </div>
                <div style="display:flex; gap:6px;">
                    <button class="btn btn-glass btn-xs" data-ref-action="jump-book" data-source-id="${b.id}">檢視條目</button>
                    <button class="btn btn-warn btn-xs" data-ref-action="delete-book" data-source-id="${b.id}" data-file-name="${escapeHtml(b.file_name)}">刪除書籍</button>
                </div>
            </div>
        `}).join("");
    } catch (e) {
        console.error("Failed to load reference books:", e);
        if (box) box.innerHTML = `<div class="muted-text" style="padding: 12px; text-align: center;">尚無已入庫之參考書籍。請由上方選擇檔案上傳。</div>`;
    }
}

async function deleteReferenceBook(sourceId, fileName) {
    if (!confirm(`確定要刪除參考書籍《${fileName}》嗎？\n此操作將永久移除該書籍及其萃取出的所有條目、標籤與情節範本！`)) {
        return;
    }
    try {
        const resp = await fetch(`/api/reference/books/${sourceId}`, { method: "DELETE" });
        const data = await resp.json();
        if (!resp.ok) {
            showToast(data.detail || "刪除書籍失敗");
            return;
        }
        showToast(`已成功刪除書籍《${fileName}》及其關聯資產！`);
        await loadReferenceOverview();
        await loadPacingTemplates();
        if (document.getElementById("ref-tab-btn-entries")?.classList.contains("active")) {
            await loadReferenceEntries(1);
            await loadReferenceHotTags();
        }
        refreshWizardGenreCatalog();
    } catch (e) {
        showToast("刪除書籍時發生錯誤");
    }
}

function jumpToBookEntries(sourceId) {
    switchRefTab("entries");
    const sourceSelect = document.getElementById("ref-source-filter");
    if (sourceSelect) {
        sourceSelect.value = sourceId;
    }
    loadReferenceEntries(1);
}

function debounceSearchEntries() {
    if (searchEntriesTimer) clearTimeout(searchEntriesTimer);
    searchEntriesTimer = setTimeout(() => {
        loadReferenceEntries(1);
    }, 300);
}

function clearEntryFilters() {
    const searchInput = document.getElementById("ref-search-input");
    if (searchInput) searchInput.value = "";
    const sourceSelect = document.getElementById("ref-source-filter");
    if (sourceSelect) sourceSelect.value = "";
    currentSelectedRefTag = "";
    loadReferenceHotTags();
    loadReferenceEntries(1);
}

async function loadReferenceEntries(page = 1) {
    refEntriesCurrentPage = page;
    const limit = 15;
    const offset = (page - 1) * limit;
    const sourceId = document.getElementById("ref-source-filter")?.value || "";
    const keyword = document.getElementById("ref-search-input")?.value.trim() || "";
    const tag = currentSelectedRefTag || "";

    const box = document.getElementById("ref-entries-list-box");
    if (box) box.innerHTML = `<div class="muted-text" style="padding: 16px; text-align: center;">正在載入技法條目...</div>`;

    const url = new URL("/api/reference/entries", window.location.origin);
    if (sourceId) url.searchParams.append("source_id", sourceId);
    if (keyword) url.searchParams.append("keyword", keyword);
    if (tag) url.searchParams.append("tag", tag);
    url.searchParams.append("limit", limit);
    url.searchParams.append("offset", offset);

    try {
        const resp = await fetch(url);
        if (!resp.ok) {
            if (box) box.innerHTML = `<div class="muted-text" style="padding: 16px; text-align: center;">載入條目失敗。</div>`;
            return;
        }
        const data = await resp.json();
        const entries = data.entries || [];
        const total = data.total || 0;
        const totalPages = Math.max(1, Math.ceil(total / limit));

        // Update pagination UI
        const pageInfo = document.getElementById("ref-page-info");
        if (pageInfo) pageInfo.textContent = `第 ${page} / ${totalPages} 頁（共 ${total} 則條目）`;
        const btnPrev = document.getElementById("ref-btn-prev-page");
        if (btnPrev) btnPrev.disabled = page <= 1;
        const btnNext = document.getElementById("ref-btn-next-page");
        if (btnNext) btnNext.disabled = page >= totalPages;

        if (entries.length === 0) {
            if (box) box.innerHTML = `<div class="muted-text" style="padding: 24px; text-align: center;">查無符合條件的技法條目。</div>`;
            return;
        }

        if (box) {
            box.innerHTML = entries.map(e => `
                <div class="ref-entry-card" id="ref-entry-${e.id}">
                    <div class="ref-entry-header">
                        <span>📘 來源：《${escapeHtml(e.file_name)}》 · 條目 #${e.id}（${e.created_at || ""}）</span>
                        <button class="btn btn-glass btn-xs" style="color:#ef4444;" onclick="deleteReferenceEntry(${e.id})">🗑️ 刪除</button>
                    </div>
                    <div class="ref-entry-desc">${escapeHtml(e.description)}</div>
                    ${e.snippet ? `<div class="ref-entry-snippet">「${escapeHtml(e.snippet)}」</div>` : ''}
                    <div class="ref-entry-tags-row">
                        ${(e.tags || []).map(t => `
                            <span class="ref-tag-pill">${escapeHtml(t)}</span>
                        `).join("")}
                        <div class="ref-tag-add-wrap">
                            <input type="text" class="ref-tag-input" placeholder="+ 新增標籤" onkeydown="handleTagInputKey(event, ${e.id}, this)" />
                        </div>
                    </div>
                </div>
            `).join("");
        }
    } catch (e) {
        console.error("Failed to load entries:", e);
        if (box) box.innerHTML = `<div class="muted-text" style="padding: 16px; text-align: center;">載入條目時發生異常。</div>`;
    }
}

function prevEntriesPage() {
    if (refEntriesCurrentPage > 1) {
        loadReferenceEntries(refEntriesCurrentPage - 1);
    }
}

function nextEntriesPage() {
    loadReferenceEntries(refEntriesCurrentPage + 1);
}

async function deleteReferenceEntry(entryId) {
    if (!confirm(`確定要刪除第 #${entryId} 則技法條目嗎？`)) return;
    try {
        const resp = await fetch(`/api/reference/entries/${entryId}`, { method: "DELETE" });
        const data = await resp.json();
        if (!resp.ok) {
            showToast(data.detail || "刪除條目失敗");
            return;
        }
        showToast(`已刪除條目 #${entryId}。`);
        await loadReferenceEntries(refEntriesCurrentPage);
        await loadReferenceHotTags();
        await loadReferenceBooks();
    } catch (e) {
        showToast("刪除條目時發生錯誤");
    }
}

async function handleTagInputKey(evt, entryId, inputEl) {
    if (evt.key === "Enter") {
        evt.preventDefault();
        const tagName = inputEl.value.trim();
        if (!tagName) return;
        try {
            const resp = await fetch(`/api/reference/entries/${entryId}/tags`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ tag_name: tagName }),
            });
            const data = await resp.json();
            if (!resp.ok) {
                showToast(data.detail || "新增標籤失敗");
                return;
            }
            inputEl.value = "";
            showToast(`已為條目 #${entryId} 新增標籤 [${tagName}]！`);
            await loadReferenceEntries(refEntriesCurrentPage);
            await loadReferenceHotTags();
            refreshWizardGenreCatalog();
        } catch (e) {
            showToast("新增標籤時發生異常");
        }
    }
}

async function loadReferenceHotTags() {
    const hotBox = document.getElementById("ref-hot-tags-box");
    try {
        const resp = await fetch("/api/reference/tags");
        if (!resp.ok) {
            if (hotBox) hotBox.innerHTML = `<span class="muted-text">尚無標籤數據</span>`;
            return;
        }
        const data = await resp.json();
        const tags = data.tags || [];

        if (hotBox) {
            if (tags.length === 0) {
                hotBox.innerHTML = `<span class="muted-text">尚無標籤數據</span>`;
            } else {
                const topTags = tags.slice(0, 16);
                hotBox.innerHTML = topTags.map(t => `
                    <span class="ref-hot-tag-pill ${currentSelectedRefTag === t.tag_name ? 'active' : ''}" data-ref-action="filter-hot-tag" data-tag="${escapeHtml(t.tag_name)}">
                        #${escapeHtml(t.tag_name)} (${t.count})
                    </span>
                `).join("");
            }
        }
    } catch (e) {
        console.error("Failed to load tags:", e);
        if (hotBox) hotBox.innerHTML = `<span class="muted-text">尚無標籤數據</span>`;
    }
}

function toggleHotTagFilter(tagName) {
    if (currentSelectedRefTag === tagName) {
        currentSelectedRefTag = "";
    } else {
        currentSelectedRefTag = tagName;
    }
    loadReferenceHotTags();
    loadReferenceEntries(1);
}

let tagClusterNodes = [];
let clusterCanvasInitialized = false;
let clusterZoom = 1.0;
let clusterPanX = 0;
let clusterPanY = 0;
let isClusterDragging = false;
let clusterDragStartX = 0;
let clusterDragStartY = 0;
let activeClusterFilter = null;
let hoveredClusterNode = null;
let lastClusterRelaxW = 0;
let lastClusterRelaxH = 0;

function renderTagAssetsList() {
    const assetsListBox = document.getElementById("ref-assets-tags-list");
    if (!assetsListBox) return;
    if (!tagClusterNodes || tagClusterNodes.length === 0) {
        assetsListBox.innerHTML = `<span class="muted-text">尚無標籤資產數據</span>`;
        return;
    }
    const visibleNodes = activeClusterFilter
        ? tagClusterNodes.filter(n => n.cluster === activeClusterFilter)
        : tagClusterNodes;
    const filterBadge = activeClusterFilter
        ? `<span class="ref-tag-pill" style="border-color:#3b82f688; background:rgba(59,130,246,0.18); color:#93c5fd;" title="點擊圖例可取消分群篩選">分群篩選：${escapeHtml(activeClusterFilter)} (${visibleNodes.length})</span>`
        : "";
    if (visibleNodes.length === 0) {
        assetsListBox.innerHTML = `${filterBadge}<span class="muted-text">此分群暫無標籤資產數據</span>`;
        return;
    }
    assetsListBox.innerHTML = filterBadge + visibleNodes.map(n => {
        const statusTag = n.is_outlier ? ` <span style="color:#ef4444;font-size:0.7rem;">[孤島]</span>` : "";
        return `
        <span class="ref-tag-pill" style="border-color:${n.color}55; background:${n.color}15; color:${n.color};" title="所屬題材：${escapeHtml(n.cluster)} ｜ 引用頻次：${n.count}${n.is_outlier ? ' ｜ 密度：孤島雜訊' : ''}">
            ${escapeHtml(n.name)}${statusTag} <strong style="opacity:0.85;">(${n.count})</strong>
        </span>`;
    }).join("");
}

async function loadReferenceTagAssets() {
    const assetsListBox = document.getElementById("ref-assets-tags-list");
    const legendBox = document.getElementById("ref-cluster-legend");
    try {
        const resp = await fetch("/api/reference/tag_assets");
        if (!resp.ok) {
            if (assetsListBox) assetsListBox.innerHTML = `<span class="muted-text">尚無標籤資產數據</span>`;
            if (legendBox) legendBox.innerHTML = `<span class="muted-text" style="font-size:0.75rem;">尚無分群圖例</span>`;
            tagClusterNodes = [];
            renderTagClusterCanvas();
            return;
        }
        const data = await resp.json();
        const stats = data.stats || {};
        tagClusterNodes = data.nodes || [];
        const clusters = data.clusters || {};

        // Update metric cards
        const uEl = document.getElementById("ref-asset-total-unique");
        if (uEl) uEl.textContent = stats.total_unique || 0;
        const usEl = document.getElementById("ref-asset-total-usages");
        if (usEl) usEl.textContent = stats.total_usages || 0;
        const gEl = document.getElementById("ref-asset-genre-count");
        if (gEl) gEl.textContent = stats.genre_tags_count || 0;
        const tEl = document.getElementById("ref-asset-technique-count");
        if (tEl) tEl.textContent = stats.technique_tags_count || 0;
        const cEl = document.getElementById("ref-asset-clusters-count");
        if (cEl) cEl.textContent = stats.clusters_count || 0;
        const oEl = document.getElementById("ref-asset-outliers-count");
        if (oEl) oEl.textContent = stats.outliers_count || 0;

        // Render dynamic cluster legend from backend node colors (clickable filter)
        if (legendBox) {
            if (Object.keys(clusters).length === 0) {
                legendBox.innerHTML = `<span class="muted-text" style="font-size:0.75rem;">尚無分群圖例</span>`;
            } else {
                const clusterColorMap = {};
                tagClusterNodes.forEach(n => {
                    if (n.cluster && n.color) {
                        clusterColorMap[n.cluster] = n.color;
                    }
                });
                legendBox.innerHTML = Object.keys(clusters).map(c => `
                    <span class="ref-legend-item ${activeClusterFilter === c ? 'active' : ''}" data-ref-action="filter-cluster" data-cluster="${escapeHtml(c)}">
                        <span class="ref-legend-dot" style="background:${clusterColorMap[c] || '#94a3b8'};"></span>
                        ${escapeHtml(c)} (${clusters[c]})
                    </span>
                `).join("");
            }
        }

        // Render read-only tag assets list (respecting any active cluster filter)
        renderTagAssetsList();

        // Run anti-collision relaxation and render HiDPI canvas
        const canvas = document.getElementById("ref-tag-cluster-canvas");
        if (canvas) {
            const rect = canvas.getBoundingClientRect();
            relaxNodePositions(tagClusterNodes, rect.width || 900, rect.height || 520);
        }

        initClusterCanvasEvents();
        renderTagClusterCanvas();
    } catch (e) {
        console.error("Failed to load tag assets:", e);
        if (assetsListBox) assetsListBox.innerHTML = `<span class="muted-text">尚無標籤資產數據</span>`;
        if (legendBox) legendBox.innerHTML = `<span class="muted-text" style="font-size:0.75rem;">尚無分群圖例</span>`;
        tagClusterNodes = [];
        renderTagClusterCanvas();
    }
}

function relaxNodePositions(nodes, width, height) {
    lastClusterRelaxW = width;
    lastClusterRelaxH = height;
    if (!nodes || nodes.length === 0) return;
    const cx = width / 2;
    const cy = height / 2;
    // Isotropic scaling factor: minimum of usable width and height to preserve Euclidean distance
    const pad = 60;
    const baseScale = Math.min((width - pad * 2) / 2, (height - pad * 2) / 2) * 0.92;

    nodes.forEach(n => {
        n.baseX = cx + (n.x || 0) * baseScale;
        n.baseY = cy + (n.y || 0) * baseScale;
        n.radius = Math.max(6, Math.min(20, 5 + Math.sqrt(n.count || 1) * 2.8));
    });

    // 45 iterations of gentle repulsive anti-collision relaxation
    const iterations = 45;
    for (let iter = 0; iter < iterations; iter++) {
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const n1 = nodes[i];
                const n2 = nodes[j];
                const dx = n2.baseX - n1.baseX;
                const dy = n2.baseY - n1.baseY;
                const dist = Math.hypot(dx, dy) || 0.001;
                const minDist = n1.radius + n2.radius + 16;
                if (dist < minDist) {
                    const overlap = (minDist - dist) * 0.45;
                    const nx = dx / dist;
                    const ny = dy / dist;
                    n1.baseX -= nx * overlap;
                    n1.baseY -= ny * overlap;
                    n2.baseX += nx * overlap;
                    n2.baseY += ny * overlap;
                }
            }
        }
        // Gravitational cohesion toward cluster center
        nodes.forEach(n => {
            n.baseX += (cx - n.baseX) * 0.003;
            n.baseY += (cy - n.baseY) * 0.003;
        });
    }
}

function renderTagClusterCanvas() {
    const canvas = document.getElementById("ref-tag-cluster-canvas");
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const cssW = rect.width || 900;
    const cssH = rect.height || 520;

    // Retina HiDPI buffer synchronization: only resize if dimensions changed to preserve GPU texture
    const targetW = Math.round(cssW * dpr);
    const targetH = Math.round(cssH * dpr);
    if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
    }

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Clear canvas
    ctx.clearRect(0, 0, cssW, cssH);

    const cx = cssW / 2;
    const cy = cssH / 2;
    const viewCx = cx + clusterPanX;
    const viewCy = cy + clusterPanY;

    // Draw background subtle coordinate grid & concentric orbits
    ctx.save();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 1;

    // Center axes
    ctx.beginPath();
    ctx.moveTo(viewCx, 16);
    ctx.lineTo(viewCx, cssH - 16);
    ctx.moveTo(16, viewCy);
    ctx.lineTo(cssW - 16, viewCy);
    ctx.stroke();

    // Concentric orbits
    const maxRadius = Math.min(cssW, cssH) * 0.45 * clusterZoom;
    [0.35, 0.7, 1.0].forEach(ratio => {
        ctx.beginPath();
        ctx.arc(viewCx, viewCy, maxRadius * ratio, 0, Math.PI * 2);
        ctx.stroke();
    });
    ctx.restore();

    if (!tagClusterNodes || tagClusterNodes.length === 0) {
        ctx.fillStyle = "rgba(148, 163, 184, 0.5)";
        ctx.font = "14px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("尚無足夠的標籤向量數據可供投影", cx, cy);
        return;
    }

    // Re-relax layout when the visible canvas size settles or changes after initial render
    if (Math.abs(cssW - lastClusterRelaxW) > 5 || Math.abs(cssH - lastClusterRelaxH) > 5) {
        relaxNodePositions(tagClusterNodes, cssW, cssH);
    }

    // Calculate canvas screen positions for each node
    tagClusterNodes.forEach(n => {
        n.screenX = cx + (n.baseX - cx) * clusterZoom + clusterPanX;
        n.screenY = cy + (n.baseY - cy) * clusterZoom + clusterPanY;
        n.drawRadius = Math.max(3, n.radius * Math.min(1.8, Math.max(0.6, Math.sqrt(clusterZoom))));
    });

    // Draw connecting webs between nodes in the same cluster if nearby
    ctx.save();
    for (let i = 0; i < tagClusterNodes.length; i++) {
        const n1 = tagClusterNodes[i];
        if (activeClusterFilter && n1.cluster !== activeClusterFilter) continue;
        for (let j = i + 1; j < tagClusterNodes.length; j++) {
            const n2 = tagClusterNodes[j];
            if (n1.cluster === n2.cluster) {
                const dist = Math.hypot(n1.screenX - n2.screenX, n1.screenY - n2.screenY);
                if (dist < 90 * clusterZoom) {
                    ctx.strokeStyle = `${n1.color}25`;
                    ctx.lineWidth = Math.min(1.5, 0.7 * clusterZoom);
                    ctx.beginPath();
                    ctx.moveTo(n1.screenX, n1.screenY);
                    ctx.lineTo(n2.screenX, n2.screenY);
                    ctx.stroke();
                }
            }
        }
    }
    ctx.restore();

    // Draw glowing nodes
    tagClusterNodes.forEach(n => {
        const isDimmed = activeClusterFilter && n.cluster !== activeClusterFilter;
        const isHovered = hoveredClusterNode === n;
        const alphaMul = isDimmed ? 0.15 : 1.0;
        const rad = isHovered ? n.drawRadius * 1.3 : n.drawRadius;

        // Outer halo
        const haloR = rad * 2.4;
        const grad = ctx.createRadialGradient(n.screenX, n.screenY, 1, n.screenX, n.screenY, haloR);
        grad.addColorStop(0, `${n.color}${Math.round(200 * alphaMul).toString(16).padStart(2, "0")}`);
        grad.addColorStop(0.5, `${n.color}${Math.round(60 * alphaMul).toString(16).padStart(2, "0")}`);
        grad.addColorStop(1, `${n.color}00`);
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(n.screenX, n.screenY, haloR, 0, Math.PI * 2);
        ctx.fill();

        // Solid core
        ctx.fillStyle = n.color;
        ctx.globalAlpha = alphaMul;
        ctx.beginPath();
        ctx.arc(n.screenX, n.screenY, rad, 0, Math.PI * 2);
        ctx.fill();

        // Outlier dashed ring indicator for noise points
        if (n.is_outlier) {
            ctx.save();
            ctx.strokeStyle = "rgba(239, 68, 68, 0.9)";
            ctx.setLineDash([2, 3]);
            ctx.lineWidth = 1.3;
            ctx.beginPath();
            ctx.arc(n.screenX, n.screenY, rad + 3, 0, Math.PI * 2);
            ctx.stroke();
            ctx.restore();
        }

        // White core spark
        ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
        ctx.beginPath();
        ctx.arc(n.screenX, n.screenY, Math.max(1.5, rad * 0.35), 0, Math.PI * 2);
        ctx.fill();

        ctx.globalAlpha = 1.0;
    });

    // Draw high-contrast text pill labels
    tagClusterNodes.forEach(n => {
        const isDimmed = activeClusterFilter && n.cluster !== activeClusterFilter;
        if (isDimmed) return;

        const isHovered = hoveredClusterNode === n;
        const shouldShowLabel = isHovered || activeClusterFilter || clusterZoom >= 1.25 || n.count >= 3 || tagClusterNodes.length <= 25;
        if (!shouldShowLabel) return;

        ctx.save();
        ctx.font = isHovered
            ? "bold 12px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
            : "600 11px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
        const textMetrics = ctx.measureText(n.name);
        const textW = textMetrics.width;
        const textH = 15;
        const pillX = n.screenX - textW / 2 - 5;
        const pillY = n.screenY + n.drawRadius + 4;

        // Dark translucent pill backdrop
        ctx.fillStyle = isHovered ? "rgba(15, 23, 42, 0.95)" : "rgba(10, 15, 30, 0.85)";
        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(pillX, pillY, textW + 10, textH, 4);
        } else {
            ctx.rect(pillX, pillY, textW + 10, textH);
        }
        ctx.fill();

        // Border outline
        ctx.strokeStyle = isHovered ? "#ffffff" : `${n.color}77`;
        ctx.lineWidth = isHovered ? 1.2 : 0.8;
        ctx.stroke();

        // Crisp white text
        ctx.fillStyle = isHovered ? "#ffffff" : "rgba(241, 245, 249, 0.95)";
        ctx.textAlign = "center";
        ctx.fillText(n.name, n.screenX, pillY + 11);
        ctx.restore();
    });
}

function zoomClusterCanvas(factor) {
    clusterZoom = Math.max(0.4, Math.min(5.5, clusterZoom * factor));
    renderTagClusterCanvas();
}

function resetClusterCanvasView() {
    clusterZoom = 1.0;
    clusterPanX = 0;
    clusterPanY = 0;
    activeClusterFilter = null;
    hoveredClusterNode = null;
    const legendItems = document.querySelectorAll(".ref-legend-item");
    legendItems.forEach(el => el.classList.remove("active"));
    const tooltip = document.getElementById("ref-canvas-tooltip");
    if (tooltip) tooltip.classList.add("hidden");
    renderTagAssetsList();
    renderTagClusterCanvas();
}

function toggleClusterLegendFilter(clusterName) {
    if (activeClusterFilter === clusterName) {
        activeClusterFilter = null;
    } else {
        activeClusterFilter = clusterName;
    }
    const legendItems = document.querySelectorAll(".ref-legend-item");
    legendItems.forEach(el => {
        if (el.dataset.cluster === activeClusterFilter) {
            el.classList.add("active");
        } else {
            el.classList.remove("active");
        }
    });
    renderTagAssetsList();
    renderTagClusterCanvas();
}

// Delegated handler for library actions. Book titles and tag names must never be
// interpolated into inline JS strings: HTML attribute decoding turns &#39; back into
// a quote, which closes the string literal and breaks the button (or injects code).
document.addEventListener("click", (event) => {
    const el = event.target && event.target.closest ? event.target.closest("[data-ref-action]") : null;
    if (!el) return;
    const action = el.dataset.refAction;
    if (action === "jump-book") {
        jumpToBookEntries(Number(el.dataset.sourceId));
    } else if (action === "delete-book") {
        deleteReferenceBook(Number(el.dataset.sourceId), el.dataset.fileName || "");
    } else if (action === "filter-hot-tag") {
        toggleHotTagFilter(el.dataset.tag || "");
    } else if (action === "filter-cluster") {
        toggleClusterLegendFilter(el.dataset.cluster || "");
    }
});

let clusterCanvasRafId = null;
function requestClusterCanvasRender() {
    if (!clusterCanvasRafId) {
        clusterCanvasRafId = requestAnimationFrame(() => {
            clusterCanvasRafId = null;
            renderTagClusterCanvas();
        });
    }
}

function initClusterCanvasEvents() {
    if (clusterCanvasInitialized) return;
    clusterCanvasInitialized = true;

    const canvas = document.getElementById("ref-tag-cluster-canvas");
    if (!canvas) return;
    const tooltip = document.getElementById("ref-canvas-tooltip");

    // Smooth mouse wheel zoom centered at cursor
    canvas.addEventListener("wheel", (evt) => {
        evt.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const mouseX = evt.clientX - rect.left;
        const mouseY = evt.clientY - rect.top;
        const cx = rect.width / 2;
        const cy = rect.height / 2;

        const zoomFactor = evt.deltaY < 0 ? 1.15 : 0.87;
        const newZoom = Math.max(0.4, Math.min(5.5, clusterZoom * zoomFactor));

        clusterPanX = mouseX - cx - (mouseX - cx - clusterPanX) * (newZoom / clusterZoom);
        clusterPanY = mouseY - cy - (mouseY - cy - clusterPanY) * (newZoom / clusterZoom);
        clusterZoom = newZoom;

        requestClusterCanvasRender();
    }, { passive: false });

    // Drag to pan
    canvas.addEventListener("mousedown", (evt) => {
        if (evt.button === 0) {
            isClusterDragging = true;
            clusterDragStartX = evt.clientX - clusterPanX;
            clusterDragStartY = evt.clientY - clusterPanY;
            canvas.classList.add("is-dragging");
        }
    });

    window.addEventListener("mousemove", (evt) => {
        if (isClusterDragging) {
            clusterPanX = evt.clientX - clusterDragStartX;
            clusterPanY = evt.clientY - clusterDragStartY;
            requestClusterCanvasRender();
            return;
        }

        const rect = canvas.getBoundingClientRect();
        if (
            evt.clientX < rect.left ||
            evt.clientX > rect.right ||
            evt.clientY < rect.top ||
            evt.clientY > rect.bottom
        ) {
            if (hoveredClusterNode) {
                hoveredClusterNode = null;
                if (tooltip) tooltip.classList.add("hidden");
                renderTagClusterCanvas();
            }
            return;
        }

        const mouseX = evt.clientX - rect.left;
        const mouseY = evt.clientY - rect.top;

        let found = null;
        let minDist = 26;
        for (const n of tagClusterNodes) {
            if (activeClusterFilter && n.cluster !== activeClusterFilter) continue;
            const dist = Math.hypot(n.screenX - mouseX, n.screenY - mouseY);
            if (dist < (n.drawRadius || 10) + 10 && dist < minDist) {
                minDist = dist;
                found = n;
            }
        }

        if (found !== hoveredClusterNode) {
            hoveredClusterNode = found;
            renderTagClusterCanvas();
        }

        if (hoveredClusterNode && tooltip) {
            tooltip.classList.remove("hidden");
            tooltip.style.left = `${hoveredClusterNode.screenX}px`;
            tooltip.style.top = `${hoveredClusterNode.screenY}px`;
            const statusBadge = hoveredClusterNode.is_outlier
                ? `<span style="color:#ef4444; font-size:0.75rem; font-weight:600; margin-left:6px;">⚠️ 孤島雜訊待整頓</span>`
                : (hoveredClusterNode.density_cluster >= 0 ? `<span style="color:#10b981; font-size:0.75rem; margin-left:6px;">[密度聚落 #${hoveredClusterNode.density_cluster}]</span>` : "");
            tooltip.innerHTML = `
                <div style="font-weight:700; color:${hoveredClusterNode.color}; font-size:0.86rem;">${escapeHtml(hoveredClusterNode.name)} ${statusBadge}</div>
                <div style="color:#94a3b8; font-size:0.74rem; margin-top:3px;">
                    語義聚類：<strong style="color:#e2e8f0;">${escapeHtml(hoveredClusterNode.cluster)}</strong> ｜ 引用頻次：<strong style="color:#e2e8f0;">${hoveredClusterNode.count} 次</strong>
                </div>
            `;
        } else if (tooltip) {
            tooltip.classList.add("hidden");
        }
    });

    window.addEventListener("mouseup", () => {
        if (isClusterDragging) {
            isClusterDragging = false;
            canvas.classList.remove("is-dragging");
        }
    });

    window.addEventListener("resize", () => {
        if (tagClusterNodes && tagClusterNodes.length > 0) {
            const rect = canvas.getBoundingClientRect();
            relaxNodePositions(tagClusterNodes, rect.width || 900, rect.height || 520);
            renderTagClusterCanvas();
        }
    });
}

async function triggerAutoCleanTags() {
    const btn = document.getElementById("btn-auto-clean-tags");
    const origText = btn ? btn.innerHTML : "⚡ 啟動 AI 智慧整頓";
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<span class="pulse-indicator" style="display:inline-block; margin-right:6px;"></span> ⏳ AI 智慧整頓中...`;
    }

    const progressCard = document.getElementById("ref-ai-clean-progress");
    const barEl = document.getElementById("ref-ai-clean-bar");
    const pctEl = document.getElementById("ref-ai-clean-pct");
    const stageEl = document.getElementById("ref-ai-clean-stage");
    const descEl = document.getElementById("ref-ai-clean-desc");

    if (progressCard) {
        progressCard.classList.remove("hidden");
        if (barEl) barEl.style.width = "15%";
        if (pctEl) pctEl.textContent = "15%";
        if (stageEl) stageEl.textContent = "階段 1/4：全庫標籤頻率與 DBSCAN 密度拓撲探測...";
        if (descEl) descEl.textContent = "正在計算餘弦距離矩陣，探測密集聚落與孤島雜訊...";
    }
    showToast("AI 正在掃描全庫標籤，結合密度拓撲執行開放題材歸納與近義詞合併...");

    const stepTimer1 = setTimeout(() => {
        if (barEl) barEl.style.width = "50%";
        if (pctEl) pctEl.textContent = "50%";
        if (stageEl) stageEl.textContent = "階段 2/4：神經網路開放分類與新題材動態歸納...";
        if (descEl) descEl.textContent = "正在根據密度拓撲自發提煉新興題材與文學技法契約...";
    }, 1000);

    const stepTimer2 = setTimeout(() => {
        if (barEl) barEl.style.width = "85%";
        if (pctEl) pctEl.textContent = "85%";
        if (stageEl) stageEl.textContent = "階段 3/4：SQLite 原子交易替換、去重與向量重算...";
        if (descEl) descEl.textContent = "正在執行資料庫去重並調用 CPU BGE 生成 512 維向量...";
    }, 2200);

    try {
        const resp = await fetch("/api/reference/tags/auto_clean", { method: "POST" });
        clearTimeout(stepTimer1);
        clearTimeout(stepTimer2);
        const data = await resp.json();
        if (handleAiModelError(data)) {
            if (progressCard) progressCard.classList.add("hidden");
            return;
        }
        if (!resp.ok) {
            showToast(data.detail || data.error || "標籤整頓失敗");
            if (progressCard) progressCard.classList.add("hidden");
            return;
        }

        if (barEl) barEl.style.width = "100%";
        if (pctEl) pctEl.textContent = "100%";
        if (stageEl) stageEl.textContent = "✔ 標籤整頓與 2D 群聚圖更新完成！";
        if (descEl) descEl.textContent = "已自動刷新 2D 語意群聚投影圖並注入創世題材庫。";

        const st = data.stats || {};
        showToast(`✔ AI 整頓完成：規範化 ${st.normalized || 0} 處、合併 ${st.merged || 0} 處、清理雜訊 ${st.dropped || 0} 處！`);

        await loadReferenceTagAssets();
        await loadReferenceHotTags();
        if (document.getElementById("ref-tab-btn-entries")?.classList.contains("active")) {
            await loadReferenceEntries(refEntriesCurrentPage);
        }
        refreshWizardGenreCatalog();

        setTimeout(() => {
            if (progressCard) progressCard.classList.add("hidden");
        }, 3000);
    } catch (e) {
        clearTimeout(stepTimer1);
        clearTimeout(stepTimer2);
        showToast("AI 標籤整頓時發生錯誤");
        if (progressCard) progressCard.classList.add("hidden");
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = origText;
        }
    }
}

function computeDefaultActionForStagedItem(item, extractSkills, extractPacing) {
    if (item.status === "CORRUPTED") return "SKIP";
    if (item.status === "NEW") return "INGEST";
    // For DUPLICATE books: if user checked a track that this book hasn't extracted yet, default to INGEST (supplement)
    const needsSkills = extractSkills && !item.has_skills;
    const needsPacing = extractPacing && !item.has_pacing;
    if (needsSkills || needsPacing) {
        return "INGEST";
    }
    return "SKIP";
}

function updateConfirmIngestButtonLabel() {
    const btnConfirm = document.getElementById("btn-confirm-ingest");
    const summaryEl = document.getElementById("ref-staged-summary");
    const validItems = refStagedFilesQueue.filter(
        (i) => i.status !== "CORRUPTED" && i.action !== "SKIP" && (i.extract_skills || i.extract_pacing)
    );
    if (summaryEl) {
        if (refStagedFilesQueue.length === 0) {
            summaryEl.textContent = "待萃取清單：0 本（可跨資料夾多次選取多部小說累加至清單，或於萃取進行中隨時追加新批次）";
        } else {
            summaryEl.textContent = `待萃取清單：共加入 ${refStagedFilesQueue.length} 本小說，預計執行萃取 ${validItems.length} 本（跳過 ${refStagedFilesQueue.length - validItems.length} 本）`;
        }
    }
    if (btnConfirm) {
        btnConfirm.disabled = validItems.length === 0;
        const isActive = !document.getElementById("ref-active-progress-card")?.classList.contains("hidden");
        if (isActive && validItems.length > 0) {
            btnConfirm.textContent = `➕ 追加至背景萃取佇列（${validItems.length} 本）`;
        } else if (validItems.length > 0) {
            btnConfirm.textContent = `▶ 確認啟動背景萃取入庫（${validItems.length} 本）`;
        } else {
            btnConfirm.textContent = `▶ 確認啟動背景萃取入庫`;
        }
    }
}

function renderStagedPreflightQueue() {
    const box = document.getElementById("ref-preflight-results");
    if (!box) return;
    if (refStagedFilesQueue.length === 0) {
        box.innerHTML = `<div class="muted-text" style="padding: 8px 4px;">請選擇一個或多個參考書籍檔案（支援分批多次選取累加）...</div>`;
        preflightNewOrDupPaths = [];
        updateConfirmIngestButtonLabel();
        return;
    }

    preflightNewOrDupPaths = refStagedFilesQueue
        .filter((i) => i.status !== "CORRUPTED")
        .map((i) => i.file_path);

    box.innerHTML = refStagedFilesQueue
        .map((i, idx) => {
            let statusBadge = `<span class="ref-pill" style="font-size:0.72rem; color:#10b981;">新書籍</span>`;
            if (i.status === "DUPLICATE") {
                const sText = i.has_skills ? `技法:${i.entry_count}則` : "未萃技法";
                const pText = i.has_pacing ? "已有情節" : "未萃情節";
                statusBadge = `<span class="ref-pill" style="font-size:0.72rem; color:#0066ff;">已入庫 (${sText} · ${pText})</span>`;
            } else if (i.status === "CORRUPTED") {
                statusBadge = `<span class="ref-pill" style="font-size:0.72rem; color:#ef4444;">損毀</span>`;
            }

            let optionsHtml = `<option value="SKIP" selected>損毀略過</option>`;
            if (i.status === "NEW") {
                optionsHtml = `
                    <option value="INGEST" ${i.action === "INGEST" ? "selected" : ""}>入庫萃取（新檔案）</option>
                    <option value="SKIP" ${i.action === "SKIP" ? "selected" : ""}>跳過此檔</option>
                `;
            } else if (i.status === "DUPLICATE") {
                optionsHtml = `
                    <option value="INGEST" ${i.action === "INGEST" ? "selected" : ""}>補齊未萃軌道（重用藍圖）</option>
                    <option value="REEXTRACT" ${i.action === "REEXTRACT" ? "selected" : ""}>強制完整重新萃取</option>
                    <option value="SKIP" ${i.action === "SKIP" ? "selected" : ""}>跳過重複檔案</option>
                `;
            }

            return `<div class="qa-tree-item" style="display:flex;justify-content:space-between;align-items:center;gap:0.5rem;flex-wrap:wrap;">
                <div style="display:flex;align-items:center;gap:0.4rem;flex-wrap:wrap;flex:1;min-width:220px;">
                    ${statusBadge}
                    <strong>${escapeHtml(i.file_name)}</strong>
                    <span class="muted-text" style="font-size:0.75rem;">（${escapeHtml(i.encoding || "未知")}）${i.error_reason ? " — " + escapeHtml(i.error_reason) : ""}</span>
                </div>
                <div style="display:flex;align-items:center;gap:0.55rem;flex-wrap:wrap;">
                    <label style="display:flex;align-items:center;gap:0.25rem;font-size:0.78rem;cursor:pointer;" title="對此書萃取寫作技能">
                        <input type="checkbox" ${i.extract_skills ? "checked" : ""} ${i.status === "CORRUPTED" ? "disabled" : ""} onchange="updateStagedItemTrack(${idx}, 'extract_skills', this.checked)" />
                        <span>✍️技能</span>
                    </label>
                    <label style="display:flex;align-items:center;gap:0.25rem;font-size:0.78rem;cursor:pointer;" title="對此書萃取情節規劃">
                        <input type="checkbox" ${i.extract_pacing ? "checked" : ""} ${i.status === "CORRUPTED" ? "disabled" : ""} onchange="updateStagedItemTrack(${idx}, 'extract_pacing', this.checked)" />
                        <span>📈情節</span>
                    </label>
                    <select class="ref-item-action" data-path="${escapeHtml(i.file_path)}" style="width:185px;" ${i.status === "CORRUPTED" ? "disabled" : ""} onchange="updateStagedItemAction(${idx}, this.value)">
                        ${optionsHtml}
                    </select>
                    <button type="button" class="btn btn-glass btn-xs" onclick="removeStagedReferenceFile(${idx})" title="從待萃清單移除">✕</button>
                </div>
            </div>`;
        })
        .join("");

    updateConfirmIngestButtonLabel();
}

function updateStagedItemAction(idx, actionVal) {
    if (!refStagedFilesQueue[idx]) return;
    refStagedFilesQueue[idx].action = actionVal;
    updateConfirmIngestButtonLabel();
}

function updateStagedItemTrack(idx, field, checked) {
    if (!refStagedFilesQueue[idx]) return;
    refStagedFilesQueue[idx][field] = Boolean(checked);
    const item = refStagedFilesQueue[idx];
    if (item.status === "DUPLICATE" && item.action !== "REEXTRACT") {
        item.action = computeDefaultActionForStagedItem(
            item,
            item.extract_skills,
            item.extract_pacing
        );
    }
    renderStagedPreflightQueue();
}

function onGlobalTrackOptionChange() {
    const extractSkills = document.getElementById("ref-opt-extract-skills")?.checked ?? true;
    const extractPacing = document.getElementById("ref-opt-extract-pacing")?.checked ?? false;
    refStagedFilesQueue.forEach((item) => {
        if (item.status === "CORRUPTED") return;
        item.extract_skills = extractSkills;
        item.extract_pacing = extractPacing;
        if (item.status === "DUPLICATE" && item.action !== "REEXTRACT") {
            item.action = computeDefaultActionForStagedItem(item, extractSkills, extractPacing);
        }
    });
    renderStagedPreflightQueue();
}

function applyBatchStagedAction(actionType) {
    if (refStagedFilesQueue.length === 0) return;
    refStagedFilesQueue.forEach((item) => {
        if (item.status === "CORRUPTED") return;
        if (actionType === "REEXTRACT") {
            item.action = item.status === "NEW" ? "INGEST" : "REEXTRACT";
        } else {
            item.action = actionType;
        }
    });
    renderStagedPreflightQueue();
}

function removeStagedReferenceFile(idx) {
    refStagedFilesQueue.splice(idx, 1);
    renderStagedPreflightQueue();
}

function clearStagedReferenceFiles() {
    refStagedFilesQueue = [];
    const fileIn = document.getElementById("ref-file-input");
    if (fileIn) fileIn.value = "";
    renderStagedPreflightQueue();
}

async function runReferenceUploadPreflight() {
    const input = document.getElementById("ref-file-input");
    if (!input || !input.files.length) return;
    const formData = new FormData();
    Array.from(input.files).forEach((f) => formData.append("files", f));
    input.value = "";

    const extractSkills = document.getElementById("ref-opt-extract-skills")?.checked ?? true;
    const extractPacing = document.getElementById("ref-opt-extract-pacing")?.checked ?? false;

    const resp = await fetch("/api/reference/upload_preflight", {
        method: "POST",
        body: formData,
    });
    const data = await resp.json();
    const items = data.items || [];

    items.forEach((incoming) => {
        const defaultAction = computeDefaultActionForStagedItem(
            incoming,
            extractSkills,
            extractPacing
        );
        const enriched = {
            ...incoming,
            action: defaultAction,
            extract_skills: extractSkills,
            extract_pacing: extractPacing,
        };
        const existingIdx = refStagedFilesQueue.findIndex(
            (e) =>
                (incoming.file_sha256 && e.file_sha256 === incoming.file_sha256) ||
                e.file_name === incoming.file_name
        );
        if (existingIdx >= 0) {
            refStagedFilesQueue[existingIdx] = enriched;
        } else {
            refStagedFilesQueue.push(enriched);
        }
    });

    renderStagedPreflightQueue();
}

async function confirmReferenceIngest() {
    const globalSkills = document.getElementById("ref-opt-extract-skills")?.checked ?? true;
    const globalPacing = document.getElementById("ref-opt-extract-pacing")?.checked ?? false;
    const jitterVal = parseFloat(document.getElementById("ref-pacing-jitter-slider")?.value || "0.0");

    let items = [];
    if (refStagedFilesQueue.length > 0) {
        items = refStagedFilesQueue.map((st) => ({
            file_path: st.file_path,
            action: st.action,
            extract_skills: Boolean(st.extract_skills),
            extract_pacing: Boolean(st.extract_pacing),
        }));
    } else {
        const actionSelects = Array.from(document.querySelectorAll(".ref-item-action"));
        items = actionSelects.map((sel) => ({
            file_path: sel.getAttribute("data-path"),
            action: sel.value,
            extract_skills: globalSkills,
            extract_pacing: globalPacing,
        }));
    }

    const activeItems = items.filter(
        (it) => it.action !== "SKIP" && (it.extract_skills || it.extract_pacing)
    );
    if (activeItems.length === 0) {
        showToast("請至少為一部小說勾選一種萃取模式（寫作技能或情節規劃）並設為入庫！");
        return;
    }

    const anySkills = activeItems.some((it) => it.extract_skills);
    const anyPacing = activeItems.some((it) => it.extract_pacing);

    const resp = await fetch("/api/reference/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            items: items,
            run_in_background: true,
            extract_skills: anySkills,
            extract_pacing: anyPacing,
            pacing_jitter: jitterVal,
        }),
    });
    const data = await resp.json();
    if (handleAiModelError(data)) {
        closeModal("modal-reference");
        return;
    }
    if (!resp.ok) {
        showToast(data.detail || data.error || "入庫萃取失敗，請重試。");
        return;
    }

    let toastMsg = "參考書籍入庫已啟動";
    wasPacingActive = false;
    pacingCompletedHandled = false;
    if (pacingHideTimer) {
        clearTimeout(pacingHideTimer);
        pacingHideTimer = null;
    }
    if (data.appended_to_active_queue) {
        toastMsg = `已將 ${activeItems.length} 本小說追加至背景萃取佇列，將於當前任務完成後自動接續執行！`;
    } else if (anySkills && anyPacing) {
        toastMsg = `已啟動 ${activeItems.length} 本小說雙軌背景佇列萃取（情節規劃波浪 + 寫作技能）！`;
    } else if (anyPacing) {
        toastMsg = `已啟動 ${activeItems.length} 本小說宏觀情節規劃波浪背景提煉！`;
    } else {
        toastMsg = `已啟動 ${activeItems.length} 本小說寫作技能背景佇列萃取...`;
    }
    showToast(toastMsg);

    refStagedFilesQueue = [];
    renderStagedPreflightQueue();
    const preResults = document.getElementById("ref-preflight-results");
    if (preResults) {
        preResults.innerHTML = `<div class="muted-text" style="padding: 8px 4px;">✔ 已排入背景佇列執行中。您可隨時繼續選擇檔案追加下一批小說！</div>`;
    }

    if (refActiveTab === "pacing") {
        await loadPacingTemplates();
    }
    await checkReferenceProgressNow();
    await refreshWarroomDashboard();
}

async function openAcceptanceModal() {
    const el = document.getElementById("modal-acceptance");
    if (el) {
        el.classList.remove("hidden");
        el.style.display = "flex";
    }
    const resp = await fetch("/api/warroom/acceptance_report");
    const rep = await resp.json();
    const box = document.getElementById("acceptance-report-box");
    if (!box) return;

    const totalChapters = (rep.details && rep.details.total_chapters) || 0;
    if (totalChapters === 0) {
        box.innerHTML = `
            <div class="score-card" style="text-align: center; padding: 1.1rem; margin-bottom: 0.65rem;">
                <div style="font-size: 1.4rem; margin-bottom: 0.25rem;">📊</div>
                <div style="font-weight: 700; color: var(--text-primary); margin-bottom: 0.2rem;">目前尚無已完成章節</div>
                <div class="muted-text" style="font-size: 0.8rem;">
                    品質驗收指標將於連載章節生成後即時計算，目前尚未產出評測數據。
                </div>
            </div>
            <div class="qa-tree-item"><strong>驗收狀態：</strong> <span style="color:var(--text-secondary);font-weight:700;">待連載正文生成後評測（已生成 0 章）</span></div>
            <div class="qa-tree-item">1. 核心設定矛盾率：<strong>--%</strong>（剛性目標：0.0%）</div>
            <div class="qa-tree-item">2. 時間線因果一致性：<strong>--%</strong>（剛性目標：100.0%）</div>
            <div class="qa-tree-item">3. 伏筆回收完成率：<strong>--%</strong>（剛性目標：&ge; 95.0%）</div>
            <div class="qa-tree-item">4. 節奏平穩度：<strong>--%</strong>（剛性目標：&ge; 95.0%）</div>
            <div class="qa-tree-item">5. 文學自審綜合均分：<strong>-- 分</strong>（剛性目標：&ge; 85.0 分）</div>
        `;
        return;
    }

    const badge = rep.passed
        ? `<span style="color:#10b981;font-weight:700;">✔ 全數達標通過（已評測 ${totalChapters} 章）</span>`
        : `<span style="color:#f59e0b;font-weight:700;">⚠ 尚有未達標項目（${(rep.failed_reasons || []).join("；")}）</span>`;
    box.innerHTML = `
        <div class="qa-tree-item"><strong>綜合驗收判定：</strong> ${badge}</div>
        <div class="qa-tree-item">1. 核心設定矛盾率：<strong>${(rep.hard_conflict_rate * 100).toFixed(1)}%</strong>（剛性目標：0.0%）</div>
        <div class="qa-tree-item">2. 時間線因果一致性：<strong>${(rep.timeline_consistency_rate * 100).toFixed(1)}%</strong>（剛性目標：100.0%）</div>
        <div class="qa-tree-item">3. 伏筆回收完成率：<strong>${(rep.foreshadow_recovery_rate * 100).toFixed(1)}%</strong>（剛性目標：&ge; 95.0%）</div>
        <div class="qa-tree-item">4. 節奏平穩度：<strong>${(rep.pacing_stability_rate * 100).toFixed(1)}%</strong>（剛性目標：&ge; 95.0%）</div>
        <div class="qa-tree-item">5. 文學自審綜合均分：<strong>${rep.mean_literary_score} 分</strong>（剛性目標：&ge; 85.0 分）</div>
    `;
}

// -------------------------------------------------------------
// TAB 4: Macro Pacing Templates Workbench
// -------------------------------------------------------------
let currentPacingTemplates = [];
let currentSelectedPacingTemplate = null;

async function loadPacingTemplates() {
    try {
        const resp = await fetch("/api/reference/pacing_templates");
        if (!resp.ok) {
            currentPacingTemplates = [];
            renderPacingTemplatesSelect();
            return;
        }
        const data = await resp.json();
        currentPacingTemplates = data.templates || [];
        renderPacingTemplatesSelect();
    } catch (err) {
        console.error("Failed to load pacing templates:", err);
        currentPacingTemplates = [];
        renderPacingTemplatesSelect();
    }
}

function renderPacingTemplatesSelect() {
    const sel = document.getElementById("ref-pacing-select");
    if (!sel) return;
    if (currentPacingTemplates.length === 0) {
        sel.innerHTML = '<option value="">(尚未萃取任何情節規劃範本)</option>';
        renderPacingDetails(null);
        return;
    }
    sel.innerHTML = currentPacingTemplates.map(t => `<option value="${t.id}">${t.name} (${t.archetype})</option>`).join("");
    if (!currentSelectedPacingTemplate || !currentPacingTemplates.find(t => t.id === currentSelectedPacingTemplate.id)) {
        currentSelectedPacingTemplate = currentPacingTemplates[0];
    } else {
        currentSelectedPacingTemplate = currentPacingTemplates.find(t => t.id === currentSelectedPacingTemplate.id);
    }
    sel.value = currentSelectedPacingTemplate.id;
    renderPacingDetails(currentSelectedPacingTemplate);
}

function onSelectPacingTemplate() {
    const sel = document.getElementById("ref-pacing-select");
    if (!sel) return;
    const tId = parseInt(sel.value, 10);
    currentSelectedPacingTemplate = currentPacingTemplates.find(t => t.id === tId) || null;
    renderPacingDetails(currentSelectedPacingTemplate);
}

function renderPacingDetails(template) {
    const archetypeEl = document.getElementById("pacing-meta-archetype");
    const genreEl = document.getElementById("pacing-meta-genre");
    const stagesEl = document.getElementById("pacing-meta-stages");
    const tagsContainer = document.getElementById("pacing-meta-tags");

    if (!template) {
        if (archetypeEl) archetypeEl.innerText = "-";
        if (genreEl) genreEl.innerText = "-";
        if (stagesEl) stagesEl.innerText = "-";
        if (tagsContainer) tagsContainer.innerHTML = "";
        const canvas = document.getElementById("ref-pacing-canvas");
        if (canvas) {
            const ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = "rgba(148, 163, 184, 0.5)";
            ctx.font = "14px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("尚未選擇情節規劃範本（請先萃取或混血合成）", canvas.width / 2, canvas.height / 2);
        }
        const tbl = document.getElementById("ref-pacing-stages-table-box");
        if (tbl) tbl.innerHTML = '<div style="padding:1rem; text-align:center; color:var(--text-secondary);">尚未選擇範本（請先萃取或混血合成情節規劃）</div>';
        return;
    }

    if (archetypeEl) archetypeEl.innerText = template.archetype || "標準流派";
    if (genreEl) genreEl.innerText = `${template.genre_primary} / ${template.genre_sub}`;
    const chCountStr = template.total_chapters ? `${template.total_chapters} 章全書波浪 / ` : "";
    if (stagesEl) stagesEl.innerText = `${chCountStr}${template.stages?.length || 0} 個里程碑`;
    if (tagsContainer) {
        tagsContainer.innerHTML = (template.pacing_tags || []).map(t => `<span class="ref-pill" style="font-size:0.75rem;">${t}</span>`).join("");
    }

    renderPacingWaveCanvas(template);
    renderPacingStagesTable(template);
}

let pacingCanvasEventsInitialized = false;
let hoveredPacingChapter = null;

function initPacingCanvasEvents() {
    if (pacingCanvasEventsInitialized) return;
    const canvas = document.getElementById("ref-pacing-canvas");
    if (!canvas) return;
    pacingCanvasEventsInitialized = true;

    canvas.addEventListener("mousemove", (evt) => {
        if (!currentSelectedPacingTemplate) return;
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const mouseX = (evt.clientX - rect.left) * scaleX;

        const padLeft = 45;
        const padRight = 30;
        const plotW = canvas.width - padLeft - padRight;

        if (mouseX < padLeft || mouseX > canvas.width - padRight) {
            hoveredPacingChapter = null;
            const tooltip = document.getElementById("ref-pacing-tooltip");
            if (tooltip) tooltip.classList.add("hidden");
            renderPacingWaveCanvas(currentSelectedPacingTemplate);
            return;
        }

        const targetPct = ((mouseX - padLeft) / plotW) * 100.0;
        const curveData = currentSelectedPacingTemplate.chapter_curve_data || [];
        const stages = currentSelectedPacingTemplate.stages || [];

        if (curveData.length > 0) {
            let closest = curveData[0];
            let minDiff = Math.abs(closest.pct - targetPct);
            for (let i = 1; i < curveData.length; i++) {
                const diff = Math.abs(curveData[i].pct - targetPct);
                if (diff < minDiff) {
                    minDiff = diff;
                    closest = curveData[i];
                }
            }
            hoveredPacingChapter = closest;
        } else if (stages.length > 0) {
            let closest = stages[0];
            let minDiff = Math.abs(closest.offset_percent - targetPct);
            for (let i = 1; i < stages.length; i++) {
                const diff = Math.abs(stages[i].offset_percent - targetPct);
                if (diff < minDiff) {
                    minDiff = diff;
                    closest = stages[i];
                }
            }
            hoveredPacingChapter = {
                ch: closest.stage_index,
                pct: closest.offset_percent,
                t: closest.target_tension,
                p: closest.target_progress,
                d: closest.target_density,
                title: closest.dramatic_purpose || "",
            };
        } else {
            hoveredPacingChapter = null;
        }

        const tooltip = document.getElementById("ref-pacing-tooltip");
        if (tooltip && hoveredPacingChapter) {
            tooltip.classList.remove("hidden");
            const tipX = Math.min(rect.width - 200, Math.max(10, evt.clientX - rect.left + 14));
            const tipY = Math.max(8, (evt.clientY - rect.top) - 52);
            tooltip.style.left = `${tipX}px`;
            tooltip.style.top = `${tipY}px`;
            tooltip.style.pointerEvents = "none";
            tooltip.style.zIndex = "100";
            tooltip.style.background = "rgba(15, 23, 42, 0.95)";
            tooltip.style.border = "1px solid rgba(56, 189, 248, 0.5)";
            tooltip.style.borderRadius = "6px";
            tooltip.style.padding = "6px 10px";
            tooltip.style.boxShadow = "0 4px 12px rgba(0,0,0,0.5)";
            tooltip.style.fontSize = "0.78rem";
            tooltip.style.color = "#f8fafc";
            const chNum = hoveredPacingChapter.ch || "-";
            const chTitle = hoveredPacingChapter.title || "";
            const titleDisplay = chTitle ? `【${escapeHtml(chTitle)}】` : "";
            tooltip.innerHTML = `
                <div style="font-weight:600; color:#38bdf8; margin-bottom:3px;">
                    第 ${chNum} 章${titleDisplay} <span style="color:#94a3b8; font-weight:normal;">(${hoveredPacingChapter.pct.toFixed(1)}%)</span>
                </div>
                <div style="display:flex; gap:12px; font-size:0.75rem;">
                    <span style="color:#ef4444;">● 張力: ${hoveredPacingChapter.t.toFixed(1)}</span>
                    <span style="color:#3b82f6;">● 推進: ${hoveredPacingChapter.p.toFixed(1)}</span>
                    <span style="color:#10b981;">● 密度: ${hoveredPacingChapter.d.toFixed(1)}</span>
                </div>
            `;
        }
        renderPacingWaveCanvas(currentSelectedPacingTemplate);
    });

    canvas.addEventListener("mouseleave", () => {
        hoveredPacingChapter = null;
        const tooltip = document.getElementById("ref-pacing-tooltip");
        if (tooltip) tooltip.classList.add("hidden");
        if (currentSelectedPacingTemplate) {
            renderPacingWaveCanvas(currentSelectedPacingTemplate);
        }
    });
}

function renderPacingWaveCanvas(template) {
    const canvas = document.getElementById("ref-pacing-canvas");
    if (!canvas || !template) return;
    initPacingCanvasEvents();
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const padLeft = 45;
    const padRight = 30;
    const padTop = 25;
    const padBottom = 35;
    const plotW = W - padLeft - padRight;
    const plotH = H - padTop - padBottom;

    // Draw grid lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
    ctx.lineWidth = 1;
    for (let v = 1; v <= 5; v++) {
        const y = padTop + plotH - ((v - 1) / 4) * plotH;
        ctx.beginPath();
        ctx.moveTo(padLeft, y);
        ctx.lineTo(W - padRight, y);
        ctx.stroke();

        ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
        ctx.font = "10px -apple-system, sans-serif";
        ctx.textAlign = "right";
        ctx.fillText(`Lv.${v}`, padLeft - 8, y + 3);
    }

    // X Axis Progress Labels
    ctx.textAlign = "center";
    for (let pct = 0; pct <= 100; pct += 25) {
        const x = padLeft + (pct / 100.0) * plotW;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
        ctx.beginPath();
        ctx.moveTo(x, padTop);
        ctx.lineTo(x, padTop + plotH);
        ctx.stroke();

        ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
        ctx.font = "11px -apple-system, sans-serif";
        ctx.fillText(`${pct}%`, x, padTop + plotH + 18);
    }

    const curveData = template.chapter_curve_data || [];
    const hasCurve = curveData.length > 0;

    // Helper to draw a curve
    function drawMetricCurve(key, color, fillGrad) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.2;
        ctx.beginPath();

        if (hasCurve) {
            curveData.forEach((pt, idx) => {
                const x = padLeft + (pt.pct / 100.0) * plotW;
                const val = Math.max(1.0, Math.min(5.0, pt[key]));
                const y = padTop + plotH - ((val - 1) / 4) * plotH;
                if (idx === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();

            if (fillGrad) {
                ctx.save();
                ctx.lineTo(padLeft + (curveData[curveData.length - 1].pct / 100.0) * plotW, padTop + plotH);
                ctx.lineTo(padLeft + (curveData[0].pct / 100.0) * plotW, padTop + plotH);
                ctx.closePath();
                const grad = ctx.createLinearGradient(0, padTop, 0, padTop + plotH);
                grad.addColorStop(0, fillGrad);
                grad.addColorStop(1, "rgba(0, 0, 0, 0)");
                ctx.fillStyle = grad;
                ctx.fill();
                ctx.restore();
            }
        } else if (template.stages && template.stages.length > 0) {
            const stageKey = key === "t" ? "target_tension" : (key === "p" ? "target_progress" : "target_density");
            template.stages.forEach((st, idx) => {
                const x = padLeft + (st.offset_percent / 100.0) * plotW;
                const val = Math.max(1.0, Math.min(5.0, st[stageKey]));
                const y = padTop + plotH - ((val - 1) / 4) * plotH;
                if (idx === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
        }
    }

    // 1. Progress Curve (Blue)
    drawMetricCurve("p", "#3b82f6", null);
    // 2. Density Curve (Green)
    drawMetricCurve("d", "#10b981", null);
    // 3. Tension Curve (Red with subtle fill)
    drawMetricCurve("t", "#ef4444", "rgba(239, 68, 68, 0.2)");

    // Draw Milestone Nodes from template.stages
    if (template.stages && template.stages.length > 0) {
        template.stages.forEach(st => {
            const x = padLeft + (st.offset_percent / 100.0) * plotW;
            const y = padTop + plotH - ((Math.max(1.0, Math.min(5.0, st.target_tension)) - 1) / 4) * plotH;

            ctx.fillStyle = "rgba(239, 68, 68, 0.35)";
            ctx.beginPath();
            ctx.arc(x, y, 7, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = "#ef4444";
            ctx.beginPath();
            ctx.arc(x, y, 4.5, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = "#ffffff";
            ctx.beginPath();
            ctx.arc(x, y, 2, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = "rgba(226, 232, 240, 0.9)";
            ctx.font = "9px -apple-system, sans-serif";
            ctx.textAlign = "center";
            const phaseLabel = st.pacing_phase === "BUILDUP" ? "蓄勢" :
                               st.pacing_phase === "COMPLICATION" ? "衝突" :
                               st.pacing_phase === "CLIMAX" ? "高潮" : "收束";
            ctx.fillText(phaseLabel, x, y - 9);
        });
    }

    // Interactive Hover Probe Indicator
    if (hoveredPacingChapter) {
        const hPct = hoveredPacingChapter.pct;
        const hX = padLeft + (hPct / 100.0) * plotW;

        ctx.save();
        ctx.strokeStyle = "rgba(56, 189, 248, 0.85)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(hX, padTop);
        ctx.lineTo(hX, padTop + plotH);
        ctx.stroke();
        ctx.restore();

        const yT = padTop + plotH - ((Math.max(1, Math.min(5, hoveredPacingChapter.t)) - 1) / 4) * plotH;
        ctx.fillStyle = "#ef4444";
        ctx.beginPath();
        ctx.arc(hX, yT, 5.5, 0, Math.PI * 2);
        ctx.fill();

        const yP = padTop + plotH - ((Math.max(1, Math.min(5, hoveredPacingChapter.p)) - 1) / 4) * plotH;
        ctx.fillStyle = "#3b82f6";
        ctx.beginPath();
        ctx.arc(hX, yP, 5.5, 0, Math.PI * 2);
        ctx.fill();

        const yD = padTop + plotH - ((Math.max(1, Math.min(5, hoveredPacingChapter.d)) - 1) / 4) * plotH;
        ctx.fillStyle = "#10b981";
        ctx.beginPath();
        ctx.arc(hX, yD, 5.5, 0, Math.PI * 2);
        ctx.fill();
    }
}

function renderPacingStagesTable(template) {
    const box = document.getElementById("ref-pacing-stages-table-box");
    if (!box || !template || !template.stages) return;
    const stages = template.stages;

    let html = `<table class="snapshot-table" style="font-size:0.82rem;">
        <thead>
            <tr>
                <th style="width:50px;">階段</th>
                <th style="width:70px;">進度</th>
                <th style="width:110px;">相位標籤</th>
                <th style="width:70px;">張力</th>
                <th style="width:70px;">推進</th>
                <th style="width:70px;">密度</th>
                <th>抽象戲劇任務描述（點擊可微調）</th>
            </tr>
        </thead>
        <tbody>`;

    stages.forEach((st, idx) => {
        html += `<tr>
            <td style="font-weight:600; text-align:center;">#${st.stage_index}</td>
            <td style="text-align:center;">${st.offset_percent.toFixed(0)}%</td>
            <td>
                <select class="ref-filter-select" style="padding:2px 6px; font-size:0.75rem;" onchange="updateStageField(${idx}, 'pacing_phase', this.value)">
                    <option value="BUILDUP" ${st.pacing_phase==='BUILDUP'?'selected':''}>蓄勢鋪陳</option>
                    <option value="COMPLICATION" ${st.pacing_phase==='COMPLICATION'?'selected':''}>衝突拉扯</option>
                    <option value="CLIMAX" ${st.pacing_phase==='CLIMAX'?'selected':''}>高潮爆發</option>
                    <option value="RESOLUTION" ${st.pacing_phase==='RESOLUTION'?'selected':''}>餘波收束</option>
                </select>
            </td>
            <td>
                <input type="number" min="1" max="5" value="${st.target_tension}" style="width:48px; padding:2px 4px; background:rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.4); color:#fff; border-radius:4px; text-align:center;" onchange="updateStageField(${idx}, 'target_tension', parseInt(this.value,10))" />
            </td>
            <td>
                <input type="number" min="1" max="5" value="${st.target_progress}" style="width:48px; padding:2px 4px; background:rgba(59,130,246,0.15); border:1px solid rgba(59,130,246,0.4); color:#fff; border-radius:4px; text-align:center;" onchange="updateStageField(${idx}, 'target_progress', parseInt(this.value,10))" />
            </td>
            <td>
                <input type="number" min="1" max="5" value="${st.target_density}" style="width:48px; padding:2px 4px; background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.4); color:#fff; border-radius:4px; text-align:center;" onchange="updateStageField(${idx}, 'target_density', parseInt(this.value,10))" />
            </td>
            <td>
                <input type="text" value="${st.dramatic_purpose}" style="width:100%; padding:2px 6px; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.15); color:#e2e8f0; border-radius:4px; font-size:0.8rem;" onchange="updateStageField(${idx}, 'dramatic_purpose', this.value)" />
            </td>
        </tr>`;
    });

    html += `</tbody></table>`;
    box.innerHTML = html;
}

function updateStageField(stageIdx, field, val) {
    if (!currentSelectedPacingTemplate || !currentSelectedPacingTemplate.stages[stageIdx]) return;
    currentSelectedPacingTemplate.stages[stageIdx][field] = val;
    renderPacingWaveCanvas(currentSelectedPacingTemplate);
}

async function savePacingStageOverrides() {
    if (!currentSelectedPacingTemplate) {
        showToast("尚未選擇任何範本。");
        return;
    }
    try {
        const resp = await fetch(`/api/reference/pacing_templates/${currentSelectedPacingTemplate.id}/stages`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ stages: currentSelectedPacingTemplate.stages }),
        });
        const data = await resp.json();
        if (resp.ok) {
            showToast("情節規劃波浪里程碑覆寫已成功儲存！");
            await loadPacingTemplates();
        } else {
            showToast(data.detail || "儲存覆寫失敗。");
        }
    } catch (err) {
        showToast("網路異常，無法儲存覆寫。");
    }
}

async function deleteCurrentPacingTemplate() {
    if (!currentSelectedPacingTemplate) return;
    if (!confirm(`確定要刪除範本《${currentSelectedPacingTemplate.name}》嗎？`)) return;
    try {
        const resp = await fetch(`/api/reference/pacing_templates/${currentSelectedPacingTemplate.id}`, {
            method: "DELETE",
        });
        if (resp.ok) {
            showToast("範本已成功刪除。");
            currentSelectedPacingTemplate = null;
            await loadPacingTemplates();
        } else {
            const data = await resp.json();
            showToast(data.detail || "刪除失敗。");
        }
    } catch (err) {
        showToast("網路連線失敗。");
    }
}

function openBlendPacingModal() {
    if (currentPacingTemplates.length < 2) {
        showToast("書庫中至少需要 2 套情節範本才能進行混血合成！");
        return;
    }
    const selA = document.getElementById("blend-select-a");
    const selB = document.getElementById("blend-select-b");
    if (!selA || !selB) return;

    const opts = currentPacingTemplates.map(t => `<option value="${t.id}">${t.name}</option>`).join("");
    selA.innerHTML = opts;
    selB.innerHTML = opts;
    if (currentPacingTemplates.length > 1) {
        selB.selectedIndex = 1;
    }
    document.getElementById("blend-weight-slider").value = "0.5";
    updateBlendSliderLabel();
    document.getElementById("modal-blend-pacing")?.classList.remove("hidden");
}

function updateBlendSliderLabel() {
    const slider = document.getElementById("blend-weight-slider");
    const label = document.getElementById("blend-ratio-label");
    if (!slider || !label) return;
    const valA = Math.round(parseFloat(slider.value) * 100);
    const valB = 100 - valA;
    label.innerText = `母本 A (${valA}%) : 母本 B (${valB}%)`;
}

async function confirmBlendPacing() {
    const selA = document.getElementById("blend-select-a");
    const selB = document.getElementById("blend-select-b");
    const slider = document.getElementById("blend-weight-slider");
    const nameInput = document.getElementById("blend-new-name");

    const idA = parseInt(selA.value, 10);
    const idB = parseInt(selB.value, 10);
    const weightA = parseFloat(slider.value);
    const newName = nameInput.value.trim();

    if (idA === idB) {
        showToast("母本 A 與母本 B 不可為同一範本！");
        return;
    }

    try {
        const resp = await fetch("/api/reference/pacing_templates/blend", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                template_id_a: idA,
                template_id_b: idB,
                weight_a: weightA,
                new_name: newName,
            }),
        });
        const data = await resp.json();
        if (resp.ok) {
            showToast("🧬 混血範本合成成功！已自動存入情節規劃庫。");
            closeModal("modal-blend-pacing");
            if (nameInput) nameInput.value = "";
            currentSelectedPacingTemplate = data.template;
            await loadPacingTemplates();
        } else {
            showToast(data.detail || "混血合成失敗。");
        }
    } catch (err) {
        showToast("網路異常，無法完成混血合成。");
    }
}


async function pauseReferenceExtraction() {
    const btnPause = document.getElementById("btn-pause-extraction");
    const btnResume = document.getElementById("btn-resume-extraction");
    if (btnPause) btnPause.disabled = true;
    try {
        const resp = await fetch("/api/reference/pause", { method: "POST" });
        const data = await resp.json();
        if (resp.ok) {
            showToast("⏸️ 已發送暫停訊號，當前切片處理完畢後將安全暫停。");
            if (btnPause) btnPause.classList.add("hidden");
            if (btnResume) {
                btnResume.classList.remove("hidden");
                btnResume.disabled = false;
            }
            const speedEl = document.getElementById("ref-prog-speed");
            if (speedEl) speedEl.textContent = "【已暫停】點擊「繼續萃取」恢復處理...";
        } else {
            showToast(data.detail || "暫停萃取失敗。");
        }
    } catch (err) {
        showToast("網路異常，無法發送暫停訊號。");
    } finally {
        if (btnPause) btnPause.disabled = false;
    }
}

async function resumeReferenceExtraction() {
    const btnPause = document.getElementById("btn-pause-extraction");
    const btnResume = document.getElementById("btn-resume-extraction");
    if (btnResume) btnResume.disabled = true;
    try {
        const resp = await fetch("/api/reference/resume", { method: "POST" });
        const data = await resp.json();
        if (resp.ok) {
            showToast("▶️ 已恢復書籍萃取工作。");
            if (btnResume) btnResume.classList.add("hidden");
            if (btnPause) {
                btnPause.classList.remove("hidden");
                btnPause.disabled = false;
            }
            const speedEl = document.getElementById("ref-prog-speed");
            if (speedEl) speedEl.textContent = "後台神經網路寫作技法分析中...";
        } else {
            showToast(data.detail || "恢復萃取失敗。");
        }
    } catch (err) {
        showToast("網路異常，無法發送恢復訊號。");
    } finally {
        if (btnResume) btnResume.disabled = false;
    }
}

async function cancelReferenceExtraction() {
    const btnCancel = document.getElementById("btn-cancel-extraction");
    const btnPause = document.getElementById("btn-pause-extraction");
    const btnResume = document.getElementById("btn-resume-extraction");
    if (btnCancel) btnCancel.disabled = true;
    if (btnPause) btnPause.disabled = true;
    if (btnResume) btnResume.disabled = true;
    try {
        const resp = await fetch("/api/reference/cancel", { method: "POST" });
        const data = await resp.json();
        if (resp.ok) {
            showToast("⏹️ 已成功取消當前萃取作業。");
            const card = document.getElementById("ref-active-progress-card");
            if (card) card.classList.add("hidden");
            const pacingCard = document.getElementById("ref-pacing-progress-card");
            if (pacingCard) pacingCard.classList.add("hidden");
            if (pacingHideTimer) {
                clearTimeout(pacingHideTimer);
                pacingHideTimer = null;
            }
            wasPacingActive = false;
            pacingCompletedHandled = false;
            const btnConfirm = document.getElementById("btn-confirm-ingest");
            if (btnConfirm) btnConfirm.disabled = false;
            await loadReferenceBooks();
            await refreshWarroomDashboard();
        } else {
            showToast(data.detail || "取消萃取失敗。");
        }
    } catch (err) {
        showToast("網路異常，無法發送取消訊號。");
    } finally {
        if (btnCancel) btnCancel.disabled = false;
        if (btnPause) btnPause.disabled = false;
        if (btnResume) btnResume.disabled = false;
    }
}

function updatePacingProgressMonitor(st) {
    const card = document.getElementById("ref-pacing-progress-card");
    if (!card) return;

    if (!st) {
        card.classList.add("hidden");
        wasPacingActive = false;
        pacingCompletedHandled = false;
        if (pacingHideTimer) {
            clearTimeout(pacingHideTimer);
            pacingHideTimer = null;
        }
        return;
    }

    // Active extraction state
    if (st.active) {
        wasPacingActive = true;
        pacingCompletedHandled = false;
        if (pacingHideTimer) {
            clearTimeout(pacingHideTimer);
            pacingHideTimer = null;
        }
        card.classList.remove("hidden");

        const fileEl = document.getElementById("ref-pacing-prog-file");
        const batchSuffix = st.batch_total > 1 ? `（第 ${st.batch_index || 1}/${st.batch_total} 本）` : "";
        if (fileEl && st.novel_title) {
            fileEl.textContent = `情節規劃分析：《${st.novel_title}》${batchSuffix}`;
        }
        const phaseEl = document.getElementById("ref-pacing-prog-phase");
        if (phaseEl) {
            if (st.phase === "TOC_PARSING") phaseEl.textContent = "（解析目錄分界）";
            else if (st.phase === "FACT_EXTRACTION" || st.phase === "CHAPTER_ANALYSIS") phaseEl.textContent = "（章節事實與張力解構）";
            else if (st.phase === "VOLUME_CONSOLIDATION") phaseEl.textContent = "（分卷劇情弧收斂）";
            else if (st.phase === "MAJOR_ARCS_CONSOLIDATION") phaseEl.textContent = "（全書大幕樹狀收斂）";
            else if (st.phase === "BLUEPRINT_SYNTHESIS") phaseEl.textContent = "（合成永久事實藍圖）";
            else if (st.phase === "NORMALIZATION") phaseEl.textContent = "（0%~100% 規一化編排）";
            else phaseEl.textContent = "（分析中）";
        }
        const pct = typeof st.percent === "number" ? Math.min(100, Math.max(0, st.percent)).toFixed(1) : "0.0";
        const pctEl = document.getElementById("ref-pacing-prog-pct");
        if (pctEl) pctEl.textContent = `${pct}%`;
        const barEl = document.getElementById("ref-pacing-prog-bar");
        if (barEl) barEl.style.width = `${pct}%`;
        const detailEl = document.getElementById("ref-pacing-prog-detail");
        if (detailEl && st.detail) detailEl.textContent = st.detail;
        return;
    }

    // Completion state: only transition and display grace period if it was actively extracting and hasn't been handled yet
    if (st.phase === "PACING_COMPLETE") {
        if (pacingCompletedHandled) {
            // Already handled completion and scheduled grace period; ignore repeated poll ticks
            return;
        }

        // Only transition to complete if we witnessed it active or card is currently open
        if (wasPacingActive || !card.classList.contains("hidden")) {
            pacingCompletedHandled = true;
            wasPacingActive = false;
            card.classList.remove("hidden");

            const fileEl = document.getElementById("ref-pacing-prog-file");
            if (fileEl && st.novel_title) fileEl.textContent = `情節規劃分析：《${st.novel_title}》`;
            const phaseEl = document.getElementById("ref-pacing-prog-phase");
            if (phaseEl) phaseEl.textContent = "（分析完成）";
            const pctEl = document.getElementById("ref-pacing-prog-pct");
            if (pctEl) pctEl.textContent = "100.0%";
            const barEl = document.getElementById("ref-pacing-prog-bar");
            if (barEl) barEl.style.width = "100%";
            const detailEl = document.getElementById("ref-pacing-prog-detail");
            if (detailEl && st.detail) detailEl.textContent = st.detail;

            loadPacingTemplates();
            loadReferenceBooks();

            if (pacingHideTimer) clearTimeout(pacingHideTimer);
            pacingHideTimer = setTimeout(() => {
                card.classList.add("hidden");
                pacingHideTimer = null;
            }, 6000);
            return;
        }

        // Stale PACING_COMPLETE on modal open when not active: keep hidden
        card.classList.add("hidden");
        return;
    }

    // Any other inactive phase (IDLE, CANCELLED, ERROR, etc.)
    wasPacingActive = false;
    pacingCompletedHandled = false;
    if (pacingHideTimer) {
        clearTimeout(pacingHideTimer);
        pacingHideTimer = null;
    }
    card.classList.add("hidden");
}
