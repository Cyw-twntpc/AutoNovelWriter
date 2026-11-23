# 程式碼審查問題追蹤（2026-10-29）

深度審查後的完整缺陷清單。狀態欄：`待修` / `修復中` / `已修` / `已驗證`。

驗證標記：●＝逐行回讀原始碼確認、△＝子代理發現但未逐行回讀（修復前需再確認）。

---

## P0 — 極高風險：不可逆資料損失或核心一致性失效

| ID | 位置 | 問題 | 狀態 |
|---|---|---|---|
| H1 | `wizard/blueprint_builder.py:151,185-186`、`web/routes_wizard.py:271-278` | 重跑藍圖會 `DELETE FROM chapters/volumes/characters`，但後端不看 `pipeline_state`，且 `chapter_metrics`/`character_history`/`timeline`/`entity_relations`/`foreshadowing` 未清 → 正文遺失 + 孤兒列；`DELETE FROM characters` 在 FK 開啟下必 IntegrityError ● | 已驗證 |
| H2 | `engine/chapter_pipeline.py:138-177`、`:72-98` | Gate A 三次重試皆失敗仍回傳 `last_gate_a`，`execute_chapter` 不檢查 `passed` 直接 commit 硬衝突草稿；且 extraction 取自 Gate A 之前的 draft，與最終正文脫節 ● | 已驗證 |
| H3 | `db/project_repo.py:192-231,424-429,493-501` | 重寫章節不清理舊 `timeline`/`foreshadowing`/`character_history` → `elapsed_days` 每重寫一次膨脹、伏筆重複 PENDING、歷史重複 ● | 已驗證 |
| H4 | `engine/daemon.py:243-280` | 切換專案時 `cancel()` 後立刻重綁所有 `.repo`，被取消任務的同步 commit 會寫進新專案；舊專案 staging 在 rollback 之後才寫入 ● | 已驗證 |
| H5 | `engine/daemon.py:197-223,232-238` | 「卸載專案」不寫 PAUSED、不 rollback → 下次啟動 `startup_recovery(auto_resume=True)` 自動開始寫作 ● | 已驗證 |
| H6 | `engine/daemon.py:444-470` | 連載迴圈例外後 `_pause_requested` 仍為 False → 不寫 PAUSED、不釋放 anti-sleep 鎖；UI 顯示連載中但實際已死 ● | 已驗證 |
| H7 | `db/snapshot_manager.py:109-117`、`web/routes_warroom.py:319-323` | 還原直接覆寫線上 DB（非原子、無備份）；守門只看 `current_writing_chapter`，章節空窗可還原且 worker 繼續寫 ● | 已驗證 |

## P1 — 高風險：品質閘門失效、契約矛盾

| ID | 位置 | 問題 | 狀態 |
|---|---|---|---|
| H8 | `schemas/llm_contracts.py:155`、`db/project_repo.py:439-448` | 伏筆 RESOLVE 需要 DB 主鍵，LLM 不可能提供 → 幾乎永不生效；模型幻覺 id 會誤標無關伏筆。連鎖：foreshadowing 只增不減、`should_conclude_novel` 永不可能 True ● | 已修 |
| H9 | `engine/volume_planner.py:125-134,150-162` | `EXPIRED` 從未寫入 → 回收率恆 1.0；`elapsed_days` 由 `MAX+duration` 產生且驗證按 `id` 排序 → 單調性恆真 ● | 已修 |
| H10 | `engine/daemon.py:412-420` | Volume 驗收失敗（重寫後）仍標 COMPLETED 並繼續下一卷 ● | 已修 |
| H11 | `schemas/llm_contracts.py:142` vs `engine/chapter_pipeline.py:270` | `new_status: Literal["ALIVE","DEAD"]` 與 prompt「2～6 字狀態術語」直接矛盾 ● | 已修 |
| M14 | `engine/volume_planner.py:250-251` | 超出大綱卷數後永遠重用最後一個 stage gate，境界上限固定 ● | 已修 |

## P2 — 高風險：參考書 pipeline 與 LLM 基礎設施

| ID | 位置 | 問題 | 狀態 |
|---|---|---|---|
| H12 | `reference/chunk_extractor.py:649-651` | `asyncio.gather` 無 `return_exceptions` → 單一 chunk 失敗丟棄整批 8 個成功結果 ● | 已驗證 |
| H13 | `reference/chunk_extractor.py:350-352` vs `626-637` | `entry_count>0` 直接 return 0，使 chunk 級續傳成為死碼 → 殘缺書永久跳過 ● | 已驗證 |
| H14 | `reference/file_parser.py:113-116` | epub 用 `sorted(html_names)` 忽略 spine → chapter10 在 chapter2 前；`ET.fromstring` 遇具名實體拋錯判「損毀」 ● | 已驗證 |
| H15 | `llm/llama_manager.py:184-211,242` | `taskkill /F /T` 任何佔用 8080 的 PID，不驗證映像名稱 ● | 已驗證 |
| H16 | `llm/llm_client.py:53-59,135,212-214` | 單次請求失敗即 `aclose()` 共用 client，打死其他並行串流 ● | 已驗證 |
| H19 | `llm/llama_manager.py:152-163`、`llm/llm_client.py:110,137` | async 路徑中 `time.sleep` 最多 30 秒、`restart_server()` 同步 → 事件迴圈凍結 ● | 已驗證 |

## P3 — 前端、Web 與次要缺陷

| ID | 位置 | 問題 | 狀態 |
|---|---|---|---|
| H17 | `web/static/js/projects.js:129,169`、`state.js:89-96` | `onclick` 內插書名，`&#39;` 被瀏覽器二次解碼 → 可跳出 JS 字串；含撇號書名使按鈕永久故障 ● | 已驗證 |
| H18 | `web/static/js/warroom.js:348-356` | anomaly 日誌 `innerHTML` 直出，內容含例外訊息與模型輸出 ● | 已驗證 |
| H20 | `web/routes_warroom.py:242-248`、`engine/daemon.py:361-371` | `/warroom/step` 與背景連載無互斥 → 相鄰章節並行生成，context 與 Gate A 基於舊狀態 ● | 已驗證 |
| M1 | `engine/daemon.py:413-419` | volume 驗收重寫無 try/except，失敗留下 WRITING 章節 ● | 已驗證 |
| M2 | `engine/pacing_monitor.py:153-162` | overdue 伏筆每章重複寫 anomaly_logs ● | 已驗證 |
| M3 | `db/project_repo.py:156-170` | `_find_character_by_name_or_alias` 全表掃描 + 無 `ORDER BY` ● | 已驗證 |
| M4 | `db/project_repo.py:30-33` | 每次建 repo 都跑完整 DDL + `BEGIN IMMEDIATE` ● | 已驗證 |
| M5 | `db/reference_repo.py:181-213` | 全庫矩陣快取無上限；維度不一致 `np.vstack` 直接 ValueError △ | 已驗證 |
| M6 | `db/reference_repo.py:571-700` | 無 tag 時 `s_tag = s_desc` 結構性偏袒 △（第二輪改為跳過＋維護自動刪除，見 R4） | 已驗證 |
| M7 | `db/reference_repo.py:999-1006` | `GROUP BY tag_name` 取任意 `tag_embedding` 後用於 prune △ | 已驗證 |
| M8 | `reference/file_parser.py:54-63` | Big5 偵測順序在 gb18030 之前，GBK 檔可能變亂碼且無告警 △ | 已驗證 |
| M9 | `reference/macro_extractor.py:74-127` | TOC 正則把列舉行當章節標頭；開頭 <600 字前言被丟棄 △ | 已驗證 |
| M10 | `reference/macro_extractor.py:896` | fallback 硬編碼 `world_rules_and_hierarchy=[]`，無 log 標記降級 ● | 已驗證 |
| M11 | `reference/chunk_extractor.py:285-317` | 暫停等待迴圈在 `llm_client is None` 時無限卡住 ● | 已驗證 |
| M12 | `reference/chunk_extractor.py:461-472` | checkpoint 每批全量重寫、非原子 △ | 已驗證 |
| M13 | `reference/chunk_extractor.py:588-649` | 同時持有全文 3-4 份副本；同檔重複解析 2-3 次 △ | 已驗證 |
| M15 | `db/project_manager.py:146-151` | `project_id` 未驗證就 `resolve()` + `mkdir` → 可跳出工作區 ● | 已驗證 |
| M16 | `db/project_manager.py:354-358` | `rmtree(ignore_errors=True)` 後仍移除 metadata → 孤兒專案 △ | 已驗證 |
| M17 | `db/snapshot_manager.py:63-79` | 快照保留／列表用檔名排序而非時間 △ | 已驗證 |
| M18 | `wizard/blueprint_builder.py:130-145` | 藍圖回寫覆蓋使用者輸入的書名 ● | 已驗證 |
| M19 | `wizard/blueprint_builder.py:322-331` | `factions`/`locations` 只 INSERT OR IGNORE，刪除不持久化 △ | 已驗證 |
| M20 | `web/routes_wizard.py:277` | 藍圖解鎖門檻僅為回應欄位，後端不比對 ● | 已驗證 |
| M21 | `llm/llm_client.py:194-198` | `ValidationError`/`JSONDecodeError` 未捕捉 → 截斷即 500 △ | 已驗證 |
| M22 | `engine/hardware_guard.py:172-189` | 降溫逾時仍回報「已恢復」並回傳 True △ | 已驗證 |
| M23 | `export/exporter.py:55-69` | scope 參數缺失時靜默退回全書 △ | 已驗證 |
| M24 | `export/exporter.py:31-33` | 批次目錄秒級精度 + 全域 export_dir → 同秒覆寫 △ | 已驗證 |
| M25 | `web/app.py:92-98` | 活動白名單漏掉 `/api/reference/progress` 等高頻輪詢 → 閒置卸載永不觸發 ● | 已驗證 |
| M26 | `web/routes_wizard.py:92-95` | GET `/wizard/genres` 觸發 LLM 生成（有副作用的 GET） ● | 已驗證 |
| M27 | `wizard/qa_engine.py:56-60`、`engine/daemon.py:318-320` | 直接改寫 `_manually_unloaded`，抵銷使用者手動卸載 △ | 已驗證 |
| M28 | `engine/context_router.py:190,209` | `json.loads` 未防護；`[:12]` 依 id 截斷丟棄新角色 ● | 已驗證 |
| M29 | `web/routes_warroom.py:1022-1036` | SSE 訂閱靠 generator finally 釋放，15 秒才偵測斷線；前端重連 timer 未去重 △ | 已驗證 |

## 低風險

| ID | 位置 | 問題 | 狀態 |
|---|---|---|---|
| L1 | `config.py:78-82` | 環境變數解析無容錯，打錯字導致 app 起不來 △ | 已驗證 |
| L2 | `db/project_schema.py:192` | `executescript()` 先隱式 COMMIT，初始化非原子 △ | 已驗證 |
| L3 | `export/exporter.py:221-230` | EPUB identifier 用章數推導會撞號；缺 `dcterms:modified` △ | 已驗證 |
| L4 | `web/static/js/review_workbench.js:327-329` | 不看 HTTP 狀態就跳「已儲存」，失敗時清空工作台 △ | 已驗證 |
| L5 | `wizard/qa_engine.py:247-252` | `step_auto_turn` 先落盤再生成，失敗重按會重複 append △ | 已驗證 |
| L6 | `db/project_manager.py:253` | 排序註解與實際行為相反 △ | 已驗證 |
| L7 | `db/reference_repo.py:82` | `hash(cluster_name)` 取色受 `PYTHONHASHSEED` 影響 △ | 已驗證 |
| L8 | `web/static/js/main.js:101`、`warroom.js:23-25` | 輪詢無 try/catch、不等待前次完成 → 未處理 rejection 與請求堆疊 △ | 已驗證 |
| L9 | `reference/chunk_extractor.py:717` | 取消時仍回傳 `total_chunks` → 進度虛高 △ | 已驗證 |
| L10 | `reference/macro_extractor.py:349-351` | pacing 曲線用列舉位置而非 `chapter_index`，缺漏時章號錯位 △ | 已驗證 |

---

## 第二輪（收尾後複核）

第一輪 59 項結案後回讀程式碼，另外發現的同類遺漏與設計取捨，已逐項處理。

| ID | 位置 | 問題 | 處理 |
|---|---|---|---|
| R1 | `web/static/js/reference.js` | H17 同類漏洞：書名／標籤名以 `onclick="fn('...')"` 內插，`escapeHtml` 產生的 `&#39;` 會在屬性解析時還原成 `'`，可跳出字串（可注入、按鈕永久故障）。書名來自上傳檔名，屬外部輸入 | 4 處改為 `data-ref-action` + `data-*`，document 層單一委派處理 `jump-book`／`delete-book`／`filter-hot-tag`／`filter-cluster` |
| R2 | `web/routes_warroom.py`、`db/reference_repo.py` | 刪書只刪 DB 列，`data/ref_uploads/` 只增不減且無清理入口 | 刪除後回收同名暫存檔（僅當無其他 source 同名），回應新增 `removed_file`；新增 `get_source()`、`count_sources_by_file_name()` |
| R3 | `pyproject.toml` | `onnxruntime`、`tokenizers` 未列依賴，乾淨環境無法執行測試（AGENTS.md 已註明） | 加入 `dependencies`：`onnxruntime>=1.17.0`、`tokenizers>=0.15.0` |
| R4 | `db/reference_repo.py`、`reference/retriever.py`、`engine/daemon.py` | M6 的「無 tag 得利」只能靠特殊計分迴避；書本題材完全未參與檢索 | ①移除手動刪標籤（route／repo method／UI ✕ 按鈕）②無 tag 列改為 `continue` 跳過，不再有「0.4×desc」的替代語意 ③`prune_marginalized_tags`／`apply_tag_rules` 在同一交易內呼叫 `_delete_tagless_entries()` 自動清除 ④計分改為 `0.4×desc + 0.45×tag + 0.15×genre`，題材由 `ref_entries.source_id → ref_book_blueprints.genre_primary/genre_sub` 取得（每來源只嵌入一次並快取） |

**A2 撤回（誤報）**：原判定「軟刪除＋`name UNIQUE` 會讓同名重新加入停在 ARCHIVED」不成立——`blueprint_builder.py:377-380`／`:391-394` 在 `INSERT OR IGNORE` 之後已執行 `UPDATE ... SET status='ACTIVE'`；而完整藍圖重建會先經 `_reset_narrative_state` 硬刪 factions/locations，不會留下幽靈列。程式未改動。

**行為變更（需知悉）**：

- `DELETE /api/reference/entries/{entry_id}/tags/{tag_name}` 已移除（回 404/405）；只保留「刪除整個條目」。
- 分數構成由 `desc 0.4 / tag 0.6` 改為 `desc 0.4 / tag 0.45 / genre 0.15`。查詢端未提供題材（測試、無專案、或 `genre_provider` 回空）時題材項為 0，滿分為 0.85。
- 無標籤條目（僅可能來自舊資料或外部寫入）不再參與帶標籤查詢，且會被標籤維護流程刪除；抽取流程本身仍保證每筆 ≥3 個標籤。
- 候選切片下界 `(threshold - 0.6)/0.4` 不變，因 `0.45 + 0.15 = 0.6`。

---

## 已知限制（本次不處理，需另案）

- 角色狀態（`characters.status`/`rank_level`）是「當前值」而非事件溯源，重寫舊章無法回滾到該章當時的狀態。
- `is_model_in_use()` 依賴 `_worker_task`，cancel 後到真正結束之間仍有空窗（已以 await 緩解，未根本解決）。
- 無 dependency lockfile（未鎖定版本／hash）；`onnxruntime`、`tokenizers` 已列入 `pyproject.toml`，但重建環境仍可能拿到不同版本。
- 參考書庫矩陣快取上限 `MAX_MATRIX_ENTRIES = 20_000`：超過後較舊條目不再參與檢索（僅記 warning），`tag_freq` 也只在保留範圍內統計。
- `get_library_stats()` 仍以 `":" in tag_name` 推估題材標籤數量，僅影響統計面板；檢索用的題材已改由 `ref_book_blueprints` 取得。
- 目前不支援多實例同時執行（未加 `data/.lock`）；兩個 app 會共用同一個 llama-server 與 SQLite。
