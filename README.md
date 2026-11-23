# AutoNovelWriter

An autonomous, local-first long-form novel generation system built around a **Dual-SQLite Consistency Engine**, a strict **16k Token Context Router**, **Two-Stage Quality Gates**, and local **`llama.cpp` + CPU ONNX** inference.

Designed to sustain multi-volume, million-word-scale Traditional Chinese (`zh-TW` / `zh-Hant`) serialization on consumer hardware (optimized for **8 GB VRAM** GPUs), AutoNovelWriter decouples background writing from browser sessions and continuously enforces worldbuilding rules, timeline causality, character state continuity, and literary craft.

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Workflow](#usage-workflow)
- [CLI Utilities](#cli-utilities)
- [Testing](#testing)
- [API Overview](#api-overview)
- [License](#license)
- [Acknowledgments & Third-Party Notices](#acknowledgments--third-party-notices)

---

## Key Features

### 1. Decoupled Background Serialization Daemon
- **`NovelWritingDaemon`** (`src/autonovel/engine/daemon.py`) executes chapter generation independently of active browser tabs.
- Real-time token streaming and lifecycle events over Server-Sent Events (`/api/warroom/stream`).
- **Graceful Pause & Crash Recovery**: Pauses cleanly at atomic chapter boundaries; on startup (`startup_recovery(auto_resume=True)`), incomplete `STAGING` chapters are rolled back automatically and `RUNNING` projects can resume safely.
- **Idle VRAM Watchdog**: Automatically unloads `llama-server` after configurable frontend inactivity (`AUTONOVEL_MODEL_IDLE_TIMEOUT_SEC`, default `600s`) while ignoring passive dashboard polling endpoints.

### 2. Dual-SQLite State & Reference Architecture
- **Per-Project Narrative Database** (`data/projects/<project_id>/novel_project.db`): Tracks `volumes`, `chapters`, `characters`, `character_history`, `factions`, `locations`, `entity_relations`, `timeline` (`elapsed_days`), bidirectional `foreshadowing` (`PLANT` / `RESOLVE` / `EXPIRED`), `hierarchy_levels`, `world_rules`, `chapter_metrics`, and `anomaly_logs`.
- **Global Reference Library** (`data/reference_library.db`): Shared across all projects; stores `ref_sources`, `ref_entries`, `ref_tags`, `ref_book_blueprints`, `ref_pacing_templates`, and `ref_pacing_stages` alongside 512-dimensional semantic embeddings.

### 3. Deterministic 16k Context Router
`ContextRouter` (`src/autonovel/engine/context_router.py`) allocates a strict 16,000-token budget per chapter (`ContextBudgetConfig`):
- **System, Style Contract & Anchors**: `1,200` tokens (writing contract + static & dynamic high-score anchors)
- **Routed Settings**: `2,300` tokens (1-hop active entities, due foreshadowing, world laws, forward retcon notes)
- **Reference Digest**: `500` tokens (compressed technique guidance from the reference library)
- **Scene Beats & Pacing Directive**: `600` tokens
- **Previous Chapter Tail**: `900` tokens (`600`–`800` trailing characters from Chapter N-1 for seamless prose continuity)
- **Generation & Revision Buffer**: `10,500` tokens (targeting `2,000`–`4,200` chars per chapter, hard cap `4,500`)

### 4. Two-Stage Quality Gates & End-of-Volume Acceptance
- **Gate A (Hard Facts, Consistency & Format Purity)**:
  - Normalizes extracted entities (`EntityResolver`) and validates hard database constraints (e.g., dead characters cannot act, monotonic timeline progression, realm/hierarchy rules).
  - Runs an LLM semantic consistency and format purity check against `world_rules` (rejecting markdown headings, beat tags, or meta-commentary in prose).
  - Injects actionable *Forward Retcon* instructions for minor drift and retries up to `3` times (`max_gate_a_retries`) before blocking commit (`GateAFailedError`).
- **Gate B (100-Point Literary Rubric)**:
  - Evaluates prose across 10 rubric items weighted by `RHETORIC` (40%), `DIALOGUE` (35%), and `SCENE` (25%).
  - Requires composite score `>= 80.0` and minimum dimension score `>= 65.0`. Failing chapters undergo targeted revision (`max_gate_b_revisions = 1`) enriched with reference library techniques, followed by delta re-extraction and hard-fact re-validation before atomic commit.
- **Periodic Maintenance & Volume Acceptance**:
  - Every `10` chapters, `DeepOrganizer` deduplicates character aliases, archives minor NPCs unseen for 30+ chapters with no pending foreshadowing, and aligns timeline sync points, while `SnapshotManager` rotates periodic SQLite backups.
  - At volume completion, `VolumePlanner` verifies `0%` hard conflicts, `100%` timeline consistency, `>= 95%` foreshadow recovery rate, `>= 95%` pacing stability, and `>= 85.0` mean literary score—automatically rewriting the bottom 20% chapters if any threshold fails before planning the next volume.

### 5. Dual-Track Reference Extraction & Multi-Track Retrieval
- **Multi-Format Parser** (`src/autonovel/reference/file_parser.py`): Supports `.txt`, `.md`, `.epub` (OPF spine-ordered), and `.docx` with Hanzi-frequency encoding detection (`utf-8`, `utf-8-sig`, `big5-hkscs`, `big5`, `gb18030`, `gbk`) and SHA-256 deduplication.
- **Track B (Macro Blueprint & Factual Pacing Curve)**: Batches chapters dynamically, extracts 3D chapter scores and entity deltas, consolidates volumes and major narrative arcs via a zero-truncation tree, and builds high-resolution pacing templates.
- **Track A (Exemplar Chunk Ingestion)**: Selects high-value chapters via Dual-Tier Z-score sampling (global outliers + per-volume peaks) and embeds writing technique entries on CPU using `BAAI/bge-small-zh-v1.5` (ONNX).
- **Weighted Retrieval**: Combines description similarity, tag similarity, and genre alignment (`0.40 * desc + 0.45 * tag + 0.15 * genre >= 0.72`, `top_k = 3`) and compresses matches into a concise guidance digest.

### 6. Traditional Chinese Prose Sanitizer & Multi-Format Export
- Built-in Simplified-to-Traditional Chinese conversion using bundled `opencc_data.json` (no external OpenCC binary required), quotation style normalization, literary term mapping, and web-novel noise stripping.
- One-click export (`NovelExporter`) across full book, single volume, or custom chapter ranges (`ALL` / `VOLUME` / `RANGE`, single-file or `PER_VOLUME` bundle) into:
  - **Markdown (`.md`)**
  - **Plain Text (`.txt`)**
  - **Styled Standalone HTML (`.html`)**
  - **EPUB 3 (`.epub`)**
  - **Microsoft Word OpenXML (`.docx`)**

### 7. Hardware & Thermal Protection
- **0 MB VRAM Embeddings**: `CpuEmbedder` runs `bge-small-zh-v1.5` entirely on CPU via `onnxruntime`, preserving 100% of GPU VRAM for `llama-server` (`-c 16000 -ngl 99 -ctk q8_0 -ctv q8_0 --flash-attn auto`).
- **Thermal Cooldown & Anti-Sleep**: `HardwareGuard` monitors GPU VRAM and temperature via NVML (`pynvml`) or `nvidia-smi`, pauses generation at `>= 83 C` until cooled to `<= 76 C`, and acquires Windows `SetThreadExecutionState` anti-sleep locks during active serialization.

---

## Architecture Overview

```text
+-------------------------------------------------------------------------+
|                    Vanilla JS Web UI (Static / No Build)                |
|   Project Hub  |  Worldbuilding Wizard  |  War Room  |  Reference Lab   |
+------------------------------------+------------------------------------+
                                     | HTTP / SSE (/api/*)
+------------------------------------v------------------------------------+
|                         FastAPI App (web/app.py)                        |
|        routes_projects  |  routes_wizard  |  routes_warroom             |
+------------------------------------+------------------------------------+
                                     |
+------------------------------------v------------------------------------+
|                  NovelWritingDaemon (engine/daemon.py)                  |
|  - Crash Recovery & Auto-Resume     - Idle VRAM Watchdog                |
|  - Graceful Pause & Anti-Sleep Lock - Hardware Thermal Guard            |
+-------+-------------------------+-----------------------------+---------+
        |                         |                             |
        v                         v                             v
+---------------+       +-------------------+        +--------------------+
| Wizard &      |       | ChapterPipeline   |        | Reference Pipeline |
| Blueprint     |       | - ContextRouter   |        | - FileParser       |
| - QAEngine    |       | - Draft Stream    |        | - MacroExtractor   |
| - Blueprint   |       | - Gate A / Gate B |        | - ChunkExtractor   |
|   Builder     |       | - Atomic Commit   |        | - Retriever        |
+-------+-------+       | - DeepOrganizer   |        +---------+----------+
        |               | - VolumePlanner   |                  |
        |               +---------+---------+                  |
        v                         v                            v
+-----------------------------------+    +--------------------------------+
|     Per-Project SQLite + Backups  |    |   Global Reference SQLite DB   |
| data/projects/<id>/novel_project  |    |   data/reference_library.db    |
+-----------------------------------+    +--------------------------------+
        |                                                      |
        +-----------------------+------------------------------+
                                |
          +---------------------+---------------------+
          v                                           v
+--------------------------+               +--------------------------+
| llama-server (GPU/GGUF)  |               | CpuEmbedder (ONNX / CPU) |
| Qwen2.5-7B-Instruct      |               | bge-small-zh-v1.5        |
| 16k Context, Q8_0 KV     |               | 512-dim, 0 MB VRAM       |
+--------------------------+               +--------------------------+
```

---

## Project Structure

```text
AutoNovelWriter/
├── main.py                      # Uvicorn server entrypoint (127.0.0.1:7860)
├── pyproject.toml               # Package metadata, dependencies, pytest config
├── AGENTS.md                    # Developer guidelines & architecture notes
├── bin/                         # Local llama-server binary directory (gitignored executables)
│   └── LICENSE-LLVM-OpenMP      # Redistributed LLVM OpenMP license notice
├── models/                      # Local LLM GGUF & embedding models
│   └── bge-small-zh-v1.5/       # Bundled CPU ONNX embedding model & tokenizer
├── scripts/                     # Standalone setup & extraction CLI scripts
│   ├── download_llama_server.py # Auto-detects CUDA and downloads llama.cpp Windows build
│   ├── setup_environment.py     # Environment inspector & Hugging Face model downloader
│   ├── run_auto_extraction.py   # End-to-end CLI reference extraction runner
│   └── slice_novel.py           # Helper to slice the first N chapters of a text novel
├── src/autonovel/
│   ├── config.py                # RuntimeConfig, ContextBudgetConfig, QualityThresholdConfig
│   ├── db/                      # Dual-SQLite schemas, repositories, project & snapshot managers
│   ├── engine/                  # Daemon, chapter pipeline, context router, quality gates, pacing
│   ├── export/                  # Multi-format exporter (.md, .txt, .html, .epub, .docx)
│   ├── llm/                     # llama-server process manager, async LLM client, CPU ONNX embedder
│   ├── reference/               # File parser, Track A/B extractors, multi-track retriever
│   ├── schemas/                 # Domain models and strict Pydantic LLM structured contracts
│   ├── web/                     # FastAPI routes and static Web UI (HTML/CSS/vanilla JS)
│   └── wizard/                  # Genre catalog, Q&A engine, and blueprint synthesizer
├── tests/                       # Pytest test suite using deterministic MockLLMClient
└── data/                        # Runtime databases, snapshots, logs, uploads, exports (gitignored)
```

---

## Requirements

- **Python**: `>= 3.11`
- **Operating System**:
  - **Windows 10/11 (64-bit)**: Primary supported platform with automatic NVIDIA CUDA detection for `llama.cpp` releases and native Windows anti-sleep locking.
  - **Linux / macOS**: Supported by installing or compiling `llama-server` separately and setting `AUTONOVEL_LLAMA_BIN` (or placing `llama-server` on `PATH`).
- **Hardware**:
  - **GPU**: NVIDIA GPU with `>= 8 GB` VRAM recommended for `Qwen2.5-7B-Instruct-Q5_K_M.gguf` at 16k context (`Q8_0` KV cache + FlashAttention). CPU fallback works with reduced throughput.
  - **RAM**: `16 GB+` system memory recommended.

---

## Quick Start

### 1. Clone and Create a Virtual Environment

```bash
git clone https://github.com/Cyw-twntpc/AutoNovelWriter.git
cd AutoNovelWriter

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate
```

### 2. Install Dependencies

Install the package in editable mode along with test dependencies:

```bash
pip install -e ".[dev]"
```

For automated Hugging Face model downloads (from the Web UI or CLI setup scripts), install `huggingface-hub` (and optionally `nvidia-ml-py` for native NVML telemetry if `nvidia-smi` CLI probing is not preferred):

```bash
pip install huggingface-hub
```

### 3. Provision Inference Engine & Models

The repository bundles `models/bge-small-zh-v1.5/` (`tokenizer.json` and `onnx/model.onnx`) for zero-VRAM CPU embeddings. You only need `llama-server` and a GGUF LLM (`Qwen2.5-7B-Instruct-Q5_K_M.gguf` by default).

#### Option A: One-Click Setup via Web UI (Recommended on Windows)
1. Start the application:
   ```bash
   python main.py
   ```
2. Open **`http://127.0.0.1:7860`** in your browser.
3. Open the Environment Setup panel and run **One-Click Deployment** (`/api/system/deploy_all`) to automatically:
   - Verify the CPU BGE ONNX model (or re-download from `Xenova/bge-small-zh-v1.5` if missing).
   - Download `Qwen2.5-7B-Instruct-Q5_K_M.gguf` from `bartowski/Qwen2.5-7B-Instruct-GGUF` with automatic retry/resume.
   - Detect your system CUDA version, download the matching `llama.cpp` Windows binary into `bin/`, and start `llama-server`.

#### Option B: Setup via CLI Scripts
```bash
# Download optimal Windows llama-server binary (auto-detects CUDA; pass --cpu-only for CPU builds)
python scripts/download_llama_server.py

# Verify environment (optionally pass --download-bge to re-fetch BGE ONNX weights)
python scripts/setup_environment.py
```

If using a custom GGUF model or non-Windows `llama-server`, place the files locally or point to them via environment variables:
```bash
# Example (PowerShell)
$env:AUTONOVEL_LLAMA_BIN = "/usr/local/bin/llama-server"
$env:AUTONOVEL_MODEL_PATH = "models/Qwen2.5-7B-Instruct-Q5_K_M.gguf"
```

### 4. Launch the Studio

Run all commands from the repository root (unless `AUTONOVEL_WORKSPACE` is explicitly set):

```bash
python main.py
```

The FastAPI server starts on **`http://127.0.0.1:7860`** with hot-reload enabled and static asset caching disabled.

---

## Configuration

### Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `AUTONOVEL_WORKSPACE` | Current working directory | Base directory for resolving `data/`, `models/`, and `bin/`. |
| `AUTONOVEL_LLAMA_URL` | `http://127.0.0.1:8080` | Base URL of the local `llama-server` HTTP endpoint. |
| `AUTONOVEL_LLAMA_BIN` | `llama-server` | Path or executable name for `llama-server` (auto-probes `bin/` on Windows). |
| `AUTONOVEL_MODEL_PATH` | `models/Qwen2.5-7B-Instruct-Q5_K_M.gguf` | Path to the GGUF model file (relative to `AUTONOVEL_WORKSPACE` or absolute). |
| `AUTONOVEL_MODEL_IDLE_TIMEOUT_SEC` | `600.0` | Seconds of user inactivity before the idle watchdog stops `llama-server` to free VRAM. |
| `AUTONOVEL_LLAMA_NP` | `1` | Number of parallel decoding slots (`-np`) passed to `llama-server`. |

### Tunable Constants (`src/autonovel/config.py`)

Core budgets and thresholds are defined as frozen dataclasses in `src/autonovel/config.py`:
- **`ContextBudgetConfig`**: 16k token allocation across prompt sections and chapter character targets (`2,000`–`4,200` soft, `4,500` hard max).
- **`QualityThresholdConfig`**:
  - `single_chapter_pass_score = 80.0` / `single_dimension_min_score = 65.0`
  - `volume_acceptance_mean_score = 85.0` / `foreshadow_recovery_min_rate = 0.95` / `pacing_stability_min_rate = 0.95`
  - `max_gate_a_retries = 3` / `max_gate_b_revisions = 1`
  - `reference_similarity_threshold = 0.72` / `reference_top_k = 3`
- **`RuntimeConfig`**: Thermal cooldown threshold (`gpu_temp_cooldown_threshold_c = 83`), periodic deep maintenance/snapshot interval (`10` chapters), and rotating log limits (`data/logs/autonovel.log`, 10 MB x 5 backups).

---

## Usage Workflow

1. **Environment & Reference Library (Optional)**:
   - Upload reference novels (`.txt`, `.md`, `.epub`, `.docx`) via the Reference Workbench to extract macro pacing curves, book blueprints, and tagged rhetorical exemplars into `data/reference_library.db`.
2. **Create Project & Worldbuilding Wizard**:
   - Create a new novel project from the Project Hub (`data/projects/<id>/`).
   - Select a primary/sub-genre in the Wizard and complete the interactive AI Q&A session (supports user input, AI-assisted autocomplete, and step rollback).
   - Generate the initial story blueprint (characters, factions, locations, hierarchy levels, world rules, and Volume 1 chapter beats).
3. **Review Workbench & Blueprint Lock**:
   - Inspect and edit characters, world laws, and chapter outlines in the Review Workbench, then lock the blueprint to unlock serialization.
4. **Autonomous War Room**:
   - Click **Start / Resume Serialization** for continuous autonomous writing, or **Step Single Chapter** to write one chapter at a time.
   - Monitor live streaming prose, token throughput, GPU VRAM/temperature gauges, Gate A/B metrics, anomaly logs, and pacing alerts.
   - Create or restore SQLite snapshots at any time, or rewrite the latest chapter with a fresh seed.
5. **Export**:
   - Export completed chapters (all, by volume, or by chapter range) to `.md`, `.txt`, `.html`, `.epub`, or `.docx`.

---

## CLI Utilities

- **Standalone Full-Novel Reference Extraction**:
  ```bash
  python scripts/run_auto_extraction.py --novel path/to/novel.txt --genre-primary "<PrimaryGenre>" --genre-sub "<SubGenre>"
  ```
  Executes Track B (batched chapter facts, entity deltas, volume consolidation, major arc reduction, macro blueprint, factual pacing curve) with JSON checkpointing and Track A (Z-score exemplar chunk ingestion), outputting an audit report to `data/extraction_audit_report.json`.
- **Chapter Slicing Helper**:
  Edit or invoke `slice_novel()` in `scripts/slice_novel.py` to extract the first *N* chapters of a long novel text file into a smaller test fixture.

---

## Testing

The test suite runs completely offline without a GPU or `llama-server` by mocking structured and streaming LLM responses via `MockLLMClient` (`tests/conftest.py`) while exercising the real SQLite schemas, pipelines, and CPU ONNX embeddings (`models/bge-small-zh-v1.5/`).

Run all tests from the repository root:

```bash
python -m pytest -q
```

Run a specific test module or case:

```bash
python -m pytest tests/test_pipeline_and_export.py -q
```

---

## API Overview

All backend endpoints are mounted under `/api`:
- **Projects & Export** (`/api/projects/*`): List, create, switch, rename, delete, and export projects.
- **Wizard & Review** (`/api/wizard/*`, `/api/review/*`): Genre catalog, Q&A turn progression, autocomplete, rollback, blueprint generation, manual review edits, and lock-and-start.
- **War Room & Snapshots** (`/api/warroom/*`): Dashboard state, chapter detail, pause/resume/step/rewrite, acceptance report, snapshot create/list/restore, and SSE stream (`/api/warroom/stream`).
- **Reference Library** (`/api/reference/*`): Upload preflight, pause/resume/cancel ingestion, books, blueprints, entries, tag management/pruning, and pacing curve blending.
- **System & Environment** (`/api/system/*`): Environment inspection, model/binary downloads, one-click deployment, `llama-server` start/unload, heartbeat, and logs.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Third-party bundled assets and runtime dependencies remain subject to their respective upstream licenses (including `bin/LICENSE-LLVM-OpenMP` for LLVM OpenMP runtime notices and `models/bge-small-zh-v1.5/` for BGE embedding weights).

---

## Acknowledgments & Third-Party Notices

- **[llama.cpp](https://github.com/ggerganov/llama.cpp)**: Local GGUF LLM inference engine (`bin/LICENSE-LLVM-OpenMP` covers redistributed LLVM OpenMP runtime notices).
- **[Qwen2.5](https://huggingface.co/Qwen) / [bartowski GGUF](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF)**: Default local base model (`Qwen2.5-7B-Instruct-Q5_K_M.gguf`).
- **[BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)** & **[Xenova ONNX](https://huggingface.co/Xenova/bge-small-zh-v1.5)**: 512-dimensional Chinese semantic embedding weights bundled under `models/bge-small-zh-v1.5/`.
- **[FastAPI](https://fastapi.tiangolo.com/)**, **[Pydantic](https://docs.pydantic.dev/)**, and **[ONNX Runtime](https://onnxruntime.ai/)**.
