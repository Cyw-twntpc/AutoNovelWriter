# AGENTS.md

## Commands

- Run everything from the repo root. `python main.py` starts the FastAPI app on `http://127.0.0.1:7860` with auto-reload.
- `python -m pytest -q` runs the full suite (~100 tests, ~20 s, mocked LLM; no llama-server/GPU needed).
- Single test: `python -m pytest tests/test_x.py::test_y -q`.
- No lint, formatter, or typecheck is configured. Only pytest settings exist in `pyproject.toml` (`pythonpath=["src"]`, `asyncio_mode="auto"`).
- Do not assume an editable install: `main.py` prepends `src/` to `sys.path`. Specific scripts under `scripts/` do the same.

## Test prerequisites and quirks

- Embedding tests construct `CpuEmbedder()` directly and require `models/bge-small-zh-v1.5/` with `tokenizer.json` + `onnx/model.onnx`, plus `onnxruntime` and `tokenizers` (installed here but absent from `pyproject.toml`). If the model is unavailable, `embed_batch` raises instead of falling back. The path is relative, so tests fail outside the repo root.
- ONNX sessions are cached process-wide per model path (`embedder._ONNX_SESSION_CACHE`), so many `CpuEmbedder()` instances share one session. Load failures are recorded per instance and reported as "failed to load", distinct from the "is not loaded" model-missing error.
- Any new test that instantiates `RuntimeConfig` must pass `workspace_dir=tmp_path`; the default is `Path.cwd()` and creates `data/` in the repo.
- `MockLLMClient.generate_structured` in `tests/conftest.py` must gain a branch for each new Pydantic contract, otherwise it raises `ValueError`. Extend the mock rather than calling a real model.

## Architecture

- Entrypoint: `main.py` → `autonovel.web.app.create_app(RuntimeConfig())`. The app lifespan runs `startup_recovery(auto_resume=True)`, starts the idle watchdog, and on shutdown stops llama-server and unloads models.
- `engine/daemon.py::NovelWritingDaemon` is the central object holding repositories and engines. Routes reach it via `request.app.state.daemon`; long-running generation is decoupled from HTTP/browser sessions.
- `src/autonovel/` layers: `web/` (FastAPI routes + static vanilla JS, no bundler/build step), `engine/` (daemon, chapter pipeline, quality gates, context router, pacing), `db/` (project/reference repositories, schemas, snapshots), `llm/` (llama-server lifecycle, LLM client, CPU embedder), `reference/`, `wizard/`, `schemas/` (Pydantic LLM contracts and domain models), `export/`.
- API endpoints live under `/api`; SSE stream is `/api/warroom/stream`; project CRUD/export is `/api/projects`. Static assets are mounted at `/static` and served with no-store headers, so frontend JS/CSS edits take effect on reload.
- Persistence: global reference library at `data/reference_library.db` (shared across projects); per-project state at `data/projects/<id>/{novel_project.db,snapshots,exports}`; project metadata in `data/projects.json`. The empty `novel_project.db` in the repo root is a leftover and is not used.
- All paths derive from `RuntimeConfig.workspace_dir` (default `Path.cwd()`). Start from the repo root or set `AUTONOVEL_WORKSPACE`, or the app will look for `models/`, `bin/`, and `data/` in the wrong place.
- Local inference: llama-server at `http://127.0.0.1:8080`, GGUF models in `models/`, binaries in `bin/`. Env overrides: `AUTONOVEL_LLAMA_URL`, `AUTONOVEL_LLAMA_BIN`, `AUTONOVEL_MODEL_PATH`, `AUTONOVEL_MODEL_IDLE_TIMEOUT_SEC`, `AUTONOVEL_LLAMA_NP` (parallel slots).
- The 16k context budget and quality thresholds are frozen dataclasses in `src/autonovel/config.py`; tune them there rather than hardcoding values at call sites.

## Conventions

- UI copy, prompts, chapter prose, and fixtures are Traditional Chinese. `engine/prose_sanitizer.py` converts Simplified input using bundled `engine/opencc_data.json` (no external OpenCC dependency); keep new fixtures Traditional.
- The workspace is currently not a git repository (there is a `.gitignore` but no `.git`), so `git diff`/history-based workflows are unavailable.
- Generated/runtime artifacts (`data/logs`, `data/exports`, `data/snapshots`, `data/projects/*`, `bin/*.exe|dll`, `models/*.gguf`) are ignored and should not be edited by hand.
