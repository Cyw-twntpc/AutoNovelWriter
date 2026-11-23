"""Application entry point for launching AutoNovelWriter local server."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
for path_str in (str(SRC_DIR), str(ROOT_DIR)):
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import uvicorn  # noqa: E402

from autonovel.config import RuntimeConfig  # noqa: E402
from autonovel.web.app import create_app  # noqa: E402

app = create_app(RuntimeConfig())


def main() -> None:
    """Start the local Uvicorn server on 127.0.0.1:7860 with auto-reload."""
    uvicorn.run("main:app", host="127.0.0.1", port=7860, reload=True, log_level="info")


if __name__ == "__main__":
    main()
