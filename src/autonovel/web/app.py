"""FastAPI application factory binding the background daemon and static Web UI."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from autonovel.config import RuntimeConfig, close_logging_handlers
from autonovel.engine.daemon import NovelWritingDaemon
from autonovel.web.routes_projects import router as projects_router
from autonovel.web.routes_warroom import router as warroom_router
from autonovel.web.routes_wizard import router as wizard_router

_NO_CACHE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}

# Browser tabs poll these endpoints continuously (every 1-5 s). Counting them as user
# activity would keep the model loaded forever while a tab is merely open.
_PASSIVE_POLL_PATHS: frozenset[str] = frozenset(
    {
        "/api/warroom/dashboard",
        "/api/system/env_status",
        "/api/events",
        "/api/reference/progress",
    }
)


def _is_passive_poll(path: str) -> bool:
    """Return True for background polling endpoints that must not reset the idle timer."""
    return path in _PASSIVE_POLL_PATHS or path.endswith("/progress")


def _attach_static_routes(app: FastAPI, static_dir: Path) -> None:
    """Mount static assets and attach cache-busting headers for instant UI updates."""
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.middleware("http")
    async def disable_static_cache(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Prevent browser disk caching and clear stale site cache on API bootstrap."""
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static/") or path.startswith("/api/"):
            for key, value in _NO_CACHE_HEADERS.items():
                response.headers[key] = value
            if "etag" in response.headers:
                del response.headers["etag"]
            if "last-modified" in response.headers:
                del response.headers["last-modified"]
        if path in {"/api/wizard/genres", "/api/warroom/dashboard"}:
            response.headers["Clear-Site-Data"] = '"cache"'
        return response

    @app.get("/")
    async def serve_index() -> FileResponse:
        """Serve the main Traditional Chinese 3-stage Web UI with no-store headers."""
        return FileResponse(str(static_dir / "index.html"), headers=_NO_CACHE_HEADERS)


def create_app(
    config: RuntimeConfig | None = None,
    daemon: NovelWritingDaemon | None = None,
) -> FastAPI:
    """Create FastAPI instance with decoupled background daemon and static assets."""
    runtime_cfg = config or RuntimeConfig()
    active_daemon = daemon or NovelWritingDaemon(runtime_cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Run crash recovery, auto-resume RUNNING projects, start idle watchdog, and release OS locks on shutdown."""
        await app.state.daemon.startup_recovery(auto_resume=True)
        app.state.daemon.start_idle_watchdog()
        yield
        if hasattr(app.state.daemon, "shutdown"):
            await app.state.daemon.shutdown()
        else:
            app.state.daemon.stop_idle_watchdog()
            app.state.daemon.request_graceful_pause()
            app.state.daemon.hardware_guard.release_anti_sleep_lock()
            app.state.daemon.llama_manager.stop_server()
        close_logging_handlers()

    app = FastAPI(
        title="長篇小說全自動寫作工具 (AutoNovelWriter)",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def track_user_activity(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Record user activity on interactive API endpoints to prevent idle model unloading."""
        path = request.url.path
        if path.startswith("/api/") and not _is_passive_poll(path):
            active_daemon.record_user_activity()
        return await call_next(request)
    app.state.config = runtime_cfg
    app.state.daemon = active_daemon
    app.include_router(projects_router)
    app.include_router(wizard_router)
    app.include_router(warroom_router)
    _attach_static_routes(app, Path(__file__).parent / "static")
    return app
