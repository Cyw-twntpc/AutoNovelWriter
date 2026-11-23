"""Multi-project directory and metadata manager for isolated novel workspaces."""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autonovel.config import RuntimeConfig, get_logger
from autonovel.db.connection import atomic_transaction, read_connection
from autonovel.db.project_schema import init_project_database

logger = get_logger(__name__)

# Project ids are generated as ``proj_<timestamp>_<hex>``; anything else is rejected
# before it can reach a filesystem join (prevents ``../`` traversal from the API).
PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def normalize_datetime(val: Any) -> str:
    """Normalize various timestamp or ISO formats to standard 'YYYY-MM-DD HH:MM:SS' string."""
    if not val:
        return "--"
    s = str(val).strip()
    try:
        if "T" in s:
            clean_s = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_s)
            return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        parts = s.split(".")[0]
        if len(parts) >= 19 and parts[4] == "-" and parts[7] == "-":
            dt_utc = datetime.strptime(parts[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            return dt_utc.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        return parts
    except Exception:
        if len(s) >= 19 and s[4] == "-" and s[7] == "-":
            return s[:19].replace("T", " ")
        return s


class ProjectManager:
    """Manages creation, switching, renaming, and deletion of independent novel projects."""

    def __init__(self, config: RuntimeConfig) -> None:
        """Initialize project manager with configuration directories."""
        self.config = config
        self.projects_dir = config.projects_dir
        self.meta_file = config.projects_meta_path
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def _load_meta(self) -> dict[str, Any]:
        """Read projects.json metadata or return fresh dictionary."""
        if not self.meta_file.exists():
            return {"active_project_id": "", "projects": []}
        try:
            with open(self.meta_file, encoding="utf-8") as fp:
                data = json.load(fp)
                if isinstance(data, dict) and "projects" in data:
                    return data
        except Exception as exc:
            logger.warning("Failed to parse projects.json, recreating: %s", exc)
        return {"active_project_id": "", "projects": []}

    def _save_meta(self, data: dict[str, Any]) -> None:
        """Persist projects.json atomically using UTF-8 encoding."""
        tmp_file = self.meta_file.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)
        tmp_file.replace(self.meta_file)

    def ensure_initialized(self, auto_create_default: bool = False) -> str:
        """Ensure initial workspace metadata exists. Only creates default project on brand new system if auto_create_default is True."""
        if not self.meta_file.exists():
            logger.info("Initializing clean novel project workspace on first install...")
            created = self.create_project(
                title="未命名長篇小說",
                genre_primary="奇幻",
                genre_sub="古典修仙",
                min_word_count=5000000,
            )
            return str(created["project_id"])

        meta = self._load_meta()
        active_id = meta.get("active_project_id", "")
        projects: list[dict[str, Any]] = meta.get("projects", [])

        if projects:
            active_exists = False
            if active_id:
                active_db = self.projects_dir / active_id / "novel_project.db"
                active_exists = active_db.exists()
            if not active_exists:
                for p in projects:
                    candidate_id = p["project_id"]
                    if (self.projects_dir / candidate_id / "novel_project.db").exists():
                        meta["active_project_id"] = candidate_id
                        self._save_meta(meta)
                        return candidate_id
                meta["projects"] = []
                meta["active_project_id"] = ""
                self._save_meta(meta)
                return ""
            return active_id

        if auto_create_default:
            created = self.create_project(
                title="未命名長篇小說",
                genre_primary="奇幻",
                genre_sub="古典修仙",
                min_word_count=5000000,
            )
            return str(created["project_id"])
        return ""

    def get_active_project_id(self) -> str:
        """Return the current active project ID or empty string if no project exists."""
        if not self.meta_file.exists():
            return self.ensure_initialized(auto_create_default=True)
        meta = self._load_meta()
        active_id = meta.get("active_project_id", "")
        if active_id and (self.projects_dir / active_id / "novel_project.db").exists():
            return active_id
        projects = meta.get("projects", [])
        if projects:
            for p in projects:
                pid = p["project_id"]
                if (self.projects_dir / pid / "novel_project.db").exists():
                    meta["active_project_id"] = pid
                    self._save_meta(meta)
                    return pid
        meta["active_project_id"] = ""
        self._save_meta(meta)
        return ""

    def _safe_project_dir(self, project_id: str) -> Path:
        """Resolve a project directory, rejecting ids that escape the projects root."""
        if not PROJECT_ID_RE.fullmatch(project_id):
            raise ValueError(f"Invalid project id: {project_id!r}")
        base = self.projects_dir.resolve()
        proj_dir = (base / project_id).resolve()
        if proj_dir.parent != base:
            raise ValueError(f"Project id escapes workspace: {project_id!r}")
        return proj_dir

    def get_project_paths(self, project_id: str | None = None) -> dict[str, Path]:
        """Resolve filesystem paths for a project database, snapshots, and exports."""
        pid = project_id if project_id is not None else self.get_active_project_id()
        if not pid:
            dummy_dir = self._safe_project_dir("_standby")
            return {
                "project_dir": dummy_dir,
                "project_db_path": dummy_dir / "novel_project.db",
                "snapshot_dir": dummy_dir / "snapshots",
                "export_dir": dummy_dir / "exports",
            }
        proj_dir = self._safe_project_dir(pid)
        proj_dir.mkdir(parents=True, exist_ok=True)
        snap_dir = proj_dir / "snapshots"
        exp_dir = proj_dir / "exports"
        snap_dir.mkdir(parents=True, exist_ok=True)
        exp_dir.mkdir(parents=True, exist_ok=True)
        return {
            "project_dir": proj_dir,
            "project_db_path": proj_dir / "novel_project.db",
            "snapshot_dir": snap_dir,
            "export_dir": exp_dir,
        }

    def list_projects(self) -> list[dict[str, Any]]:
        """Return detailed list of all projects with live chapter counts, word metrics, and status."""
        meta = self._load_meta()
        active_id = meta.get("active_project_id", "")
        results: list[dict[str, Any]] = []

        for p in meta.get("projects", []):
            pid = p["project_id"]
            db_path = self.projects_dir / pid / "novel_project.db"
            if not db_path.exists():
                continue

            info: dict[str, Any] = {
                "project_id": pid,
                "title": p.get("title", "未命名小說"),
                "genre_primary": p.get("genre_primary", "奇幻"),
                "genre_sub": p.get("genre_sub", ""),
                "min_word_count": p.get("min_word_count", 5000000),
                "created_at": p.get("created_at", ""),
                "updated_at": p.get("updated_at", ""),
                "is_active": (pid == active_id),
                "pipeline_state": "WIZARD",
                "current_stage": "WIZARD",
                "completed_chapters": 0,
                "total_chapters": 0,
                "chapters_count": 0,
                "volumes_count": 0,
                "total_words": 0,
                "current_words": 0,
                "target_words": p.get("min_word_count", 5000000),
                "progress_percent": 0.0,
            }

            try:
                with read_connection(db_path) as conn:
                    cfg_row = conn.execute(
                        "SELECT title, genre_primary, genre_sub, min_word_count, "
                        "pipeline_state, updated_at FROM project_config WHERE id = 1;"
                    ).fetchone()
                    if cfg_row:
                        if cfg_row["title"]:
                            info["title"] = cfg_row["title"]
                        info["genre_primary"] = cfg_row["genre_primary"] or info["genre_primary"]
                        info["genre_sub"] = cfg_row["genre_sub"] or info["genre_sub"]
                        info["min_word_count"] = int(cfg_row["min_word_count"] or info["min_word_count"])
                        info["pipeline_state"] = cfg_row["pipeline_state"] or "WIZARD"
                        info["current_stage"] = info["pipeline_state"]
                        if cfg_row["updated_at"]:
                            info["updated_at"] = cfg_row["updated_at"]

                    ch_up_row = conn.execute(
                        "SELECT MAX(updated_at) AS max_up FROM chapters;"
                    ).fetchone()
                    if ch_up_row and ch_up_row["max_up"]:
                        if not info["updated_at"] or str(ch_up_row["max_up"]) > str(info["updated_at"]):
                            info["updated_at"] = ch_up_row["max_up"]

                    ch_row = conn.execute(
                        "SELECT COUNT(*) AS total_chs, "
                        "SUM(CASE WHEN status IN ('COMPLETED', 'REVISED') THEN 1 ELSE 0 END) AS done_chs "
                        "FROM chapters;"
                    ).fetchone()
                    if ch_row:
                        total_chs = int(ch_row["total_chs"] or 0)
                        done_chs = int(ch_row["done_chs"] or 0)
                        info["total_chapters"] = total_chs
                        info["chapters_count"] = total_chs
                        info["completed_chapters"] = done_chs

                    vol_row = conn.execute(
                        "SELECT COUNT(*) AS total_vols FROM volumes;"
                    ).fetchone()
                    if vol_row:
                        info["volumes_count"] = int(vol_row["total_vols"] or 0)

                    w_row = conn.execute(
                        "SELECT COALESCE(SUM(word_count), 0) AS words FROM chapter_metrics;"
                    ).fetchone()
                    if w_row:
                        tot_words = int(w_row["words"] or 0)
                        info["total_words"] = tot_words
                        info["current_words"] = tot_words

                    target_words = max(100000, info["min_word_count"])
                    info["target_words"] = target_words
                    info["progress_percent"] = round((info["total_words"] / target_words) * 100.0, 2)
            except sqlite3.Error as exc:
                logger.warning("Could not read project stats for '%s': %s", pid, exc)

            info["created_at"] = normalize_datetime(info["created_at"])
            info["updated_at"] = normalize_datetime(info["updated_at"] or info["created_at"])
            results.append(info)

        # Sort: newest first, then float the active project to the top (stable sort).
        results.sort(key=lambda x: x["updated_at"], reverse=True)
        results.sort(key=lambda x: not x["is_active"])
        return results

    def create_project(
        self,
        title: str,
        genre_primary: str = "奇幻",
        genre_sub: str = "",
        min_word_count: int = 5000000,
    ) -> dict[str, Any]:
        """Create a new novel project directory, initialize database, and set as active."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:6]
        project_id = f"proj_{ts}_{uid}"
        paths = self.get_project_paths(project_id)
        db_path = paths["project_db_path"]

        init_project_database(db_path)
        with atomic_transaction(db_path) as conn:
            conn.execute(
                "UPDATE project_config SET title = ?, genre_primary = ?, "
                "genre_sub = ?, min_word_count = ?, pipeline_state = 'WIZARD', "
                "updated_at = datetime('now') WHERE id = 1;",
                (title.strip() or "未命名長篇小說", genre_primary, genre_sub, min_word_count),
            )

        now_iso = datetime.now(timezone.utc).isoformat()
        proj_meta = {
            "project_id": project_id,
            "title": title.strip() or "未命名長篇小說",
            "genre_primary": genre_primary,
            "genre_sub": genre_sub,
            "min_word_count": min_word_count,
            "created_at": now_iso,
            "updated_at": now_iso,
        }

        meta = self._load_meta()
        meta["projects"] = [p for p in meta.get("projects", []) if p["project_id"] != project_id]
        meta["projects"].append(proj_meta)
        meta["active_project_id"] = project_id
        self._save_meta(meta)

        logger.info("Created new novel project: '%s' (%s)", title, project_id)
        return proj_meta

    def switch_project(self, project_id: str) -> dict[str, Any]:
        """Change the active project ID in projects.json."""
        paths = self.get_project_paths(project_id)
        if not paths["project_db_path"].exists():
            raise FileNotFoundError(f"Project database '{project_id}' not found.")

        meta = self._load_meta()
        found = any(p["project_id"] == project_id for p in meta.get("projects", []))
        if not found:
            raise KeyError(f"Project ID '{project_id}' not registered in metadata.")

        meta["active_project_id"] = project_id
        self._save_meta(meta)
        logger.info("Switched active novel project to: %s", project_id)
        return paths

    def rename_project(self, project_id: str, new_title: str) -> dict[str, Any]:
        """Update the title of a novel project both in metadata and project_config table."""
        cleaned_title = new_title.strip()
        if not cleaned_title:
            raise ValueError("小說書名不能為空。")

        meta = self._load_meta()
        matched = False
        for p in meta.get("projects", []):
            if p["project_id"] == project_id:
                p["title"] = cleaned_title
                p["updated_at"] = datetime.now(timezone.utc).isoformat()
                matched = True
                break

        if not matched:
            raise KeyError(f"Project ID '{project_id}' not found.")

        self._save_meta(meta)

        paths = self.get_project_paths(project_id)
        if paths["project_db_path"].exists():
            with atomic_transaction(paths["project_db_path"]) as conn:
                conn.execute(
                    "UPDATE project_config SET title = ?, updated_at = datetime('now') WHERE id = 1;",
                    (cleaned_title,),
                )

        logger.info("Renamed project '%s' to '%s'", project_id, cleaned_title)
        return {"project_id": project_id, "title": cleaned_title}

    def delete_project(self, project_id: str) -> str:
        """Delete project files from disk, prune from metadata, and return current/new active ID."""
        meta = self._load_meta()
        projects: list[dict[str, Any]] = meta.get("projects", [])
        if not any(p["project_id"] == project_id for p in projects):
            raise KeyError(f"Project ID '{project_id}' not found.")

        # Remove from disk
        proj_dir = self._safe_project_dir(project_id)
        if proj_dir.exists() and proj_dir.is_dir():
            shutil.rmtree(proj_dir, ignore_errors=True)
            if proj_dir.exists():
                # Never drop metadata while files survive: that would orphan the data
                # with no UI entry left to delete it.
                raise OSError(
                    f"Failed to remove project directory for '{project_id}'; metadata kept."
                )

        meta["projects"] = [p for p in projects if p["project_id"] != project_id]

        # If deleted project was active, reassign active
        new_active = meta.get("active_project_id", "")
        if new_active == project_id:
            if meta["projects"]:
                new_active = meta["projects"][0]["project_id"]
                meta["active_project_id"] = new_active
            else:
                new_active = ""
                meta["active_project_id"] = ""

        self._save_meta(meta)
        logger.info("Deleted novel project: %s (remaining active: '%s')", project_id, new_active)
        return new_active
