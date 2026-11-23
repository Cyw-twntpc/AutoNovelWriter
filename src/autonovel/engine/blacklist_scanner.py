"""Pass-through style guard interface (literary cliché and style evaluation delegated to AI Gate B Rubric)."""

from __future__ import annotations

import json

from autonovel.db.project_repo import ProjectRepository
from autonovel.llm.llm_client import LLMClient
from autonovel.schemas.domain import BlacklistConfig


class BlacklistScanner:
    """Pass-through interface preserving pipeline contract without programmatic keyword scanning or sentence surgery."""

    def __init__(
        self, repo: ProjectRepository, llm_client: LLMClient | None = None
    ) -> None:
        """Initialize scanner with project repository and optional LLM client."""
        self.repo = repo
        self.llm_client = llm_client

    def load_active_terms(self) -> list[str]:
        """Load user-configured terms from project_config.blacklist_json."""
        cfg = self.repo.get_project_config()
        raw_json = cfg.get("blacklist_json") or "{}"
        data = json.loads(raw_json)
        config = BlacklistConfig.model_validate(data)
        return config.merged_terms()

    def scan_violations(self, text: str, terms: list[str] | None = None) -> list[str]:
        """No-op keyword scan; stylistic quality is evaluated holistically by AI in Gate B Rubric."""
        return []

    async def sanitize_text(
        self, text: str, terms: list[str] | None = None
    ) -> tuple[str, int, list[str]]:
        """Return text untouched without mechanical sentence splitting or rule-based rewriting."""
        return text, 0, []
