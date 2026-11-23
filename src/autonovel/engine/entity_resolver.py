"""Entity disambiguation and alias resolution layer for SQLite foreign-key binding."""

from __future__ import annotations

import json

from autonovel.db.project_repo import ProjectRepository
from autonovel.schemas.llm_contracts import ChapterExtractionContract


class EntityResolver:
    """Normalizes extracted entity names against canonical names and aliases in SQLite."""

    def __init__(self, repo: ProjectRepository) -> None:
        """Initialize resolver with project repository."""
        self.repo = repo

    def normalize_extraction(
        self, extraction: ChapterExtractionContract
    ) -> ChapterExtractionContract:
        """Map extracted character, faction, and location names in deltas to canonical DB names."""
        char_map = self._build_entity_alias_map("characters")
        fac_map = self._build_entity_alias_map("factions")
        loc_map = self._build_entity_alias_map("locations")
        for c_delta in extraction.character_deltas:
            canonical = self._resolve_name(c_delta.character_name, char_map)
            if canonical != c_delta.character_name:
                if c_delta.character_name not in c_delta.aliases:
                    c_delta.aliases.append(c_delta.character_name)
                c_delta.character_name = canonical
            if c_delta.faction_name:
                c_delta.faction_name = self._resolve_name(c_delta.faction_name, fac_map)
            if c_delta.location_name:
                c_delta.location_name = self._resolve_name(c_delta.location_name, loc_map)
        for rel in extraction.relation_deltas:
            rel.source_name = self._resolve_name(rel.source_name, char_map)
            rel.target_name = self._resolve_name(rel.target_name, char_map)
        for fs in extraction.foreshadow_deltas:
            fs.related_characters = [
                self._resolve_name(name, char_map) for name in fs.related_characters
            ]
        for tl in extraction.timeline_deltas:
            if tl.location_name:
                tl.location_name = self._resolve_name(tl.location_name, loc_map)
        return extraction

    def _build_entity_alias_map(self, table: str) -> dict[str, str]:
        """Build lookup dictionary mapping canonical names and aliases to canonical name."""
        if table not in ("characters", "factions", "locations"):
            raise ValueError(f"Unsupported entity table: {table}")
        rows = self.repo.fetch_all(f"SELECT name, aliases FROM {table};")
        mapping: dict[str, str] = {}
        for row in rows:
            canonical = str(row["name"]).strip()
            mapping[canonical] = canonical
            for alias in json.loads(row.get("aliases") or "[]"):
                cleaned = str(alias).strip()
                if cleaned:
                    mapping[cleaned] = canonical
        return mapping

    @staticmethod
    def _resolve_name(raw_name: str, alias_map: dict[str, str]) -> str:
        """Return canonical entity name only on exact match against canonical names or registered aliases."""
        cleaned = raw_name.strip()
        return alias_map.get(cleaned, cleaned)
