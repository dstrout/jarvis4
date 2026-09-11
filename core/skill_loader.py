"""
Skill loader — progressive disclosure for Agent Skills.

Provides two levels of skill access:
1. Catalog: lightweight name+description index for system prompts
2. Full load: complete SKILL.md instructions on demand

Uses agent-skills-sdk for spec-compliant parsing. Skills are discovered
from the project's skills/ directory.
"""

import logging
from pathlib import Path
from typing import Optional

from agent_skills_sdk import AgentSkillsClient
from agent_skills_sdk.models import Skill, SkillMetadata

log = logging.getLogger("jarvis.skills")

# Default skills directory
SKILLS_DIR = Path(__file__).parent.parent / "skills"

# Singleton loader instance
_loader: Optional["SkillLoader"] = None


class SkillLoader:
    """Two-tier skill loader with progressive disclosure."""

    def __init__(self, skills_dir: Path = SKILLS_DIR):
        self._client = AgentSkillsClient(
            skill_paths=[str(skills_dir)],
            auto_discover=False,
        )
        self._metadata: list[SkillMetadata] = []
        self._skills_cache: dict[str, Skill] = {}
        self._discover()

    def _discover(self):
        """Scan skills directory and build metadata index."""
        self._metadata = self._client.discover_metadata()
        log.info("Discovered %d skills", len(self._metadata))

    @property
    def catalog(self) -> list[dict]:
        """Return lightweight catalog: [{name, description, category}]."""
        entries = []
        for m in self._metadata:
            category = None
            if m.metadata and isinstance(m.metadata, dict):
                oc = m.metadata.get("openclaw", {})
                category = oc.get("category")
            entries.append({
                "name": m.name,
                "description": m.description,
                "category": category,
            })
        return entries

    def catalog_for_prompt(
        self,
        categories: list[str] | None = None,
        name_filter: list[str] | None = None,
    ) -> str:
        """Format catalog as a compact string for system prompts.

        Args:
            categories: If provided, only include skills in these categories.
                        Common values: "productivity", "recipe"
            name_filter: If provided, only include skills whose names contain
                         any of these substrings. Useful for scoping to an
                         agent's domain (e.g., ["gmail", "email"] for mail agent).
        """
        lines = []
        for entry in self.catalog:
            if categories and entry["category"] not in categories:
                continue
            if name_filter:
                if not any(f in entry["name"] for f in name_filter):
                    # Also check description for domain relevance
                    if not any(
                        f in (entry["description"] or "").lower()
                        for f in name_filter
                    ):
                        continue
            lines.append(f"- {entry['name']}: {entry['description']}")
        return "\n".join(sorted(lines))

    def get_skill(self, name: str) -> Optional[Skill]:
        """Load full skill instructions by name (cached)."""
        if name in self._skills_cache:
            return self._skills_cache[name]

        # Load all skills and cache (SDK doesn't support single-skill load)
        if not self._skills_cache:
            for skill in self._client.discover_skills():
                self._skills_cache[skill.name] = skill

        return self._skills_cache.get(name)

    def get_instructions(self, name: str) -> str | None:
        """Get just the instruction text for a skill."""
        skill = self.get_skill(name)
        if skill:
            return skill.instructions
        return None

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Simple keyword search across skill names and descriptions."""
        query_lower = query.lower()
        scored = []
        for entry in self.catalog:
            name_score = 0
            desc_score = 0
            for word in query_lower.split():
                if word in entry["name"]:
                    name_score += 2
                if word in (entry["description"] or "").lower():
                    desc_score += 1
            total = name_score + desc_score
            if total > 0:
                scored.append((total, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:limit]]


def get_skill_loader() -> SkillLoader:
    """Get or create the singleton SkillLoader."""
    global _loader
    if _loader is None:
        _loader = SkillLoader()
    return _loader
