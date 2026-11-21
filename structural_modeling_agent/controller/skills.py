"""Lazy-loaded domain skill playbooks for ModelingAgent.

Skills encode scientific heuristics, caveats, and multi-step tool workflows in
plain markdown files under ``structural_modeling_agent/skills/*.md``.
Hard provenance checks, file hashing, resource bounds, and execution remain in
typed tools registered in ``tool_registry.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"


@dataclass(frozen=True)
class SkillSpec:
    """One domain skill loaded from a markdown file with frontmatter."""

    name: str
    description: str
    domains: frozenset[str]
    path: Path
    body: str


def _parse_frontmatter(text: str, path: Path) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"Skill file {path.name} must start with YAML frontmatter '---'")
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        raise ValueError(f"Skill file {path.name} is missing closing frontmatter '---'")
    meta: dict[str, Any] = {}
    for line in lines[1:end]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"Invalid frontmatter line in {path.name}: {line!r}")
        key, raw = stripped.split(":", 1)
        key = key.strip()
        val = raw.strip()
        if val.startswith("[") and val.endswith("]"):
            items = [item.strip().strip("'\"") for item in val[1:-1].split(",") if item.strip()]
            meta[key] = items
        else:
            meta[key] = val.strip("'\"")
    body = "\n".join(lines[end + 1 :]).strip()
    return meta, body


_SKILLS_CACHE: dict[str, SkillSpec] | None = None


def load_all_skills(skills_dir: Path = SKILLS_DIR) -> dict[str, SkillSpec]:
    """Load and cache all markdown skill specifications from ``skills_dir``."""
    global _SKILLS_CACHE
    if skills_dir == SKILLS_DIR and _SKILLS_CACHE is not None:
        return _SKILLS_CACHE
    specs: dict[str, SkillSpec] = {}
    if skills_dir.is_dir():
        for path in sorted(skills_dir.glob("*.md")):
            meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"), path)
            name = meta.get("name") or path.stem
            description = meta.get("description") or ""
            domains = frozenset(meta.get("domains") or ())
            if not description:
                raise ValueError(f"Skill {name!r} ({path.name}) requires a non-empty description")
            if name in specs:
                raise ValueError(f"Duplicate skill name: {name!r}")
            specs[name] = SkillSpec(
                name=name,
                description=description,
                domains=domains,
                path=path,
                body=body,
            )
    if skills_dir == SKILLS_DIR:
        _SKILLS_CACHE = specs
    return specs


def list_skills() -> list[dict[str, Any]]:
    """Return a compact, prompt-ready index of all available skills."""
    return [
        {
            "name": s.name,
            "description": s.description,
            "domains": sorted(s.domains),
        }
        for s in load_all_skills().values()
    ]


def get_skill(name: str) -> SkillSpec:
    """Return one SkillSpec by name, raising ValueError with valid choices if unknown."""
    skills = load_all_skills()
    if name not in skills:
        raise ValueError(f"unknown skill {name!r}; choose one of {sorted(skills)}")
    return skills[name]


def skill_index_prompt() -> str:
    """Render a compact 1-line-per-skill index for inclusion in the agent system prompt."""
    entries = list_skills()
    if not entries:
        return ""
    lines = [
        "Available domain skills (call load_skill(name) when entering a modality to load its checklist and unlock its tools):"
    ]
    for item in entries:
        lines.append(f"- {item['name']}: {item['description']}")
    return "\n".join(lines)
