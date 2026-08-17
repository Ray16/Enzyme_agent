"""Research notebook: persistent working memory across agent iterations.

Provides both the Notebook data model and the @tool-decorated functions
(update_notebook, get_notebook) that the LLM calls.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from enzyme_discovery_agent.tool_registry import tool

logger = logging.getLogger("enzyme_agent")

NOTEBOOK_SECTIONS = [
    "reaction_analysis",
    "enzyme_landscape",
    "candidates",
    "evaluation_notes",
    "structural_analysis",
    "decisions",
    "final_ranking",
]

_DECISION_SECTIONS = frozenset({"decisions", "candidates", "final_ranking"})


def load_notebook(run_dir: Path) -> dict:
    """Load or initialize the notebook from run_dir/notebook.json."""
    path = run_dir / "notebook.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    now = datetime.now().astimezone().isoformat()
    return {
        "sections": {s: "" for s in NOTEBOOK_SECTIONS},
        "created_at": now,
        "last_updated": now,
        "revision_log": [],
    }


def save_notebook(run_dir: Path, notebook: dict) -> None:
    """Persist the notebook to run_dir/notebook.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "notebook.json", "w") as f:
        json.dump(notebook, f, indent=2)


def get_notebook_text_for_report(run_dir: Path) -> str:
    """Load the notebook and return formatted text for report generation.

    This consolidates the duplicated notebook-loading logic that was
    in both _generate_explorative_report and _generate_confidence_report.
    """
    notebook_path = run_dir / "notebook.json"
    if not notebook_path.exists():
        return ""
    try:
        with open(notebook_path) as f:
            nb = json.load(f)
        sections = nb.get("sections", {})
        if not isinstance(sections, dict):
            return ""
        text = ""
        for section_name, content in sections.items():
            if content:
                text += f"\n### {section_name}\n{content}\n"
        return text
    except Exception:
        return ""


@tool(
    description=(
        "Write synthesized analysis into a notebook section. Use this to persist "
        "key insights, decisions, and candidate data across iterations. Write your "
        "OWN markdown analysis — not raw data dumps."
    ),
    category="reasoning",
    parallel_safe=False,
    injected_params=["run_dir"],
    params={
        "run_dir": {"type": "string", "description": "Agent run directory (injected)."},
        "section": {
            "type": "string",
            "description": "Notebook section to write to.",
            "enum": NOTEBOOK_SECTIONS,
        },
        "content": {
            "type": "string",
            "description": "Optional Markdown analysis to write. Supply non-empty content only when you have a concise, evidence-backed note to persist.",
            "default": "",
        },
        "append": {
            "type": "boolean",
            "description": "If true, append with timestamp separator. If false, replace.",
            "default": True,
        },
    },
)
def update_notebook(
    run_dir: Path | str,
    section: str,
    content: str = "",
    append: bool = True,
) -> dict:
    """Write synthesized analysis into a notebook section."""
    from enzyme_discovery_agent.models.exploration_log import append_exploration_log
    from enzyme_discovery_agent.models.reasoning_trace import record_trace_event

    run_dir = Path(run_dir)

    if section not in NOTEBOOK_SECTIONS:
        return {
            "status": "error",
            "summary": f"Unknown section '{section}'. Valid: {NOTEBOOK_SECTIONS}",
            "data": None,
        }

    # Some providers occasionally emit a section-only notebook call.  It is
    # not a discovery failure and should neither write an empty section nor
    # consume the run's error budget.
    if not content.strip():
        return {
            "status": "partial",
            "summary": "Notebook update skipped: no content supplied; no state changed.",
            "data": {"section": section, "written": False},
        }

    notebook = load_notebook(run_dir)
    now = datetime.now().astimezone().isoformat()

    if not append and notebook["sections"][section]:
        old_content = notebook["sections"][section]
        append_exploration_log(
            run_dir,
            entry_type="NOTEBOOK_ARCHIVE",
            title=f"Section '{section}' replaced",
            details={"archived_chars": len(old_content), "preview": old_content[:300]},
        )

    if append and notebook["sections"][section]:
        notebook["sections"][section] += f"\n\n---\n*[{now}]*\n\n{content}"
    else:
        notebook["sections"][section] = content

    notebook["last_updated"] = now
    notebook["revision_log"].append({
        "timestamp": now,
        "section": section,
        "action": "append" if append else "replace",
    })

    save_notebook(run_dir, notebook)

    if section in _DECISION_SECTIONS:
        record_trace_event(
            run_dir=run_dir,
            stage="selection" if section == "final_ranking" else "candidate_screening",
            event_type="decision_made",
            data={"section": section, "content_snippet": content[:500]},
        )

    section_lengths = {s: len(v) for s, v in notebook["sections"].items()}
    return {
        "status": "success",
        "summary": f"Updated notebook section '{section}' ({len(content)} chars)",
        "data": {"section": section, "section_lengths": section_lengths},
    }


@tool(
    description=(
        "Recall earlier analysis from the research notebook. Use this before making "
        "decisions to review accumulated insights rather than relying on context that "
        "may have been compressed."
    ),
    category="reasoning",
    parallel_safe=True,
    injected_params=["run_dir"],
    params={
        "run_dir": {"type": "string", "description": "Agent run directory (injected)."},
        "sections": {
            "type": "array",
            "items": {"type": "string"},
            "description": f"Sections to retrieve (default: all). Options: {NOTEBOOK_SECTIONS}",
            "default": [],
        },
    },
)
def get_notebook(
    run_dir: Path | str,
    sections: list[str] | None = None,
) -> dict:
    """Recall earlier analysis from the research notebook."""
    run_dir = Path(run_dir)
    notebook = load_notebook(run_dir)

    if sections is not None and len(sections) == 0:
        sections = None

    if sections is not None:
        invalid = [s for s in sections if s not in NOTEBOOK_SECTIONS]
        if invalid:
            return {
                "status": "error",
                "summary": f"Unknown sections: {invalid}. Valid: {NOTEBOOK_SECTIONS}",
                "data": None,
            }
        content = {s: notebook["sections"][s] for s in sections}
    else:
        content = notebook["sections"]

    section_lengths = {s: len(v) for s, v in notebook["sections"].items()}
    return {
        "status": "success",
        "summary": f"Notebook retrieved ({sum(section_lengths.values())} total chars)",
        "data": {
            "sections": content,
            "section_lengths": section_lengths,
            "last_updated": notebook["last_updated"],
        },
    }
