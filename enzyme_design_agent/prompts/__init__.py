"""Assembly of the design agent's prompts."""
from __future__ import annotations

from .sections import (
    ANTI_PATTERNS, EVIDENCE_GUIDANCE, MOTIF_GUIDANCE, POCKET_GUIDANCE,
    REPORT_FORMAT, RESOURCE_GUIDANCE, ROLE, SATURATION_GUIDANCE, STRATEGY_CHOICE,
)

_SECTIONS = (
    ROLE,
    STRATEGY_CHOICE,
    MOTIF_GUIDANCE,
    POCKET_GUIDANCE,
    SATURATION_GUIDANCE,
    EVIDENCE_GUIDANCE,
    RESOURCE_GUIDANCE,
    ANTI_PATTERNS,
    REPORT_FORMAT,
)


def build_system_prompt(available_tools: list[str] | None = None,
                        capabilities: dict[str, bool] | None = None) -> str:
    """The design agent's standing instructions.

    ``available_tools`` is appended as a plain list, not as schemas: the full
    schemas reach the model through the API's tool array, and repeating them in
    the prompt only invites the two copies to drift. The list is here so the
    strategy section can be read against what is actually callable right now --
    an agent told to choose between two strategies should be able to see when
    one of them is missing.
    """
    parts = list(_SECTIONS)
    if available_tools:
        parts.append(
            "=== TOOLS AVAILABLE THIS RUN ===\n\n"
            + "\n".join(f"  - {t}" for t in sorted(available_tools))
        )
    caps = capabilities or {}
    unavailable = sorted(k for k, v in caps.items() if not v)
    if unavailable:
        parts.append(
            "=== NOT AVAILABLE ON THIS HOST ===\n\n"
            + "\n".join(f"  - {k}" for k in unavailable)
            + "\n\nPlan around these rather than attempting them. If a strategy "
              "needs one, say so instead of substituting a weaker step silently."
        )
    return "\n\n".join(parts)


def build_state_header(iteration: int, max_iterations: int,
                       strategy: str = "undecided",
                       designs: int = 0,
                       tools_used: list[str] | None = None) -> str:
    """One compact line of run state, refreshed each turn."""
    recent = ", ".join((tools_used or [])[-8:])
    return (
        f"[STATE: iter={iteration}/{max_iterations}, strategy={strategy}, "
        f"designs={designs}, recent_tools=[{recent}]]"
    )


REFLECTION_PROMPT = """\
Pause and check yourself before the next call:

  - Is the strategy you chose still the right one given what you have seen? If
    the evidence has turned against it, switch and say why.
  - Is your motif or mutable set justified residue by residue, or has something
    crept in unexamined?
  - What is the strongest argument that your current best design is wrong?
  - Are your negative controls still in the run?

Answer briefly, then continue.\
"""

FINALIZE_PROMPT = """\
Stop designing and write the final design report now, in the required format.
Report what you actually ran, including what is unfinished. Do not start new
work. If the campaign did not produce a design you trust, say that plainly --
it is a valid outcome.\
"""

__all__ = [
    "build_system_prompt", "build_state_header",
    "REFLECTION_PROMPT", "FINALIZE_PROMPT",
]
