"""Enzyme Design Agent — generative protein design, as distinct from discovery.

Where discovery *retrieves* an existing protein and asks whether it can
accommodate the target chemistry, this agent decides what to *build* and builds
it. Two strategies, and choosing between them is the agent's job:

  generative  -- construct a backbone around a catalytic motif, when no
                 scaffold holds the required geometry
  redesign    -- saturate the pocket of a scaffold that already binds, when one
                 exists in a pose worth believing

Infrastructure (``config``, ``tool_registry``, ``llm_client``, run-dir
conventions) is shared with ``enzyme_discovery_agent`` rather than forked, so a
single run can reach both toolkits. See README.md.
"""

__all__ = ["EnzymeDesignAgent", "register_design_tools"]


def __getattr__(name):
    # Lazy, so `import enzyme_design_agent` stays cheap for tool-only users and
    # does not drag in the LLM client.
    if name == "EnzymeDesignAgent":
        from .agent import EnzymeDesignAgent

        return EnzymeDesignAgent
    if name == "register_design_tools":
        from .bootstrap import register_design_tools

        return register_design_tools
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
