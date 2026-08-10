"""Pre-evaluation gate pipeline.

Each gate has should_fire(state) and execute(state, messages) methods.
The pipeline runs gates in order and returns the first that fires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class AgentState:
    """Snapshot of agent state passed to gates (avoids coupling to EnzymeAgent)."""
    run_dir: Path
    candidate_count: int = 0
    lit_search_count: int = 0
    entity_extraction_done: bool = False
    literature_harvest_done: bool = False
    critique_injected: bool = False
    eval_gate_fired: bool = False
    ec_triage_done: bool = False
    homolog_expansion_done: bool = False
    evidence_selection_prompted: bool = False
    discovery_coverage_done: bool = False
    discovery_coverage_incomplete: bool = False
    discovery_coverage_redirected: bool = False
    accumulated_papers: list[dict] = field(default_factory=list)
    search_log: list[str] = field(default_factory=list)


@dataclass
class GateResult:
    """Result from a gate execution."""
    intercept: bool
    new_messages: list[dict] = field(default_factory=list)
    log_message: str = ""
    state_updates: dict = field(default_factory=dict)


class Gate(Protocol):
    def should_fire(self, state: AgentState, has_eval_calls: bool, messages: list[dict]) -> bool: ...
    def execute(self, state: AgentState, messages: list[dict]) -> GateResult: ...


class GatePipeline:
    """Runs gates in order, returning the first interception with accumulated state.

    A gate that declines to intercept can still report a state transition (for
    example, "discovery coverage is now satisfied").  Those updates must survive
    the pipeline, otherwise a one-time redirect flag can never be cleared and a
    later gate is permanently locked out of firing.
    """

    def __init__(self, gates: list[Gate]):
        self.gates = gates

    def check(
        self,
        state: AgentState,
        has_eval_calls: bool,
        messages: list[dict],
    ) -> GateResult | None:
        """Run gates in order and return the first interception, or the
        accumulated state updates from the gates that ran without intercepting.

        Returns None only when no gate fired and no state changed.
        """
        accumulated: dict = {}
        for gate in self.gates:
            if not gate.should_fire(state, has_eval_calls, messages):
                continue
            result = gate.execute(state, messages)
            accumulated.update(result.state_updates)
            if result.intercept:
                return GateResult(
                    intercept=True,
                    new_messages=result.new_messages,
                    log_message=result.log_message,
                    state_updates=accumulated,
                )
            # Apply immediately: a later gate's should_fire may depend on the
            # transition this gate just reported (coverage -> triage ordering).
            for key, value in result.state_updates.items():
                setattr(state, key, value)
        if accumulated:
            return GateResult(intercept=False, state_updates=accumulated)
        return None
