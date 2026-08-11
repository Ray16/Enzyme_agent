"""Final output assembly for an enzyme-discovery run.

This module deliberately owns post-run persistence and report generation so
``EnzymeAgent`` can focus on the LLM/tool loop.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from enzyme_discovery_agent.gates.ec_triage import _auto_evaluate_all, auto_screen_candidates
from enzyme_discovery_agent.gates.discovery_coverage import assess_discovery_coverage
from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry
from enzyme_discovery_agent.output.csv_generator import generate_candidates_csv
from enzyme_discovery_agent.output.exploration_summary import write_exploration_summary
from enzyme_discovery_agent.output.report_generator import (
    generate_two_pass_explorative,
    generate_two_pass_report,
)
from enzyme_discovery_agent.output.structure_downloader import download_candidate_structures
from enzyme_discovery_agent.output.structural_modeling_handoff import write_structural_modeling_handoff
from enzyme_discovery_agent.tools.citation_verification import verify_report_citations

logger = logging.getLogger("enzyme_agent")


class RunFinalizer:
    """Persist a completed run and produce its reports and deliverables."""

    def __init__(self, run_dir: Path, llm: Any, log: Callable[[str], None]):
        self.run_dir = run_dir
        self.artifacts_dir = run_dir / "artifacts"
        self.llm = llm
        self.log = log

    def _load_evaluations(self) -> list[dict]:
        evaluations = []
        for path in sorted((self.artifacts_dir / "candidates").glob("evaluate_candidate_*.json")):
            try:
                with path.open() as handle:
                    artifact = json.load(handle)
                result = artifact.get("result", artifact)
                data = result.get("data", result)
                # Drop failed evaluations (e.g. "could not fetch UniProt entry"):
                # their data is None, they are not candidates, and the report /
                # CSV / enrichment consumers all assume a dict here.
                if not isinstance(data, dict):
                    continue
                # Exclude non-enzymes (regulators/transporters that co-occur in the
                # same gene clusters); they are not candidates and must not appear
                # as families in the report.
                if data.get("is_enzyme") is False:
                    continue
                evaluations.append(result)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Skipping unreadable evaluation artifact %s: %s", path, exc)

        if evaluations:
            return evaluations

        registry = load_candidate_registry(self.run_dir)
        for accession, entry in registry.get("candidates", {}).items():
            if "evaluated" not in [s.get("status") for s in entry.get("statuses", [])]:
                continue
            metadata = entry.get("metadata", {})
            recovered = {"accession": accession, "_recovered_from_registry": True}
            keys = (
                "protein_name", "gene_names", "organism", "sequence_length", "ec_numbers",
                "catalytic_activities", "interpro_family", "interpro_family_id", "literature",
                "structural_data", "substrate_similarity", "reaction_chemistry", "strengths",
                "weaknesses", "critical_warnings", "is_swissprot",
            )
            recovered.update({key: metadata[key] for key in keys if key in metadata})
            evaluations.append({"status": "success", "summary": f"Recovered: {accession}", "data": recovered})
        if evaluations:
            self.log(f"  Recovered {len(evaluations)} evaluations from registry")
        return evaluations

    def _verify_confidence_report(self, report_path: Path) -> None:
        try:
            annotated, results = verify_report_citations(report_path.read_text(), self.llm)
            report_path.write_text(annotated)
            flagged = [result for result in results if result.get("supported") is not True]
            if flagged:
                self.log(f"  Citation verification: {len(flagged)}/{len(results)} claims flagged")
                (self.run_dir / "citation_verification.json").write_text(
                    json.dumps(results, indent=2)
                )
            else:
                self.log(f"  Citation verification: {len(results)} claims verified OK")
        except (OSError, ValueError, TypeError) as exc:
            self.log(f"  Citation verification failed: {exc}")

    @property
    def _finalization_status_path(self) -> Path:
        return self.run_dir / "finalization_status.json"

    def _load_finalization_status(self) -> dict[str, Any]:
        try:
            if self._finalization_status_path.exists():
                return json.loads(self._finalization_status_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
        return {"stages": {}}

    def _record_stage(
        self,
        stage: str,
        status: str,
        error: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Persist finalization progress so report delivery can be resumed safely."""
        payload = self._load_finalization_status()
        payload.setdefault("stages", {})[stage] = {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **({"error": error} if error else {}),
            **({"detail": detail} if detail else {}),
        }
        self._finalization_status_path.write_text(json.dumps(payload, indent=2) + "\n")

    def _record_run_outcome(self, **fields: Any) -> None:
        """Record a top-level verdict so an empty run is not mistaken for a good one."""
        payload = self._load_finalization_status()
        payload["run_outcome"] = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        self._finalization_status_path.write_text(json.dumps(payload, indent=2) + "\n")

    def _run_finalization_stage(
        self,
        stage: str,
        operation: Callable[[], Any],
        produces: Path | None = None,
    ) -> Any:
        """Log and checkpoint one independently recoverable finalization stage.

        ``produces`` names the artifact the stage is expected to leave behind.
        A stage that returns without raising but without writing its artifact is
        recorded as ``incomplete`` rather than ``completed``: a report skipped
        for lack of evaluations is not a delivered report, and marking it
        completed makes a failed run look successful.
        """
        self.log(f"  Finalization stage: {stage}")
        self._record_stage(stage, "running")
        try:
            result = operation()
        except Exception as exc:  # Continue producing independent deliverables.
            logger.exception("Finalization stage %s failed", stage)
            self._record_stage(stage, "failed", error=str(exc))
            self.log(f"  Finalization stage failed: {stage}: {exc}", level="warning")
            return None
        if produces is not None and not produces.exists():
            self._record_stage(
                stage, "incomplete",
                detail=f"expected artifact not written: {produces.name}",
            )
            self.log(
                f"  Finalization stage produced no output: {stage} ({produces.name})",
                level="warning",
            )
            return result
        self._record_stage(stage, "completed")
        return result

    def finalize(
        self,
        messages: list[dict],
        token_usage: Any,
        query: str,
        final_text: str,
        target_context: dict | None = None,
        force_completion: bool = False,
    ) -> tuple[str, int]:
        """Write durable artifacts and return the final user-facing report and count."""
        # Import lazily: report finalization should not implicitly register tools.
        from enzyme_discovery_agent.tools.literature import cleanup_rerank_model

        cleanup_rerank_model()
        self.log("\n  FINALIZING RUN")
        try:
            (self.run_dir / "agent_state" / "messages.json").write_text(
                json.dumps(messages, indent=2, default=str)
            )
        except (OSError, TypeError) as exc:
            logger.warning("Failed to save messages: %s", exc)

        coverage = assess_discovery_coverage(self.run_dir)
        (self.run_dir / "discovery_coverage.json").write_text(
            json.dumps(coverage, indent=2, default=str) + "\n"
        )
        # Finalization is the LAST backstop — there is no later turn — so always
        # evaluate the screened pool.  A run that discovered and screened
        # candidates must never emit an empty report merely because coverage is
        # incomplete; the gaps are recorded as report limitations instead.
        if not coverage["covered"]:
            self.log(
                "  Discovery coverage incomplete; evaluating the screened pool anyway "
                "and recording the gaps as report limitations: "
                + "; ".join(coverage["missing"]),
                level="warning",
            )
        # Mine deposited sequence accessions from the top papers' FULL TEXT and
        # register them as candidates.  Deterministic (regex + REST, no full text
        # in the LLM context), so it adds no prompt tokens; it is the only route
        # to characterized enzymes that carry no gene name and only a generic
        # protein annotation, reachable solely via the GenBank accession
        # deposited in their paper.  Runs BEFORE screen so the resolved
        # candidates are screened and evaluated with the rest.
        try:
            from enzyme_discovery_agent.tools.fulltext import (
                mine_and_register_fulltext_accessions,
            )
            mine_and_register_fulltext_accessions(
                self.run_dir, query, log_fn=self.log, llm_client=self.llm
            )
        except Exception as exc:  # never let mining block finalization
            logger.warning("Full-text accession mining failed: %s", exc)
        auto_screen_candidates(self.run_dir, log_fn=self.log)
        _auto_evaluate_all(self.run_dir, log_fn=self.log)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        candidate_dir = self.artifacts_dir / "candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        evaluations = self._load_evaluations()
        # Write a provenance file per evaluated candidate, and for fact-poor leads
        # that carry a paper link, recover their native substrates + mechanism
        # from the paper's full text (curation misses them).  Mutates the
        # evaluations in place so ranking sees the recovered substrates.  Fully
        # defensive — never blocks finalization.
        try:
            from enzyme_discovery_agent.models.candidate_provenance import (
                enrich_leads_and_write_provenance,
            )
            enrich_leads_and_write_provenance(
                self.run_dir, evaluations, query, self.llm, log_fn=self.log
            )
        except Exception as exc:
            logger.warning("Provenance/enrichment step failed: %s", exc)
        # Categorise the harvested full text by enzyme family (papers_enzymes.csv
        # + families_index.json), now that the registry carries each candidate's
        # family.  Fully defensive — never blocks finalization.
        try:
            from enzyme_discovery_agent.tools.fulltext import write_fulltext_family_index
            write_fulltext_family_index(self.run_dir, log_fn=self.log)
        except Exception as exc:
            logger.warning("Full-text family index failed: %s", exc)
        if evaluations:
            (candidate_dir / "all_evaluations.json").write_text(
                json.dumps(evaluations, indent=2, default=str)
            )
        (self.run_dir / "token_usage.json").write_text(json.dumps(token_usage.to_dict(), indent=2))

        self._run_finalization_stage(
            "confidence_report",
            lambda: generate_two_pass_report(
                self.run_dir, evaluations, query, self.llm, token_usage, self.log
            ),
            produces=self.run_dir / "report_confidence.md",
        )
        self._run_finalization_stage(
            "explorative_report",
            lambda: generate_report(
                "explorative", self.run_dir, evaluations, query, self.llm, token_usage, self.log
            ),
            produces=self.run_dir / "report_explorative.md",
        )

        confidence_path = self.run_dir / "report_confidence.md"
        if confidence_path.exists():
            self._run_finalization_stage(
                "citation_verification",
                lambda: self._verify_confidence_report(confidence_path),
            )
            self._run_finalization_stage(
                "structure_downloads",
                lambda: download_candidate_structures(
                    self.run_dir, confidence_path.read_text(), self.artifacts_dir, self.log
                ),
            )
        self._run_finalization_stage(
            "exploration_summary",
            lambda: write_exploration_summary(self.run_dir, log_fn=self.log),
            produces=self.run_dir / "exploration_summary.json",
        )
        self._run_finalization_stage(
            "candidates_csv",
            lambda: generate_candidates_csv(self.run_dir, evaluations, log_fn=self.log),
            produces=self.run_dir / "ranked_candidates.csv",
        )
        handoff_path = self._run_finalization_stage(
            "structural_modeling_handoff",
            lambda: write_structural_modeling_handoff(
                self.run_dir,
                evaluations,
                target=target_context or {"query": query},
            ),
            produces=self.run_dir / "structural_modeling_handoff.json",
        )
        if handoff_path:
            self.log(f"  Structural-modeling handoff: {handoff_path}")

        # A run that evaluated nothing produced no deliverable, whatever the
        # per-stage bookkeeping says.  Record that verdict explicitly.
        stages = self._load_finalization_status().get("stages", {})
        delivered = confidence_path.exists()
        if not evaluations:
            outcome = "no_candidates_evaluated"
        elif not delivered:
            outcome = "evaluated_but_no_report"
        else:
            outcome = "reports_delivered"
        self._record_run_outcome(
            outcome=outcome,
            evaluations=len(evaluations),
            discovery_coverage_met=coverage["covered"],
            coverage_gaps=coverage["missing"],
            forced_completion=force_completion,
            stages_not_completed=sorted(
                name for name, info in stages.items()
                if info.get("status") != "completed"
            ),
        )
        if outcome != "reports_delivered":
            self.log(f"  Run outcome: {outcome} ({len(evaluations)} evaluations)", level="warning")

        return (confidence_path.read_text() if confidence_path.exists() else final_text, len(evaluations))
