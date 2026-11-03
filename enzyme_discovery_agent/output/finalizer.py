"""Final output assembly for an enzyme-discovery run.

This module deliberately owns post-run persistence and report generation so
``EnzymeAgent`` can focus on the LLM/tool loop.
"""

from __future__ import annotations

import json
from collections import Counter
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from enzyme_discovery_agent import config
from enzyme_discovery_agent.gates.ec_triage import _auto_evaluate_all, auto_screen_candidates
from enzyme_discovery_agent.gates.discovery_coverage import assess_discovery_coverage
from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry
from enzyme_discovery_agent.models.evaluations import current_evaluations
from enzyme_discovery_agent.state_store import atomic_write_json, state_lock
from enzyme_discovery_agent.output.candidate_map_plot import render_candidate_map
from enzyme_discovery_agent.output.csv_generator import generate_candidates_csv
from enzyme_discovery_agent.output.exploration_summary import write_exploration_summary
from enzyme_discovery_agent.output.family_fit_plot import render_family_fit_plot
from enzyme_discovery_agent.output.report_generator import (
    generate_two_pass_explorative,
    generate_two_pass_report,
)
from enzyme_discovery_agent.output.structure_downloader import download_candidate_structures
from enzyme_discovery_agent.output.structural_modeling_handoff import write_structural_modeling_handoff
from enzyme_discovery_agent.tools.citation_verification import verify_report_citations
from enzyme_discovery_agent.run_layout import run_file

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
        for path in sorted((self.artifacts_dir / "candidates").glob("evaluate_candidate_*.json"),
                           key=lambda p: (p.stat().st_mtime_ns, p.name)):
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
                evaluations.append(result)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Skipping unreadable evaluation artifact %s: %s", path, exc)

        # Include registry recovery even when only part of the artifact set
        # survived. The later artifact for an accession remains authoritative.
        registry = load_candidate_registry(self.run_dir)
        artifact_accessions = {e.get("data", e).get("accession") for e in evaluations}
        for accession, entry in registry.get("candidates", {}).items():
            if accession in artifact_accessions:
                continue
            statuses = [s.get("status") for s in entry.get("statuses", [])]
            if "evaluated" not in statuses or (statuses and statuses[-1] == "rejected"):
                continue
            metadata = entry.get("metadata", {})
            validation = metadata.get("claim_validation")
            if validation and validation.get("promotion_eligible") is not True:
                continue
            recovered = {"accession": accession, "_recovered_from_registry": True}
            if isinstance(metadata.get("evaluation_data"), dict):
                recovered.update(metadata["evaluation_data"])
            keys = (
                "protein_name", "gene_names", "organism", "sequence_length", "ec_numbers",
                "catalytic_activities", "interpro_family", "interpro_family_id", "literature",
                "structural_data", "substrate_similarity", "reaction_chemistry", "strengths",
                "weaknesses", "critical_warnings", "is_swissprot",
                "expression_data",
            )
            recovered.update({key: metadata[key] for key in keys if key in metadata})
            evaluations.append({"status": "success", "summary": f"Recovered: {accession}", "data": recovered})
        if evaluations:
            self.log(f"  Recovered {len(evaluations)} evaluations from registry")
        return current_evaluations(self.run_dir, evaluations)

    def _resource_reason(self) -> str | None:
        check = getattr(self.llm, "budget_check", None)
        reason = check() if callable(check) else None
        return reason if isinstance(reason, str) else None

    def snapshot(self, *, query: str, reason: str) -> tuple[str, int]:
        """Write a budget checkpoint report using only already saved evidence."""
        from enzyme_discovery_agent.output.report_generator import (
            _deterministic_mechanistic_ranking, _write_deterministic_report,
            _family_relevance_partition, _family_rank_map,
        )
        from enzyme_discovery_agent.models.candidate_registry import discovery_status
        from enzyme_discovery_agent.models.evaluations import registered_hypothesis
        registry = load_candidate_registry(self.run_dir).get("candidates", {})

        def carries_evidence(evaluation):
            """Admit exactly what the live ranking admits.

            A candidate whose case rests on a registered hypothesis is still
            pending routine evaluation, so a status-only filter dropped every one
            of them here while the live refresh ranked them. The budget-stop
            report then overwrote a populated decision list with an empty one.
            """
            accession = evaluation.get("data", evaluation).get("accession")
            entry = registry.get(accession, {})
            return (discovery_status(entry) in {"evaluated", "ranked"}
                    or registered_hypothesis(self.run_dir, accession, entry.get("metadata") or {}))

        evaluations = [e for e in self._load_evaluations() if carries_evidence(e)]
        stopped, on_target, _ = _family_relevance_partition(self.run_dir, evaluations)
        ranking = _deterministic_mechanistic_ranking(
            evaluations, {}, family_rank_by_acc=_family_rank_map(stopped, on_target))
        _write_deterministic_report(self.run_dir, "report_interim.md", query, evaluations,
                                    ranking=ranking)
        path = run_file(self.run_dir, 'report_interim.md')
        text = path.read_text()
        text = text.replace(
            "This report was rendered directly from the completed evaluation artifacts because "
            "the report-model response was empty. Rankings and evidence are preserved; no "
            "composite score has been introduced.",
            "This interim report uses saved evaluations. Discovery remains resumable. "
            "Stop reason: " + reason + ".",
        )
        counts = Counter(discovery_status(entry) or "unknown" for entry in registry.values())
        pending = sum(counts.get(status, 0) for status in ("discovered", "screened", "evidence_pending"))
        disclosure = (
            f"**Checkpoint coverage:** {len(evaluations)} evaluated record(s) available for assessment; "
            f"{pending} discovered, screened, or evidence-pending records remain outside "
            "this ranking. Ranks and tiers describe only the evaluated subset and do "
            "not establish activity on the target or complete discovery coverage.\n\n"
        )
        marker = "## Ranked candidates\n"
        text = text.replace(marker, marker + "\n" + disclosure, 1) if marker in text else disclosure + text
        path.write_text(text)
        # Always provide a stable report artifact, with the incomplete status
        # preserved in its contents; no extra provider work after budget stop.
        atomic_write_json(self.run_dir / "agent_state" / "report_delivery.json", {
            "report": "report.md", "status": "incomplete", "reason": reason,
            "evaluated_records": len(evaluations), "pending_records": pending})
        (self.run_dir / "report.md").write_text(text)
        generate_candidates_csv(self.run_dir, evaluations, log_fn=self.log)
        discovery_file=self.run_dir/'discovery_candidates.csv'
        if discovery_file.exists():
            self.log(f"  Discovery modeling candidates: {discovery_file}")
        try:
            target = json.loads(run_file(self.run_dir, 'target_context.json').read_text())
        except (OSError, ValueError):
            target = {'query': query}
        write_structural_modeling_handoff(
            self.run_dir, evaluations, target,
            discovery_status_value='incomplete', coverage={'status': 'incomplete', 'reason': reason})
        from enzyme_discovery_agent.export_reasoning_trace import export_trace
        trace=export_trace(self.run_dir)
        assessments=sum(event['type']=='target_assessment_recorded' for event in trace['events'])
        self.log(f"  Scientific reasoning trace: {len(trace['events'])} events, "
                 f"{len(trace['examples'])} planned tests, {assessments} candidate assessments; "
                 f"source integrity {trace['source_files_intact']}")
        return text, len(evaluations)

    def _verify_confidence_report(self, report_path: Path) -> None:
        if self._resource_reason():
            return
        try:
            annotated, results = verify_report_citations(report_path.read_text(), self.llm)
            report_path.write_text(annotated)
            flagged = [result for result in results if result.get("supported") is not True]
            # Distinguish "nothing to check" from "checked N, all passed": a bare
            # "0 claims verified OK" reads as a silent no-op and hides whether the
            # verifier found no citations or genuinely cleared them all.
            if not results:
                self.log("  Citation verification: no verifiable citations found in report")
            elif flagged:
                self.log(f"  Citation verification: {len(flagged)}/{len(results)} claims flagged")
                (run_file(self.run_dir, 'citation_verification.json')).write_text(
                    json.dumps(results, indent=2)
                )
            else:
                self.log(f"  Citation verification: {len(results)}/{len(results)} claims verified OK")
            self._emit_event(
                "citation_verification",
                claims_checked=len(results), claims_flagged=len(flagged),
            )
        except (OSError, ValueError, TypeError) as exc:
            self._emit_event("citation_verification", error=str(exc))
            self.log(f"  Citation verification failed: {exc}", level="warning")

    @property
    def _finalization_status_path(self) -> Path:
        return run_file(self.run_dir, 'finalization_status.json')

    def _load_finalization_status(self) -> dict[str, Any]:
        try:
            if self._finalization_status_path.exists():
                return json.loads(self._finalization_status_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
        return {"stages": {}}

    def _emit_event(self, event_type: str, **fields: Any) -> None:
        """Mirror a finalization event into the structured run event stream.

        ``finalization_status.json`` is the resumable checkpoint; ``events.jsonl``
        is the append-only audit stream.  Emitting failures/degradations here too
        keeps the event stream a complete health record — a stage failure is
        queryable without cross-reading the status file.  Best-effort.
        """
        try:
            from enzyme_discovery_agent.models.event_log import append_event
            append_event(self.run_dir, event_type, **fields)
        except Exception:  # noqa: BLE001 - telemetry must never break finalize
            pass

    def _record_stage(
        self,
        stage: str,
        status: str,
        error: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Persist finalization progress so report delivery can be resumed safely."""
        with state_lock(self._finalization_status_path):
            payload = self._load_finalization_status()
            payload.setdefault("stages", {})[stage] = {
                "status": status,
                "updated_at": datetime.now().astimezone().isoformat(),
                **({"error": error} if error else {}),
                **({"detail": detail} if detail else {}),
            }
            atomic_write_json(self._finalization_status_path, payload)

    def _record_run_outcome(self, **fields: Any) -> None:
        """Record a top-level verdict so an empty run is not mistaken for a good one."""
        with state_lock(self._finalization_status_path):
            payload = self._load_finalization_status()
            payload["run_outcome"] = {
                "updated_at": datetime.now().astimezone().isoformat(),
                **fields,
            }
            atomic_write_json(self._finalization_status_path, payload)

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
        reason = self._resource_reason()
        if reason:
            self._record_stage(stage, "deferred", detail=reason)
            return None
        self._record_stage(stage, "running")
        previous_stat = (produces.stat().st_mtime_ns, produces.stat().st_size) if produces and produces.exists() else None
        try:
            result = operation()
        except Exception as exc:  # Continue producing independent deliverables.
            logger.exception("Finalization stage %s failed", stage)
            self._record_stage(stage, "failed", error=str(exc))
            self._emit_event("stage_failed", stage=stage, error=str(exc))
            self.log(f"  Finalization stage failed: {stage}: {exc}", level="warning")
            return None
        current_stat = (produces.stat().st_mtime_ns, produces.stat().st_size) if produces and produces.exists() else None
        if produces is not None and (current_stat is None or current_stat == previous_stat):
            self._record_stage(
                stage, "incomplete",
                detail=f"expected artifact not freshly written: {produces.name}",
            )
            self._emit_event(
                "stage_incomplete", stage=stage,
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
        (run_file(self.run_dir, 'discovery_coverage.json')).write_text(
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
        # Last-chance full-text named/cited-enzyme harvest, BUT only if the
        # literature-harvest gate never ran one during discovery -- otherwise this
        # is a duplicate pass over the same (already-read) paper pool that profiling
        # showed costs minutes for ~0 new candidates right before the report, with no
        # time left to evaluate anything it finds.  (The old deposit-accession regex
        # channel this block was written for has since been removed from the harvest,
        # so re-running it here no longer buys the thing the comment once claimed.)
        harvest_manifest = self.run_dir / "artifacts" / "fulltext" / "manifest.json"
        already_harvested = harvest_manifest.exists() and harvest_manifest.stat().st_size > 2
        if not already_harvested:
            try:
                from enzyme_discovery_agent.tools.fulltext import (
                    mine_and_register_fulltext_accessions,
                )
                mine_and_register_fulltext_accessions(
                    self.run_dir, query, log_fn=self.log, llm_client=self.llm
                )
            except Exception as exc:  # never let mining block finalization
                logger.warning("Full-text accession mining failed: %s", exc)
        reason = self._resource_reason()
        if reason:
            return self.snapshot(query=query, reason=reason)
        auto_screen_candidates(self.run_dir, log_fn=self.log)
        reason = self._resource_reason()
        if reason:
            return self.snapshot(query=query, reason=reason)
        _auto_evaluate_all(self.run_dir, log_fn=self.log, llm_client=self.llm)
        reason = self._resource_reason()
        if reason:
            return self.snapshot(query=query, reason=reason)
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

        self._run_finalization_stage(
            "confidence_report",
            lambda: generate_two_pass_report(
                self.run_dir, evaluations, query, self.llm, token_usage, self.log
            ),
            produces=self.run_dir / "report_confidence.md",
        )
        self._run_finalization_stage(
            "explorative_report",
            lambda: generate_two_pass_explorative(
                self.run_dir, evaluations, query, self.llm, token_usage, self.log
            ),
            produces=self.run_dir / "report_explorative.md",
        )

        confidence_path = self.run_dir / "report_confidence.md"
        confidence_delivered = self._load_finalization_status().get("stages", {}).get(
            "confidence_report", {}).get("status") == "completed"
        if confidence_path.exists() and confidence_delivered:
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
            produces=run_file(self.run_dir, 'exploration_summary.json'),
        )
        # Native-substrate-fit picture for experimentalists: a radar of per-axis
        # fit + native pocket-size ranges for the top families (best-effort).
        self._run_finalization_stage(
            "family_fit_plot",
            lambda: render_family_fit_plot(self.run_dir, log_fn=self.log),
            produces=self.run_dir / "family_substrate_fit.png",
        )
        # Candidate map in ESM embedding space (best-effort; only when the
        # agent-invoked embedding map was enabled for this run).
        from enzyme_discovery_agent.models.embedding_store import load_store
        if len(load_store(self.run_dir)) >= 3:
            from enzyme_discovery_agent.output.evolving_map import update_candidate_map
            self._run_finalization_stage(
                "candidate_map_plot",
                lambda: update_candidate_map(self.run_dir, log_fn=self.log),
                produces=self.run_dir / "candidate_map.png",
            )
        self._run_finalization_stage(
            "candidates_csv",
            lambda: generate_candidates_csv(self.run_dir, evaluations, log_fn=self.log),
            produces=self.run_dir / "ranked_candidates.csv",
        )
        # ESM neighbors of promising candidates feed the cofold shortlist.
        neighbor_map = None
        try:
            neighbor_map = json.loads((run_file(self.run_dir, 'candidate_map.json')).read_text())
        except (OSError, json.JSONDecodeError):
            neighbor_map = None
        handoff_path = self._run_finalization_stage(
            "structural_modeling_handoff",
            lambda: write_structural_modeling_handoff(
                self.run_dir,
                evaluations,
                target=target_context or {"query": query},
                neighbor_map=neighbor_map,
                discovery_status_value="completed" if coverage.get("covered") and not force_completion else "incomplete",
                coverage=coverage,
            ),
            produces=self.run_dir / "structural_modeling_handoff.json",
        )
        if handoff_path:
            self.log(f"  Structural-modeling handoff: {handoff_path}")
        from enzyme_discovery_agent.export_reasoning_trace import export_trace
        self._run_finalization_stage(
            'scientific_reasoning_trace',lambda:export_trace(self.run_dir),
            produces=self.run_dir/'reasoning/trace.json')

        # Written LAST, after the report stages above have recorded their own LLM
        # calls into ``token_usage``.  Writing it earlier undercounts the run: the
        # confidence/explorative report generation is a substantial fraction of
        # total tokens, and an early snapshot disagrees with the ``run_complete``
        # event and ``manifest.json`` (both taken after finalize returns).
        atomic_write_json(run_file(self.run_dir, 'token_usage.json'), {
            **token_usage.to_dict(), "request_counts": getattr(self.llm, "request_counts", {}),
        })

        # A run that evaluated nothing produced no deliverable, whatever the
        # per-stage bookkeeping says.  Record that verdict explicitly.
        stages = self._load_finalization_status().get("stages", {})
        delivered = confidence_path.exists() and confidence_delivered
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
            self._emit_event(
                "run_degraded", outcome=outcome, evaluations=len(evaluations),
            )
            self.log(f"  Run outcome: {outcome} ({len(evaluations)} evaluations)", level="warning")

        return (confidence_path.read_text() if evaluations and delivered else final_text, len(evaluations))
