"""OpenAI-compatible tool JSON schemas exposed by ModelingAgent."""
from __future__ import annotations


def _schema(name, description, properties=None, required=()):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


def _string(description):
    return {"type": "string", "description": description}


def build_core_schemas():
    from ..barrier.api import tool_schemas as barrier_schemas
    from ..sampling.setup import declaration_schema
    from enzyme_discovery_agent.tools import structural_proxy_planning
    from enzyme_discovery_agent.tool_registry import get_tool_info

    acc = {"accession": _string("Candidate identifier from this task")}
    source_schema = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": [
                    "primary_literature",
                    "experimental_structure",
                    "solution_measurement",
                    "curated_annotation",
                ],
            },
            **{
                key: _string(description)
                for key, description in {
                    "source_id": "Stable source identifier",
                    "path": "Authorized local source path",
                    "sha256": "Actual source file SHA-256",
                    "quote": "Exact supporting passage read from the source",
                    "rationale": "Interpret the passage and its limitations",
                }.items()
            },
            "reviewed": {"type": "boolean"},
            "derived_from_prediction": {"type": "boolean"},
        },
        "required": ["kind", "source_id", "path", "sha256", "quote", "rationale", "reviewed"],
    }
    test_schema = {
        "type": "object",
        "properties": {
            **{
                key: _string(description)
                for key, description in {
                    "measurement": "Discriminating measurement",
                    "expected_if_selected": "Outcome under the selected hypothesis",
                    "expected_if_alternative": "Different outcome under the competing explanation",
                }.items()
            },
            "controls": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        },
        "required": ["measurement", "expected_if_selected", "expected_if_alternative", "controls"],
    }
    observation_schema = {
        "type": "object",
        "properties": {
            **{
                key: _string(description)
                for key, description in {
                    "id": "New observation ID",
                    "metric": "Exact predicted observable",
                    "unit": "Exact measurement units",
                    "endpoint": "Exact assay/structural endpoint",
                    "interpretation": "Explain the source-grounded measurement and uncertainty",
                    "transfer_rationale": "Required for structural candidate applicability",
                }.items()
            },
            "scope": {"type": "string", "enum": ["structural", "experimental"]},
            "conditions": {"type": "object"},
            "value": {"type": "number"},
            "measurement_interval": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            "applicability": {"type": "string", "enum": ["model", "candidate"]},
            "source": source_schema,
        },
        "required": ["id", "metric", "scope", "unit", "endpoint", "conditions", "value", "interpretation", "source"],
    }
    shared = [
        get_tool_info(name)["schema"]
        for name in (
            "get_structural_proxy_guidance",
            "validate_structural_proxy_plan",
            "inspect_experimental_proxy_bundle",
        )
    ]
    return shared + barrier_schemas() + [
        _schema("inspect_result_locations", "Find this system/run's output destinations, current revisions and recorded result paths; folder existence is not completion", acc),
        _schema("inspect_workflow_policy", "Inspect default cofolding and the distinct scientific purposes of optional dynamics and de novo design", acc, ("accession",)),
        _schema("inspect_mlmm_engine_options", "Inspect reviewed UMA/Nitro, precision and embedding options and the separate evidence needed before a chemical-energetics run; no model loading"),
        _schema("record_preparation_findings", "Persist compact scientific working memory with verified source references and the next concrete action; replaces current notes, history is retained", {
            **acc,
            "summary": _string("Evidence-based findings, unresolved alternatives and decisions; maximum 6000 characters"),
            "sources": {"type": "array", "items": {"type": "object"}},
            "next_action": _string("Next discriminating preparation or investigation operation"),
        }, ("accession", "summary", "sources", "next_action")),
        _schema("inspect_preparation_contract", "Inspect the preparation/validation loop and alternative-state evidence contract"),
        _schema("inspect_preparation_structure", "Inspect an authorized single-model PDB for hydrogens, waters and metal neighbors; observations only", {"path": _string("Authorized PDB path")}, ("path",)),
        _schema("declare_preparation_study", "Record preparation alternatives, measured validation, failures and concrete next actions", {
            **acc,
            "study": {
                "type": "object",
                "description": "Preparation study object (see inspect_preparation_contract); includes variants referencing declared setup_id and stage measurements",
                "properties": {
                    "variants": {"type": "array", "items": {"type": "object"}},
                    "next_actions": {"type": "array", "items": {"type": "string"}},
                },
            },
        }, ("accession", "study")),
        _schema("inspect_preparation_study", "Recheck latest preparation evidence and upstream dependencies", acc, ("accession",)),
        _schema("run_preparation_operator", "Run a short CPU Python preparation or diagnostic operator; saves code, inputs, logs and outputs for review. Not a GPU launcher.", {
            **acc,
            "python": _string("Absolute validated Python interpreter"),
            "code": _string("Explicit Python source; opens each declared input at inputs/<name> and writes the declared outputs in the working directory"),
            "inputs": {
                "type": "array",
                "description": "Hash-verified evidence files, each exposed to the script as inputs/<name>",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": _string("Authorized absolute evidence path"),
                        "sha256": _string("Expected digest of that file"),
                        "name": _string("Local file name under inputs/; defaults to the path basename"),
                    },
                    "required": ["path", "sha256"],
                },
            },
            "outputs": {"type": "array", "items": {"type": "string"}},
            "seconds": {"type": "integer", "minimum": 1, "maximum": 60},
        }, ("accession", "python", "code", "inputs", "outputs", "seconds")),
        _schema("inspect_sampling_setup_contract", "Inspect the evidence ledger for agent-selected MD/enhanced-sampling protocols; supplies questions, no chemical defaults"),
        _schema("inspect_processivity_toolchain", "Inspect configured MD, enhanced sampling, tunnel, QM/MM and ML/MM tools and exact validation scope; no model loading or submission"),
        _schema("inspect_processivity_contract", "Inspect the transport/chemistry/release network contract and limits of processivity claims"),
        _schema("assess_processivity_network", "Evaluate a source-grounded kinetic network without filling unknown rates or counting release/rebinding as continuation", {
            **acc,
            "network": {"type": "object", "description": "processivity-network-1.0 declaration with exact sequence_sha256 and target_sha256"},
        }, ("accession", "network")),
        _schema("declare_sampling_setup", "Save candidate-specific MD setup decisions, alternatives, source evidence and pending questions; does not submit compute", {
            **acc,
            "setup": declaration_schema(),
        }, ("accession", "setup")),
        _schema("inspect_sampling_setup", "Recheck an agent-selected sampling setup against current evidence and model context; paginate the full declaration", {
            **acc,
            "setup_id": _string("Saved setup ID; omit for latest"),
            "offset": {"type": "integer", "minimum": 0},
        }, ("accession",)),
        _schema("plan_followup", "Record an evidence-grounded optional follow-up after primary cofold inspection", {
            **acc,
            "plan": {
                "type": "object",
                "properties": {
                    "purpose": {"type": "string", "enum": ["dynamics", "de_novo_design"]},
                    **{
                        key: _string(description)
                        for key, description in {
                            "question": "Scientific question motivating the follow-up",
                            "why_cofold_insufficient": "Why primary cofold evidence cannot answer it",
                            "expected_information": "What the tool could distinguish",
                            "decision_if_supported": "What would change if supported",
                            "decision_if_refuted": "What would change if refuted",
                            "assessment_sha256": "Current inspect_reasoning assessment fingerprint",
                        }.items()
                    },
                    "hypothesis_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "num_samples": {"type": "integer", "minimum": 1},
                },
                "required": [
                    "purpose", "question", "why_cofold_insufficient", "expected_information",
                    "decision_if_supported", "decision_if_refuted", "assessment_sha256", "hypothesis_ids",
                ],
            },
        }, ("accession", "plan")),
        _schema("compare_hypothesis_tests", "Compare competing predictions and identify discriminating readouts, unresolved pairs and required controls", acc, ("accession",)),
        _schema("inspect_reasoning", "Inspect current hypotheses, prediction comparisons, contradictions and feedback fingerprint", acc, ("accession",)),
        _schema("read_reasoning_section", "Paginate full reasoning details omitted from compact batch context", {
            **acc,
            "section": {"type": "string", "enum": ["scientific_reasoning", "assessment", "observations", "response", "followups"]},
            "offset": {"type": "integer", "minimum": 0},
        }, ("accession", "section")),
        _schema("record_reasoning_observation", "Append a source-grounded numerical observation; never overwrite contradictory evidence", {
            **acc,
            "observation": observation_schema,
        }, ("accession", "observation")),
        _schema("respond_to_reasoning_feedback", "Acknowledge current evidence and choose an action with a discriminating next test", {
            **acc,
            "assessment_sha256": _string("Fingerprint from inspect_reasoning"),
            "action": {"type": "string", "enum": ["retain", "revise", "investigate", "abstain"]},
            "rationale": _string("Interpret contradictions, alternatives and remaining uncertainty"),
            "addressed_hypotheses": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "next_test": test_schema,
        }, ("accession", "assessment_sha256", "action", "rationale", "addressed_hypotheses", "next_test")),
        _schema("declare_proxy_plan", "Save a validated proxy hypothesis plan before extracting its measurements", {
            **acc,
            "plan": {
                "type": "object",
                "description": "Reviewed proxy plan (see get_structural_proxy_guidance); requires mechanism, reaction_step, assay_endpoint, context, and hypotheses",
                "properties": {
                    "mechanism": _string("Exact mechanism matching selected hypothesis"),
                    "reaction_step": _string("Exact reaction step description"),
                    "assay_endpoint": _string("Assay endpoint being modeled"),
                    "context": {"type": "object"},
                    "hypotheses": {"type": "array", "items": {"type": "object"}},
                },
            },
        }, ("accession", "plan")),
        _schema("evaluate_proxy_plan", "Evaluate the declared proxy plan on the current inspected prediction files; no activity labels or fitting", acc, ("accession",)),
        _schema("inspect_candidate", "Inspect candidate evidence, sequence and current modeling decisions", acc, ("accession",)),
        _schema("review_existing_predictions", "Verify an original shared-pool prediction report and return its measurements without a new model or compute submission", {
            **acc,
            "setup_id": _string("Exact setup_id from the supplied existing prediction index"),
            "applicability": {"type": "string", "enum": ["same_molecular_question", "limited_transfer", "unsupported_transfer"]},
            "interpretation": _string("Explain correspondence to the requested inputs and step, and the limits of the original model"),
            "unresolved": {"type": "array", "items": {"type": "string"}},
        }, ("accession", "setup_id", "applicability", "interpretation", "unresolved")),
        _schema("read_candidate_section", "Read a paginated candidate section when inspection is too large", {
            **acc,
            "section": {"type": "string", "enum": ["identity", "evidence", "modeling_input", "selection"]},
            "pointer": _string("Optional JSON pointer within that section, e.g. /validation_plan/setup/assembly"),
            "offset": {"type": "integer", "minimum": 0},
        }, ("accession", "section")),
        _schema("inspect_investigation_lead", "Read an unadmitted discovery lead without selecting it for modeling", {
            **acc,
            "offset": {"type": "integer", "minimum": 0},
        }, ("accession",)),
        _schema("inspect_validation_contract", "Read focused schema sections; use scientific_reasoning for hypothesis fields and reasoning_rules for feedback rules", {
            "section": {"type": "string", "enum": ["all", "scientific_reasoning", "reasoning_rules", "setup", "poses", "source_record", "proposal"]},
        }),
        _schema("inspect_molecule", "Compute molecular charge, stereochemistry and optional reactive-atom matches", {
            "smiles": _string("Proposed modeled molecular structure"),
            "smarts": _string("Optional reaction-site selector"),
        }, ("smiles",)),
        _schema("read_evidence", "Read an existing local source with its fingerprint and pagination", {
            "path": _string("Path to supplied or retrieved evidence"),
            "offset": {"type": "integer", "minimum": 0},
        }, ("path",)),
        _schema("retrieve_reference", "Retrieve a primary source snapshot", {
            "kind": {"type": "string", "enum": ["uniprot", "pdb", "paper"]},
            "identifier": _string("Authoritative accession, structure identifier or open full-text identifier"),
        }, ("kind", "identifier")),
        _schema("search_literature", "Find literature addressing a modeling uncertainty", {
            "query": _string("Evidence-derived search query"),
        }, ("query",)),
        _schema("assess_assembly", "Gather assembly/site evidence using a UniProt accession; NCBI identifiers require an explicit sequence-matched UniProt mapping. Suggestions remain provisional", {
            **acc,
            "uniprot_accession": _string("Optional explicit UniProt mapping for the candidate; exact sequence is verified"),
            "ligand_heavy_atoms": {"type": "integer", "minimum": 1},
            "catalytic_position": {"type": "integer", "minimum": 1},
        }, ("accession", "ligand_heavy_atoms")),
        _schema("propose_model", "Save or revise a model with rationale, alternatives and a validation plan", {
            **acc,
            "modeling_input": {"type": "object"},
            "rationale": _string("Evidence-based reasons and unresolved issues"),
            "modeling_input_from": {"type": "string", "enum": ["candidate", "current"], "description": "Instead of repeating modeling_input, explicitly copy the supplied candidate proposal or current revision; neither bypasses validation"},
            "modeling_input_changes": {"type": "object", "description": "Optional JSON merge patch when using modeling_input_from; nested objects merge, arrays replace, null deletes a field"},
            "scientific_reasoning": {"type": "object", "description": "Competing hypotheses and testable predictions per inspect_validation_contract"},
            "alternatives": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "description": _string("Concrete alternative setup"),
                        "reason_not_selected": _string("Evidence favoring the chosen setup or explicit unresolved comparison"),
                    },
                    "required": ["description", "reason_not_selected"],
                },
            },
        }, ("accession", "rationale", "alternatives", "scientific_reasoning")),
        _schema("validate_setup", "Recompute evidence and setup validation for the current revision", acc, ("accession",)),
        _schema("render_inputs", "Compile the validated current decision into unrestrained cofold inputs", {
            **acc,
            "msa_index": {
                "type": "object",
                "description": "Map candidate ACCESSION to the full precursor MSA path, not entity/chain IDs. Omit to use this candidate's explicitly supplied evidence.msa_path; no MSA is generated.",
                "additionalProperties": {"type": "string"},
            },
        }, ("accession",)),
        _schema("prepare_bioemu2", "Prepare a BioEmu2 DYNAMICS follow-up after a justified follow-up plan and primary cofold inspection; no GPU run", {
            **acc,
            "checkpoint_path": _string("Existing local BioEmu2 checkpoint, with config.yaml beside it"),
            "followup_id": _string("ID returned by plan_followup for dynamics"),
            "seed": {"type": "integer"},
        }, ("accession", "followup_id", "checkpoint_path", "seed")),
        _schema("inspect_dynamics", "Inspect BioEmu2 follow-up outputs without changing primary cofold evidence", {
            **acc,
            "followup_id": _string("Saved dynamics follow-up ID"),
            "prediction_manifest": _string("BioEmu2 runner prediction manifest"),
        }, ("accession", "followup_id", "prediction_manifest")),
        _schema("register_design_run", "Link a design-agent run under its allocated follow-up output root; does not validate generated enzymes", {
            **acc,
            "followup_id": _string("Saved de novo follow-up ID"),
            "design_run_dir": _string("Actual child run directory created by the design agent"),
        }, ("accession", "followup_id", "design_run_dir")),
        _schema("submit_cofold", "Submit current validated inputs through the configured reservation-aware adapter", acc, ("accession",)),
        _schema("check_cofold", "Reconcile or inspect the current job using its idempotency key", acc, ("accession",)),
        _schema("validate_predictions", "Check supplied or completed predictions and feed results back into the decision loop", {
            **acc,
            "directory": _string("Directory containing this revision's prediction files"),
            "prediction_manifest": _string("Producer manifest binding input/execution hashes to the prediction file hashes"),
        }, ("accession", "directory")),
        _schema("finish_task", "Finish only when supported by saved artifacts, or record an inconclusive result", {
            "outcome": {"type": "string", "enum": ["complete", "inconclusive"]},
            "rationale": _string("Interpret results and identify unresolved choices"),
        }, ("outcome", "rationale")),
    ]


def tool_schemas():
    """Return the cached list of all registered tool schemas from the unified ToolRegistry."""
    from .tool_registry import get_registry
    return get_registry().schemas()
