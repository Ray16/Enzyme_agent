"""Tests for the once-per-family bounded LLM protocol resolver."""

import copy
import csv
import json
from pathlib import Path

import pytest

from structural_modeling_agent.family import cofold as family
from structural_modeling_agent.family import resolver


@pytest.fixture
def base_spec():
    acid = {'source_smiles': 'CC(=O)O', 'states': {}}
    partner = {'source_smiles': 'CCN', 'states': {}}
    intermediates = {
        'acyl_adenylate': 'CC(=O)OP(=O)([O-])OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O',
        'acyl_phosphate': 'CC(=O)OP(=O)([O-])[O-]',
    }
    templates = {}
    for mode, intermediate in intermediates.items():
        acid['states'][mode] = {
            '1': {'role': 'carboxylate_substrate', 'include_in_model': True, 'modeling_smiles': 'CC(=O)[O-]'},
            '2': {'role': mode + '_intermediate', 'include_in_model': True, 'modeling_smiles': intermediate},
        }
        partner['states'][mode] = {
            '1': {'role': 'amine_substrate', 'include_in_model': False, 'modeling_smiles': 'CCN'},
            '2': {'role': 'amine_substrate', 'include_in_model': True, 'modeling_smiles': 'CCN'},
        }
        templates[mode] = {
            'nucleotide_ccd': 'ATP',
            'cofactors': [{'ccd': 'ATP', 'count': 1, 'role': 'nucleotide'}],
            'metal': {'metal': 'MG', 'per_site_metal_count': 1 if mode == 'acyl_adenylate' else 2, 'rationale': 'Test metal'},
        }
    return {
        'components': [acid, partner],
        'mechanism_templates': templates,
        'protein_copies': 1,
        'assay_pH': 8.0,
        'assembly_basis': 'Default single-chain unit',
        'step_rationale': {'1': 'Step 1', '2': 'Step 2'},
        'assignment_rules': [
            {'mode': 'acyl_adenylate', 'required': ['AMP', 'diphosphate'], 'excluded': ['ADP']},
        ],
    }


def test_resolve_pending_family_once_and_cache(tmp_path, monkeypatch, base_spec):
    monkeypatch.setattr(
        'enzyme_discovery_agent.models.target_review.target_contract',
        lambda _run: {'query': 'Synthetic ligation', 'substrate_smiles': 'CC(=O)O.CCN'},
    )
    spec_path = tmp_path / 'spec.json'
    spec_path.write_text(json.dumps(base_spec))
    handoff = {
        'pending_families': [{
            'accession': 'SYNTH_DIMER_1',
            'enzyme_name': 'Synthetic Interfacial Ligase',
            'family_id': 'IPR_SYNTH_DIMER',
            'rank': 1,
            'recorded_activities': [],
        }]
    }
    calls = []

    def fake_llm(system_prompt, user_prompt):
        calls.append(user_prompt)
        return json.dumps({
            'decision': 'resolve_protocol',
            'rationale': 'Interfacial two-metal phosphoryl-transfer family.',
            'mode': 'acyl_phosphate',
            'assay_pH': 8.0,
            'protein_copies': 2,
            'active_sites': 1,
            'assembly_basis': 'Minimal functional dimer interface.',
            'assignment_rule': {'mode': 'acyl_phosphate', 'family_regex': '^IPR_SYNTH_DIMER$'},
        }), {'total_tokens': 1200}

    updated, report = resolver.resolve_pending_families(
        tmp_path, handoff, base_spec, spec_path, tmp_path / 'out', llm_caller=fake_llm
    )
    assert report['resolved'] == ['IPR_SYNTH_DIMER']
    assert report['llm_calls'] == 1
    assert report['total_tokens'] == 1200
    assert updated['family_overrides']['IPR_SYNTH_DIMER']['protein_copies'] == 2
    assert updated['family_overrides']['IPR_SYNTH_DIMER']['active_sites'] == 1
    assert any(r.get('family_regex') == '^IPR_SYNTH_DIMER$' for r in updated['assignment_rules'])

    # Second pass on same pending family hits cache -> 0 LLM calls
    _, report2 = resolver.resolve_pending_families(
        tmp_path, handoff, updated, spec_path, tmp_path / 'out', llm_caller=fake_llm
    )
    assert report2['llm_calls'] == 0
    assert len(calls) == 1


def test_resolver_repairs_chemistry_error_then_validates(base_spec):
    # Attempt 1 proposes neutral -COOH at pH 8 (blocked by RDKit chemistry check);
    # Attempt 2 repairs to deprotonated -COO- and succeeds.
    responses = [
        json.dumps({
            'decision': 'resolve_protocol',
            'rationale': 'Initial attempt with protonated acid',
            'mode': 'acyl_phosphate',
            'assay_pH': 8.0,
            'protein_copies': 1,
            'assembly_basis': 'Monomer',
            'components': [{
                'source_smiles': 'CC(=O)O',
                'states': {
                    'acyl_phosphate': {
                        '1': {'role': 'carboxylate_substrate', 'include_in_model': True, 'modeling_smiles': 'CC(=O)O'},
                        '2': {'role': 'acyl_phosphate_intermediate', 'include_in_model': True, 'modeling_smiles': 'CC(=O)OP(=O)([O-])[O-]'},
                    }
                },
            }],
        }),
        json.dumps({
            'decision': 'resolve_protocol',
            'rationale': 'Repaired to deprotonated carboxylate at pH 8',
            'mode': 'acyl_phosphate',
            'assay_pH': 8.0,
            'protein_copies': 1,
            'assembly_basis': 'Monomer',
            'components': [{
                'source_smiles': 'CC(=O)O',
                'states': {
                    'acyl_phosphate': {
                        '1': {'role': 'carboxylate_substrate', 'include_in_model': True, 'modeling_smiles': 'CC(=O)[O-]'},
                        '2': {'role': 'acyl_phosphate_intermediate', 'include_in_model': True, 'modeling_smiles': 'CC(=O)OP(=O)([O-])[O-]'},
                    }
                },
            }],
        }),
    ]
    prompts = []

    def fake_llm(_sys, user):
        prompts.append(user)
        return responses.pop(0), {'total_tokens': 500}

    outcome = resolver.resolve_one_family(
        target={'query': 'Synthetic ligation', 'substrate_smiles': 'CC(=O)O.CCN'},
        family_id='IPR_REPAIR_TEST',
        members=[{'accession': 'SYNTH_1', 'enzyme_name': 'Synthetic Ligase'}],
        base_spec=base_spec,
        llm_caller=fake_llm,
        compile_fn=family.compile_protocol,
        max_repair_turns=1,
    )
    assert outcome['status'] == 'resolved'
    assert outcome['llm_calls'] == 2
    assert 'protonated carboxylic acid' in prompts[1]


def test_resolver_fails_closed_on_persistent_invalid_chemistry(tmp_path, monkeypatch, base_spec):
    monkeypatch.setattr(
        'enzyme_discovery_agent.models.target_review.target_contract',
        lambda _run: {'query': 'Synthetic ligation', 'substrate_smiles': 'CC(=O)O.CCN'},
    )
    spec_path = tmp_path / 'spec.json'
    spec_path.write_text(json.dumps(base_spec))
    handoff = {'pending_families': [{'accession': 'SYNTH_BAD', 'enzyme_name': 'Bad', 'family_id': 'IPR_BAD'}]}

    def always_bad(_sys, _user):
        return json.dumps({
            'decision': 'resolve_protocol',
            'mode': 'nonexistent_mode_xyz',
            'rationale': 'Hallucinated mode',
        }), {'total_tokens': 300}

    updated, report = resolver.resolve_pending_families(
        tmp_path, handoff, base_spec, spec_path, tmp_path / 'out', llm_caller=always_bad
    )
    assert report['resolved'] == []
    assert len(report['failed']) == 1
    assert report['llm_calls'] == 2  # initial + 1 repair, then fail closed
    assert updated == base_spec  # untouched


def test_resolver_honors_defer_unsupported(tmp_path, monkeypatch, base_spec):
    monkeypatch.setattr(
        'enzyme_discovery_agent.models.target_review.target_contract',
        lambda _run: {'query': 'Synthetic ligation', 'substrate_smiles': 'CC(=O)O.CCN'},
    )
    spec_path = tmp_path / 'spec.json'
    spec_path.write_text(json.dumps(base_spec))
    handoff = {'pending_families': [{'accession': 'SYNTH_UNSUP', 'enzyme_name': 'Unsupported', 'family_id': 'IPR_UNSUP'}]}

    def defer_llm(_sys, _user):
        return json.dumps({
            'decision': 'defer_unsupported',
            'rationale': 'Requires separate pre-activation enzyme not modeled in a single-enzyme protocol.',
        }), {'total_tokens': 400}

    updated, report = resolver.resolve_pending_families(
        tmp_path, handoff, base_spec, spec_path, tmp_path / 'out', llm_caller=defer_llm
    )
    assert report['resolved'] == []
    assert len(report['deferred']) == 1
    assert report['llm_calls'] == 1
    assert updated == base_spec


def test_resolve_target_protocol_bootstraps_and_registers(tmp_path, monkeypatch, base_spec):
    run_dir = tmp_path / 'runs' / 'synthetic_run'
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(
        'enzyme_discovery_agent.models.target_review.target_contract',
        lambda _run: {'query': 'Synthetic new target', 'substrate_smiles': 'CC(=O)O.CCN'},
    )
    spec_path = run_dir / 'agent_state' / 'family_modeling_protocol.json'
    out_dir = run_dir / 'family_cofold'

    def bootstrap_llm(_sys, _user):
        return json.dumps({
            'decision': 'resolve_protocol',
            'rationale': 'Bootstrap acyl-phosphate protocol for new target.',
            'mode': 'acyl_phosphate',
            'assay_pH': 8.0,
            'protein_copies': 1,
            'assembly_basis': 'Monomer unit',
            'mechanism_template': base_spec['mechanism_templates']['acyl_phosphate'],
            'components': base_spec['components'],
            'step_rationale': base_spec['step_rationale'],
            'assignment_rule': {'mode': 'acyl_phosphate', 'required': ['ADP'], 'excluded': ['AMP']},
        }), {'total_tokens': 1500}

    spec, info = resolver.resolve_target_protocol(run_dir, spec_path, out_dir, llm_caller=bootstrap_llm)
    assert spec is not None
    assert info['status'] == 'resolved'
    assert spec_path.is_file()
    assert (tmp_path / 'runs' / '.family_protocols' / 'index.json').is_file()
