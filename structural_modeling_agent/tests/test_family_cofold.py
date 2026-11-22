"""Shared protocol compilation, new family members and chemical representation."""
import copy
import csv
import json
from pathlib import Path

import pytest

from structural_modeling_agent.family import cofold as family
from structural_modeling_agent.modeling_reasoning.chemistry import characterize_smiles


@pytest.fixture
def specification():
    acid={'source_smiles':'CC(=O)O','states':{}}
    partner={'source_smiles':'CCN','states':{}}
    intermediates={'acyl_adenylate':'CC(=O)OP(=O)([O-])OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O',
                   'acyl_phosphate':'CC(=O)OP(=O)([O-])[O-]'}
    templates={}
    for mode,intermediate in intermediates.items():
        acid['states'][mode]={'1':{'role':'carboxylate_substrate','include_in_model':True,'modeling_smiles':'CC(=O)[O-]'},
                              '2':{'role':mode+'_intermediate','include_in_model':True,'modeling_smiles':intermediate}}
        partner['states'][mode]={'1':{'role':'amine_substrate','include_in_model':False,'modeling_smiles':'CCN'},
                                 '2':{'role':'amine_substrate','include_in_model':True,'modeling_smiles':'CCN'}}
        templates[mode]={'nucleotide_ccd':'ATP','cofactors':[{'ccd':'ATP','count':1,'role':'nucleotide'}],
                         'metal':{'metal':'MG','per_site_metal_count':1 if mode=='acyl_adenylate' else 2,'rationale':'Explicit shared test protocol'}}
    return {'components':[acid,partner],'mechanism_templates':templates,'protein_copies':1,
            'assembly_basis':'Shared test unit','step_rationale':{'1':'First declared step','2':'Second declared step'},
            'assignment_rules':[{'mode':'acyl_adenylate','required':['AMP','diphosphate'],'excluded':['ADP']},
                                {'mode':'acyl_phosphate','required':['ADP'],'excluded':['AMP']}]}


def test_both_steps_share_mechanism_metal_but_not_consumed_nucleotide(specification):
    for mode,metal in [('acyl_adenylate',1),('acyl_phosphate',2)]:
        protocol=family.compile_protocol(mode,specification)
        first,second=(protocol['steps'][str(i)] for i in [1,2])
        assert {c['ccd']:c['count'] for c in first['cofactors']}=={'ATP':1,'MG':metal}
        assert {c['ccd']:c['count'] for c in second['cofactors']}=={'MG':metal}
        assert len(family.modeled_substrates(first))==1
        assert len(family.modeled_substrates(second))==2
        assert protocol['setup_llm_calls']==0


def test_new_members_share_one_protocol_and_conflicting_mechanisms_split(tmp_path,monkeypatch,specification):
    records={identifier:{'metadata':{'interpro_family_id':'SYNTHETIC_FAMILY','catalytic_activities':[reaction]}}
             for identifier,reaction in [('SYNTH_A','ATP = AMP + diphosphate'),('SYNTH_B','ATP = AMP + diphosphate')]}
    monkeypatch.setattr(family,'load_candidate_registry',lambda *a,**k:{'candidates':records})
    def write():
        with (tmp_path/'ranked_candidates.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=['rank','accession','protein_name']);writer.writeheader()
            for index,identifier in enumerate(records,1):writer.writerow({'rank':index,'accession':identifier,'protein_name':'Synthetic catalyst'})
    write();handoff=family.build_handoff(tmp_path,specification)
    assert len(handoff['protocols'])==len(handoff['families'])==1
    assert len(handoff['families'][0]['members'])==2
    records['SYNTH_C']={'metadata':{'interpro_family_id':'SYNTHETIC_FAMILY','catalytic_activities':['ATP = ADP']}}
    write();handoff=family.build_handoff(tmp_path,specification)
    assert len(handoff['protocols'])==len(handoff['families'])==2
    assert sum(len(g['members']) for g in handoff['families'])==3


def test_unknown_family_is_pending_without_a_guessed_protocol(tmp_path,monkeypatch,specification):
    monkeypatch.setattr(family,'load_candidate_registry',lambda *a,**k:{'candidates':{'SYNTH_X':{'metadata':{}}}})
    (tmp_path/'ranked_candidates.csv').write_text('rank,accession,protein_name\n1,SYNTH_X,Unknown synthetic protein\n')
    handoff=family.build_handoff(tmp_path,specification)
    assert not handoff['protocols'] and len(handoff['pending_families'])==1


def test_phosphate_resonance_rule_is_not_an_enzyme_or_substrate_rule():
    ordinary=characterize_smiles('COP(=O)([O-])OCC')
    isotope=characterize_smiles('COP(=[18O])([O-])OCC')
    sulfur=characterize_smiles('COP(=S)([O-])OCC')
    carbon=characterize_smiles('CC(O)CC')
    assert ordinary['stereochemistry']['status']=='not_applicable'
    assert isotope['stereochemistry']['status']=='unspecified'
    assert sulfur['stereochemistry']['status']=='unspecified'
    assert carbon['stereochemistry']['status']=='unspecified'


def test_physical_reuse_key_ignores_labels_but_preserves_required_input_differences():
    from campaign_catalog.lookup import physical_system_key
    original={'sequences':[{'protein':{'id':'A','sequence':'AGGA'}},
        {'ligand':{'id':'B','smiles':'CC(=O)[O-]'}},{'ligand':{'id':'C','ccd':'MG'}}]}
    renamed=copy.deepcopy(original);renamed['sequences'][0]['protein']['id']='X'
    renamed['sequences']=renamed['sequences'][::-1]
    assert physical_system_key(renamed)==physical_system_key(original)
    changed=copy.deepcopy(original);changed['sequences'][1]['ligand']['smiles']='CC(=O)O'
    assert physical_system_key(changed)!=physical_system_key(original)
    changed=copy.deepcopy(original);changed['sequences'][0]['protein']['id']=['A','D']
    assert physical_system_key(changed)!=physical_system_key(original)
    changed=copy.deepcopy(original);changed['sequences'].append({'ligand':{'id':'D','ccd':'MG'}})
    assert physical_system_key(changed)!=physical_system_key(original)


def test_sequence_preparation_batches_only_new_members(tmp_path,monkeypatch):
    from structural_modeling_agent.family import sequences as family_sequences
    handoff={'families':[{'members':[{'accession':'P12345','enzyme_name':'Synthetic enzyme A'},
                                   {'accession':'Q12345','enzyme_name':'Synthetic enzyme B'}]}]}
    fetches=[];batches=[]
    def fetch(accession):fetches.append(accession);return {'primaryAccession':accession,'sequence':{'value':'AGGA'}}
    def prepare(fasta,out,cache):
        batches.append(fasta.read_text())
        return [{'id':accession,'csv':'synthetic-msa'} for accession in fetches]
    monkeypatch.setattr(family_sequences,'prepare',prepare)
    first=family_sequences.prepare_members(handoff,{},tmp_path/'sequence',tmp_path/'cache',fetch)
    assert len(first)==2 and len(batches)==1 and len(fetches)==2
    family_sequences.prepare_members(handoff,{},tmp_path/'sequence',tmp_path/'cache',fetch)
    assert len(batches)==1 and len(fetches)==2


def test_same_job_id_in_an_older_run_is_reused_not_skipped(tmp_path,monkeypatch):
    import hashlib
    import yaml
    old=tmp_path/'old_run';new=tmp_path/'new_run';new.mkdir()
    document={'sequences':[{'protein':{'id':'A','sequence':'AGGA'}},{'ligand':{'id':'B','smiles':'CCO'}}]}
    old.mkdir();old_input=old/'input.yaml';old_input.write_text(yaml.safe_dump(document))
    new_input=new/'input.yaml';new_input.write_text(yaml.safe_dump(document))
    old_job={'id':'same-id','input_yaml':str(old_input),'input_sha256':hashlib.sha256(old_input.read_bytes()).hexdigest(),
             'modeling_input':{'geometry':{}}}
    saved=old/'family_cofold/jobs/same-id/complete.json';saved.parent.mkdir(parents=True)
    saved.write_text(json.dumps({'job':old_job,'receipt':{'status':'complete'}}))
    job={**old_job,'input_yaml':str(new_input)}
    monkeypatch.setattr(family,'analyze_outputs',lambda *a:{'identity_pass_poses':50})
    result=family.find_reusable(job,new,{'prediction_pools':[]})
    assert result['status']=='reused' and result['job']['discovery_run_id']=='old_run'


def test_registered_protocol_does_not_copy_old_cohort_pilot_selectors(tmp_path,specification):
    specification['fleet_pilots']={'standard':'OLD_COHORT_SELECTOR'}
    path=tmp_path/'protocol.json';path.write_text(json.dumps(specification))
    registry=tmp_path/'registry/index.json'
    family.register_protocol(path,registry)
    record=next(iter(json.loads(registry.read_text()).values()))
    saved=json.loads(Path(record['path']).read_text())
    assert 'fleet_pilots' not in saved
    assert saved['components']==specification['components']


def test_readable_input_index_has_the_consuming_run_identity(tmp_path):
    path=tmp_path/'model.yaml';path.write_text('version: 1\n')
    job={'accession':'SYNTH_A','enzyme_name':'Synthetic enzyme','family_id':'SYNTH_FAMILY','step':1,
         'discovery_run_id':'synthetic_run','input_yaml':str(path),'modeling_input':{
             'cofactors':[],'target_substrates':[{'role':'substrate','modeling_smiles':'CCO','include_in_model':True}]}}
    family.write_input_index([job],tmp_path)
    rows=list(csv.DictReader((tmp_path/'input_index.csv').open()))
    assert rows[0]['discovery_run_id']=='synthetic_run'
    assert Path(rows[0]['input_yaml']).resolve()==path


def test_interfacial_fold_triggers_dimer_override_with_single_active_site(tmp_path, monkeypatch, specification):
    records = {
        'SYNTH_INT1': {'metadata': {
            'interpro_family_id': 'IPR_SYNTH_A',
            'catalytic_activities': ['ATP = ADP'],
            'is_swissprot': True,
            'architecture_and_tractability': {'subunit_structure': 'Homohexamer; active site lies at the subunit interface between adjacent chains'},
        }},
        'SYNTH_INT2': {'metadata': {
            'interpro_family_id': 'IPR_SYNTH_B',
            'catalytic_activities': ['ATP = ADP'],
            'is_swissprot': True,
            'architecture_and_tractability': {'interfacial_active_site': True, 'required_copies': 2},
        }},
        'SYNTH_MONO': {'metadata': {
            'interpro_family_id': 'SYNTH_MONOMER_FOLD',
            'catalytic_activities': ['ATP = AMP + diphosphate'],
            'is_swissprot': True,
        }},
    }
    monkeypatch.setattr(family, 'load_candidate_registry', lambda *a, **k: {'candidates': records})
    with (tmp_path / 'ranked_candidates.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['rank', 'accession', 'protein_name', 'mechanism_family', 'is_swissprot'])
        writer.writeheader()
        writer.writerow({'rank': 1, 'accession': 'SYNTH_INT1', 'protein_name': 'Synthetic Interfacial A', 'mechanism_family': 'Interfacial ligase fold', 'is_swissprot': 'True'})
        writer.writerow({'rank': 2, 'accession': 'SYNTH_INT2', 'protein_name': 'Synthetic Interfacial B', 'mechanism_family': 'Interfacial ligase fold', 'is_swissprot': 'True'})
        writer.writerow({'rank': 3, 'accession': 'SYNTH_MONO', 'protein_name': 'Synthetic Monomer', 'mechanism_family': 'Monomeric adenylate ligase', 'is_swissprot': 'True'})

    handoff = family.build_handoff(tmp_path, specification)
    # Canonical family grouping keeps SYNTH_INT1 (IPR_SYNTH_A) and SYNTH_INT2 (IPR_SYNTH_B) together
    int_groups = [g for g in handoff['families'] if g['family_id'] == 'Interfacial ligase fold']
    assert len(int_groups) == 1
    assert len(int_groups[0]['members']) == 2
    assert int_groups[0]['protein_copies'] == 2

    int_proto = handoff['protocols'][int_groups[0]['protocol_id']]
    step1 = int_proto['steps']['1']
    step2 = int_proto['steps']['2']
    assert len(step1['enzyme_state']['chains']) == 2
    assert len(step2['enzyme_state']['chains']) == 2
    # 1 modeled interfacial active site => 2 Mg (not 4 Mg) and 1 ATP
    assert {c['ccd']: c['count'] for c in step1['cofactors']} == {'ATP': 1, 'MG': 2}
    assert {c['ccd']: c['count'] for c in step2['cofactors']} == {'MG': 2}

    mono_groups = [g for g in handoff['families'] if g['family_id'] == 'Monomeric adenylate ligase']
    assert len(mono_groups) == 1
    assert mono_groups[0]['protein_copies'] == 1


def test_protocol_supplied_interfacial_assembly_rules_are_generic(tmp_path, monkeypatch, specification):
    spec = copy.deepcopy(specification)
    spec['interfacial_assembly_rules'] = [
        {'family_ids': ['IPR_CUSTOM_DIMER'], 'protein_copies': 2, 'active_sites': 1}
    ]
    records = {
        'SYNTH_C1': {'metadata': {'interpro_family_id': 'IPR_CUSTOM_DIMER', 'catalytic_activities': ['ATP = ADP'], 'is_swissprot': True}},
    }
    monkeypatch.setattr(family, 'load_candidate_registry', lambda *a, **k: {'candidates': records})
    (tmp_path / 'ranked_candidates.csv').write_text('rank,accession,protein_name\n1,SYNTH_C1,Synthetic Custom Dimer\n')
    handoff = family.build_handoff(tmp_path, spec)
    assert handoff['families'][0]['protein_copies'] == 2
    step1 = handoff['protocols'][handoff['families'][0]['protocol_id']]['steps']['1']
    assert {c['ccd']: c['count'] for c in step1['cofactors']} == {'ATP': 1, 'MG': 2}


def test_handoff_ortholog_dedup_collapses_near_identical_and_keeps_divergent(tmp_path, monkeypatch, specification):
    base_seq = 'ACDEFGHIKLMNPQRSTVWY' * 10  # 200 aa
    near_seq = base_seq[:-3] + 'AAA'        # 98.5% identical
    div_seq = 'WYVTSRQPNMLKIHGFEDCA' * 10   # divergent
    records = {
        'SYNTH_TREMBL': {'metadata': {'interpro_family_id': 'SYNTH_FAM', 'catalytic_activities': ['ATP = AMP + diphosphate'], 'sequence': near_seq}},
        'SYNTH_SP': {'metadata': {'interpro_family_id': 'SYNTH_FAM', 'catalytic_activities': ['ATP = AMP + diphosphate'], 'sequence': base_seq}},
        'SYNTH_DIV': {'metadata': {'interpro_family_id': 'SYNTH_FAM', 'catalytic_activities': ['ATP = AMP + diphosphate'], 'sequence': div_seq}},
    }
    monkeypatch.setattr(family, 'load_candidate_registry', lambda *a, **k: {'candidates': records})
    with (tmp_path / 'ranked_candidates.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['rank', 'accession', 'protein_name', 'is_swissprot'])
        writer.writeheader()
        writer.writerow({'rank': 1, 'accession': 'SYNTH_TREMBL', 'protein_name': 'Unreviewed copy', 'is_swissprot': 'False'})
        writer.writerow({'rank': 2, 'accession': 'SYNTH_SP', 'protein_name': 'Reviewed rep', 'is_swissprot': 'True'})
        writer.writerow({'rank': 3, 'accession': 'SYNTH_DIV', 'protein_name': 'Divergent ortholog', 'is_swissprot': 'False'})

    handoff = family.build_handoff(tmp_path, specification)
    kept_accs = [m['accession'] for m in handoff['families'][0]['members']]
    # Reviewed Swiss-Prot representative is retained over lower-rank TrEMBL duplicate; divergent ortholog is kept.
    assert kept_accs == ['SYNTH_SP', 'SYNTH_DIV']
    assert len(handoff['collapsed_orthologs']) == 1
    assert handoff['collapsed_orthologs'][0]['accession'] == 'SYNTH_TREMBL'
    assert handoff['collapsed_orthologs'][0]['representative_accession'] == 'SYNTH_SP'


def test_prioritize_modeling_jobs_round_robin_and_reviewed_first():
    jobs = [
        {'id': 'a1', 'accession': 'A_TREMBL', 'family_id': 'FAM_A', 'rank': 1, 'step': 1, 'is_swissprot': False, 'recorded_activities': []},
        {'id': 'a2', 'accession': 'A_TREMBL', 'family_id': 'FAM_A', 'rank': 1, 'step': 2, 'is_swissprot': False, 'recorded_activities': []},
        {'id': 'a3', 'accession': 'A_SP', 'family_id': 'FAM_A', 'rank': 5, 'step': 1, 'is_swissprot': True, 'recorded_activities': ['rxn']},
        {'id': 'a4', 'accession': 'A_SP', 'family_id': 'FAM_A', 'rank': 5, 'step': 2, 'is_swissprot': True, 'recorded_activities': ['rxn']},
        {'id': 'b1', 'accession': 'B_SP', 'family_id': 'FAM_B', 'rank': 8, 'step': 1, 'is_swissprot': True, 'recorded_activities': ['rxn']},
        {'id': 'b2', 'accession': 'B_SP', 'family_id': 'FAM_B', 'rank': 8, 'step': 2, 'is_swissprot': True, 'recorded_activities': ['rxn']},
    ]
    ordered = family.prioritize_modeling_jobs(jobs)
    # Slot 0: A_SP (steps 1, 2), then B_SP (steps 1, 2); Slot 1: A_TREMBL (steps 1, 2)
    assert [(j['accession'], j['step']) for j in ordered] == [
        ('A_SP', 1), ('A_SP', 2),
        ('B_SP', 1), ('B_SP', 2),
        ('A_TREMBL', 1), ('A_TREMBL', 2),
    ]


def test_comparative_synthesis_ingests_structural_geometry(tmp_path):
    from enzyme_discovery_agent.models.comparative_synthesis import _load_structural_geometry, render_comparative_markdown
    analysis_dir = tmp_path / 'family_cofold' / 'analysis'
    analysis_dir.mkdir(parents=True)
    (analysis_dir / 'both_steps_summary.csv').write_text(
        'enzyme_name,accession,step_1_harmonic_distance_A,step_1_near_attack_fraction,step_2_harmonic_distance_A,step_2_near_attack_fraction,larger_step_distance_A\n'
        'Synthetic Ligase,SYNTH_1,3.75,0.80,4.90,0.12,4.90\n'
    )
    geom = _load_structural_geometry(tmp_path)
    assert 'SYNTH_1' in geom
    assert geom['SYNTH_1']['larger_step_distance_A'] == 4.9
    md = render_comparative_markdown({
        'total_admitted_candidates': 1,
        'total_families': 1,
        'families': [{'family': 'SYNTH_FAM', 'n_prioritized': 1, 'n_investigate': 0, 'activation_modes': ['acyl_phosphate'], 'molecularities': ['bimolecular'], 'representatives': [{'accession': 'SYNTH_1', 'protein_name': 'Synthetic Ligase', 'decision': 'prioritize'}]}],
        'structural_modeling_summary': {'total_modeled_admitted': 1, 'top_by_two_step_distance': [{'accession': 'SYNTH_1', 'protein_name': 'Synthetic Ligase', 'family': 'SYNTH_FAM', **geom['SYNTH_1']}]},
    })
    assert 'Completed Two-Step Structural Geometry (Not Demonstrated Activity)' in md
    assert 'SYNTH_1' in md and '4.90' in md


def test_family_handoff_markdown_is_human_readable(tmp_path, monkeypatch, specification):
    records = {
        'SYNTH_INT': {'metadata': {
            'interpro_family_id': 'IPR_SYNTH_INT',
            'catalytic_activities': ['ATP = ADP'],
            'is_swissprot': True,
            'architecture_and_tractability': {'interfacial_active_site': True, 'required_copies': 2},
        }},
        'SYNTH_UNK': {'metadata': {}},
    }
    monkeypatch.setattr(family, 'load_candidate_registry', lambda *a, **k: {'candidates': records})
    with (tmp_path / 'ranked_candidates.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['rank', 'accession', 'protein_name', 'mechanism_family', 'is_swissprot'])
        writer.writeheader()
        writer.writerow({'rank': 1, 'accession': 'SYNTH_INT', 'protein_name': 'Synthetic Interfacial Ligase', 'mechanism_family': 'Interfacial fold', 'is_swissprot': 'True'})
        writer.writerow({'rank': 2, 'accession': 'SYNTH_UNK', 'protein_name': 'Unresolved Protein', 'mechanism_family': '', 'is_swissprot': 'False'})

    handoff = family.build_handoff(tmp_path, specification)
    md_path = family.write_family_handoff_markdown(handoff, tmp_path / 'family_handoff.md')
    text = md_path.read_text(encoding='utf-8')
    assert '# Family Structural-Modeling Handoff' in text
    assert '2 (dimer)' in text
    assert 'Synthetic Interfacial Ligase (`SYNTH_INT`)' in text
    assert 'Unresolved Protein (`SYNTH_UNK`)' in text
    assert '## Pending / Unresolved Families' in text


def test_stereochemistry_and_ph_role_charge_validation():
    from structural_modeling_agent.modeling_reasoning.chemistry import characterize_smiles, substrate_errors

    # 1. Chiral stereoisomer preservation (synthetic 2-hydroxybutanoate: R vs S vs unspecified)
    src_r = 'CC[C@@H](O)C(=O)O'       # (R)
    mod_r = 'CC[C@@H](O)C(=O)[O-]'    # (R), deprotonated carboxylate
    mod_s = 'CC[C@H](O)C(=O)[O-]'     # (S), inverted!
    mod_unk = 'CCC(O)C(=O)[O-]'       # unspecified!
    assert substrate_errors(mod_r, characterize_smiles(mod_r), role='carboxylate_substrate', source_smiles=src_r, assay_pH=8.0) == []
    inv_errs = substrate_errors(mod_s, characterize_smiles(mod_s), role='carboxylate_substrate', source_smiles=src_r, assay_pH=8.0)
    assert any('stereochemistry does not match source stereoisomer' in e for e in inv_errs)
    unk_errs = substrate_errors(mod_unk, characterize_smiles(mod_unk), role='carboxylate_substrate', source_smiles=src_r, assay_pH=8.0)
    assert any('unspecified' in e for e in unk_errs)

    # 2. Step-1 dicarboxylic acid at pH 8: both carboxylates must be deprotonated (-2)
    diacid_m2 = '[O-]C(=O)CCC(=O)[O-]'
    diacid_m1 = 'OC(=O)CCC(=O)[O-]'
    diacid_0 = 'OC(=O)CCC(=O)O'
    assert substrate_errors(diacid_m2, characterize_smiles(diacid_m2), role='carboxylate_substrate', assay_pH=8.0, expected_formal_charge=-2) == []
    assert any('protonated carboxylic acid' in e for e in substrate_errors(diacid_m1, characterize_smiles(diacid_m1), role='carboxylate_substrate', assay_pH=8.0))
    assert any('protonated carboxylic acid' in e for e in substrate_errors(diacid_0, characterize_smiles(diacid_0), role='carboxylate_substrate', assay_pH=8.0))

    # 3. Step-2 diamine nucleophile at pH 8: 1 neutral -NH2 (nucleophile) + 1 protonated -NH3+ (spectator) -> charge +1
    diamine_p1 = 'NCCCC[NH3+]'
    diamine_p2 = '[NH3+]CCCC[NH3+]'
    diamine_0 = 'NCCCCN'
    assert substrate_errors(diamine_p1, characterize_smiles(diamine_p1), role='amine_substrate', assay_pH=8.0, expected_formal_charge=1) == []
    assert any('cannot act as a nucleophile' in e for e in substrate_errors(diamine_p2, characterize_smiles(diamine_p2), role='amine_substrate', assay_pH=8.0))
    assert any('spectator aliphatic amine' in e for e in substrate_errors(diamine_0, characterize_smiles(diamine_0), role='amine_substrate', assay_pH=8.0))
    # Monoamine nucleophile (1 neutral -NH2, no spectator amine) still passes cleanly
    assert substrate_errors('CCN', characterize_smiles('CCN'), role='amine_substrate', assay_pH=8.0) == []
