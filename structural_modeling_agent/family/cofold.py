"""Compile shared family protocols and execute cofolds without per-member LLM calls."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
import string
import socket
import shlex
import subprocess
import sys
import time

from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry
from enzyme_discovery_agent.state_store import atomic_write_json, state_lock
from ..env_config import EXACT_BOLTZ_PYTHON
from ..io_utils import bounded_workers, file_digest, init_single_thread_worker
from ..modeling_reasoning.chemistry import characterize_smiles
from ..modeling_reasoning.literature import LiteratureKnowledge
from ..modeling_reasoning.reaction import ReactionCoordinates
from ..modeling_reasoning.resolve_handoff import resolve_candidate, _load_family_metal
from ..msa import valid_csv
from .memory_retry import MemoryRetryPending, is_memory_failure, record_failure, retry_backend

# Arm-directory names scanned for cross-run cache reuse and in-flight dedup.
# `cofolded` is the current convention (paired with `cofolded_restrained` for the
# pocket-guided step-2 arm); `family_cofold` is the legacy name, kept here so runs
# produced before the rename stay reusable. Only the unrestrained arm participates in
# reuse/dedup (the restrained arm carries a pocket constraint -> different geometry).
COFOLD_ARM_NAMES = ('cofolded', 'family_cofold')
from ..scripts.build_step1_inputs import enzyme_chains, modeled_substrates, yaml_for


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


file_hash = file_digest


def target_key(smiles):
    from rdkit import Chem
    return digest([Chem.MolToSmiles(Chem.MolFromSmiles(value),isomericSmiles=True) for value in smiles])


def register_protocol(path,registry):
    """Cache an explicit shared target protocol, never candidate discoveries."""
    path=Path(path);registry=Path(registry);spec=json.loads(path.read_text())
    spec.pop('fleet_pilots',None)  # Operational cohort selectors never enter a reusable chemistry protocol.
    key=target_key([c['source_smiles'] for c in spec['components']])
    directory=registry.parent/'protocols';directory.mkdir(parents=True,exist_ok=True)
    content=(json.dumps(spec,indent=2)+'\n').encode()
    destination=directory/(hashlib.sha256(content).hexdigest()+'.json')
    if not destination.exists():destination.write_bytes(content)
    with state_lock(registry):
        data=json.loads(registry.read_text()) if registry.exists() else {}
        data[key]={'path':str(destination.resolve()),'sha256':file_hash(destination),'scope':'User-supplied conditional shared protocol; not experimentally demonstrated activity.'}
        atomic_write_json(registry,data)


def adopt_registered_protocol(run,path):
    from enzyme_discovery_agent.models.target_review import target_contract
    registry=Path(run).parent/'.family_protocols/index.json'
    if not registry.exists():return False
    target=[s for s in (target_contract(run).get('substrate_smiles') or '').split('.') if s]
    if not target:return False
    reference=json.loads(registry.read_text()).get(target_key(target))
    if not reference:return False
    source=Path(reference['path'])
    if file_hash(source)!=reference['sha256']:raise ValueError('Registered protocol changed')
    atomic_write_json(path,json.loads(source.read_text()));return True


def compile_protocol(mode, specification):
    """Resolve both steps once for a mechanism/unit/target protocol."""
    coordinates=(ReactionCoordinates.load(specification['reaction_coordinates_path'])
                 if specification.get('reaction_coordinates_path') else ReactionCoordinates.load())
    key,coordinate,_=coordinates.select(activation_mode=mode)
    if coordinate is None:raise ValueError('No shared reaction coordinate for '+str(mode))
    template=specification['mechanism_templates'][mode]
    metal=specification.get('metal_override') or template.get('metal')
    components=specification['components'];stages={}
    knowledge=LiteratureKnowledge.load()
    for stage in coordinate['steps']:
        index=stage['index']
        partners=[];directives=[]
        for number,component in enumerate(components,1):
            state=component['states'][mode][str(index)]
            included=state['include_in_model'];smiles=state['modeling_smiles'];role=state['role']
            partners.append({'component_index':number,'source_smiles':component['source_smiles'],
                'modeling_smiles':smiles,'role':role,'include_in_model':included})
            directives.append({'match':{'component_index':number},'include_in_model':included,
                'modeling_smiles':smiles,'role':role,'review_status':'reviewed',
                'origin':'user_protocol','rationale':'Shared stage-specific representation from the supplied protocol.',
                **({'expected_formal_charge':state['expected_formal_charge']} if state.get('expected_formal_charge') is not None else {}),
                **({'assay_pH':state['assay_pH']} if state.get('assay_pH') is not None else {}),
                **({'allow_all_neutral_polyamine':bool(state['allow_all_neutral_polyamine'])} if 'allow_all_neutral_polyamine' in state else {})})
        nucleotide=template.get('nucleotide_ccd')
        candidate={'accession':'shared-protocol','evidence':{},'modeling_input':{
            'activation':{'status':'assigned','activation_mode':mode,'nucleotide':nucleotide},
            'target_substrates':partners,'cofactors':copy.deepcopy(template.get('step_cofactors',{}).get(str(index),template.get('cofactors',[])))}}
        metal_resolution = (
            {'strategy':'explicit','ccd':metal['metal'],
             'per_site_metal_count':metal['per_site_metal_count'],'uniprot_cross_check':False,
             'origin':'existing_mechanism_protocol','rationale':metal['rationale'],
             **({'active_sites':int(specification['active_sites'])} if specification.get('active_sites') is not None else {})}
            if metal else {'strategy':'omit'}
        )
        declaration={'step':{'index':index,'mechanism':key,'chosen_because':specification['step_rationale'][str(index)]},
            'scoring_policy':'unrestrained',
            **({'assay_pH':specification['assay_pH']} if specification.get('assay_pH') is not None else {}),
            'resolutions':{
                'enzyme_state':{'chains':[{'source':'full_sequence'}],'copies':specification.get('protein_copies',1),
                    'state':'intact','origin':'shared_protocol','rationale':specification['assembly_basis']},
                'substrates':directives,'catalytic_metal':metal_resolution}}
        resolved=resolve_candidate(candidate,declaration,knowledge,step=index,mechanism_override=key,coordinates=coordinates)
        if resolved['status']!='ready':raise ValueError(str(resolved['errors']))
        stages[str(index)]=candidate['modeling_input']
    return {'mechanism':mode,'coordinate':key,'steps':stages,
        'scope':'Shared conditional structural test, not established target activity.',
        'assembly_basis':specification['assembly_basis'],'setup_llm_calls':0}


# Generic structural-annotation pattern detecting obligate inter-subunit active
# sites from UniProt/PDB subunit or architecture evidence (never enzyme-specific).
_INTERFACIAL_ARCH_RE = re.compile(
    r'active\s+site.{0,40}(?:interface|between\s+(?:adjacent\s+)?subunits|inter[\s\-]?subunit)'
    r'|inter[\s\-]?subunit\s+active\s+site'
    r'|subunit[\s\-]+interface.{0,40}active\s+site',
    re.I,
)


def _family(data):
    value=data.get('interpro_family_id') or data.get('screening_family_id') or data.get('interpro_family')
    return json.dumps(value,sort_keys=True) if isinstance(value,(list,dict)) else str(value or 'unresolved')


def _canonical_family(data, row):
    """Group related InterPro sub-tags that share a canonical mechanism/fold family."""
    raw = _family(data)
    mech_family = str(row.get('mechanism_family') or data.get('mechanism_family') or '').strip()
    if mech_family:
        # Use a short deterministic slug for grouping while keeping raw InterPro ID accessible.
        return mech_family, raw
    return raw, raw


def _load_discovery_handoff_index(root: Path) -> dict:
    """Read-only lookup of discovery's structural_modeling_handoff.json if present."""
    path = root / 'structural_modeling_handoff.json'
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    by_acc = {}
    for cand in doc.get('candidates') or []:
        if isinstance(cand, dict) and cand.get('accession'):
            by_acc[cand['accession']] = cand
    return by_acc


def _interfacial_assembly_override(raw_family, canonical_family, data, row, specification):
    """Return a minimal multi-chain override when generic evidence indicates an interfacial active site.

    Resolution order (all enzyme-agnostic; no hardcoded enzyme/family names in code):
    1. Explicit ``specification['family_overrides']`` (handled by caller; takes precedence).
    2. Optional protocol-supplied ``specification['interfacial_assembly_rules']`` matching
       InterPro family IDs or family regex.
    3. Candidate's own structural/subunit evidence (``architecture_and_tractability``,
       ``oligomeric_state``, ``subunit_structure``, or ``function_comments``) indicating
       an inter-subunit / interfacial active site.
    """
    overrides = specification.get('family_overrides') or {}
    explicit = overrides.get(raw_family) or overrides.get(canonical_family) or {}
    if explicit.get('protein_copies') is not None:
        return {}
    if specification.get('disable_interfacial_assembly_override'):
        return {}

    interpro_ids = {raw_family}
    for key in ('interpro_family_id', 'screening_family_id'):
        val = data.get(key)
        if isinstance(val, str) and val:
            interpro_ids.add(val)
        elif isinstance(val, list):
            interpro_ids.update(str(v) for v in val if v)
    fold_text = ' '.join(
        str(x or '') for x in (
            data.get('interpro_family'),
            row.get('interpro_family'),
            row.get('mechanism_family'),
            data.get('mechanism_family'),
        )
    )
    for rule in specification.get('interfacial_assembly_rules') or []:
        if not isinstance(rule, dict):
            continue
        rule_ids = set(rule.get('family_ids') or [])
        rule_regex = rule.get('family_regex')
        if (rule_ids and (interpro_ids & rule_ids)) or (rule_regex and re.search(rule_regex, fold_text, re.I)):
            return {
                'protein_copies': int(rule.get('protein_copies', 2) or 2),
                'active_sites': int(rule.get('active_sites', 1) or 1),
                'assembly_basis': rule.get('assembly_basis') or (
                    'Minimal functional oligomer interface from protocol assembly rule '
                    f"({int(rule.get('protein_copies', 2) or 2)} protein copies, "
                    f"{int(rule.get('active_sites', 1) or 1)} modeled catalytic site)."
                ),
            }

    # Check structured or text evidence from the candidate's own record
    arch = data.get('architecture_and_tractability') or {}
    if isinstance(arch, dict) and arch.get('interfacial_active_site') is True:
        copies = int(arch.get('required_copies') or 2)
        return {
            'protein_copies': copies,
            'active_sites': 1,
            'assembly_basis': (
                f'Minimal functional interface override from candidate structural evidence '
                f'({copies} protein copies, 1 modeled catalytic site).'
            ),
        }
    evidence_text = ' '.join(
        json.dumps(x) if isinstance(x, (dict, list)) else str(x or '')
        for x in (arch, data.get('subunit_structure'), data.get('function_comments'))
    )
    if _INTERFACIAL_ARCH_RE.search(evidence_text):
        return {
            'protein_copies': 2,
            'active_sites': 1,
            'assembly_basis': (
                'Minimal functional dimer interface override: candidate subunit/architecture evidence '
                'indicates an inter-subunit active site (2 protein copies, 1 modeled catalytic site).'
            ),
        }
    return {}


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'true', '1', 'yes'}
    return bool(value)


def deduplicate_handoff_orthologs(handoff, candidates=None, *, identity_threshold: float | None = 0.85):
    """Collapse near-identical orthologs (>= identity_threshold) within each family group.

    Retains all distinct families and divergent orthologs (< identity_threshold),
    prioritizing reviewed Swiss-Prot and reaction-annotated representatives first.
    Collapsed members are recorded in ``handoff['collapsed_orthologs']`` for auditability.
    """
    if identity_threshold is None or identity_threshold <= 0:
        handoff.setdefault('collapsed_orthologs', [])
        return handoff
    from enzyme_discovery_agent.models.ortholog import filter_highly_similar_orthologs

    candidates = candidates or {}
    collapsed_all = list(handoff.get('collapsed_orthologs') or [])
    seen_collapsed = {c.get('accession') for c in collapsed_all if isinstance(c, dict)}
    for group in handoff.get('families') or []:
        members = list(group.get('members') or [])
        members.sort(key=lambda m: (
            0 if m.get('is_swissprot') else 1,
            0 if m.get('recorded_activities') else 1,
            int(m.get('rank') or 999999),
            m.get('accession') or '',
        ))
        kept, collapsed = filter_highly_similar_orthologs(
            members,
            identity_threshold=float(identity_threshold),
            seq_getter=lambda m: (candidates.get(m.get('accession')) or {}).get('sequence') or m.get('_sequence'),
            family_getter=lambda m: group.get('family_id') or 'same_family',
        )
        group['members'] = kept
        for item in collapsed:
            if item.get('accession') not in seen_collapsed:
                seen_collapsed.add(item.get('accession'))
                collapsed_all.append(item)
    handoff['collapsed_orthologs'] = collapsed_all
    return handoff


def prioritize_modeling_jobs(jobs):
    """Order cofold jobs round-robin across families, prioritizing reviewed/annotated leads first.

    Within each family, candidates are ordered by:
    1. Swiss-Prot reviewed status,
    2. Recorded native reaction annotation,
    3. Discovery rank,
    keeping Step 1 and Step 2 of the same candidate adjacent.
    """
    by_family = defaultdict(list)
    seen_member = {}
    for job in jobs:
        fam = job.get('family_id') or 'unresolved'
        acc = job.get('accession') or ''
        key = (fam, acc)
        if key not in seen_member:
            seen_member[key] = {
                'is_swissprot': bool(job.get('is_swissprot')),
                'has_reaction': bool(job.get('recorded_activities')),
                'rank': int(job.get('rank') or 999999),
                'accession': acc,
                'jobs': [],
            }
            by_family[fam].append(seen_member[key])
        seen_member[key]['jobs'].append(job)
    # Sort members within each family and families by their top member's priority.
    for fam in by_family:
        by_family[fam].sort(key=lambda m: (
            0 if m['is_swissprot'] else 1,
            0 if m['has_reaction'] else 1,
            m['rank'],
            m['accession'],
        ))
        for m in by_family[fam]:
            m['jobs'].sort(key=lambda j: int(j.get('step') or 0))
    family_order = sorted(
        by_family,
        key=lambda f: (
            0 if by_family[f][0]['is_swissprot'] else 1,
            0 if by_family[f][0]['has_reaction'] else 1,
            by_family[f][0]['rank'],
            f,
        ),
    )
    ordered = []
    max_depth = max((len(v) for v in by_family.values()), default=0)
    for slot in range(max_depth):
        for fam in family_order:
            if slot < len(by_family[fam]):
                ordered.extend(by_family[fam][slot]['jobs'])
    return ordered


def build_handoff(run_dir, specification):
    """Group published candidates; direct contradictory evidence always splits groups."""
    root=Path(run_dir);registry=load_candidate_registry(root,strict=True)['candidates']
    discovery_handoff=_load_discovery_handoff_index(root)
    ranking_path=root/'ranked_candidates.csv'
    rows=list(csv.DictReader(ranking_path.open())) if ranking_path.is_file() else []
    assignments=[];known=defaultdict(set)
    coordinates=(ReactionCoordinates.load(specification['reaction_coordinates_path'])
                 if specification.get('reaction_coordinates_path') else ReactionCoordinates.load())
    for row in rows:
        meta=(registry.get(row['accession']) or {}).get('metadata') or {}
        dh_cand=discovery_handoff.get(row['accession']) or {}
        dh_ev=dh_cand.get('evidence') or {}
        data={**(meta.get('evaluation_data') or {}),**meta}
        if dh_ev.get('architecture_and_tractability') and not data.get('architecture_and_tractability'):
            data['architecture_and_tractability']=dh_ev['architecture_and_tractability']
        if dh_cand.get('sequence') and not data.get('sequence'):
            data['sequence']=dh_cand['sequence']
        canonical_fam, raw_fam = _canonical_family(data, row)
        text=' '.join(data.get('catalytic_activities') or [])
        matched=[rule['mode'] for rule in specification.get('assignment_rules',[])
                 if all(re.search(pattern,text,re.I) for pattern in rule.get('required',[]))
                 and not any(re.search(pattern,text,re.I) for pattern in rule.get('excluded',[]))
                 and rule.get('required')]
        basis='recorded_reaction'
        if len(set(matched))==1:
            mode=matched[0];known[canonical_fam].add(mode);known[raw_fam].add(mode)
        else:
            mode=((data.get('modeling_input') or {}).get('activation') or {}).get('activation_mode')
            basis='source_activation_annotation'
            if mode not in specification['mechanism_templates']:
                label=str(data.get('interpro_family') or row.get('mechanism_family') or '')
                modes={rule['mode'] for rule in specification.get('assignment_rules',[])
                       if rule.get('family_regex') and re.search(rule['family_regex'],label,re.I)}
                mode=next(iter(modes)) if len(modes)==1 else None
                basis='conditional_family_transfer'
            if not mode:
                ids=[raw_fam] if raw_fam.startswith('IPR') else []
                _,coordinate,_=coordinates.select(family_ids=ids)
                modes=(coordinate or {}).get('keys',{}).get('activation_mode',[])
                mode=modes[0] if len(modes)==1 else None
                basis='conditional_family_coordinate'
        assignments.append((row,data,canonical_fam,raw_fam,mode,basis,dh_ev))
    protocols={};groups={};pending=[];invalid={}
    for row,data,canonical_fam,raw_fam,mode,basis,dh_ev in assignments:
        if not mode:
            fam_modes = known[canonical_fam] or known[raw_fam]
            if len(fam_modes)==1:
                mode=next(iter(fam_modes));basis='conditional_family_member_transfer'
        is_sp = _as_bool(row.get('is_swissprot')) if row.get('is_swissprot') not in (None, '') else _as_bool(data.get('is_swissprot'))
        ta_summary = dh_ev.get('target_assessment_summary') or {}
        if not ta_summary and isinstance(data.get('target_assessment'), dict):
            ta = data['target_assessment']
            ta_summary = {k: ta.get(k) for k in ('decision', 'evidence_level', 'mechanism_fit', 'pocket_engineering_outlook') if ta.get(k) is not None}
        member={'accession':row['accession'],'enzyme_name':row.get('protein_name') or row.get('enzyme_name') or data.get('protein_name'),
            'rank':int(row.get('rank') or len(rows)),'assignment_basis':basis,
            'hypothesis_class':'native_mechanism_match' if basis=='recorded_reaction' else 'exploratory_transfer',
            'is_swissprot':is_sp,'raw_family_id':raw_fam,
            'recorded_activities':data.get('catalytic_activities') or [],
            **({'architecture_and_tractability':data['architecture_and_tractability']} if data.get('architecture_and_tractability') else {}),
            **({'target_assessment_summary':ta_summary} if ta_summary else {}),
            **({'_sequence':data['sequence']} if isinstance(data.get('sequence'),str) and data['sequence'] else {})}
        if mode not in specification['mechanism_templates']:
            pending.append({k:v for k,v in {**member,'family_id':canonical_fam,'reason':'No applicable shared mechanism protocol; resolve this family once.'}.items() if not k.startswith('_')});continue
        overrides=specification.get('family_overrides') or {}
        explicit_override=overrides.get(raw_fam) or overrides.get(canonical_fam) or {}
        auto_override=_interfacial_assembly_override(raw_fam, canonical_fam, data, row, specification)
        shared={**specification,**auto_override,**explicit_override}
        member['protein_copies']=int(shared.get('protein_copies',1) or 1)
        protocol_id=digest({'mode':mode,'specification':shared})[:20]
        if protocol_id not in protocols and protocol_id not in invalid:
            try:protocols[protocol_id]=compile_protocol(mode,shared)
            except (KeyError,ValueError) as exc:invalid[protocol_id]=str(exc)
        if protocol_id in invalid:
            pending.append({k:v for k,v in {**member,'family_id':canonical_fam,'reason':'Shared protocol needs review: '+invalid[protocol_id]}.items() if not k.startswith('_')});continue
        group_id=canonical_fam+':'+mode+':c'+str(member['protein_copies'])
        group=groups.setdefault(group_id,{'family_id':canonical_fam,'protocol_id':protocol_id,'protein_copies':member['protein_copies'],'members':[]})
        group['members'].append(member)
    handoff={'schema_version':'family-cofold-1','discovery_run_id':root.name,'source_run':str(root.resolve()),'specification_sha256':digest(specification),
        'protocols':protocols,'families':list(groups.values()),'pending_families':pending,
        'collapsed_orthologs':[],'ranked_candidates':len(rows),'setup_llm_calls':0}
    threshold = specification.get('ortholog_identity_threshold', 0.85)
    deduplicate_handoff_orthologs(handoff, identity_threshold=threshold)
    # Strip private _sequence helper before persisting
    for group in handoff['families']:
        group['members'] = [{k: v for k, v in m.items() if not k.startswith('_')} for m in group['members']]
    return handoff


def _prepared_candidates(root):
    result={}
    for path in (root/'agent_state/production_modeling').glob('*/item.json'):
        item=json.loads(path.read_text())
        task_path=Path(item.get('task',''))
        if task_path.is_file():
            task=json.loads(task_path.read_text())
            for candidate in task.get('candidates',[]):result[candidate['accession']]=candidate
    return result


def write_family_handoff_markdown(handoff, path):
    """Write a concise, human-readable Markdown companion to family_handoff.json."""
    path = Path(path)
    families = handoff.get('families') or []
    protocols = handoff.get('protocols') or {}
    pending = handoff.get('pending_families') or []
    collapsed = handoff.get('collapsed_orthologs') or []
    total_members = sum(len(g.get('members') or []) for g in families)

    def _step_desc(step_model):
        if not isinstance(step_model, dict):
            return '-'
        cofs = ', '.join(f"{c['ccd']}×{c['count']}" for c in (step_model.get('cofactors') or []))
        sub_parts = []
        for s in (step_model.get('target_substrates') or []):
            if not s.get('include_in_model', True):
                continue
            role_txt = s.get('role') or 'substrate'
            rep = s.get('modeling_representation') or {}
            tags = []
            if rep.get('formal_charge') is not None:
                q = int(rep['formal_charge'])
                tags.append(f"q={q:+d}" if q else "q=0")
            stereo = rep.get('stereochemistry') or {}
            if stereo.get('status') == 'specified':
                cips = ''.join(c.get('cip', '') for c in (stereo.get('assigned_centers') or []))
                if cips:
                    tags.append(f"CIP={cips}")
            sub_parts.append(f"{role_txt}[{','.join(tags)}]" if tags else role_txt)
        subs = ', '.join(sub_parts)
        return f"{subs} ({cofs})" if cofs else subs

    lines = [
        '# Family Structural-Modeling Handoff',
        '',
        f"- **Discovery run**: `{handoff.get('discovery_run_id', '-')}`",
        f"- **Ranked candidates**: **{handoff.get('ranked_candidates', 0)}** → "
        f"**{total_members}** scheduled across **{len(families)}** family group(s) · "
        f"**{len(collapsed)}** near-identical ortholog(s) collapsed · "
        f"**{len(pending)}** pending family resolution",
        f"- **Setup LLM calls**: `{handoff.get('setup_llm_calls', 0)}`",
        '',
        '## Resolved Family Protocols',
        '',
        '| Family | Mechanism | Copies | Step 1 Setup | Step 2 Setup | Members |',
        '| --- | --- | :---: | --- | --- | ---: |',
    ]
    for group in families:
        proto = protocols.get(group.get('protocol_id')) or {}
        steps = proto.get('steps') or {}
        copies = group.get('protein_copies', 1)
        copy_label = f'{copies} (dimer)' if copies == 2 else str(copies)
        lines.append(
            f"| {group.get('family_id')} | `{proto.get('mechanism', '-')}` | {copy_label} | "
            f"{_step_desc(steps.get('1'))} | {_step_desc(steps.get('2'))} | {len(group.get('members') or [])} |"
        )
    lines.append('')

    lines.extend(['## Scheduled Members by Family', ''])
    for group in families:
        proto = protocols.get(group.get('protocol_id')) or {}
        lines.append(f"### {group.get('family_id')} (`{proto.get('mechanism', '-')}`, {group.get('protein_copies', 1)} chain(s))")
        if proto.get('assembly_basis'):
            lines.append(f"> {proto['assembly_basis']}")
        lines.append('')
        for m in group.get('members') or []:
            sp = 'Swiss-Prot' if m.get('is_swissprot') else 'TrEMBL'
            hyp = m.get('hypothesis_class') or m.get('assignment_basis') or '-'
            lines.append(
                f"- **#{m.get('rank', '-')} {m.get('enzyme_name') or 'Unresolved'} (`{m.get('accession')}`)** "
                f"— {sp}, `{hyp}`"
            )
        lines.append('')

    if collapsed:
        lines.extend([
            '## Collapsed Near-Identical Orthologs (Retained Representative Modeled)',
            '',
            '| Collapsed Enzyme (Accession) | Representative | Seq Identity | Reason |',
            '| --- | --- | ---: | --- |',
        ])
        for c in collapsed:
            ident = f"{float(c['sequence_identity']):.1%}" if c.get('sequence_identity') is not None else '-'
            lines.append(
                f"| {c.get('protein_name') or 'Unresolved'} (`{c.get('accession')}`) | "
                f"`{c.get('representative_accession')}` | {ident} | {c.get('reason', '-')} |"
            )
        lines.append('')

    if pending:
        lines.extend(['## Pending / Unresolved Families (Not Yet Cofolded)', ''])
        for p in pending:
            lines.append(
                f"- **{p.get('enzyme_name') or 'Unresolved'} (`{p.get('accession')}`)** "
                f"[{p.get('family_id', '-')}]: {p.get('reason', '-')}"
            )
        lines.append('')

    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    os.replace(temporary, path)
    return path


def render_jobs(run_dir, handoff, directory,candidates=None):
    """Only sequence/identity/MSA expansion is per member; no scientific replanning."""
    root=Path(run_dir);directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    candidates=candidates if candidates is not None else _prepared_candidates(root);jobs=[];pending=[]
    for family in handoff['families']:
        protocol=handoff['protocols'][family['protocol_id']]
        for member in family['members']:
            accession=member['accession'];candidate=candidates.get(accession)
            try:
                if not candidate:raise ValueError('Exact sequence/MSA preparation pending')
                evidence=candidate['evidence'];sequence=candidate['sequence']
                identity=Path(evidence['identity_source_path'])
                if file_hash(identity)!=evidence['identity_source_sha256']:raise ValueError('Sequence source changed')
                record=json.loads(identity.read_text())
                if record.get('primaryAccession')!=accession or record.get('sequence',{}).get('value')!=sequence:raise ValueError('Sequence identity mismatch')
                msa=Path(evidence['msa_path'])
                if not valid_csv(msa,sequence):raise ValueError('A validated exact-sequence MSA is required')
                for step,model in protocol['steps'].items():
                    chains=enzyme_chains(sequence,str(msa),model['enzyme_state']['chains'])
                    text=yaml_for(chains,model['cofactors'],modeled_substrates(model))
                    job_id=digest({'yaml':text,'msa_sha256':file_hash(msa),'samples':50,'profile':'exact','seed':42})
                    path=directory/(job_id+'.yaml')
                    if not path.exists():path.write_text(text)
                    if path.read_text()!=text:raise ValueError('Existing rendered input changed')
                    labels=iter(string.ascii_uppercase);entities={}
                    for _ in chains:next(labels)
                    for cofactor in model['cofactors']:
                        for _ in range(cofactor['count']):entities.setdefault(cofactor['ccd'],[]).append(next(labels))
                    for substrate in modeled_substrates(model):entities[substrate['role']]=next(labels)
                    jobs.append({**member,'id':job_id,'family_id':family['family_id'],'protocol_id':family['protocol_id'],
                        'discovery_run_id':root.name,'discovery_run_dir':str(root.resolve()),
                        'step':int(step),'input_yaml':str(path.resolve()),'input_sha256':file_hash(path),'msa_path':str(msa),
                        'msa_sha256':file_hash(msa),'sequence_sha256':hashlib.sha256(sequence.encode()).hexdigest(),
                        'expected_predictions':50,'entities':entities,'modeling_input':model})
            except (KeyError,ValueError,OSError) as exc:pending.append({**member,'reason':str(exc)})
    return jobs,pending


def write_input_index(jobs,directory):
    """Expose readable member/stage aliases without changing fingerprinted inputs."""
    from rdkit import Chem
    directory=Path(directory);rows=[]
    for job in jobs:
        name=job['accession'] if re.fullmatch('[A-Za-z0-9_.-]+',job['accession']) else digest(job['accession'])[:20]
        folder=directory/'inputs/by_accession'/name;folder.mkdir(parents=True,exist_ok=True)
        alias=folder/('step_'+str(job['step'])+'.yaml')
        temporary=alias.with_name(alias.name+'.'+str(os.getpid())+'.tmp')
        temporary.symlink_to(Path(job['input_yaml']).resolve());os.replace(temporary,alias)
        model=job['modeling_input']
        rows.append({'discovery_run_id':job.get('discovery_run_id',directory.parent.name),
            'enzyme_name':job['enzyme_name'],'accession':job['accession'],'family_id':job['family_id'],
            'step':job['step'],'input_yaml':str(alias),
            'cofactors':'; '.join(f"{v['ccd']} x {v['count']}" for v in model['cofactors']),
            'ligands':'; '.join(f"{v['role']}: {v['modeling_smiles']} (charge {Chem.GetFormalCharge(Chem.MolFromSmiles(v['modeling_smiles']))})" for v in model['target_substrates'] if v.get('include_in_model'))})
    if rows:
        path=directory/'input_index.csv';temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
        with temporary.open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        os.replace(temporary,path)


def provider_owner_live(receipt):
    """A terminal receipt still needs PID/start-time reconciliation before retry."""
    code="from pathlib import Path;import json,sys;r=json.load(sys.stdin);p=Path('/proc/'+str(r['pid'])+'/stat');print(json.dumps((p.exists() and p.read_text().rsplit(')',1)[1].split()[19]==r['process_start_ticks']) or (bool(r.get('predictor_pid')) and Path('/proc/'+str(r['predictor_pid'])).exists())))"
    if receipt.get('host')==socket.gethostname():
        path=Path('/proc/'+str(receipt['pid'])+'/stat')
        return ((path.exists() and path.read_text().rsplit(')',1)[1].split()[19]==receipt['process_start_ticks'])
                or bool(receipt.get('predictor_pid') and Path('/proc/'+str(receipt['predictor_pid'])).exists()))
    command = shlex.join([EXACT_BOLTZ_PYTHON, '-c', code])
    try:
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',receipt['host'],command],
            input=json.dumps(receipt),text=True,capture_output=True,timeout=20,check=True)
        return json.loads(result.stdout)
    except (subprocess.SubprocessError,ValueError,OSError) as exc:
        raise MemoryRetryPending('Cannot verify previous producer ownership on '+receipt['host']+'; preserve attempt and retry the ownership check') from exc


def execute_job(job, backend, directory,resource_attempt=0):
    """Use the existing reserved native provider, without a modeling-agent session."""
    directory=Path(directory);receipt_path=directory/'receipt.json'
    backend=retry_backend(directory,backend,provider_owner_live,job)
    state_dir=Path(backend['argv'][backend['argv'].index('--state-dir')+1])
    if not receipt_path.exists():
        for path in directory.parents[1].glob('workers/*/provider/backend_jobs/'+job['id']+'/receipt.json'):
            native=json.loads(path.read_text());request_path=path.parent/'request.json'
            if native.get('status') not in {'queued','running','unknown','complete'} or not request_path.exists():continue
            if json.loads(request_path.read_text()).get('input_sha256')!=job['input_sha256']:continue
            native['provider_state_dir']=str(path.parents[2]);atomic_write_json(receipt_path,native);break
    if receipt_path.exists():
        previous=json.loads(receipt_path.read_text())
        if previous.get('status')=='failed' and previous.get('provider_state_dir') not in {None,str(state_dir)}:
            attempts=directory/'backend_attempts';attempts.mkdir(exist_ok=True)
            os.rename(receipt_path,attempts/(str(time.time_ns())+'.json'))
    request={'operation':'submit','idempotency_key':job['id'],'input_yaml':job['input_yaml'],
        'input_sha256':job['input_sha256'],'expected_predictions':50,'resource_policy':backend['resource_policy'],
        'execution_sha256':digest({'job':job['id'],'input':job['input_sha256'],'msa':job['msa_sha256'],'profile':'exact','samples':50}),
        'msa_evidence':[{'path':job['msa_path'],'sha256':job['msa_sha256']}]}
    if not receipt_path.exists():
        result=subprocess.run(backend['argv'],input=json.dumps(request),text=True,capture_output=True,timeout=90)
        if result.returncode:raise RuntimeError(result.stderr[-2000:])
        receipt=json.loads(result.stdout);receipt['provider_state_dir']=str(state_dir);atomic_write_json(receipt_path,receipt)
    else:receipt=json.loads(receipt_path.read_text())
    producer_state=Path(receipt.get('provider_state_dir',state_dir))
    live_receipt=producer_state/'backend_jobs'/job['id']/'receipt.json'
    while receipt.get('status') in {'queued','running','unknown'}:
        if live_receipt.exists():receipt={**json.loads(live_receipt.read_text()),'provider_state_dir':str(producer_state)}
        atomic_write_json(receipt_path,receipt)
        if receipt.get('status') not in {'queued','running','unknown'}:break
        time.sleep(5)
    if (receipt.get('status')=='failed' and receipt.get('error')=='Foreign resident compute process appeared'
            and resource_attempt<backend['resource_policy'].get('max_resource_retries',6)):
        for _ in range(6):
            if not provider_owner_live(receipt):break
            time.sleep(2)
        else:raise RuntimeError('Stopped native worker still owns its attempt; reconciliation required')
        stamp=str(time.time_ns());native=producer_state/'backend_jobs'/job['id']
        archive=producer_state/'resource_stops'/job['id']/stamp;archive.parent.mkdir(parents=True,exist_ok=True)
        os.rename(native,archive)
        local=directory/'resource_stops'/stamp;local.mkdir(parents=True,exist_ok=True)
        os.rename(receipt_path,local/'receipt.json')
        atomic_write_json(local/'reason.json',{'reason':'Actual foreign GPU contention; no completed ensemble replaced.',
            'native_archive':str(archive),'resource_attempt':resource_attempt+1})
        time.sleep(5)
        return execute_job(job,backend,directory,resource_attempt+1)
    if is_memory_failure(receipt):
        record_failure(directory,receipt,backend['resource_policy'])
        raise MemoryRetryPending('GPU memory exhausted; queued for a larger device')
    if receipt.get('status')!='complete':raise RuntimeError(receipt.get('error') or str(receipt))
    return receipt


def build_reusable_index(run):
    """Scan completed sibling runs once and index valid entries by physical_system_key."""
    import yaml
    from campaign_catalog.lookup import physical_system_key

    index = defaultdict(list)
    parent = Path(run).parent
    for arm in COFOLD_ARM_NAMES:
        for path in parent.glob('*/' + arm + '/jobs/*/complete.json'):
            resolved = path.resolve()
            try:
                saved = json.loads(path.read_text())
                old = saved.get('reuse_source_job') or saved['job']
                old.setdefault('discovery_run_id', Path(saved.get('reuse_source') or path).parents[3].name)
                input_path = Path(old['input_yaml'])
                if not input_path.is_file() or file_hash(input_path) != old['input_sha256']:
                    continue
                key = physical_system_key(yaml.safe_load(input_path.read_text()))
            except (OSError, ValueError, KeyError):
                continue
            index[key].append((resolved, saved, old))
    return index


def find_reusable(job, run, specification, *, reusable_index=None):
    """Consult completed family results and the existing registered prediction pools."""
    import yaml
    from campaign_catalog.lookup import physical_system_key, lookup_exact_system

    document = yaml.safe_load(Path(job['input_yaml']).read_text())
    wanted = physical_system_key(document)
    self_paths = {(Path(run) / arm / 'jobs' / job['id'] / 'complete.json').resolve() for arm in COFOLD_ARM_NAMES}
    if reusable_index is not None:
        candidates = reusable_index.get(wanted, ())
    else:
        candidates = build_reusable_index(run).get(wanted, ())
    job_geom_digest = digest(job['modeling_input']['geometry'])
    for resolved, saved, old in candidates:
        if resolved in self_paths:
            continue
        if digest(old['modeling_input']['geometry']) != job_geom_digest:
            return {'status': 'exact_coordinates_available_analysis_pending', 'source': str(resolved), 'saved': saved}
        analysis = analyze_outputs(old, saved['receipt'])
        return {'status': 'reused', 'source': str(resolved), 'job': old, 'receipt': saved['receipt'], 'analysis': analysis}
    for pool in specification.get('prediction_pools', []):
        if not (Path(pool) / 'catalog/current.json').is_file():
            continue
        result = lookup_exact_system(pool, document)
        if result['exact_system_matches']:
            return {
                'status': 'exact_coordinates_available_analysis_pending',
                'pool': str(pool),
                'matches': result['exact_system_matches'],
                'physical_system_key': wanted,
            }
    return None


def publish_results(directory,jobs):
    directory=Path(directory);rows=[]
    for job in jobs:
        folder=directory/'jobs'/job['id'];complete=folder/'complete.json';pending=folder/'reuse_pending_analysis.json'
        saved=json.loads(complete.read_text()) if complete.exists() else {}
        score=(saved.get('analysis') or {}).get('geometry') or {}
        rows.append({'discovery_run_id':job.get('discovery_run_id',directory.parent.name),
            'producer_discovery_run_id':(saved.get('reuse_source_job') or job).get('discovery_run_id',directory.parent.name),
            'enzyme_name':job['enzyme_name'],'accession':job['accession'],'family_id':job['family_id'],
            'step':job['step'],'status':'complete' if saved else 'cached_analysis_pending' if pending.exists() else 'pending',
            'n_poses':score.get('n_poses'),'weighted_attack_angle_deg':score.get('weighted_attack_angle_deg'),
            'harmonic_distance_A':score.get('harmonic_distance_A'),'reused':bool(saved.get('reuse_source')),
            'numerical_profile':saved.get('receipt',{}).get('numerical_profile'),
            'input_yaml':job['input_yaml'],'result':str(complete) if saved else str(pending) if pending.exists() else ''})
    if rows:
        path=directory/'family_results.csv';temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
        with temporary.open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        os.replace(temporary,path)
    from .analysis import refresh
    try:refresh(directory)
    except ImportError as exc:
        atomic_write_json(directory/'analysis_backend_pending.json',{'error':str(exc),'scope':'Plotting unavailable on this frontend; completed coordinates remain available for the central analyzer.'})


# Periodic-table catalytic metal elements for generic holo-seating checks
# (works for any metalloenzyme: Mg, Mn, Zn, Fe, Ca, Co, Cu, Ni, Mo, W, etc.).
_METAL_ELEMENTS = frozenset({'MG', 'MN', 'ZN', 'FE', 'CA', 'CO', 'CU', 'NI', 'MO', 'W', 'V', 'K', 'NA'})
_POLAR_COORD_ELEMENTS = frozenset({'O', 'N', 'S', 'P'})


def _holo_seating_metrics(atoms, by_chain, entities):
    """Generic, enzyme-agnostic cofactor and catalytic-metal seating check for one pose."""
    import numpy as np

    protein_xyz = np.array([v['xyz'] for v in atoms.values() if v.get('protein')])
    protein_polar_xyz = np.array([
        v['xyz'] for v in atoms.values()
        if v.get('protein') and str(v.get('element', '')).upper() in _POLAR_COORD_ELEMENTS
    ])

    metal_chains = []
    organic_cofactor_chains = []
    substrate_chains = []
    for key, val in (entities or {}).items():
        if isinstance(val, list):
            for ch in val:
                chain_atoms = by_chain.get(ch, ())
                elems = {str(v.get('element', '')).upper() for _, v in chain_atoms}
                if chain_atoms and elems <= _METAL_ELEMENTS:
                    metal_chains.append(ch)
                elif chain_atoms:
                    organic_cofactor_chains.append(ch)
        elif isinstance(val, str):
            if by_chain.get(val):
                substrate_chains.append(val)

    result = {}
    if organic_cofactor_chains and len(protein_xyz):
        min_contacts = None
        for ch in organic_cofactor_chains:
            c_xyz = np.array([v['xyz'] for _, v in by_chain[ch]])
            dists = np.linalg.norm(protein_xyz[:, None, :] - c_xyz[None, :, :], axis=2)
            n_contacts = int((dists.min(axis=1) <= 4.5).sum())
            min_contacts = n_contacts if min_contacts is None else min(min_contacts, n_contacts)
        result['cofactor_min_protein_contacts'] = min_contacts
        result['cofactor_seated'] = bool(min_contacts is not None and min_contacts >= 4)

    if metal_chains:
        cof_polar_list = [
            v['xyz'] for ch in organic_cofactor_chains for _, v in by_chain.get(ch, ())
            if str(v.get('element', '')).upper() in _POLAR_COORD_ELEMENTS
        ]
        sub_polar_list = [
            v['xyz'] for ch in substrate_chains for _, v in by_chain.get(ch, ())
            if str(v.get('element', '')).upper() in _POLAR_COORD_ELEMENTS
        ]
        all_partner_list = (
            (list(protein_polar_xyz) if len(protein_polar_xyz) else [])
            + cof_polar_list + sub_polar_list
        )
        all_partner_xyz = np.array(all_partner_list) if all_partner_list else np.empty((0, 3))
        cof_polar_xyz = np.array(cof_polar_list) if cof_polar_list else np.empty((0, 3))

        worst_any = 0.0
        worst_cof = 0.0
        for ch in metal_chains:
            m_xyz = np.array([v['xyz'] for _, v in by_chain[ch]])
            if len(all_partner_xyz):
                d_any = float(np.linalg.norm(all_partner_xyz[:, None, :] - m_xyz[None, :, :], axis=2).min())
                worst_any = max(worst_any, d_any)
            else:
                worst_any = float('inf')
            if len(cof_polar_xyz):
                d_cof = float(np.linalg.norm(cof_polar_xyz[:, None, :] - m_xyz[None, :, :], axis=2).min())
                worst_cof = max(worst_cof, d_cof)
        result['metal_nearest_polar_A'] = round(worst_any, 3) if math.isfinite(worst_any) else None
        result['metal_coordinated'] = bool(math.isfinite(worst_any) and worst_any <= 3.2)
        if len(cof_polar_xyz):
            result['metal_to_cofactor_A'] = round(worst_cof, 3)
            result['metal_cofactor_coordinated'] = bool(worst_cof <= 3.5)

    return result


def _summarize_holo_seating(poses):
    """Aggregate per-pose cofactor/metal seating into an ensemble-level diagnostic."""
    n = len(poses)
    if not n:
        return {'status': 'not_applicable'}
    cof_vals = [p['cofactor_seated'] for p in poses if 'cofactor_seated' in p]
    met_vals = [p['metal_coordinated'] for p in poses if 'metal_coordinated' in p]
    mc_vals = [p['metal_cofactor_coordinated'] for p in poses if 'metal_cofactor_coordinated' in p]
    if not cof_vals and not met_vals and not mc_vals:
        return {'status': 'not_applicable'}
    cof_frac = round(sum(cof_vals) / len(cof_vals), 4) if cof_vals else None
    met_frac = round(sum(met_vals) / len(met_vals), 4) if met_vals else None
    mc_frac = round(sum(mc_vals) / len(mc_vals), 4) if mc_vals else None
    checks = [v for v in (cof_frac, met_frac, mc_frac) if v is not None]
    status = 'pass' if all(v >= 0.5 for v in checks) else 'warning_seating_artifact'
    summary = {
        'status': status,
        'cofactor_seated_fraction': cof_frac,
        'metal_coordinated_fraction': met_frac,
        'metal_cofactor_coordinated_fraction': mc_frac,
    }
    if status != 'pass':
        summary['note'] = (
            'Cofactor and/or catalytic metal failed to seat properly in >50% of poses; '
            'poor target attack geometry may reflect a cofolding seating artifact rather than '
            'true substrate incompatibility (verify with a native-substrate control).'
        )
    return summary


def _analyze_single_prediction(args):
    """Validate identity and compute attack geometry for one prediction file."""
    (prediction, sequence_sha256, expected_protein_copies, entities, ccd_atoms,
     ligand_atom_maps, center_ref, leaving_ref, donor_refs, pocket_protocol) = args
    from ..validation import _read_structure, _protein_completeness
    import numpy as np

    path = Path(prediction['path'])
    if file_hash(path) != prediction['sha256']:
        raise ValueError('Prediction changed')
    atoms, sequences = _read_structure(path)
    by_chain = defaultdict(list)
    by_chain_atom = defaultdict(list)
    for key, val in atoms.items():
        by_chain[key[0]].append((key, val))
        by_chain_atom[(key[0], key[2])].append(val['xyz'])

    issues = []
    for chain, sequence in sequences.items():
        if hashlib.sha256(sequence.encode()).hexdigest() != sequence_sha256:
            issues.append('protein_sequence:' + chain)
        if any(c['status'] != 'pass' for c in _protein_completeness(atoms, chain, {})):
            issues.append('protein_heavy_atoms:' + chain)
    if len(sequences) != expected_protein_copies:
        issues.append('protein_copy_count')
    observed = set(by_chain)
    expected = set(sequences)
    for entity in entities.values():
        expected.update(entity if isinstance(entity, list) else [entity])
    if observed != expected:
        issues.append('entity_inventory')
    for ccd, chains in ((k, v) for k, v in entities.items() if isinstance(v, list)):
        expected_ccd = ccd_atoms.get(ccd)
        for chain in chains:
            actual = {k[2]: v['element'] for k, v in by_chain.get(chain, ())}
            if actual != expected_ccd:
                issues.append('cofactor_identity:' + chain)
    for chain, definition in ligand_atom_maps.items():
        actual = {k[2]: v['element'] for k, v in by_chain.get(chain, ())}
        desired = {v['name']: v['element'] for v in definition['atoms'].values()}
        if actual != desired:
            issues.append('ligand_atom_inventory:' + chain)

    def coordinate(ref):
        matches = by_chain_atom.get(ref, ())
        if len(matches) != 1:
            raise ValueError('Missing/ambiguous reaction atom ' + str(ref))
        return matches[0]

    center = coordinate(center_ref)
    axis = coordinate(leaving_ref) - center
    donors = [(ref, coordinate(ref)) for ref in donor_refs]
    ref, donor = min(donors, key=lambda pair: np.linalg.norm(pair[1] - center))
    vector = donor - center
    distance = float(np.linalg.norm(vector))
    angle = math.degrees(math.acos(float(np.clip(np.dot(vector, axis) / (distance * np.linalg.norm(axis)), -1, 1))))
    pose = {
        'path': str(path),
        'sha256': prediction['sha256'],
        'distance_A': distance,
        'angle_deg': angle,
        'identity_issues': issues,
        'nucleophile': ref,
        **_holo_seating_metrics(atoms, by_chain, entities),
    }
    if pocket_protocol:
        contacts = {(chain, str(number)) for chain, number in pocket_protocol['contacts']}
        pocket = np.array([value['xyz'] for key, value in atoms.items() if (key[0], key[1]) in contacts and value['protein']])
        if not len(pocket):
            raise ValueError('Mapped pocket residues are absent from guided prediction')
        ligand = np.array([value['xyz'] for _, value in by_chain.get(ref[0], ())])
        pose['nucleophile_to_pocket_A'] = float(np.linalg.norm(pocket - donor, axis=1).min())
        pose['ligand_to_pocket_A'] = float(np.linalg.norm(ligand[:, None, :] - pocket[None, :, :], axis=2).min())
        pose['pocket_localized'] = pose['ligand_to_pocket_A'] <= pocket_protocol['max_distance_A']
    return pose


def analyze_outputs(job, receipt, *, max_workers=None):
    """Measure every pose with the existing geometry metric; retain identity failures."""
    from ..proxies.attack_angle import weighted_attack_angle
    import gemmi

    source = json.loads(Path(receipt['prediction_manifest']).read_text())
    if source['input_sha256'] != job['input_sha256']:
        raise ValueError('Producer input identity changed')
    predictions = source['predictions']
    if len(predictions) != 50:
        raise ValueError('Require the complete 50-pose ensemble')
    geometry = job['modeling_input']['geometry']
    entities = job['entities']
    ccd_atoms = {}
    for ccd, reference in source.get('chemical_component_sources', {}).items():
        if file_hash(reference['path']) != reference['sha256']:
            raise ValueError('CCD source changed')
        block = gemmi.cif.read_file(reference['path']).sole_block()
        ccd_atoms[ccd] = {
            gemmi.cif.as_string(name): gemmi.cif.as_string(element).upper()
            for name, element in zip(
                block.find_values('_chem_comp_atom.atom_id'),
                block.find_values('_chem_comp_atom.type_symbol'),
            )
            if gemmi.cif.as_string(element).upper() not in {'H', 'D'}
        }

    def references(participant):
        spec = geometry[participant]
        if spec['kind'] == 'cofactor':
            return [(entities[spec['ccd']][0], spec['atom'])]
        chain = entities[spec['role']]
        mapping = source['ligand_atom_maps'][chain]['atoms']
        return [(chain, mapping[str(index)]['name']) for index in spec['reactive_atom_indices']]

    centers = references('electrophile')
    leaving = references('leaving_group')
    if len(centers) != 1 or len(leaving) != 1:
        raise ValueError('Reaction center is ambiguous')
    donor_refs = references('nucleophile')
    expected_protein_copies = len(job['modeling_input']['enzyme_state']['chains'])
    ligand_atom_maps = source.get('ligand_atom_maps', {})
    pocket_protocol = job.get('pocket_protocol')
    tasks = [
        (
            prediction, job['sequence_sha256'], expected_protein_copies, entities,
            ccd_atoms, ligand_atom_maps, centers[0], leaving[0], donor_refs, pocket_protocol,
        )
        for prediction in predictions
    ]
    first_path = Path(predictions[0]['path']) if predictions else None
    first_size = first_path.stat().st_size if first_path and first_path.is_file() else 0
    if max_workers is None and (len(tasks) < 16 or first_size < 65536):
        workers = 1
    else:
        workers = bounded_workers(max_workers, n_items=len(tasks), min_items_for_pool=8, max_cap=8)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_single_thread_worker) as pool:
            poses = list(pool.map(_analyze_single_prediction, tasks))
    else:
        poses = [_analyze_single_prediction(t) for t in tasks]
    score = weighted_attack_angle([p['distance_A'] for p in poses], [p['angle_deg'] for p in poses])
    holo_seating = _summarize_holo_seating(poses)
    return {
        'poses': poses,
        'geometry': score,
        'holo_seating': holo_seating,
        'identity_pass_poses': sum(not p['identity_issues'] for p in poses),
        **({'pocket_localized_poses': sum(p['pocket_localized'] for p in poses)} if pocket_protocol else {}),
        'step': job['step'],
        'ideal_angle': geometry.get('ideal_angle'),
        'scope': 'Raw step-specific structural observations; no enzyme-activity threshold or cross-step activity ranking.',
    }


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--specification',type=Path,required=True)
    parser.add_argument('--backend',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--pilot-accession');parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--pilot-steps',default='1,2')
    parser.add_argument('--max-protein-length',type=int)
    parser.add_argument('--expand-after-pilot',action='store_true')
    parser.add_argument('--worker-id',default='worker-0')
    parser.add_argument('--msa-cache',type=Path)
    parser.add_argument('--watch-seconds',type=int,default=0)
    parser.add_argument('--resolver-model',help='Optional LLM model for once-per-family protocol resolution of unresolved targets or pending families')
    parser.add_argument('--no-llm-resolver',action='store_true',help='Disable automatic once-per-family LLM protocol resolution')
    parser.add_argument('--memory-retries-only',action='store_true',help='Recover only saved GPU-memory failures; reuse the existing prepared queue')
    parser.add_argument('--pocket-guided-from',type=Path,help='Prepare a separate broad-pocket arm from this saved unrestrained family queue')
    args=parser.parse_args(argv);args.out.mkdir(parents=True,exist_ok=True)
    if not re.fullmatch('[A-Za-z0-9_.-]+',args.worker_id):raise ValueError('Unsafe worker identity')
    failed=0;seen=set()
    with state_lock(args.out/'workers'/args.worker_id/'lock'):
        atomic_write_json(args.out/'workers'/args.worker_id/'owner.json',{'host':socket.gethostname(),'pid':os.getpid(),
            'process_start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
            'cpu_affinity':sorted(os.sched_getaffinity(0)),'started_at':time.time()})
        while True:
            resolver_model = None if args.no_llm_resolver else args.resolver_model
            setup_llm_calls = 0
            if not args.specification.exists():
                if adopt_registered_protocol(args.run,args.specification):continue
                if resolver_model:
                    from .resolver import resolve_target_protocol
                    resolved_spec, res_info = resolve_target_protocol(
                        args.run, args.specification, args.out, model=resolver_model
                    )
                    setup_llm_calls += int(res_info.get('llm_calls') or 0)
                    if resolved_spec is not None:
                        continue
                atomic_write_json(args.out/'workers'/args.worker_id/'current.json',{
                    'status':'awaiting_shared_family_protocol','path':str(args.specification),'setup_llm_calls':setup_llm_calls})
                if not args.watch_seconds:return
                time.sleep(args.watch_seconds);continue
            specification=json.loads(args.specification.read_text())
            if args.pocket_guided_from:
                from .pockets import prepare
                jobs=prepare(args.run,args.pocket_guided_from,args.out,specification.get('pocket_max_distance_A',10.),specification.get('pocket_rescue_all',False))
            elif args.memory_retries_only:
                jobs=json.loads((args.out/'queue.json').read_text())['items']
            else:
                handoff=build_handoff(args.run,specification)
                if resolver_model and handoff.get('pending_families'):
                    from .resolver import resolve_pending_families
                    specification, res_report = resolve_pending_families(
                        args.run, handoff, specification, args.specification, args.out, model=resolver_model
                    )
                    setup_llm_calls += int(res_report.get('llm_calls') or 0)
                    if res_report.get('resolved'):
                        handoff = build_handoff(args.run, specification)
                handoff['setup_llm_calls'] = setup_llm_calls
                from .sequences import prepare_members
                candidates=prepare_members(handoff,_prepared_candidates(args.run),args.out/'sequences',
                    args.msa_cache or args.run.parent/'.sequence_msa_cache')
                deduplicate_handoff_orthologs(
                    handoff,
                    candidates=candidates,
                    identity_threshold=specification.get('ortholog_identity_threshold', 0.85),
                )
                atomic_write_json(args.out/'family_handoff.json',handoff)
                write_family_handoff_markdown(handoff, args.out/'family_handoff.md')
                jobs,pending=render_jobs(args.run,handoff,args.out/'inputs',candidates)
                atomic_write_json(args.out/'queue.json',{'items':jobs,'pending':pending,'setup_llm_calls':setup_llm_calls})
                write_input_index(jobs,args.out)
            publish_results(args.out,jobs)
            if args.prepare_only:return
            pilot_steps={int(value) for value in args.pilot_steps.split(',')}
            pilot_jobs=[j for j in jobs if j['accession']==args.pilot_accession and j['step'] in pilot_steps]
            pilot_passed=bool(pilot_jobs) and all((args.out/'jobs'/j['id']/'complete.json').is_file() for j in pilot_jobs)
            if pilot_passed:
                pilot_passed=all(json.loads((args.out/'jobs'/j['id']/'complete.json').read_text())['analysis']['identity_pass_poses']==50 for j in pilot_jobs)
            selected=[j for j in jobs if not args.pilot_accession or (j['accession']==args.pilot_accession and j['step'] in pilot_steps) or (args.expand_after_pilot and pilot_passed)]
            if args.memory_retries_only:
                def memory_pending(job):
                    folder=args.out/'jobs'/job['id'];receipt=folder/'receipt.json'
                    return (not (folder/'complete.json').exists() and ((folder/'memory_retry.json').exists()
                        or (receipt.exists() and is_memory_failure(json.loads(receipt.read_text())))))
                selected=[j for j in selected if memory_pending(j)]
            backend_definition=json.loads(args.backend.read_text())
            provider_root=Path(backend_definition['argv'][backend_definition['argv'].index('--state-dir')+1])
            owned_native={p.parent.name for p in (provider_root/'backend_jobs').glob('*/receipt.json')
                if json.loads(p.read_text()).get('status') in {'queued','running','unknown','complete'}
                and not (args.out/'jobs'/p.parent.name/'complete.json').exists()}
            if owned_native:selected=[j for j in selected if j['id'] in owned_native]
            if args.max_protein_length:
                import yaml
                selected=[j for j in selected if max(len(e['protein']['sequence']) for e in yaml.safe_load(Path(j['input_yaml']).read_text())['sequences'] if e.get('protein'))<=args.max_protein_length]
            selected = prioritize_modeling_jobs(selected)
            deferred = []
            reusable_index = build_reusable_index(args.run)
            for job in {j['id']: j for j in selected}.values():
                if job['id'] in seen:
                    continue
                folder = args.out / 'jobs' / job['id']
                folder.mkdir(parents=True, exist_ok=True)
                complete = folder / 'complete.json'
                if complete.exists():
                    saved = json.loads(complete.read_text())
                    if saved['job']['input_sha256'] != job['input_sha256']:
                        raise ValueError('Cached input identity changed')
                    analyze_outputs(saved.get('reuse_source_job') or job, saved['receipt'])
                    seen.add(job['id'])
                    continue
                reused = find_reusable(job, args.run, specification, reusable_index=reusable_index)
                if reused:
                    if reused['status']=='reused':
                        atomic_write_json(complete,{'job':job,'reuse_source_job':reused['job'],'receipt':reused['receipt'],
                            'analysis':reused['analysis'],'reuse_source':reused['source']})
                    else:atomic_write_json(folder/'reuse_pending_analysis.json',reused)
                    seen.add(job['id']);continue
                claim=args.out/'claims'/job['id'];claim.parent.mkdir(exist_ok=True)
                try:claim.mkdir()
                except FileExistsError:
                    owner_path=claim/'owner.json'
                    if owner_path.exists():
                        owner=json.loads(owner_path.read_text())
                        if owner.get('host')==socket.gethostname():
                            try:actual=Path(f'/proc/{owner["pid"]}/stat').read_text().rsplit(')',1)[1].split()[19]
                            except FileNotFoundError:actual=None
                            native=[json.loads(p.read_text()) for p in args.out.glob('workers/*/provider/backend_jobs/'+job['id']+'/receipt.json')]
                            if actual!=owner.get('process_start_ticks') and not any(r.get('status') in {'queued','running','unknown'} for r in native):
                                archive=args.out/'stale_claims';archive.mkdir(exist_ok=True)
                                try:os.rename(claim,archive/(job['id']+'_'+str(time.time_ns())))
                                except FileNotFoundError:pass
                    continue
                atomic_write_json(claim/'owner.json',{'pid':os.getpid(),'host':socket.gethostname(),'worker_id':args.worker_id,
                    'process_start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]})
                atomic_write_json(args.out/'workers'/args.worker_id/'current.json',{'status':'running','accession':job['accession'],
                    'enzyme_name':job['enzyme_name'],'step':job['step'],'job':job['id'],'setup_llm_calls':0})
                shared_claim=None
                try:
                    from .inflight import acquire,release
                    shared_claim=acquire(args.run,job,args.worker_id)
                    if shared_claim is None:
                        atomic_write_json(folder/'inflight_wait.json',{'status':'waiting_for_exact_existing_system',
                            'discovery_run_id':args.run.name,'job':job['id']})
                        continue
                    receipt=execute_job(job,json.loads(args.backend.read_text()),folder)
                    analysis=analyze_outputs(job,receipt)
                    atomic_write_json(complete,{'job':job,'receipt':receipt,'analysis':analysis,'scope':'Complete ensemble with identity checks and step-specific geometry; not experimental activity.'})
                    memory_state=folder/'memory_retry.json'
                    if memory_state.exists():
                        record=json.loads(memory_state.read_text());record['status']='complete';atomic_write_json(memory_state,record)
                    publish_results(args.out,jobs)
                    failed=0;seen.add(job['id'])
                except MemoryRetryPending as exc:
                    deferred.append({'job':job['id'],'reason':str(exc)})
                except Exception as exc:
                    atomic_write_json(folder/'failure.json',{'error':str(exc),'job':job})
                    failed+=1
                    if (args.pilot_accession and not pilot_passed) or failed>=3:raise
                finally:
                    if shared_claim is not None:release(shared_claim)
                    (claim/'owner.json').unlink();claim.rmdir()
            atomic_write_json(args.out/'workers'/args.worker_id/'current.json',{'status':'waiting_for_new_ranked_candidates' if args.watch_seconds else 'finished',
                'completed_jobs':len(seen),'memory_deferred':deferred,'setup_llm_calls':0})
            if args.memory_retries_only and not selected and not args.watch_seconds:return
            if not args.watch_seconds:return
            time.sleep(args.watch_seconds)


if __name__=='__main__':main()
