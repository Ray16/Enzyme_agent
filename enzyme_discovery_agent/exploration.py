"""Bounded, resumable discovery from an explicitly resumed campaign's sources.

One nomination per native-reaction group amortizes reasoning across sequences.
This controller retrieves additional records and exports exploration hypotheses;
it does not require target-activity review or submit structural computations.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import time
import urllib.parse

from .llm_client import LLMClient, TokenUsage
from .models.native_annotation import summarize_native_annotation
from .models.target_review import target_contract, digest
from .run_layout import run_file
from .state_store import atomic_write_json, atomic_write_text, state_lock


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def protein_name(record):
    description = record.get('proteinDescription') or {}
    names = [description.get('recommendedName') or {}] + description.get('submissionNames', [])
    return next((n['fullName']['value'] for n in names if n.get('fullName')), '')


def add_group(groups, annotation, member, source, families=()):
    reactions = annotation.get('catalytic_activities') or []
    name = annotation.get('protein_name') or member.get('name') or ''
    if not member.get('accession') or not (name or reactions):
        return
    # Names/reactions are scheduling groups; neither proves sequence orthology.
    key = digest({'name': name.casefold().strip(), 'reactions': sorted(reactions)})[:16]
    group = groups.setdefault(key, {'id': key, 'name': name, 'native_reactions': reactions,
        'function': annotation.get('function', ''), 'family_ids': [], 'members': {}, 'sources': []})
    for family in families:
        if family not in group['family_ids']:
            group['family_ids'].append(family)
    group['members'].setdefault(member['accession'], member)
    if source not in group['sources']:
        group['sources'].append(source)


def collect_groups(run):
    groups = {}
    families = {}
    for item in read(run_file(run, 'family_landscape.json'), []):
        hypotheses = item.get('derived_family_hypotheses') or []
        if item.get('derived_family_hypothesis'):
            hypotheses += [item['derived_family_hypothesis']]
        families[item['fold_id']] = {'fold_name': item.get('fold_name'), 'hypotheses': hypotheses}
    for path in sorted((run / 'artifacts/family_surveys').glob('*.json')):
        data = (read(path, {}).get('result') or {}).get('data') or {}
        source = ref(path)
        for member in data.get('members', []):
            claims = (member.get('claim_validation') or {}).get('claims') or []
            if not all(any(c.get('claim_type') == kind and c.get('verdict') == 'supported'
                           for c in claims) for kind in ('accession_identity', 'record_active')):
                continue
            add_group(groups, member.get('screening_annotation') or {}, member, source,
                      [data['fold_id']] if data.get('fold_id') else [])
    # Explicit continuation may reuse independently retrieved records in this
    # campaign. Development artifacts and other campaigns are never searched.
    for path in sorted((run / 'artifacts/discovery_nomination').glob('*.json')):
        saved = read(path, {})
        for record in saved.get('records', []):
            add_record(groups, record, ref(path))
    return groups, families


def add_record(groups, record, source):
    if record.get('inactiveReason') or not record.get('primaryAccession'):
        return
    annotation = {'protein_name': protein_name(record), 'catalytic_activities': [], 'function': ''}
    for comment in record.get('comments', []):
        if comment.get('commentType') == 'CATALYTIC ACTIVITY':
            name = (comment.get('reaction') or {}).get('name')
            if name:
                annotation['catalytic_activities'].append(name)
        if comment.get('commentType') == 'FUNCTION':
            annotation['function'] += ' '.join(t.get('value', '') for t in comment.get('texts', []))
    member = {'accession': record['primaryAccession'], 'name': annotation['protein_name'],
              'reviewed': 'Swiss-Prot' in record.get('entryType', ''), 'record': record}
    folds = [x['id'] for x in record.get('uniProtKBCrossReferences', [])
             if x.get('database') == 'InterPro']
    add_group(groups, annotation, member, source, folds)


def compact_groups(groups):
    return [{k: v for k, v in group.items() if k not in {'members', 'sources'}} |
            {'available_sequences': len(group['members']),
             'example_accessions': list(group['members'])[:2]} for group in groups.values()]


def compact_families(families):
    hypotheses, folds = {}, {}
    for fold, source in families.items():
        ids = []
        for hypothesis in source.get('hypotheses', []):
            key = digest(hypothesis)[:16]
            hypotheses[key] = hypothesis
            if key not in ids:
                ids.append(key)
        folds[fold] = {'fold_name': source.get('fold_name'), 'hypothesis_ids': ids}
    return {'hypotheses': hypotheses, 'folds': folds}


def initial_tranche(groups, families, limit=112):
    """Bound attention by source family and recorded reaction evidence.

    This is retrieval scheduling, not a target-activity ranking. Unselected
    groups stay in initial_groups.json and can be explored in another tranche.
    """
    counts = Counter()
    remaining = list(groups.items()); chosen = {}
    def family_key(group):
        return next((f for f in group['family_ids'] if (families.get(f) or {}).get('hypotheses')),
                    'unresolved')
    def source_priority(group):
        members = list(group['members'].values())
        return (not any(m.get('direction_vs_target') == 'same' and m.get('shared_bond_type') for m in members),
                not any(m.get('reviewed') for m in members),
                -max(((m.get('substrate') or {}).get('functional_group_similarity') or 0) for m in members),
                group['id'])
    while remaining and len(chosen) < limit:
        index = min(range(len(remaining)), key=lambda i: (
            counts[family_key(remaining[i][1])], source_priority(remaining[i][1])))
        key, group = remaining.pop(index)
        chosen[key] = group; counts[family_key(group)] += 1
    return chosen


def parse_nomination(response, group_ids):
    if response is None:
        raise RuntimeError('No nomination response; saved sources remain resumable')
    message = response.choices[0].message
    calls, _, _ = LLMClient.extract_tool_calls(message)
    values = []
    for call in calls:
        function = call.function if hasattr(call, 'function') else call['function']
        name = function.name if hasattr(function, 'name') else function['name']
        arguments = function.arguments if hasattr(function, 'arguments') else function['arguments']
        if name == 'propose_exploration':
            values.append(json.loads(arguments) if isinstance(arguments, str) else arguments)
    if len(values) != 1:
        raise ValueError('Exactly one structured exploration proposal is required')
    value = values[0]
    seen = set()
    for row in value['nominations']:
        if row['group_id'] not in group_ids or row['group_id'] in seen:
            raise ValueError('Unknown or duplicated nomination group')
        if row['route'] not in {'direct_candidate', 'requires_activation_partner', 'mechanistic_extension'}:
            raise ValueError('Unsupported exploration route')
        for key in ('family', 'hypothesis', 'structural_question'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError('Incomplete exploration hypothesis')
        seen.add(row['group_id'])
    for query in value.get('searches', []):
        if not query.get('basis_group_ids') or not set(query['basis_group_ids']) <= set(group_ids):
            raise ValueError('Additional searches must reference retrieved source groups')
        if not isinstance(query.get('query'), str) or not query['query'].strip():
            raise ValueError('Missing source search query')
    return value


def conservative_charge(calls, estimated_input):
    """Budget each metered attempt even when a proxy underreports input."""
    return sum(max(call.get('prompt_tokens', 0), estimated_input) +
               max(call.get('total_tokens', 0)-call.get('prompt_tokens', 0),
                   call.get('completion_tokens', 0)) for call in calls)


SCHEMA = {'type': 'function', 'function': {'name': 'propose_exploration',
    'description': 'Nominate source-grounded reaction groups and focused database searches.',
    'parameters': {'type': 'object', 'properties': {
        'nominations': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'group_id': {'type': 'string'}, 'family': {'type': 'string'},
            'hypothesis': {'type': 'string'}, 'structural_question': {'type': 'string'},
            'route': {'type': 'string', 'enum': ['direct_candidate', 'requires_activation_partner', 'mechanistic_extension']}},
            'required': ['group_id', 'family', 'hypothesis', 'structural_question', 'route']}},
        'searches': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'query': {'type': 'string'}, 'reason': {'type': 'string'},
            'basis_group_ids': {'type': 'array', 'items': {'type': 'string'}}},
            'required': ['query', 'reason', 'basis_group_ids']}},
        'gaps': {'type': 'array', 'items': {'type': 'string'}}},
        'required': ['nominations', 'searches', 'gaps']}}}

PROMPT = (
    'Explore a diverse source-linked protein panel for the current target. This is discovery for structural '
    'investigation, not an activity review of each sequence. Nominate native-reaction groups with a concrete '
    'catalytic transfer hypothesis; unresolved substrate acceptance is expected. Preserve both reacting '
    'partner roles, activation, required carriers and recognition differences. A broad fold or an unrelated '
    'local bond change alone is insufficient. Do not pool participants from independent reactions. '
    'Source-associated family labels are hypotheses and can be incorrect: use the actual native reaction '
    'and function when assigning the catalytic family. Give the same family label to related reaction '
    'groups, and avoid treating functional subgroups or orthologs as independent folds. '
    'Separate direct candidates, routes needing an activation partner, and mechanistic extensions. '
    'Keep immediate product formation separate from later extension. Missing native annotation is an '
    'unknown, not evidence of inactivity. Prefer concrete plausible hypotheses across families over '
    'exhaustive review or a numerical quota. Use at most 65 words total for hypothesis and structural '
    'question per group. The requested panel size is a capacity, not a scientific result. '
    'Propose at most four focused UniProt search queries for useful gaps or sequence diversity, grounded '
    'in supplied groups; use valid UniProt query syntax. Do not broaden to generic terms alone. '
    'Return exactly one propose_exploration call. No target activity scores.'
)


def nominate(client, usage, groups, families, target, output, label, budget, deadline, size,
             output_tokens=10000):
    path = output / (label + '.json')
    cache_context = {'prompt_sha256': digest(PROMPT), 'schema_sha256': digest(SCHEMA),
                     'panel_capacity': size, 'output_tokens': output_tokens}
    if path.exists():
        saved = read(path)
        if saved['input_sha256'] != digest([groups, families, target, client.requested_model]):
            raise ValueError('Saved nomination input changed; use a new output directory')
        if saved.get('cache_context') and saved['cache_context'] != cache_context:
            raise ValueError('Saved nomination protocol changed; preserve it and use a new stage label')
        return saved['result']
    messages = [{'role': 'system', 'content': PROMPT}, {'role': 'user', 'content': json.dumps({
        'target': target, 'panel_capacity': size, 'families': compact_families(families),
        'groups': compact_groups(groups)}, separators=(',', ':'))}]
    # Include the transport's hidden-reasoning output allowance, not only the
    # requested visible answer. Never start a request that cannot fit its cap.
    from . import config
    reasoning_reserve = config.ARGO_HIGH_REASONING_OUTPUT_RESERVE if client.requested_model == config.ARGO_MODEL else 0
    input_estimate = client._estimate_input_tokens(messages, [SCHEMA])
    reserve = input_estimate + output_tokens + reasoning_reserve
    charged = max(usage.total_tokens, sum(read(p, {}).get('budget_charge', 0)
                                        for p in output.glob('*_dispatch.json')))
    if charged + reserve > budget or time.monotonic() >= deadline:
        raise RuntimeError('Remaining allowance cannot cover another nomination request')
    print(json.dumps({'event': 'nomination_started', 'label': label, 'groups': len(groups),
                      'reserved_tokens': reserve}), flush=True)
    first_call = len(usage.calls)
    previous_check = getattr(client, 'budget_check', None)
    def dispatch_check():
        if previous_check and previous_check():
            return 'Production allowance reached'
        if time.monotonic() >= deadline or charged + conservative_charge(usage.calls[first_call:], input_estimate) + reserve > budget:
            return 'Remaining allowance cannot cover another provider attempt'
        return None
    client.budget_check = dispatch_check
    try:
        response = client.call(messages=messages, tools=[SCHEMA], max_tokens=output_tokens,
                               timeout=min(300, max(1, int(deadline-time.monotonic()))),
                               max_retries=1, usage_label=label)
    finally:
        client.budget_check = previous_check
        attempts = usage.calls[first_call:]
        atomic_write_json(output/(label+'_dispatch.json'), {
            'estimated_input_tokens_per_attempt': input_estimate,
            'reserved_output_tokens': output_tokens+reasoning_reserve,
            'metered_attempts': len(attempts),
            'budget_charge': conservative_charge(attempts, input_estimate),
            'scope': 'Conservative admission accounting; provider-reported usage is separately preserved.'})
    # Persist a portable message before validating; dataclass and SDK envelopes
    # intentionally share this message interface, not a response.model_dump API.
    atomic_write_json(output / (label + '_response.json'), {
        'message': response.choices[0].message.model_dump() if response else None,
        'provider_usage': response.usage.model_dump() if response and hasattr(response.usage, 'model_dump') else None,
        'estimated_input_tokens': reserve-output_tokens-reasoning_reserve,
        'serving_model': getattr(response, '_enzyme_serving_model', None),
        'usage': usage.to_dict()})
    result = parse_nomination(response, groups)
    atomic_write_json(path, {'input_sha256': digest([groups, families, target, client.requested_model]),
                             'cache_context': cache_context,
                             'serving_model': getattr(response, '_enzyme_serving_model', None),
                             'result': result})
    print(json.dumps({'event': 'nomination_complete', 'label': label,
                      'nominations': len(result['nominations']), 'tokens': usage.total_tokens}), flush=True)
    return result


def fetch_json(url):
    import requests
    response = requests.get(url, timeout=(15, 45))
    response.raise_for_status()
    return response.json()


def retrieve(searches, output, groups, deadline):
    fresh = {}
    for search in searches[:4]:
        if time.monotonic() >= deadline:
            break
        path = output / 'searches' / (digest(search) + '.json')
        saved = read(path)
        if saved is None:
            url = 'https://rest.uniprot.org/uniprotkb/search?' + urllib.parse.urlencode({
                'query': search['query'], 'size': 60, 'format': 'json'})
            try:
                records = fetch_json(url).get('results', [])
                saved = {'query': search, 'url': url, 'records': records,
                         'retrieved_at': datetime.now(timezone.utc).isoformat()}
                atomic_write_json(path, saved)
            except Exception as exc:
                atomic_write_json(output / 'searches' / (digest(search) + '_error.json'),
                                  {'query': search, 'error': str(exc)})
                continue
        source = ref(path)
        for record in saved['records']:
            add_record(fresh, record, source)
        print(json.dumps({'event': 'retrieved', 'query': search['query'],
                          'records': len(saved['records'])}), flush=True)
    for key, group in list(fresh.items()):
        if key in groups:
            groups[key]['members'].update(group['members'])
            groups[key]['sources'].extend(s for s in group['sources'] if s not in groups[key]['sources'])
            del fresh[key]
    return fresh


def search_from_source_groups(groups, nominations):
    """Compile valid queries from retrieved names, avoiding invented API fields."""
    searches = []
    for key in sorted(nominations, key=lambda k: (
            nominations[k]['route'] != 'direct_candidate', len(groups[k]['members']), k)):
        group = groups[key]
        if len(group['members']) >= 24:
            continue
        name = group['name'].replace('"', ' ')
        searches.append({'query': 'protein_name:"'+name+'"',
                         'reason': 'Resolve sequence diversity for an already nominated source reaction group.',
                         'basis_group_ids': [key]})
    return searches[:8]


def resolve_member(member, output):
    accession = member['accession']
    path = output / 'records' / (accession + '.json')
    record = read(path) or member.get('record')
    if not record or not (record.get('sequence') or {}).get('value'):
        record = fetch_json('https://rest.uniprot.org/uniprotkb/' + urllib.parse.quote(accession, safe='') + '.json')
    sequence = (record.get('sequence') or {}).get('value', '')
    if record.get('primaryAccession') != accession or record.get('inactiveReason'):
        raise ValueError('Primary accession mismatch or inactive sequence')
    if not sequence or any(c not in 'ACDEFGHIKLMNPQRSTVWY' for c in sequence):
        raise ValueError('Missing or ambiguous amino-acid sequence')
    if (record.get('proteinDescription') or {}).get('flag') in {'Fragment', 'Fragments'}:
        raise ValueError('Source describes a fragment rather than a complete protein')
    if any(c.get('commentType') == 'CAUTION' and c.get('cautionType') == 'Erroneous initiation'
           for c in record.get('comments', [])):
        raise ValueError('Source flags erroneous initiation')
    if not path.exists():
        atomic_write_json(path, record)
    return {'accession': accession, 'sequence': sequence, 'record': record, 'source': ref(path),
            'annotation': summarize_native_annotation(record)}


def sequence_identity(left, right):
    from Bio.Align import PairwiseAligner
    aligner = PairwiseAligner(mode='global', match_score=2, mismatch_score=-1,
                              open_gap_score=-5, extend_gap_score=-.5)
    alignment = aligner.align(left, right)[0]
    # Matches / alignment columns penalizes length differences and gaps. Save
    # this exact definition alongside results; it is not a phylogenetic claim.
    return float(alignment.counts().identities / alignment.shape[1])


def select_panel(pool, nominations, size, identity_cap, group_cap=None):
    selected = []
    families, group_counts = Counter(), Counter()
    identical = set()
    pairwise = []
    # Exact duplicate sequences never occupy multiple experimental slots.
    for member in pool:
        member['sequence_sha256'] = hashlib.sha256(member['sequence'].encode()).hexdigest()
    remaining = sorted(pool, key=lambda m: (
        m['annotation']['native_activity_evidence'] != 'experimental_evidence_cited',
        'Swiss-Prot' not in m['record'].get('entryType', ''), m['accession']))
    while remaining and len(selected) < size:
        index = min(range(len(remaining)), key=lambda i: (
            families[nominations[remaining[i]['group_id']]['family']],
            group_counts[remaining[i]['group_id']], i))
        member = remaining.pop(index)
        if group_cap is not None and group_counts[member['group_id']] >= group_cap:
            continue
        if member['sequence_sha256'] in identical:
            continue
        comparisons = []
        for previous in selected:
            identity = sequence_identity(member['sequence'], previous['sequence'])
            comparisons.append({'left': member['accession'], 'right': previous['accession'], 'identity': identity})
            if identity >= identity_cap:
                break
        if any(p['identity'] >= identity_cap for p in comparisons):
            continue
        member['maximum_identity_to_earlier_selection'] = max((p['identity'] for p in comparisons), default=None)
        selected.append(member)
        pairwise.extend(comparisons)
        identical.add(member['sequence_sha256'])
        families[nominations[member['group_id']]['family']] += 1
        group_counts[member['group_id']] += 1
    return selected, pairwise


def expand_variants(source, output, *, size=48, identity_cap=.8):
    """Resolve homolog hypotheses from the already nominated source queries.

    Source names, shared database family annotations and sequence comparison
    bound transfer of an exploration hypothesis. This never asserts that a
    homolog shares measured activity with its comparator.
    """
    import shutil
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    groups = read(source/'initial_groups.json')
    nominations = {row['group_id']: row for row in read(source/'initial_nomination.json')['result']['nominations']}
    for path in (source/'records').glob('*.json'):
        destination = output/'records'/path.name
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            shutil.copy2(path, destination)
    seeds = {}
    for key in nominations:
        member = next(iter(groups[key]['members'].values()))
        resolved = resolve_member(member, output)
        seeds[key] = resolved
    admitted, refused = [], []
    for path in sorted((source/'searches').glob('*.json')):
        saved = read(path)
        if not saved.get('records'):
            continue
        for key in (saved.get('query') or {}).get('basis_group_ids', []):
            if key not in nominations:
                continue
            seed = seeds[key]
            seed_folds = {x['id'] for x in seed['record'].get('uniProtKBCrossReferences', []) if x.get('database') == 'InterPro'}
            for record in saved['records']:
                accession = record.get('primaryAccession')
                name = protein_name(record)
                sequence = (record.get('sequence') or {}).get('value', '')
                folds = {x['id'] for x in record.get('uniProtKBCrossReferences', []) if x.get('database') == 'InterPro'}
                if not sequence or not folds & seed_folds:
                    continue
                identity = sequence_identity(seed['sequence'], sequence)
                if identity < .35:
                    refused.append({'enzyme': name+' ('+str(accession)+')', 'source_group': key,
                                    'reason': 'Below the conservative homolog scheduling threshold', 'identity': identity})
                    continue
                member = {'accession': accession, 'name': name, 'record': record,
                          'reviewed': 'Swiss-Prot' in record.get('entryType', '')}
                groups[key]['members'][accession] = member
                source_ref = ref(path)
                if source_ref not in groups[key]['sources']:
                    groups[key]['sources'].append(source_ref)
                admitted.append({'enzyme': name+' ('+accession+')', 'source_group': key,
                    'comparator': protein_name(seed['record'])+' ('+seed['accession']+')',
                    'identity_to_comparator': identity, 'shared_interpro_ids': sorted(folds & seed_folds),
                    'scope': 'Homolog exploration hypothesis; native and target activity are not transferred as observations.'})
    atomic_write_json(output/'homolog_transfer.json', {'admitted': admitted, 'refused': refused,
        'policy': 'A named query from the nominated source group, at least one shared InterPro entry, and >=35% global identity to its source comparator. Scheduling evidence only.'})
    pool, errors, seen = [], [], set()
    work = [(key, member) for key in nominations for member in groups[key]['members'].values()]
    def resolve(item):
        key, member = item
        try:
            return {**resolve_member(member, output), 'group_id': key}, None
        except Exception as exc:
            return None, {'enzyme': (member.get('name') or '')+' ('+member['accession']+')', 'error': str(exc)}
    with ThreadPoolExecutor(max_workers=min(2, len(os.sched_getaffinity(0)))) as executor:
        for member, error in executor.map(resolve, work):
            if error:
                errors.append(error)
            elif member['accession'] not in seen:
                pool.append(member); seen.add(member['accession'])
    print(json.dumps({'event': 'homolog_sequences_resolved', 'resolved': len(pool), 'errors': len(errors)}), flush=True)
    selected, pairs = select_panel(pool, nominations, size, identity_cap, group_cap=12)
    settings = {'size': size, 'identity_cap': identity_cap, 'group_cap': 12,
                'source_panel': str(source), 'new_llm_calls': 0}
    atomic_write_json(output/'nominations.json', nominations)
    atomic_write_json(output/'groups.json', groups)
    result = export_panel(selected, pairs, nominations, groups, output, settings, TokenUsage(), errors,
                          read(source/'initial_nomination.json')['result'].get('gaps', []))
    atomic_write_json(output/'status.json', {'status': 'complete' if len(selected) >= size else 'incomplete',
        'selected': len(selected), 'new_llm_calls': 0, 'pid': os.getpid()})
    print(json.dumps({k:v for k,v in result.items() if k not in {'candidates','settings'}}), flush=True)
    return result


def apply_panel_scope(output, policy_path):
    """Re-select cached sequences after an explicit source-linked scope decision."""
    import shutil
    output = Path(output).resolve()
    policy = read(policy_path)
    saved = read(output/'panel.json'); settings = saved['settings']
    groups, nominations = read(output/'groups.json'), read(output/'nominations.json')
    supporting = set(policy.get('supporting_only_groups', {}))
    if not supporting <= set(nominations):
        raise ValueError('Scope policy references an unknown nominated group')
    pool, errors, seen = [], [], set()
    for key, group in groups.items():
        if key not in nominations:
            continue
        for member in group['members'].values():
            accession = member['accession']
            if accession in seen:
                continue
            path = output/'records'/(accession+'.json')
            if not path.exists():
                continue
            try:
                resolved = resolve_member({'accession': accession, 'record': read(path)}, output)
            except ValueError as exc:
                errors.append({'enzyme': member.get('name', '')+' ('+accession+')', 'error': str(exc)})
                continue
            pool.append({**resolved, 'group_id': key}); seen.add(accession)
    selected, pairs = select_panel([m for m in pool if m['group_id'] not in supporting], nominations,
        settings['size'], settings['identity_cap'], group_cap=settings.get('group_cap', 12))
    # Keep related hypotheses together. Position is a panel layout, not an
    # activity rank; update the preceding-selection comparison after reordering.
    selected.sort(key=lambda m: (nominations[m['group_id']]['family'], m['group_id'], m['accession']))
    for index, member in enumerate(selected):
        previous = {m['accession'] for m in selected[:index]}
        member['maximum_identity_to_earlier_selection'] = max((p['identity'] for p in pairs
            if (p['left'] == member['accession'] and p['right'] in previous) or
               (p['right'] == member['accession'] and p['left'] in previous)), default=None)
    history = output/'history'/datetime.now().strftime('%Y%m%d_%H%M%S_scope')
    history.mkdir(parents=True)
    for name in ('panel.json', 'candidates.csv', 'sequences.fasta', 'pairwise_identity.json'):
        shutil.copy2(output/name, history/name)
    settings = {**settings, 'scope_policy': ref(policy_path),
                'ordering': 'Family hypothesis then source reaction group; panel position is not an activity rank.'}
    result = export_panel(selected, pairs, nominations, groups, output, settings, TokenUsage(), errors, saved['gaps'])
    support_rows = [r for r in saved['candidates'] if r['annotation_group'] in supporting]
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=list(saved['candidates'][0]))
    writer.writeheader(); writer.writerows(support_rows)
    atomic_write_text(output/'supporting_partners.csv', stream.getvalue())
    atomic_write_json(output/'status.json', {'status': 'complete' if len(selected) >= settings['size'] else 'incomplete',
        'selected': len(selected), 'supporting_only': len(support_rows), 'new_llm_calls': 0})
    return result


def extend(settings):
    """Continue an existing production panel without repeating its paid call."""
    import shutil
    from . import config
    output = Path(settings['output']).resolve()
    deadline = time.monotonic()+settings['runtime_seconds']
    with state_lock(output/'worker'):
        history = output/'history'/datetime.now().strftime('%Y%m%d_%H%M%S')
        history.mkdir(parents=True)
        for name in ('candidates.csv', 'sequences.fasta', 'panel.json', 'pairwise_identity.json', 'status.json'):
            if (output/name).exists():
                shutil.copy2(output/name, history/name)
        atomic_write_json(output/'status.json', {'status': 'expanding', 'pid': os.getpid(),
            'cpu_affinity': sorted(os.sched_getaffinity(0)), 'previous_outputs': str(history)})
        groups = read(output/'initial_groups.json')
        first = read(output/'initial_nomination.json')['result']
        nominations = {row['group_id']: row for row in first['nominations']}
        # Keep API syntax outside model-generated text. Every search retains its
        # retrieved source group, original full name and exact request URL.
        fresh = {}
        searches = search_from_source_groups(groups, nominations)
        for start in range(0, len(searches), 4):
            fresh.update(retrieve(searches[start:start+4], output, groups, deadline))
        groups.update(fresh)
        atomic_write_json(output/'extension_retrieved_groups.json', fresh)
        source_run = Path(settings['resume_run'])
        _, families = collect_groups(source_run)
        candidates = {key: group for key, group in groups.items() if key not in nominations and (
            key in fresh or any(m.get('direction_vs_target') == 'same' and m.get('shared_bond_type')
                               for m in group['members'].values()))}
        tranche = initial_tranche(candidates, families, limit=48)
        atomic_write_json(output/'extension_tranche.json', tranche)
        contract = target_contract(source_run)
        target = {'substrate_smiles': contract['substrate_smiles'], 'target_profile': contract['target_profile'],
                  'request': (contract.get('target_reasoning') or {}).get('query'),
                  'current_direction': 'Expand the immediate-transformation exploration set across plausible catalytic families. '
                    'Substrate-recognition differences belong in structural questions. They do not require repeated activity review. '
                    'Nominate complementary native reaction groups, including related annotation variants. '
                    'Prior work already nominated '+json.dumps(list(nominations.values()))}
        usage = TokenUsage.from_dict(read(output/'token_usage.json'))
        client = LLMClient(settings['model']); client.attach_usage(usage, output)
        client.budget_check = lambda: 'Production allowance reached' if usage.total_tokens >= settings['token_budget'] or time.monotonic() >= deadline else None
        # Bound the entire output, including reasoning, for this compact pass.
        config.ARGO_HIGH_REASONING_OUTPUT_RESERVE = 4096
        try:
            second = nominate(client, usage, tranche, families, target, output, 'extension_nomination',
                              settings['token_budget'], deadline, settings['size'], output_tokens=8192)
        except (ValueError, RuntimeError) as exc:
            atomic_write_json(output/'extension_nomination_rejected.json', {'error': str(exc),
                'scope': 'No new nominations admitted; continue resolving the existing source hypotheses.'})
            second = {'nominations': [], 'gaps': first.get('gaps', [])+[str(exc)]}
        nominations.update({row['group_id']: row for row in second['nominations']})
        atomic_write_json(output/'groups.json', groups)
        atomic_write_json(output/'nominations.json', nominations)
        work, seen = [], set()
        for key in nominations:
            members = sorted(groups[key]['members'].values(), key=lambda m: (not m.get('reviewed'), m['accession']))
            for member in members[:32]:
                if member['accession'] not in seen:
                    work.append((key, member)); seen.add(member['accession'])
        pool, errors = [], []
        def resolve(item):
            key, member = item
            try:
                return {**resolve_member(member, output), 'group_id': key}, None
            except Exception as exc:
                return None, {'accession': member['accession'], 'enzyme_name': member.get('name'), 'error': str(exc)}
        with ThreadPoolExecutor(max_workers=2) as executor:
            for member, error in executor.map(resolve, work):
                if error: errors.append(error)
                else: pool.append(member)
        print(json.dumps({'event': 'extension_sequences_resolved', 'resolved': len(pool), 'errors': len(errors)}), flush=True)
        selected, pairs = select_panel(pool, nominations, settings['size'], settings['identity_cap'])
        summary = export_panel(selected, pairs, nominations, groups, output, settings, usage, errors, second.get('gaps', []))
        atomic_write_json(output/'status.json', {'status': 'complete' if len(selected) >= settings['size'] else 'incomplete',
            'pid': os.getpid(), 'selected': len(selected), 'tokens': usage.total_tokens,
            'finished_at': datetime.now(timezone.utc).isoformat()})
        print(json.dumps({k: v for k, v in summary.items() if k not in {'candidates', 'settings'}}), flush=True)
        return summary


def export_panel(selected, pairwise, nominations, groups, output, settings, usage, errors, gaps):
    rows = []
    fasta = []
    for member in selected:
        accession = member['accession']; group = groups[member['group_id']]
        nomination = nominations[member['group_id']]
        annotation = member['annotation']; record = member['record']
        row = {'selection_order': len(rows)+1, 'accession': accession,
            'protein_name': protein_name(record), 'organism': (record.get('organism') or {}).get('scientificName', ''),
            'enzyme': protein_name(record)+' ('+accession+')',
            'interpro_ids': '; '.join(x['id'] for x in record.get('uniProtKBCrossReferences', [])
                                      if x.get('database') == 'InterPro'),
            'family_hypothesis': nomination['family'], 'annotation_group': member['group_id'],
            'route': nomination['route'], 'hypothesis': nomination['hypothesis'],
            'structural_question': nomination['structural_question'],
            'native_reaction': ' | '.join(r['reaction'] or '' for r in annotation['reactions']),
            'native_activity_evidence': annotation['native_activity_evidence'],
            'reaction_evidence_json': json.dumps(annotation['reactions'], separators=(',', ':')),
            'target_activity': 'unestablished', 'sequence_length': len(member['sequence']),
            'sequence_sha256': member['sequence_sha256'],
            'maximum_identity_to_earlier_selection': member['maximum_identity_to_earlier_selection'],
            'record_source': member['source']['path'], 'record_sha256': member['source']['sha256'],
            'discovery_sources': json.dumps(group['sources'], separators=(',', ':')),
            'uniprot_url': 'https://www.uniprot.org/uniprotkb/'+accession+'/entry',
            'modeling_setup_status': 'requires_step_partner_cofactor_and_assembly_validation'}
        rows.append(row)
        fasta.append('>'+accession+' '+protein_name(record)+'\n'+member['sequence']+'\n')
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['accession'])
    writer.writeheader(); writer.writerows(rows)
    atomic_write_text(output/'candidates.csv', stream.getvalue())
    atomic_write_text(output/'sequences.fasta', ''.join(fasta))
    atomic_write_json(output/'pairwise_identity.json', {'method': 'global alignment: matches / all alignment columns; match 2, mismatch -1, gap open -5, gap extend -0.5', 'pairs': pairwise})
    summary = {'selected': len(rows), 'requested': settings['size'],
        'families': dict(Counter(r['family_hypothesis'] for r in rows)),
        'annotation_groups': len({r['annotation_group'] for r in rows}),
        'routes': dict(Counter(r['route'] for r in rows)),
        'native_evidence': dict(Counter(r['native_activity_evidence'] for r in rows)),
        'tokens': usage.total_tokens, 'errors': errors, 'gaps': gaps,
        'scientific_status': 'Source-backed exploration panel. Target activity remains unestablished. Family labels are hypotheses, not validated fold assignments.',
        'settings': settings, 'candidates': rows}
    atomic_write_json(output/'panel.json', summary)
    return summary


def run(settings):
    output = Path(settings['output']).resolve(); output.mkdir(parents=True, exist_ok=True)
    with state_lock(output/'worker'):
        return run_locked(settings, output)


def run_locked(settings, output):
    started = time.monotonic(); deadline = started+settings['runtime_seconds']
    source_run = Path(settings['resume_run']).resolve()
    contract = target_contract(source_run)
    target = {'substrate_smiles': contract['substrate_smiles'], 'target_profile': contract['target_profile'],
              'request': (contract.get('target_reasoning') or {}).get('query')}
    identity = {**settings, 'target_sha256': digest(contract)}
    previous = read(output/'settings.json')
    if previous and previous != identity:
        raise ValueError('Production settings changed; use a new output directory')
    atomic_write_json(output/'settings.json', identity)
    atomic_write_json(output/'status.json', {'status': 'running', 'pid': os.getpid(),
        'started_at': datetime.now(timezone.utc).isoformat(), 'cpu_affinity': sorted(os.sched_getaffinity(0))})
    usage = TokenUsage.from_dict(read(output/'token_usage.json')) if (output/'token_usage.json').exists() else TokenUsage(model=settings['model'])
    client = LLMClient(model=settings['model']); client.attach_usage(usage, output)
    client.budget_check = lambda: 'Production allowance reached' if usage.total_tokens >= settings['token_budget'] or time.monotonic() >= deadline else None
    try:
        all_groups, families = collect_groups(source_run)
        atomic_write_json(output/'initial_groups.json', all_groups)
        groups = initial_tranche(all_groups, families)
        first = nominate(client, usage, groups, families, target, output, 'initial_nomination',
                         settings['token_budget'], deadline, settings['size'])
        nominations = {n['group_id']: n for n in first['nominations']}
        fresh = {}
        searches = search_from_source_groups(groups, nominations)
        for start in range(0, len(searches), 4):
            fresh.update(retrieve(searches[start:start+4], output, groups, deadline))
        # Bound the second context as well; preserve the entire retrieval pool.
        atomic_write_json(output/'retrieved_groups.json', fresh)
        fresh = initial_tranche(fresh, families, limit=80)
        if fresh:
            try:
                second = nominate(client, usage, fresh, families, target, output, 'retrieval_nomination',
                                  settings['token_budget'], deadline, settings['size'])
            except (ValueError, RuntimeError) as exc:
                atomic_write_json(output/'retrieval_nomination_rejected.json', {'error': str(exc),
                    'scope': 'No new nominations admitted; existing hypotheses remain available.'})
                gaps = first.get('gaps', []) + [str(exc) + '; retrieved groups preserved for continuation.']
            else:
                nominations.update({n['group_id']: n for n in second['nominations']})
                gaps = second.get('gaps', [])
        else:
            gaps = first.get('gaps', [])
        groups.update(fresh)
        atomic_write_json(output/'groups.json', groups)
        atomic_write_json(output/'nominations.json', nominations)
        work = []; seen = set()
        for key in nominations:
            members = sorted(groups[key]['members'].values(), key=lambda m: (not m.get('reviewed'), m['accession']))
            for member in members[:12]:
                if member['accession'] not in seen:
                    work.append((key, member)); seen.add(member['accession'])
        pool, errors = [], []
        def resolve(item):
            key, member = item
            if time.monotonic() >= deadline:
                return None, {'accession': member['accession'], 'error': 'runtime allowance reached'}
            try:
                return {**resolve_member(member, output), 'group_id': key}, None
            except Exception as exc:
                return None, {'accession': member['accession'], 'enzyme_name': member.get('name'), 'error': str(exc)}
        with ThreadPoolExecutor(max_workers=min(2, len(os.sched_getaffinity(0)))) as executor:
            for member, error in executor.map(resolve, work):
                if error: errors.append(error)
                else: pool.append(member)
        print(json.dumps({'event': 'sequences_resolved', 'resolved': len(pool), 'errors': len(errors)}), flush=True)
        selected, pairwise = select_panel(pool, nominations, settings['size'], settings['identity_cap'])
        summary = export_panel(selected, pairwise, nominations, groups, output, settings, usage, errors, gaps)
        atomic_write_json(output/'status.json', {'status': 'complete' if len(selected) >= settings['size'] else 'incomplete',
            'pid': os.getpid(), 'selected': len(selected), 'tokens': usage.total_tokens,
            'elapsed_seconds': time.monotonic()-started, 'finished_at': datetime.now(timezone.utc).isoformat()})
        print(json.dumps({k: v for k, v in summary.items() if k not in {'candidates', 'settings'}}), flush=True)
        return summary
    except BaseException as exc:
        atomic_write_json(output/'status.json', {'status': 'failed', 'pid': os.getpid(), 'error': str(exc),
            'tokens': usage.total_tokens, 'elapsed_seconds': time.monotonic()-started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--size', type=int, default=48)
    parser.add_argument('--token-budget', type=int, default=80000)
    parser.add_argument('--runtime-seconds', type=int, default=1200)
    parser.add_argument('--identity-cap', type=float, default=.8)
    parser.add_argument('--extend-panel', action='store_true')
    args = parser.parse_args()
    if args.size < 1 or args.token_budget < 1 or args.runtime_seconds < 1 or not 0 < args.identity_cap <= 1:
        parser.error('Positive capacity and allowances, and an identity cap in (0, 1], are required')
    settings = vars(args); extension = settings.pop('extend_panel')
    if extension:
        try:
            extend(settings)
        except BaseException as exc:
            atomic_write_json(Path(settings['output'])/'status.json', {
                'status': 'failed', 'pid': os.getpid(), 'error': str(exc),
                'finished_at': datetime.now(timezone.utc).isoformat()})
            raise
    else:
        run(settings)


if __name__ == '__main__':
    main()
