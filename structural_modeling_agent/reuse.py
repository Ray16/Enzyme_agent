"""Review verified historical predictions without inventing a fresh modeling experiment."""
import hashlib
import json
from pathlib import Path

from .validation import file_digest


def _legacy_evidence(review):
    """Measure saved coordinates under their declared selector, retaining missing producer provenance."""
    from Bio.SeqUtils import seq1
    from Bio.PDB.Polypeptide import is_aa
    from .proxies.attack_angle import geometry_from_pdb,weighted_attack_angle
    inventory_path=Path(review['source_inventory']['path'])
    if file_digest(inventory_path)!=review['source_inventory']['sha256']:
        raise ValueError('Legacy review inventory changed')
    inventory=json.loads(inventory_path.read_text())
    for ref in inventory['artifacts']+[inventory['source']]:
        if file_digest(ref['path'])!=ref['sha256']:raise ValueError('Legacy prediction artifact changed')
    poses=[ref for ref in inventory['artifacts'] if Path(ref['path']).suffix.lower() in {'.pdb','.cif'}]
    confidences=[ref for ref in inventory['artifacts'] if Path(ref['path']).name.startswith('confidence')]
    expected=inventory['expected_poses']
    if len(poses)!=expected or len(confidences)!=expected or len({ref['sha256'] for ref in poses})!=expected:
        raise ValueError('Legacy ensemble slots or distinct coordinate files are incomplete')
    measurements=[]
    for ref in poses:
        path=Path(ref['path'])
        if path.suffix.lower()!='.pdb':raise ValueError('This legacy selector requires PDB coordinates; retain the unsupported format explicitly')
        chains={}
        for line in path.read_text().splitlines():
            if not line.startswith(('ATOM  ','HETATM')) or line[16] not in (' ','A'):continue
            residue=line[17:20].strip()
            if is_aa(residue,standard=False):chains.setdefault(line[21],{})[line[22:27]]=residue
        sequences=[''.join(seq1(name,custom_map={'MSE':'M'}) for name in residues.values()) for residues in chains.values()]
        if not any(hashlib.sha256(sequence.encode()).hexdigest()==review['sequence_sha256'] for sequence in sequences):
            raise ValueError('Legacy coordinates do not contain the exact requested full protein sequence')
        measurements.append(geometry_from_pdb(path,inventory['phosphate']))
    distances,angles=zip(*measurements)
    metrics={'site.legacy_declared_activation':{**weighted_attack_angle(distances,angles),
        'complete_observed_ensemble':True,'mapping_scope':'Recorded legacy selector; mechanistic applicability and original molecular state remain unverified.'}}
    return {'accession':review['source_accession'],'status':'incomplete','execution_status':'unverified_legacy',
        'pose_qc_status':'not_assessed','scientific_status':'conditional_geometry_only',
        'geometry_metrics':metrics,'pose_validation':{'n_predictions':expected,'status':'not_assessed'},
        'sequence_sha256':review['sequence_sha256'],'experimental_activity':'not_established_by_modeling',
        'review_errors':['Original input-to-prediction binding, molecular state and numerical profile are unverified. These are conditional coordinate descriptors, not a new validated cofold or an activity result.']}


def checked_evidence(review):
    """Recheck the original report and every bound artifact under its saved model."""
    from .evidence import modeling_evidence
    if review.get('kind')=='legacy_conditional_geometry':return _legacy_evidence(review)
    source = review['source_report']
    path = Path(source['path'])
    if file_digest(path) != source['sha256']:
        raise ValueError('The original prediction report changed after reuse review')
    original = json.loads(path.read_text())
    if any(candidate.get('existing_prediction_review') for candidate in original['candidates']):
        raise ValueError('Review the original producer report, not a chain of review reports')
    row, = [candidate for candidate in modeling_evidence(path)['candidates']
            if candidate['accession'] == review['source_accession']]
    if row.get('sequence_sha256') != review['sequence_sha256']:
        raise ValueError('Cached prediction sequence does not match the requested candidate')
    if file_digest(path) != source['sha256']:
        raise ValueError('The prediction report changed during reuse verification')
    return row


def review_existing(agent, *, accession, setup_id, applicability, interpretation, unresolved):
    if applicability not in {'same_molecular_question', 'limited_transfer', 'unsupported_transfer'}:
        raise ValueError('State applicability to the current molecular question')
    if not isinstance(interpretation, str) or len(interpretation.strip()) < 20:
        raise ValueError('Explain the source-bound applicability and limitations')
    if not isinstance(unresolved, list) or any(not isinstance(value, str) for value in unresolved):
        raise ValueError('Retain unresolved assumptions as a list of strings')
    candidate = agent._candidate(accession)
    matches = [entry for entry in (candidate.get('evidence') or {}).get('existing_cofolds', [])
               if entry.get('setup', {}).get('setup_id') == setup_id]
    if len(matches) != 1:
        raise ValueError('Choose one supplied shared-pool setup identifier')
    match = matches[0]
    source = match['execution']['source']
    path = agent._authorized_evidence(source['path'])
    if file_digest(path) != source['sha256']:
        raise ValueError('Shared-pool report identity changed; refresh the lookup')
    if path.stat().st_size>16*1024*1024:raise ValueError('Source report exceeds the local evidence inspection limit')
    if 'declared_legacy_recipe' in match['setup']:
        recipe=match['setup']['declared_legacy_recipe']
        if recipe.get('phosphate') not in {'PA','PG'}:
            raise ValueError('No supported reactive-atom selector is declared for this legacy ensemble')
        inventory={'artifacts':match['execution']['artifacts'],'source':source,
                   'expected_poses':match['execution']['expected_poses'],'phosphate':recipe['phosphate']}
        for ref in inventory['artifacts']:agent._authorized_evidence(ref['path'])
        from enzyme_discovery_agent.state_store import atomic_write_json
        inventory_path=agent.run_dir/'artifacts'/('legacy_inventory_'+hashlib.sha256(setup_id.encode()).hexdigest()[:20]+'.json')
        atomic_write_json(inventory_path,inventory)
        review={'kind':'legacy_conditional_geometry','source_setup_id':setup_id,
                'source_execution_id':match['execution']['execution_id'],'source_accession':accession,
                'source_inventory':{'path':str(inventory_path),'sha256':file_digest(inventory_path)},
                'sequence_sha256':hashlib.sha256(candidate['sequence'].encode()).hexdigest(),
                'applicability':applicability,'interpretation':interpretation,'unresolved':unresolved,
                'scope':'Conditional descriptors of historical coordinates. Original producer/input binding remains unverified.'}
        evidence=checked_evidence(review)
        review.update(geometry_metrics=evidence['geometry_metrics'],source_execution_status=evidence['execution_status'],
                      source_scientific_status=evidence['scientific_status'],pose_counts={'n_predictions':inventory['expected_poses']})
        agent.state['candidates'][accession]['existing_prediction_review']=review
        return {'status':'reviewed','geometry_metrics':review['geometry_metrics'],'pose_counts':review['pose_counts'],
                'source_execution_status':evidence['execution_status'],'review_errors':evidence['review_errors'],
                'review':review,'compute_submitted':False}
    report = json.loads(path.read_text())
    if report.get('schema_version') != 'modeling-agent-1.0':
        raise ValueError('This legacy pool entry has no source-bound agent report; retain its inventory and record the provenance gap')
    input_source = match['setup']['input_source']
    original, = [row for row in report['candidates'] if row.get('input_yaml')
                  and Path(row['input_yaml']).resolve() == Path(input_source['path']).resolve()]
    sequence_hash = hashlib.sha256(candidate['sequence'].encode()).hexdigest()
    if original.get('sequence_sha256') != sequence_hash:
        raise ValueError('The saved construct is not the supplied exact protein sequence')
    references = [input_source] + (original.get('msa_evidence') or [])
    poses = original.get('pose_validation') or {}
    references += poses.get('poses') or []
    producer = poses.get('prediction_provenance') or {}
    references += [producer] + (producer.get('supporting_files') or [])
    plan = (original.get('modeling_input') or {}).get('validation_plan') or {}
    for declaration in list((plan.get('setup') or {}).values()) + [plan.get('poses') or {}]:
        references += declaration.get('evidence') or []
    for reference in references:
        agent._authorized_evidence(reference['path'])
    review = {'source_setup_id': setup_id, 'source_execution_id': match['execution']['execution_id'],
              'source_report': {'path': str(path), 'sha256': source['sha256']},
              'source_accession': original['accession'], 'sequence_sha256': sequence_hash,
              'original_decision_origin': report.get('decision_origin'),
              'applicability': applicability, 'interpretation': interpretation, 'unresolved': unresolved,
              'scope': 'Historical computation under its original molecular model. Applicability is the current agent interpretation; native activity is not established.'}
    evidence = checked_evidence(review)
    review['geometry_metrics'] = evidence['geometry_metrics']
    review['source_execution_status'] = evidence['execution_status']
    review['source_scientific_status'] = evidence['scientific_status']
    review['pose_counts'] = {key: poses.get(key) for key in ('n_predictions', 'n_passing', 'passing_fraction')}
    agent.state['candidates'][accession]['existing_prediction_review'] = review
    return {'status': 'reviewed', 'geometry_metrics': review['geometry_metrics'],
            'pose_counts': review['pose_counts'], 'source_execution_status': evidence['execution_status'],
            'source_scientific_status': evidence['scientific_status'], 'applicability': applicability,
            'original_modeling_input': original.get('modeling_input'),
            'original_reasoning_assessment': original.get('reasoning_assessment'), 'review': review,
            'compute_submitted': False}
