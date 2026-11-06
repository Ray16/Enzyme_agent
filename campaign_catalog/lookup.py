"""Read the shared prediction pool without supplying historical discovery answers."""
import hashlib
import json
from pathlib import Path

import yaml


def physical_system_key(document):
    """Exact molecular inventory, charge/stereo, copies and restraints; ignore labels."""
    from collections import Counter
    from rdkit import Chem
    entities=[]
    for item in document.get('sequences',[]):
        protein=item.get('protein');ligand=item.get('ligand')
        value=protein or ligand
        if not value:continue
        copies=len(value['id']) if isinstance(value.get('id'),list) else 1
        if protein:
            identity=('protein',protein['sequence'],bool(protein.get('cyclic')),
                      json.dumps(protein.get('modifications') or [],sort_keys=True))
        elif ligand.get('ccd'):identity=('ccd',ligand['ccd'])
        else:
            molecule=Chem.MolFromSmiles(ligand['smiles'])
            if molecule is None:raise ValueError('Invalid cached molecular structure')
            identity=('smiles',Chem.MolToSmiles(molecule,isomericSmiles=True))
        entities.extend([identity]*copies)
    inventory=sorted((list(key),count) for key,count in Counter(entities).items())
    return hashlib.sha256(json.dumps({'entities':inventory,'constraints':document.get('constraints') or []},sort_keys=True).encode()).hexdigest()


def lookup_exact_system(campaign_dir,document,minimum_poses=50):
    """Reuse eligibility requires fingerprinted inputs and a complete saved ensemble."""
    proteins=[item['protein']['sequence'] for item in document['sequences'] if item.get('protein')]
    result=lookup_predictions(campaign_dir,sequence=proteins[0]);wanted=physical_system_key(document);matches=[]
    for match in result['matches']:
        source=(match.get('setup') or {}).get('input_source') or {};path=Path(source.get('path',''))
        execution=match['execution']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=source.get('sha256'):continue
        if physical_system_key(yaml.safe_load(path.read_text()))!=wanted:continue
        if execution.get('output_presence')!='complete' or (execution.get('expected_poses') or 0)<minimum_poses:continue
        coordinates=[a for a in execution.get('artifacts') or [] if Path(a.get('path','')).suffix.lower() in {'.pdb','.cif'}]
        if len(coordinates)<minimum_poses:continue
        if any(not Path(a['path']).is_file() or hashlib.sha256(Path(a['path']).read_bytes()).hexdigest()!=a.get('sha256') for a in coordinates):continue
        matches.append(match)
    return {**result,'exact_system_matches':matches,'physical_system_key':wanted}


def lookup_predictions(campaign_dir, *, accession=None, sequence=None, target_id=None,
                       required_ligand_smiles=None):
    """Find candidate systems for review; an identity match does not authorize reuse."""
    if not accession and not sequence:
        raise ValueError('Supply a candidate identifier or exact protein sequence')
    from .catalog import digest

    root = Path(campaign_dir).resolve()
    pointer = json.loads((root / 'catalog/current.json').read_text())
    manifest_path = Path(pointer['manifest'])
    manifest = json.loads(manifest_path.read_text())
    identity = digest({'config': manifest['config_sha256'], 'files': manifest['files']})
    if (identity != manifest['content_sha256'] or identity[:24] != pointer['snapshot_id']
            or manifest['snapshot_id'] != pointer['snapshot_id']):
        raise ValueError('Prediction-pool snapshot identity changed')
    snapshot = manifest_path.parent

    def table(name):
        filename = name + '.jsonl'
        content = (snapshot / filename).read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest['files'][filename]:
            raise ValueError('Prediction-pool table changed: ' + filename)
        return [json.loads(line) for line in content.splitlines() if line]

    setups = {row['setup_id']: row for row in table('modeling_setups')}
    proteins = {row['protein_id']: row for row in table('proteins')}
    sequence_hash = hashlib.sha256(sequence.encode()).hexdigest() if sequence else None
    alias_ids = {row['protein_id'] for row in proteins.values() if sequence_hash and
                 any(version.get('sha256') == sequence_hash for version in row.get('sequence_versions', []))}
    def molecular_identity(smiles):
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError('Invalid molecular identity in pool comparison')
        return Chem.MolToSmiles(rdMolStandardize.Uncharger().uncharge(molecule))

    required = {molecular_identity(value) for value in required_ligand_smiles or []}
    matches = []
    other_target_matches = []
    for execution in table('modeling_executions'):
        setup = setups[execution['setup_id']]
        source = setup.get('input_source') or {}
        identifier = proteins[execution['protein_id']]['accession']
        basis = 'accession' if accession and identifier == accession else None
        integrity = 'unavailable'
        document = None
        if source.get('path'):
            path = Path(source['path'])
            if path.is_file():
                content = path.read_bytes()
                integrity = 'verified' if hashlib.sha256(content).hexdigest() == source.get('sha256') else 'changed'
                if integrity == 'verified':
                    document = yaml.safe_load(content)
                    sequences = {entity['protein']['sequence'] for entity in document.get('sequences', [])
                                 if 'protein' in entity}
                    if sequence and sequence in sequences:
                        basis = basis or 'exact_input_sequence'
        if basis is None and execution['protein_id'] in alias_ids:
            basis = 'recorded_sequence_alias_requires_setup_review'
        if basis:
            match = {'identity_match': basis, 'input_integrity': integrity,
                     'execution': execution, 'setup': setup}
            observed = None
            if required and document is not None:
                observed = {molecular_identity(entity['ligand']['smiles'])
                            for entity in document.get('sequences', [])
                            if (entity.get('ligand') or {}).get('smiles')}
            if observed is not None and not required <= observed:
                match['target_applicability'] = 'required_molecular_input_absent'
                other_target_matches.append(match)
            elif target_id and setup.get('target_id') != target_id and not (required and observed is not None):
                match['target_applicability'] = 'other_target_unverified_molecular_inputs'
                other_target_matches.append(match)
            else:
                match['target_applicability'] = 'requires_complete_setup_review'
                matches.append(match)
    return {'snapshot_id': pointer['snapshot_id'], 'manifest': str(manifest_path),
            'lookup_tables_verified': True, 'matches': matches,
            'other_target_matches': other_target_matches,
            'reuse_policy': 'Candidate identity is a lookup aid. Compare complete molecular inputs, numerical profile, MSA and sample provenance before reuse. New scoring alone never justifies another prediction.'}
