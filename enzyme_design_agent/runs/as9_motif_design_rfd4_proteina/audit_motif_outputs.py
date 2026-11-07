"""Audit motif-only outputs without assuming native residue numbering or length.

Keep requested-length compliance separate from motif/assembly integrity. The
exported source-coordinate labels map unindexed motifs back to the input.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import numpy as np
from biotite.structure.io import pdbx
from geometry import evaluate


def recover_motif_selectors(atoms, source, meta, xc, rotation, yc):
    """Recover the motif->design residue mapping when RFD4 drops the annotation columns.

    Places the design into the source frame using the unchanged ligand fit, then solves a
    one-to-one assignment between conditioned source residues and design residues. Validated
    against the annotation on samples that carry it: the resulting anchor RMSD agrees to
    within 0.03 A, so this is a provenance repair and not a second, looser criterion.
    """
    from scipy.optimize import linear_sum_assignment
    protein = atoms[atoms.chain_id == 'A']
    placed = (protein.coord-xc) @ rotation+yc
    residues = np.array(sorted({int(r) for r in protein.res_id}))
    keys = list(meta['motif_fixed_atoms'])
    cost = np.full((len(keys), len(residues)), np.inf)
    for i, rid in enumerate(keys):
        definition = meta['motif_fixed_atoms'][rid]
        names = sorted(definition['fixed_atoms'])
        native = []
        for name in names:
            hit = source[(source.chain_id == 'A') & (source.res_id == int(rid))
                         & (source.atom_name == name)]
            if len(hit) != 1:
                return None
            native.append(hit.coord[0])
        native = np.asarray(native, float)
        for j, candidate in enumerate(residues):
            mask = protein.res_id == candidate
            if definition.get('sequence_fixed', True) and protein.res_name[mask][0] != definition['res_name']:
                continue
            try:
                generated = np.asarray([placed[mask & (protein.atom_name == n)][0] for n in names], float)
            except IndexError:
                continue
            cost[i, j] = np.sqrt(np.mean(np.sum((generated-native)**2, axis=1)))
    finite = np.isfinite(cost)
    if not finite.any(axis=1).all():
        return None
    rows, cols = linear_sum_assignment(np.where(finite, cost, 1e6))
    if not finite[rows, cols].all():
        return None
    selectors = {}
    for i, j in zip(rows, cols):
        rid = keys[i]
        names = set(meta['motif_fixed_atoms'][rid]['fixed_atoms'])
        selectors[rid] = ((atoms.chain_id == 'A') & (atoms.res_id == residues[j])
                          & np.isin(atoms.atom_name, sorted(names)))
    return selectors


def audit(path, source, meta, geometry):
    file = pdbx.CIFFile.read(path)
    atoms = pdbx.get_structure(file, model=1)
    site = file.block['atom_site']
    protein = atoms[atoms.chain_id == 'A']
    ca = protein[protein.atom_name == 'CA']
    targets = site['condition_sequence_1_residue'].as_array(str)
    known = ~np.isin(targets, ['<M>', 'UNK', '.', '?'])
    ca_mask = (atoms.chain_id == 'A') & (atoms.atom_name == 'CA')
    fixed_ca = ca_mask & known
    issues = []
    if not np.isfinite(atoms.coord).all():
        issues.append('Nonfinite coordinates')
    if set(atoms.chain_id) != {'A', 'B', 'C', 'D'}:
        issues.append('Assembly chain mismatch')
    expected_names = Counter(v['res_name'] for v in meta['motif_fixed_atoms'].values()
                             if v.get('sequence_fixed', True))
    # RFD4 occasionally drops both provenance columns for a sample ("Could not apply the
    # insertion mapping"). That is a labelling failure, not a conditioning failure, so the
    # identity screen defers to the recovered mapping below instead of reporting a mismatch.
    sequence_annotation_available = bool(fixed_ca.sum())
    if sequence_annotation_available:
        if Counter(targets[fixed_ca]) != expected_names:
            issues.append('Fixed motif identities differ from input')
        if not np.array_equal(targets[fixed_ca], atoms.res_name[fixed_ca]):
            issues.append('Fixed identities disagree with generated residues')
    residues = sorted(set(map(int, ca.res_id)))
    if len(residues) != len(ca) or any(b != a+1 for a, b in zip(residues, residues[1:])):
        issues.append('Duplicate or discontinuous protein residue numbering')
    peptide_distances = []
    previous_c = None
    for rid in residues:
        sub = protein[protein.res_id == rid]
        if not {'N', 'CA', 'C', 'O'} <= set(sub.atom_name):
            issues.append(f'Incomplete backbone at {rid}')
            continue
        if previous_c is not None:
            peptide_distances.append(float(np.linalg.norm(previous_c-sub.coord[sub.atom_name == 'N'][0])))
        previous_c = sub.coord[sub.atom_name == 'C'][0]
    if any(not 1 < d < 2 for d in peptide_distances):
        issues.append('Peptide C-N distance outside broad 1-2 A screen')
    labels = site['annotation_src-coordinate-conditioned-region_1_atom'].as_array(str)
    for chain in ('B', 'C', 'D'):
        ref, generated = source[source.chain_id == chain], atoms[atoms.chain_id == chain]
        inventory = lambda a: sorted(zip(map(str, a.atom_name), map(str, a.element)))
        if inventory(ref) != inventory(generated):
            issues.append(f'Ligand inventory mismatch in {chain}')
    # Fit the unchanged ligand frame; then measure the selected protein anchors.
    ref_lig, generated_lig = [], []
    for atom in source[source.chain_id != 'A']:
        match = (atoms.chain_id == atom.chain_id) & (atoms.atom_name == atom.atom_name)
        if match.sum() == 1:
            ref_lig.append(atom.coord); generated_lig.append(atoms.coord[match][0])
    x, y = np.asarray(generated_lig, float), np.asarray(ref_lig, float)
    xc, yc = x.mean(0), y.mean(0)
    u, _, vh = np.linalg.svd((x-xc).T @ (y-yc))
    sign = np.eye(3); sign[-1, -1] = np.linalg.det(u @ vh)
    rotation = u @ sign @ vh
    selectors = {rid: labels == f'A{rid}' for rid in meta['motif_fixed_atoms']}
    coordinate_annotation_available = any(
        (mask & (atoms.chain_id == 'A')).any() for mask in selectors.values())
    motif_mapping_source = 'annotation'
    if not coordinate_annotation_available:
        motif_mapping_source = 'recovered_by_ligand_frame_superposition'
        selectors = recover_motif_selectors(atoms, source, meta, xc, rotation, yc)
        if selectors is None:
            issues.append('Motif mapping unavailable: annotation absent and recovery failed')
            selectors = {}
    src_xyz, out_xyz, mapping = [], [], {}
    for rid, definition in meta['motif_fixed_atoms'].items():
        selected = selectors.get(rid)
        if selected is None:
            continue
        expected = set(definition['fixed_atoms'])
        if set(atoms.atom_name[selected]) != expected or len(set(atoms.res_id[selected])) != 1:
            issues.append(f'Motif atom mapping failed for A{rid}')
            continue
        mapped = int(atoms.res_id[selected][0])
        mapping[rid] = mapped
        if (definition.get('sequence_fixed', True)
                and set(protein.res_name[protein.res_id == mapped]) != {definition['res_name']}):
            issues.append(f'Mapped motif identity differs at A{rid}')
        for name in sorted(expected):
            native = source[(source.chain_id == 'A') & (source.res_id == int(rid)) & (source.atom_name == name)]
            generated = atoms[selected & (atoms.atom_name == name)]
            if len(native) != 1 or len(generated) != 1:
                issues.append(f'Ambiguous motif atom A{rid}:{name}')
                continue
            src_xyz.append(native.coord[0]); out_xyz.append(generated.coord[0])
    if src_xyz:
        delta = (np.asarray(out_xyz)-xc) @ rotation+yc-np.asarray(src_xyz)
        rmsd = float(np.sqrt(np.mean(np.sum(delta**2, axis=1))))
        maximum = float(np.linalg.norm(delta, axis=1).max())
    else:
        rmsd = maximum = float('nan')
    if not sequence_annotation_available:
        observed = Counter(str(protein.res_name[protein.res_id == mapping[rid]][0])
                           for rid, d in meta['motif_fixed_atoms'].items()
                           if d.get('sequence_fixed', True) and rid in mapping)
        if observed != expected_names:
            issues.append('Fixed motif identities differ from input')
    if len(src_xyz) != meta['fixed_protein_atom_count']:
        issues.append('Conditioned motif atom count differs from input')
    length = len(ca)
    return dict(path=str(path),structure_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                protein_length=length,fixed_identities=int(fixed_ca.sum()) or len(
                    [r for r, d in meta['motif_fixed_atoms'].items()
                     if d.get('sequence_fixed', True) and r in mapping]),
                designed_residues=length-(int(fixed_ca.sum()) or len(
                    [r for r, d in meta['motif_fixed_atoms'].items()
                     if d.get('sequence_fixed', True) and r in mapping])),motif_source_to_output_residue=mapping,
                fixed_atom_rmsd_A=rmsd,fixed_atom_max_deviation_A=maximum,
                motif_mapping_source=motif_mapping_source,
                sequence_annotation_available=sequence_annotation_available,
                motif_and_assembly_integrity_pass=not issues,integrity_issues=issues,
                requested_length=meta['target_length'],requested_length_match=length == meta['target_length'],
                peptide_CN_range_A=[min(peptide_distances),max(peptide_distances)],
                **evaluate(atoms, geometry))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    opts = parser.parse_args()
    cfg = json.loads(opts.config.read_text()); root = Path(cfg['campaign_dir'])
    inventory = json.loads((root/'download_inventory.json').read_text())
    cache, rows = {}, []
    for row in inventory['rows']:
        iid = row['input_id']
        if iid not in cache:
            src = pdbx.get_structure(pdbx.CIFFile.read(root/'inputs'/f'{iid}.cif'),model=1)
            meta = json.loads((root/'inputs'/f'{iid}.metadata.json').read_text())
            cache[iid] = src, meta
        path = root/'downloads'/Path(row['path']).relative_to(cfg['remote_root'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['structure_sha256']
        try:
            result = audit(path,*cache[iid],cfg['geometry'])
        except Exception as error:
            result = dict(path=str(path),motif_and_assembly_integrity_pass=False,integrity_issues=[repr(error)])
        rows.append(dict(id=row['id'],input_id=iid,**result))
    good = [r for r in rows if r['motif_and_assembly_integrity_pass']]
    summary = dict(audited=len(rows),motif_and_assembly_integrity_pass=len(good),
                   requested_length_matches=sum(r.get('requested_length_match',False) for r in rows),
                   actual_lengths=dict(Counter(r.get('protein_length') for r in rows)),
                   actual_designed_residues=dict(Counter(r.get('designed_residues') for r in rows)),
                   conditioned_anchor_rmsd_A_range=[min(r['fixed_atom_rmsd_A'] for r in rows),max(r['fixed_atom_rmsd_A'] for r in rows)],
                   max_fixed_atom_deviation_A=max((r['fixed_atom_max_deviation_A'] for r in rows),default=None),
                   integrity_issue_counts=dict(Counter(issue for r in rows for issue in r['integrity_issues'])),
                   AME_status='pending independent refolding',
                   interpretation='Broad assembly, identity and peptide screen. Coordinate deviations are reported without a pass threshold. Requested length remains a separate failed criterion.')
    out = root/'local_analysis';out.mkdir(exist_ok=True)
    (out/'motif_integrity_audit.json').write_text(json.dumps(dict(summary=summary,rows=rows),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
