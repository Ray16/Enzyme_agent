"""Select residues contacting two explicitly specified atom groups in one pose."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import numpy as np
from ..structure import load_structure, min_distance


def group_coordinates(path, selector):
    """Select one non-polymer component by chain and optional exact atom names.

    Atom names of a generated ligand follow the predictor's own numbering, which is
    not stable across inputs; ``elements`` selects the group by chemistry instead and
    ``n_atoms`` asserts the expected size so a renumbering cannot pass silently.
    """
    names = set(selector.get('atom_names', []))
    elements = {e.upper() for e in selector.get('elements', [])}
    atoms = {}
    residues = set()
    for line in Path(path).read_text().splitlines():
        if line[:6] != 'HETATM' or line[21] != selector['chain']:
            continue
        name = line[12:16].strip()
        if names and name not in names:
            continue
        element = line[76:78].strip().upper() or name[0]
        if element in ('H', 'D'):
            continue
        if elements and element not in elements:
            continue
        if line[16] not in (' ', 'A'):
            continue
        residues.add(line[17:27])
        if name in atoms:
            raise ValueError(f'{path}: ambiguous atom {name}')
        atoms[name] = [float(line[a:b]) for a,b in ((30,38),(38,46),(46,54))]
    expected = selector.get('n_atoms')
    if (not atoms or len(residues) != 1 or (names and names != set(atoms))
            or (expected is not None and len(atoms) != expected)):
        raise ValueError(f'{path}: missing or ambiguous fragment selector {selector}')
    return np.asarray(list(atoms.values()))


def interface_positions(paths, protein_chain, fragment_a, fragment_b, cutoff=5.0,
                        min_contact_poses=1):
    """Union of same-pose intersections; never intersect separate pose unions.

    Contact counts allow downstream prioritization without hiding rare contacts.
    The caller explicitly chooses the required number of supporting poses.
    """
    paths = [Path(p) for p in paths]
    if not paths or not 1 <= min_contact_poses <= len(paths) or cutoff <= 0:
        raise ValueError('Invalid ensemble size, support count or cutoff')
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError('Duplicate ensemble paths')
    counts = Counter(); distances = {}; sequence = None
    for path in paths:
        st = load_structure(path); seq = st.sequence(protein_chain)
        if sequence is not None and seq != sequence:
            raise ValueError('WT ensemble sequences differ')
        sequence = seq
        a = group_coordinates(path, fragment_a); b = group_coordinates(path, fragment_b)
        for r in st.residues:
            if r.chain != protein_chain:
                continue
            da, db = min_distance(r.xyz,a), min_distance(r.xyz,b)
            if da <= cutoff and db <= cutoff:
                counts[r.resnum] += 1
                distances.setdefault(r.resnum, []).append((da,db))
    rows = []
    for rn,n in sorted(counts.items()):
        if n < min_contact_poses:
            continue
        rows.append({'chain':protein_chain,'resnum':rn,'wt_aa':sequence[rn-1],
                     'n_contact_poses':n,'n_poses':len(paths),'contact_fraction':n/len(paths),
                     'min_joint_max_distance_A':min(max(a,b) for a,b in distances[rn])})
    return rows
