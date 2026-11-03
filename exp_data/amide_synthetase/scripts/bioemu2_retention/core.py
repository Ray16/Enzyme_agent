"""Geometry-only substrate retention measurements; distances are in angstroms."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def align_to(reference, mobile, selection):
    """Least-squares rigid alignment; row-vector coordinates, no outlier removal."""
    a, b = reference[selection], mobile[selection]
    if len(a) < 3 or np.linalg.matrix_rank(a-a.mean(0)) < 2:
        raise ValueError('Insufficient non-collinear pocket C-alpha atoms')
    u, _, vt = np.linalg.svd((b-b.mean(0)).T @ (a-a.mean(0)))
    parity = np.eye(3)
    parity[-1, -1] = np.linalg.det(u @ vt)
    return (mobile-b.mean(0)) @ (u @ parity @ vt) + a.mean(0)


def measure(start, end, *, protein, ligand, residue_ids, ca, atp_pa, atp_bridge,
            ligand_oxygens, mg, contact_cutoff=4.5, pocket_cutoff=6.0):
    """Pocket defined separately from each starting pose, before endpoint inspection.

    Native contacts are protein residues within 4.5 A of the substrate. The
    alignment pocket contains residues within 6 A, with a 10 A fallback recorded
    when needed. Endpoint contact retention uses residue identity, allowing ligand
    symmetry and side-chain movement without an arbitrary atom-name penalty.
    """
    start, end = np.asarray(start), np.asarray(end)
    protein, ligand = np.asarray(protein), np.asarray(ligand)
    residue_ids, ca = np.asarray(residue_ids), np.asarray(ca)
    d0 = cdist(start[protein], start[ligand]).min(axis=1)
    native = set(residue_ids[protein[d0 <= contact_cutoff]].tolist())
    if not native:
        raise ValueError('Starting substrate has no protein contacts')
    pocket = set(residue_ids[protein[d0 <= pocket_cutoff]].tolist())
    pocket_ca = ca[np.isin(residue_ids[ca], list(pocket))]
    alignment_cutoff = pocket_cutoff
    if len(pocket_ca) < 3:
        pocket = set(residue_ids[protein[d0 <= 10.0]].tolist())
        pocket_ca = ca[np.isin(residue_ids[ca], list(pocket))]
        alignment_cutoff = 10.0
    aligned = align_to(start, end, pocket_ca)
    d1 = cdist(end[protein], end[ligand]).min(axis=1)
    final = set(residue_ids[protein[d1 <= contact_cutoff]].tolist())
    displacement = float(np.linalg.norm(aligned[ligand].mean(0)-start[ligand].mean(0)))
    nearest = np.argmin(np.linalg.norm(end[ligand_oxygens]-end[atp_pa], axis=1))
    oxygen = end[ligand_oxygens[nearest]]
    v, w = oxygen-end[atp_pa], end[atp_bridge]-end[atp_pa]
    distance = float(np.linalg.norm(v))
    angle = float(np.degrees(np.arccos(np.clip(v@w/(np.linalg.norm(v)*np.linalg.norm(w)), -1, 1))))
    prot_heavy_distances = cdist(end[protein], end[ligand])
    native_contact_fraction = len(native & final)/len(native)
    return dict(native_contact_retention=native_contact_fraction,
                native_contact_count=len(native), endpoint_contact_count=len(final),
                pocket_centroid_displacement_A=displacement,
                pocket_retained=bool(native_contact_fraction >= .5 and displacement <= 5),
                pocket_ca_rmsd_A=float(np.sqrt(np.mean(np.sum((aligned[pocket_ca]-start[pocket_ca])**2, axis=1)))),
                global_ca_rmsd_A=float(np.sqrt(np.mean(np.sum((aligned[ca]-start[ca])**2, axis=1)))),
                alignment_ca_count=len(pocket_ca), alignment_cutoff_A=alignment_cutoff,
                substrate_protein_min_A=float(prot_heavy_distances.min()),
                substrate_protein_pairs_below_1p5A=int((prot_heavy_distances < 1.5).sum()),
                substrate_ATP_PA_distance_A=distance, substrate_ATP_attack_angle_deg=angle,
                substrate_Mg_min_A=float(cdist(end[ligand], end[mg]).min()),
                native_residue_indices=sorted(native), endpoint_residue_indices=sorted(final))


def selections(topology):
    atoms = list(topology.atoms)
    chain = lambda a: a.residue.chain.chain_id
    return dict(protein=[a.index for a in atoms if chain(a)=='A' and a.element.symbol!='H'],
                ligand=[a.index for a in atoms if chain(a)=='D'],
                residue_ids=[a.residue.index for a in atoms],
                ca=[a.index for a in atoms if chain(a)=='A' and a.name=='CA'],
                atp_pa=next(a.index for a in atoms if chain(a)=='B' and a.name=='PA'),
                atp_bridge=next(a.index for a in atoms if chain(a)=='B' and a.name=='O3A'),
                ligand_oxygens=[a.index for a in atoms if chain(a)=='D' and a.element.symbol=='O'],
                mg=[a.index for a in atoms if chain(a)=='C'])


def frame_metrics(start, endpoint):
    if endpoint.n_atoms != start.n_atoms:
        raise ValueError('Starting and endpoint atom counts differ')
    return measure(start.xyz[0]*10, endpoint.xyz[0]*10, **selections(start.topology))
