"""Radius convergence of the finite electrostatic-embedding shell.

The finite embedding shell retains a nonzero net charge because outer
counterions and charged residues beyond the cutoff are omitted. A net-charged
shell imposes a spurious monopole field on the QM region, and because the
retained net charge differs between systems it also breaks cross-enzyme
comparability. This module quantifies the retained-shell net charge versus
embedding radius, toward the neutral full-cell limit where the retained MM
charge equals -QM_charge and the embedded total is zero.

No GPU/DFT: region point charges and geometry only. No charge is added to force
neutrality -- that would define a different model to validate. This measures the
real finite model so radius convergence (or a decision to embed the whole
neutral cell) can be made on evidence.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .jobs import atomic_json


def _redistributed_charges(parm, region):
    """Original FF charges with the region's recorded redistribution applied.

    mm_charge_updates already zeroes QM/boundary charges and moves the residual
    onto remaining MM backbone sites, so the resulting vector is the exact set
    of MM point charges the embedded Hamiltonian would use at any radius.
    """
    charges = np.array([a.charge for a in parm.atoms], float)
    for key, val in region.get('mm_charge_updates', {}).items():
        charges[int(key)] = float(val)
    return charges


def radius_scan(snapshot, radii=(20, 24, 28, 32, 36, 40, 1e9)):
    import parmed as pmd
    root = Path(snapshot)
    region = json.loads((root / 'region.json').read_text())
    manifest = json.loads((root / 'input.json').read_text())
    parm = pmd.load_file(manifest['prmtop'])
    x = np.load(root / 'positions.npy')
    if x.shape[0] != len(parm.atoms):
        raise ValueError('positions/prmtop atom-count mismatch')
    charges = _redistributed_charges(parm, region)
    qm = np.asarray(region['indices'], int)
    qm_charge = int(region['charge'])
    if not np.allclose(charges[qm], 0.0, atol=1e-9):
        raise ValueError('QM atoms are not charge-zeroed in the redistributed set')
    # The neutral full-cell target: retained MM charge == -QM charge.
    target = -qm_charge
    full_cell_total = float(charges.sum()) + qm_charge  # ~0 for a neutral cell

    is_qm = np.zeros(len(parm.atoms), bool)
    is_qm[qm] = True
    mm_mask = (~is_qm) & (np.abs(charges) > 1e-12)
    mm_idx = np.where(mm_mask)[0]
    qxyz = x[qm]
    # minimum distance from each charged MM atom to any QM atom (chunked)
    mind = np.empty(len(mm_idx))
    for s in range(0, len(mm_idx), 2000):
        ids = mm_idx[s:s + 2000]
        d = np.linalg.norm(x[ids][:, None, :] - qxyz[None, :, :], axis=2).min(axis=1)
        mind[s:s + 2000] = d

    rows = []
    for R in radii:
        sel = mind <= R
        esum = float(charges[mm_idx[sel]].sum())
        rows.append(dict(
            radius_A=float(R),
            n_embedding=int(sel.sum()),
            retained_mm_sum_e=esum,
            embedded_total_charge_e=esum + qm_charge,
            deviation_from_neutral_e=esum - target,
        ))
    return dict(
        snapshot=str(root.resolve()),
        prmtop=manifest['prmtop'],
        qm_charge=qm_charge,
        full_cell_total_charge_e=full_cell_total,
        neutral_target_retained_sum_e=float(target),
        n_charged_mm_atoms=int(len(mm_idx)),
        scan=rows,
        note=('deviation_from_neutral_e -> 0 as the retained shell approaches the '
              'neutral primary cell; embedded_total_charge_e is the net charge the '
              'QM density actually sits in. No charge is added to force neutrality.'),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--snapshot', required=True, action='append',
                    help='snapshot dir (input.json + region.json + positions.npy); repeatable')
    ap.add_argument('--radii', default='20,24,28,32,36,40,1e9')
    ap.add_argument('--out', default=None, help='optional JSON output path')
    args = ap.parse_args()
    radii = [float(v) for v in args.radii.split(',')]
    results = [radius_scan(s, radii) for s in args.snapshot]
    for r in results:
        print(f"\n### {Path(r['snapshot']).name}  (QM={r['qm_charge']:+d}, "
              f"neutral target retained sum = {r['neutral_target_retained_sum_e']:+.0f} e)")
        print(f"  {'radius_A':>9} {'n_emb':>7} {'retained_sum':>13} {'embedded_total':>15} {'dev_from_neutral':>17}")
        for row in r['scan']:
            rr = 'full' if row['radius_A'] > 1e8 else f"{row['radius_A']:.0f}"
            print(f"  {rr:>9} {row['n_embedding']:>7} {row['retained_mm_sum_e']:>13.3f} "
                  f"{row['embedded_total_charge_e']:>15.3f} {row['deviation_from_neutral_e']:>17.3f}")
    if args.out:
        atomic_json(args.out, dict(results=results, radii=radii))
        print(f"\nwrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
