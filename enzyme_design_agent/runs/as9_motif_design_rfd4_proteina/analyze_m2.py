import json, collections
import numpy as np
from pathlib import Path
from scipy import stats

CAMPAIGNS = {
    'M0M1': Path('/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/enzyme_design_agent/runs/rfd4_campaigns/as9_motif_m0_m1_20261106/boltz_refolding_20261106'),
    'M2':   Path('/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/enzyme_design_agent/runs/rfd4_campaigns/as9_motif_m2_substrate_20261107/boltz_refolding_20261107'),
}

def load(R):
    arm = {r['id']: Path(r['motif_input']).stem for r in json.loads((R/'refold_inputs.json').read_text())}
    rows = []
    for f in sorted((R/'metrics').glob('*.json')):
        d = json.loads(f.read_text()); i = f.stem
        g = [p['geometry'] for p in d['poses'] if p.get('geometry')]
        if not g: continue
        ds = np.array([x['distance_A'] for x in g]); an = np.array([x['angle_deg'] for x in g])
        st = [p['structural'] for p in d['poses'] if p.get('structural')]
        bb = np.array([s['backbone']['rmsd_A'] for s in st if 'backbone' in s])
        cond = np.array([s['conditioned']['selection_aligned_to_input']['rmsd_A'] for s in st
                         if s.get('conditioned', {}).get('selection_aligned_to_input')])
        side = []
        for s in st:
            v = s.get('fixed_sidechains') or s.get('sidechains') or {}
            v = v.get('selection_aligned_to_input') if isinstance(v, dict) else None
            if v: side.append(v['rmsd_A'])
        ens = d['ensemble'].get('all_expected_poses') or d['ensemble'].get('available_pose_subset') or {}
        a = arm.get(i, '?')
        rows.append(dict(id=i, arm=a[:2], pose=a, n=len(g),
            A=ens.get('weighted_attack_angle_deg'), dh=ens.get('harmonic_distance_A'),
            frac4=float((ds < 4).mean()), near=int(((ds < 4) & (an > 150)).sum()),
            bb=float(bb.mean()) if len(bb) else None,
            cond=float(cond.mean()) if len(cond) else None,
            side=float(np.mean(side)) if side else None))
    return rows

rows = []
for R in CAMPAIGNS.values():
    rows += load(R)

def st_(v):
    v = [x for x in v if x is not None]
    if not v: return (float('nan'),)*3
    return (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0, len(v))

by = collections.defaultdict(list)
for r in rows: by[r['arm']].append(r)

print(f"scored designs: {len(rows)}  by arm: {dict((k,len(v)) for k,v in sorted(by.items()))}\n")
print(f"{'arm':6s} {'n':>3s} {'A mean±sd':>16s} {'d_h':>7s} {'frac<4A':>8s} {'near/poses':>11s} {'bbRMSD':>7s} {'motifRMSD':>9s} {'sideRMSD':>8s}")
for a in sorted(by):
    g = by[a]; A = st_([r['A'] for r in g])
    near = sum(r['near'] for r in g); npose = sum(r['n'] for r in g)
    print(f"{a:6s} {len(g):3d} {A[0]:8.2f}±{A[1]:5.2f} {st_([r['dh'] for r in g])[0]:7.2f} "
          f"{np.mean([r['frac4'] for r in g]):7.1%} {near:5d}/{npose:<5d} "
          f"{st_([r['bb'] for r in g])[0]:7.3f} {st_([r['cond'] for r in g])[0]:9.3f} {st_([r['side'] for r in g])[0]:8.3f}")

# M2 vs M0/M1 pooled
m01 = [r for r in rows if r['arm'] in ('M0', 'M1')]
m2 = [r for r in rows if r['arm'] == 'M2']
print(f"\n=== M2 (n={len(m2)}) vs M0/M1 pooled (n={len(m01)}) ===")
for k, lab in [('A', 'A (deg)'), ('dh', 'harmonic d_h (A)'), ('frac4', 'frac d<4A')]:
    a = [r[k] for r in m01 if r[k] is not None]; b = [r[k] for r in m2 if r[k] is not None]
    u = stats.mannwhitneyu(b, a, alternative='two-sided')
    print(f"  {lab:18s} M0/M1 {np.mean(a):8.3f}  M2 {np.mean(b):8.3f}  diff {np.mean(b)-np.mean(a):+8.3f}  MWU p={u.pvalue:.4f}")

# motif reproduction: M2 conditions sidechains, so sideRMSD is the relevant fidelity
print("\n=== Does conditioning the pocket move docking toward WT? ===")
WT_A, WT_D, WT_F = 156.6, 4.242, 0.28  # charged anchor
print(f"WT_AS9 charged anchor: A {WT_A}  d_h {WT_D} A  frac<4A {WT_F:.0%}")
for a in ('M0', 'M1', 'M2'):
    g = by[a]
    print(f"  {a}: A>150 designs {sum(1 for r in g if r['A'] and r['A']>150)}/{len(g)}  "
          f"d_h<4.5 {sum(1 for r in g if r['dh'] and r['dh']<4.5)}/{len(g)}  "
          f"near-attack poses {sum(r['near'] for r in g)}/{sum(r['n'] for r in g)}  "
          f"({sum(r['near'] for r in g)/max(sum(r['n'] for r in g),1):.1%})")

# best M2 designs
print("\nTop 5 M2 designs by d_h (closest docking):")
for r in sorted([r for r in m2 if r['dh']], key=lambda r: r['dh'])[:5]:
    print(f"  {r['id']} {r['pose']}  A={r['A']:.1f}  d_h={r['dh']:.2f}  frac<4A={r['frac4']:.0%}  near={r['near']}  motifRMSD={r['cond']:.2f}  sideRMSD={r['side']}")

# Among M2 designs that reproduce the pocket faithfully (side RMSD < 1.5), does docking improve?
faith = [r for r in m2 if r['side'] is not None and r['side'] < 1.5]
print(f"\nM2 designs reproducing fixed sidechains <1.5A: {len(faith)}/{len(m2)}")
if faith:
    print(f"  their docking: A {np.mean([r['A'] for r in faith]):.1f}  d_h {np.mean([r['dh'] for r in faith]):.2f}  "
          f"frac<4A {np.mean([r['frac4'] for r in faith]):.1%}  near {sum(r['near'] for r in faith)}/{sum(r['n'] for r in faith)}")

# ---- robust medians + faithful-subset diagnostic (replicate M0/M1 decisive test) ----
print("\n\n=== ROBUST: per-arm medians (mean skewed by refold-diverged poses) ===")
print(f"{'arm':6s} {'medBB':>7s} {'medMotif':>9s} {'medSide':>8s}")
def med(v): 
    v=[x for x in v if x is not None]; return float(np.median(v)) if v else float('nan')
for a in sorted(by):
    g=by[a]
    print(f"{a:6s} {med([r['bb'] for r in g]):7.3f} {med([r['cond'] for r in g]):9.3f} {med([r['side'] for r in g]):8.3f}")

print("\n=== Faithful-motif subset (motifRMSD<1.5 A) — the M0/M1 decisive diagnostic ===")
for a in ('M0','M1','M2'):
    g=[r for r in by[a] if r['cond'] is not None and r['cond']<1.5]
    gall=by[a]
    if g:
        print(f"  {a}: {len(g)}/{len(gall)} faithful | their docking A {np.mean([r['A'] for r in g]):.1f}  "
              f"d_h {np.mean([r['dh'] for r in g]):.2f}  frac<4A {np.mean([r['frac4'] for r in g]):.1%}  "
              f"near {sum(r['near'] for r in g)}/{sum(r['n'] for r in g)}")
    else:
        print(f"  {a}: 0/{len(gall)} faithful (motifRMSD<1.5)")
print("\n=== Faithful <2.0 A (looser) ===")
for a in ('M0','M1','M2'):
    g=[r for r in by[a] if r['cond'] is not None and r['cond']<2.0]; gall=by[a]
    if g:
        print(f"  {a}: {len(g)}/{len(gall)} | docking A {np.mean([r['A'] for r in g]):.1f}  d_h {np.mean([r['dh'] for r in g]):.2f}  frac<4A {np.mean([r['frac4'] for r in g]):.1%}")
    else:
        print(f"  {a}: 0/{len(gall)}")
