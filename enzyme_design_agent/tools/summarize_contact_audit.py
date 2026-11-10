"""Summarize full-campaign contact sensitivity and export candidate audit tables."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr


def correlation(x, y, seed=764, bootstrap=2000):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 4:
        return dict(n=len(pairs), rho=None, p_unadjusted=None, ci95=None)
    a, b = np.asarray(pairs, float).T
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return dict(n=len(pairs), rho=None, p_unadjusted=None, ci95=None)
    r, p = spearmanr(a, b)
    idx = np.random.default_rng(seed).integers(0, len(a), size=(bootstrap, len(a)))
    ar, br = rankdata(a[idx], axis=1), rankdata(b[idx], axis=1)
    ar -= ar.mean(axis=1, keepdims=True)
    br -= br.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.sum(ar*ar, axis=1)*np.sum(br*br, axis=1))
    valid = denom > 0
    boot = np.sum(ar[valid]*br[valid], axis=1) / denom[valid]
    return dict(n=len(pairs), rho=float(r), p_unadjusted=float(p),
                ci95=[float(v) for v in np.quantile(boot, [.025, .975])],
                bootstrap_unit='design', bootstrap_replicates=bootstrap)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(Path(p).read_text()) for p in json.loads((args.audit/'result_index.json').read_text())]
    summary, tests = {}, []
    for camp in ['M3', 'M4']:
        subset = [r for r in rows if r['campaign'] == camp and r['ident'] != 'WT_AS9']
        for predictor in ['n_old_flat36', 'n_typed_with_ambiguous', 'n_contacts', 'n_strict', 'n_loose']:
            for outcome in ['A', 'dh', 'nac']:
                test = dict(campaign=camp, predictor=predictor, outcome=outcome,
                            **correlation([r[predictor] for r in subset], [r['geometry'][outcome] for r in subset]))
                tests.append(test)
        for confidence in ['mean_global_plddt', 'mean_pocket_plddt']:
            tests.append(dict(campaign=camp, predictor='mean_consistency', outcome=confidence,
                              **correlation([r['mean_consistency'] for r in subset], [r[confidence] for r in subset])))
        panel = [r for r in rows if r['campaign'] == camp and r['panel_cell']]
        cons = [c['consistency'] for r in subset for c in r['group_contacts'] if not c['conditional']]
        hist, edges = np.histogram(cons, bins=np.linspace(0, 1, 11))
        summary[camp] = dict(designs=len(subset), panel_cells=len(panel),
                             unique_poses=sum(r['valid_unique_poses'] for r in subset),
                             panel_old_allpairs_lines=sum(r['old_allpairs38_lines'] for r in panel),
                             panel_new_contact_lines=sum(r['new_display_lines'] for r in panel),
                             panel_ambiguous_groups=sum(r['n_conditional'] for r in panel),
                             histogram=hist.tolist(), histogram_edges=edges.tolist(),
                             median_consistency=float(np.median(cons)),
                             counts={k: float(np.mean([r[k] for r in subset])) for k in
                                     ['n_old_flat36', 'n_typed_with_ambiguous', 'n_contacts', 'n_strict', 'n_loose']})
    # Report every comparison; corrections include all nonconstant comparisons.
    valid = [t for t in tests if t['p_unadjusted'] is not None]
    order = sorted(range(len(valid)), key=lambda i: valid[i]['p_unadjusted'])
    previous = 1.
    for rank in range(len(order), 0, -1):
        test = valid[order[rank-1]]
        previous = min(previous, test['p_unadjusted']*len(order)/rank)
        test['p_bh'] = previous
    (args.audit/'summary.json').write_text(json.dumps(dict(campaigns=summary, correlations=tests), indent=2)+'\n')
    panel = [r for r in rows if r['panel_cell']]
    group_rows, atom_rows = [], []
    for r in rows:
        for c in r['group_contacts']:
            p = c['pairs'][0]
            group_rows.append(dict(campaign=r['campaign'], design=r['ident'], panel=bool(r['panel_cell']),
                                   residue=c['res']+c['ri'], role=c['where'], classification=c['kind'],
                                   plotted=not c['conditional'], conditional=c['conditional'],
                                   uncertainty_reasons='; '.join(c['uncertainty_reasons']),
                                   donor_atom=p['atom'], acceptor_atom=p['O'], distance_A=p['d'],
                                   prediction_consistency=c['consistency'], n=c['consistency_n'],
                                   denominator=c['consistency_denominator'],
                                   fixed_group_consistency=c['fixed_group_consistency'],
                                   qualifying_atom_pairs=c['atom_pair_count'],
                                   independent_atom_pairs=len(c['independent_pairs']),
                                   drawn_lines=0 if c['conditional'] else 1,
                                   raw_pairs=json.dumps(c['independent_pairs'])))
        for c in r['raw_contacts']:
            atom_rows.append(dict(campaign=r['campaign'], design=r['ident'], panel=bool(r['panel_cell']), **c))
    for name, records in [('group_contacts.csv', group_rows), ('atom_pair_audit.csv', atom_rows)]:
        with (args.audit/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    text = ['# Candidate contact audit', '',
            'Primary contacts use the versioned heavy-atom policy. Conditional His/Cys and anomalously short groups are listed separately and not drawn. '
            'A two-point pattern describes atom-disjoint close pairs, not proof of bidentate hydrogen bonding. '
            'Prediction consistency is frequency across valid unique Boltz samples, not occupancy. '
            'For a flagged/omitted group, the displayed fraction instead records recurrence of that flagged status.', '']
    for r in sorted(panel, key=lambda r: (r['campaign'], r['ident'] != 'WT_AS9', r['ident'])):
        text += [f'## {r["campaign"]} {r["ident"]}', '',
                 f'{r["n_distal"]} distal + {r["n_attack"]} attack contacts; {r["n_contacts"]} distinct residues. '
                 f'{r["n_conditional"]} conditional groups omitted. {r["valid_unique_poses"]} unique poses.', '',
                 f'![Local contacts](cells/{r["campaign"]}/{r["ident"]}.png)', '',
                 '| Residue | End | Class | Closest atoms | Distance (Å) | Consistency | Independent pairs | Drawn |',
                 '|---|---|---|---|---:|---:|---:|---|']
        for c in sorted(r['group_contacts'], key=lambda c: (c['conditional'], c['where'], c['pairs'][0]['d'])):
            p = c['pairs'][0]
            kind = 'potential ionic' if c['kind'] == 'potential_ionic' else 'polar donor–acceptor'
            reason = '; '.join(c['uncertainty_reasons']) if c['conditional'] else 'yes'
            text.append(f'| {c["res"]}{c["ri"]} | {c["where"]} | {kind} | {p["atom"]}···{p["O"]} | '
                        f'{p["d"]:.2f} | {c["consistency_n"]}/{c["consistency_denominator"]} | '
                        f'{len(c["independent_pairs"])} | {reason} |')
        if not r['group_contacts']:
            text.append('| — | — | No compatible sidechain contact in scope | — | — | — | — | — |')
        text.append('')
    (args.audit/'candidate_audit.md').write_text('\n'.join(text))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.5), layout='constrained')
    for col, camp in enumerate(['M3', 'M4']):
        subset = [r for r in rows if r['campaign'] == camp and r['ident'] != 'WT_AS9']
        cons = [c['consistency'] for r in subset for c in r['group_contacts'] if not c['conditional']]
        axes[0, col].hist(cons, bins=np.linspace(0, 1, 11), color='#0072B2', edgecolor='white')
        axes[0, col].axvline(.5, ls=':', color='black', lw=1)
        axes[0, col].set(xlabel='Prediction consistency', ylabel='Representative group contacts', title=f'{camp}: all 100 designs')
        available = [r for r in subset if r['mean_consistency'] is not None]
        axes[1, col].scatter([r['mean_pocket_plddt'] for r in available], [r['mean_consistency'] for r in available],
                             s=12, alpha=.7, color='#D55E00')
        axes[1, col].set(xlabel='Pocket-local pLDDT (0–100)', ylabel='Mean contact prediction consistency', ylim=(0, 1.02))
    for ax in axes.ravel():
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
    fig.savefig(args.audit/'consistency_diagnostics.png', dpi=300)
    fig.savefig(args.audit/'consistency_diagnostics.pdf')
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
