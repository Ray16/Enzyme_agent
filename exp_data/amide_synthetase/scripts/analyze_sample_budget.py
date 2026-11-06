"""Retrospective sample-budget study of the frozen one-Mg activation panel.

Run from any directory with the repository scientific Python environment:
  .venv/bin/python exp_data/amide_synthetase/scripts/analyze_sample_budget.py

Each repetition independently permutes the 50 poses within each enzyme. Prefixes
of those permutations give nested uniform subsets WITHOUT replacement. The same
subsets are used for every descriptor. Experimental labels and all descriptor
definitions remain fixed. Quantiles describe pose-selection variability on this
panel, not confidence intervals for new enzymes or experimental measurements.
"""

import os

THREAD_KEYS = ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OPENBLAS_NUM_THREADS',
               'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
for key in THREAD_KEYS:
    os.environ[key] = '1'
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
DEFAULT_OUT = BASE / 'results/analysis/sample_budget_20261105'
COUNTS = tuple(range(5, 51, 5))
PRIMARY = 'weighted_angle'


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources(out):
    paths = [BASE / 'results/structural_proxies/tables/continuous_pose_geometry.csv',
             BASE / 'assay_data/panel_activity.csv',
             BASE / 'results/structural_proxies/attempts/geometry_proxy_screen/matched_pose_features.npz',
             BASE / 'results/structural_proxies/attempts/proxy_performance_20261011/per_enzyme_scores.csv',
             BASE / 'results/structural_proxies/tables/synthetase_fact_check.json']
    groups = {}
    for row in read_csv(paths[0]):
        if row['group'] == 'calibration' and row['id'] != 'SfaB':
            groups.setdefault(row['id'], []).append(row)
    ids = sorted(groups, key=lambda name: int(name[2:]))
    assert len(ids) == 48
    assert all(len(groups[i]) == 50 and len({r['model'] for r in groups[i]}) == 50 for i in ids)
    assert all({int(r['model'].split('_model_')[1].split('.')[0]) for r in groups[i]} == set(range(50)) for i in ids)
    d = np.array([[float(r['distance_A']) for r in groups[i]] for i in ids])
    a = np.array([[float(r['angle_deg']) for r in groups[i]] for i in ids])
    assert np.isfinite(d).all() and np.isfinite(a).all() and (d > 0).all()
    assert ((a >= 0) & (a <= 180)).all()
    assay = {r['id']: r for r in read_csv(paths[1])}
    activity = np.array([float(assay[i]['activity_N66L1_ugml_avg']) for i in ids])
    reference = float(assay['SfaB']['activity_N66L1_ugml_avg'])
    positive = activity > reference
    strong = np.isin(ids, ['AS9', 'AS17', 'AS23'])
    assert positive.sum() == 7 and strong.sum() == 3
    assert set(np.array(ids)[np.argsort(-activity)[:3]]) == set(np.array(ids)[strong])
    features = np.load(paths[2])
    assert features['ids'].tolist() == ids
    # The cache uses the original table's within-enzyme row order (lexical model
    # filenames), not numeric model order. Keep that correspondence unchanged.
    arrays = {k: features[k] for k in ['center', 'direction', 'polar_contact', 'overlap']}
    arrays.update(distance=d, angle=a)
    audit = json.loads(paths[4].read_text())['calibration']['raw_geometry']
    confidence_hash = hashlib.sha256()
    confidence = []
    exported = []
    for j, name in enumerate(ids):
        directory = ROOT / audit[name]['directory']
        values = []
        for k, row in enumerate(groups[name]):
            path = directory / ('confidence_' + row['model'].replace('.pdb', '.json'))
            raw = path.read_bytes()
            confidence_hash.update(raw)
            matrix = json.loads(raw)['pair_chains_iptm']
            value = (matrix['0']['3'] + matrix['3']['0']) / 2
            values.append(value)
            record = dict(id=name, model=row['model'], distance_A=d[j, k],
                          angle_deg=a[j, k], interface_confidence=value)
            for key in ['center', 'direction']:
                record.update({f'{key}_{axis}': arrays[key][j, k, z] for z, axis in enumerate('xyz')})
            record.update(polar_contact=arrays['polar_contact'][j, k], overlap=arrays['overlap'][j, k])
            exported.append(record)
        confidence.append(values)
    arrays['confidence'] = np.array(confidence)
    assert all(np.isfinite(value).all() for value in arrays.values())
    assert np.allclose(np.linalg.norm(arrays['direction'], axis=-1), 1)
    write_csv(out / 'matched_pose_features.csv', exported)
    prior = {r['id']: r for r in read_csv(paths[3])}
    return ids, arrays, activity, reference, positive, strong, prior, {
        str(p.relative_to(ROOT)): digest(p) for p in paths
    }, confidence_hash.hexdigest()


def permuted_statistics(arrays, indices):
    """Cumulative sufficient statistics, shared by all requested prefix sizes."""
    batch, enzymes, poses = indices.shape
    result = {}
    for key, value in arrays.items():
        ix = indices[..., None] if value.ndim == 3 else indices
        selected = np.take_along_axis(value[None], ix, axis=2)
        if value.ndim == 3:
            result[key] = np.cumsum(selected, axis=2)
            if key == 'center':
                result['center_square'] = np.cumsum((selected ** 2).sum(axis=-1), axis=2)
        else:
            result[key] = np.cumsum(selected, axis=2)
        if key == 'distance':
            d = selected
        elif key == 'angle':
            a = selected
    w = 1 / d
    near = d < 4
    for key, value in dict(weight=w, weight_square=w*w, angle_weight=a*w,
                           cosine_weight=-np.cos(np.deg2rad(a))*w,
                           near_count=near, near_weight=w*near,
                           near_angle_weight=a*w*near,
                           near_attack=near & (a >= 150)).items():
        result[key] = np.cumsum(value, axis=2)
    result['minimum_distance'] = np.minimum.accumulate(d, axis=2)
    return result


def prefix_scores(stats, n):
    s = {key: value[:, :, n-1] for key, value in stats.items()}
    near_angle = np.divide(s['near_angle_weight'], s['near_weight'],
                           out=np.full_like(s['near_weight'], np.nan), where=s['near_count'] >= 5)
    spread = np.sqrt(np.maximum(0, s['center_square']/n - ((s['center']/n)**2).sum(axis=-1)))
    return {
        'weighted_angle': s['angle_weight']/s['weight'],
        'mean_angle': s['angle']/n,
        'weighted_cosine': s['cosine_weight']/s['weight'],
        'angle_over_distance': s['angle_weight']/n,
        'cosine_over_distance': s['cosine_weight']/n,
        'gated_angle': near_angle,
        'near_attack_fraction': s['near_attack']/n,
        'nearby_fraction': s['near_count']/n,
        'minimum_distance': -s['minimum_distance'],
        'interface_confidence': s['confidence']/n,
        'direction_consistency': np.linalg.norm(s['direction']/n, axis=-1),
        'donor_dispersion': -spread,
        'polar_contact_fraction': s['polar_contact']/n,
        'overlap': -s['overlap']/n,
    }, n/s['weight'], s['weight']**2/s['weight_square']


def membership(scores, k):
    s = np.where(np.isfinite(scores), scores, -np.inf)
    boundary = np.partition(s, -k, axis=-1)[:, -k: -k+1] if k != 1 else s.max(axis=-1, keepdims=True)
    above, tied = s > boundary, s == boundary
    return above + tied * ((k-above.sum(axis=-1))/tied.sum(axis=-1))[:, None]


def auc(scores, labels):
    s = np.where(np.isfinite(scores), scores, -np.inf)
    p, n = s[:, labels], s[:, ~labels]
    return ((p[:, :, None] > n[:, None, :]) + .5*(p[:, :, None] == n[:, None, :])).mean(axis=(1, 2))


def evaluation(scores, positive, strong):
    inclusion = membership(scores, 10)
    return dict(hits7_at10=inclusion[:, positive].sum(axis=1),
                hits3_at10=inclusion[:, strong].sum(axis=1),
                auc7=auc(scores, positive), auc3=auc(scores, strong))


def row_correlation(left, right):
    l = left-left.mean(axis=-1, keepdims=True)
    r = right-right.mean(axis=-1, keepdims=True)
    return (l*r).sum(axis=-1)/np.sqrt((l*l).sum(axis=-1)*(r*r).sum(axis=-1))


def summarize(values):
    v = np.asarray(values)
    return dict(mean=float(v.mean()), q025=float(np.quantile(v, .025)),
                median=float(np.median(v)), q975=float(np.quantile(v, .975)))


def minimum_count(summary, key, threshold):
    return next((int(r['samples']) for r in summary if r[key] >= threshold), None)


def draw_plots(out, summary, inclusion_rows, baseline):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    x = np.array([r['samples'] for r in summary])
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8), constrained_layout=True)
    axes[0, 0].plot(x, [r['p_all3_at10'] for r in summary], 'o-', color='#0072B2', label='All three strongest producers')
    axes[0, 0].plot(x, [r['p_all3_at15'] for r in summary], '^-', color='#009E73', label='All three, shortlist widened to 15')
    axes[0, 0].plot(x, [r['p_at_least4_of7_at10'] for r in summary], 's-', color='#D55E00', label='At least 4 of 7 above SfaB')
    axes[0, 0].axhline(.95, color='.4', ls=':', label='95% of subsets')
    axes[0, 0].set(ylabel='Fraction of random subsets', ylim=(0, 1.03), title='A   Preserve experimental shortlist signal')
    axes[0, 0].legend(fontsize=8, loc='lower right')
    for key, color, label in [('auc3', '#0072B2', 'Three strongest'), ('auc7', '#D55E00', 'Seven above SfaB')]:
        axes[0, 1].plot(x, [r[f'{key}_mean'] for r in summary], 'o-', color=color, label=label)
        axes[0, 1].fill_between(x, [r[f'{key}_q025'] for r in summary], [r[f'{key}_q975'] for r in summary], color=color, alpha=.15)
        axes[0, 1].axhline(baseline[key], color=color, ls=':', lw=1)
    axes[0, 1].set(ylabel='AUROC', ylim=(.45, 1.01), title='B   Discriminate experimental producers')
    axes[0, 1].legend(fontsize=8, loc='lower right')
    for name, color in [('AS9', '#0072B2'), ('AS17', '#009E73'), ('AS23', '#D55E00')]:
        rows = [r for r in inclusion_rows if r['id'] == name]
        axes[1, 0].plot(x, [r['p_top10'] for r in rows], 'o-', label=name, color=color)
    axes[1, 0].axhline(.95, color='.4', ls=':')
    axes[1, 0].set(ylabel='Probability of inclusion in top 10', ylim=(0, 1.03), title='C   Recover each strongest producer')
    axes[1, 0].legend(fontsize=8, loc='lower right')
    axes[1, 1].plot(x, [r['overlap10_mean'] for r in summary], 'o-', color='#8064A2')
    axes[1, 1].fill_between(x, [r['overlap10_q025'] for r in summary], [r['overlap10_q975'] for r in summary], color='#8064A2', alpha=.15)
    axes[1, 1].set(ylabel='Shared candidates with full-50 top 10', ylim=(0, 10.3), title='D   Reproduce the exact shortlist')
    for ax in axes.flat:
        ax.set(xlabel='Structures per enzyme', xticks=x)
        ax.grid(alpha=.15)
    fig.suptitle('Frozen proximity-weighted attack angle on 48 assayed enzymes\n10,000 random subsets per budget; shading = central 95% of subset outcomes', fontsize=12)
    # Keep the actual repetition count correct for small verification runs.
    fig._suptitle.set_text(fig._suptitle.get_text().replace('10,000', f"{int(summary[0]['repetitions']):,}"))
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(out/f'sample_budget.{extension}', dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions', type=int, default=10000)
    parser.add_argument('--bootstrap-repetitions', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=20261105)
    parser.add_argument('--batch-size', type=int, default=250)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    assert args.repetitions > 0 and args.bootstrap_repetitions >= 0 and args.batch_size > 0
    start = time.perf_counter()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ids, arrays, activity, reference, positive, strong, prior, source_hashes, confidence_hash = load_sources(out)
    identity = np.broadcast_to(np.arange(50), (1, 48, 50))
    baseline_scores, full_dh, full_ess = prefix_scores(permuted_statistics(arrays, identity), 50)
    for key, value in baseline_scores.items():
        expected = np.array([float(prior[i][key]) for i in ids])
        assert np.allclose(value[0], expected, atol=1e-9, equal_nan=True), key
    baseline = {k: float(v[0]) for k, v in evaluation(baseline_scores[PRIMARY], positive, strong).items()}
    assert baseline['hits3_at10'] == 3 and baseline['hits7_at10'] == 4
    assert np.isclose(baseline['auc3'], 127/135) and np.isclose(baseline['auc7'], 29/41)
    full_score = baseline_scores[PRIMARY][0]
    full_rank = rankdata(-full_score)
    full_top10 = membership(full_score[None], 10)[0]
    activity_rank = rankdata(activity)
    baseline['spearman_activity'] = float(row_correlation(rankdata(full_score)[None], activity_rank[None])[0])
    reference_rows = []
    for j, name in enumerate(ids):
        reference_rows.append(dict(id=name, mean_yield_ugml=activity[j], above_sfab=int(positive[j]),
                                   top3_producer=int(strong[j]), angle_deg=full_score[j],
                                   harmonic_distance_A=full_dh[0, j], weight_effective_poses=full_ess[0, j],
                                   rank=full_rank[j], in_top10=full_top10[j]))
    write_csv(out/'reference_ranking.csv', sorted(reference_rows, key=lambda r: r['rank']))
    print('Validated all 14 full-50 descriptors against the prior analysis.', flush=True)
    accum = {n: {key: [] for key in baseline_scores} for n in COUNTS}
    primary_scores = {n: [] for n in COUNTS}
    harmonic = {n: [] for n in COUNTS}
    complementary = {n: [] for n in COUNTS if n <= 25}
    rng = np.random.default_rng(args.seed)
    for offset in range(0, args.repetitions, args.batch_size):
        size = min(args.batch_size, args.repetitions-offset)
        # Independent random keys give a uniform permutation for every enzyme.
        indices = rng.random((size, 48, 50)).argsort(axis=-1)
        assert np.all(np.sort(indices, axis=-1) == np.arange(50))
        stats = permuted_statistics(arrays, indices)
        for n in COUNTS:
            scores, dh, _ = prefix_scores(stats, n)
            primary_scores[n].append(scores[PRIMARY])
            harmonic[n].append(dh)
            for key, value in scores.items():
                accum[n][key].append(evaluation(value, positive, strong))
        # Compare disjoint sets: useful check that agreement with the overlapping
        # full-50 reference is easier than agreement with separate predictions.
        reverse = permuted_statistics({'distance': arrays['distance'], 'angle': arrays['angle']}, indices[:, :, ::-1])
        for n in complementary:
            k = 50-n
            complementary[n].append(reverse['angle_weight'][:, :, k-1]/reverse['weight'][:, :, k-1])
        if (offset+size) % 2500 == 0 or offset+size == args.repetitions:
            print(f'Subsampling: {offset+size}/{args.repetitions}', flush=True)
    summary, candidate_rows, comparison, paired, disjoint, two_stage = [], [], [], [], [], []
    draws = {}
    for n in COUNTS:
        metrics = {key: {metric: np.concatenate([part[metric] for part in pieces])
                         for metric in pieces[0]} for key, pieces in accum[n].items()}
        score = np.concatenate(primary_scores[n])
        dh = np.concatenate(harmonic[n])
        ranks = rankdata(-score, axis=-1)
        included = membership(score, 10)
        included5, included15 = membership(score, 5), membership(score, 15)
        assert np.all((included == 0) | (included == 1)), 'Primary score requires explicit tie handling'
        m = metrics[PRIMARY]
        m.update(overlap10=(included*full_top10).sum(axis=1),
                 rank_rho=row_correlation(ranks, full_rank[None]),
                 spearman_activity=row_correlation(rankdata(score, axis=-1), activity_rank[None]),
                 score_mae_deg=np.abs(score-full_score).mean(axis=1),
                 harmonic_mae_A=np.abs(dh-full_dh).mean(axis=1))
        for k, selected in [(5, included5), (15, included15)]:
            m[f'hits3_at{k}'] = selected[:, strong].sum(axis=1)
            m[f'hits7_at{k}'] = selected[:, positive].sum(axis=1)
        row = dict(samples=n, repetitions=args.repetitions,
                   nominal_sample_reduction=1-n/50, ideal_sampling_throughput_multiplier=50/n,
                   p_all3_at10=float(np.mean(m['hits3_at10'] == 3)),
                   p_at_least4_of7_at10=float(np.mean(m['hits7_at10'] >= 4)),
                   p_same_signal=float(np.mean((m['hits3_at10'] == 3) & (m['hits7_at10'] >= 4))),
                   p_exact_top10=float(np.mean(m['overlap10'] == 10)),
                   p_auc3_ge_090=float(np.mean(m['auc3'] >= .9)))
        row['p_all3_at10_mcse'] = float(np.sqrt(row['p_all3_at10']*(1-row['p_all3_at10'])/args.repetitions))
        row['p_all3_at5'] = float(np.mean(m['hits3_at5'] == 3))
        row['p_all3_at15'] = float(np.mean(m['hits3_at15'] == 3))
        for key, value in m.items():
            row.update({f'{key}_{stat}': val for stat, val in summarize(value).items()})
        row['enrichment7_at10_mean'] = row['hits7_at10_mean']/(10*7/48)
        row['enrichment3_at10_mean'] = row['hits3_at10_mean']/(10*3/48)
        summary.append(row)
        draws[f'n{n}_angle_deg'] = score
        draws[f'n{n}_harmonic_distance_A'] = dh
        for j, name in enumerate(ids):
            cr = dict(samples=n, id=name, above_sfab=int(positive[j]), top3_producer=int(strong[j]),
                      full50_rank=full_rank[j], full50_angle_deg=full_score[j],
                      p_top5=float(included5[:, j].mean()), p_top10=float(included[:, j].mean()),
                      p_top15=float(included15[:, j].mean()))
            for key, values in [('rank', ranks[:, j]), ('angle_deg', score[:, j]), ('harmonic_distance_A', dh[:, j])]:
                cr.update({f'{key}_{stat}': val for stat, val in summarize(values).items()})
            candidate_rows.append(cr)
        for key, values in metrics.items():
            cr = dict(samples=n, proxy=key)
            for metric in ['hits3_at10', 'hits7_at10', 'auc3', 'auc7']:
                cr.update({f'{metric}_{stat}': val for stat, val in summarize(values[metric]).items()})
            cr['p_full_expected_3_at10'] = float(np.mean(np.isclose(values['hits3_at10'], 3)))
            comparison.append(cr)
            if key != PRIMARY:
                pr = dict(samples=n, comparison=key)
                for metric in ['hits3_at10', 'hits7_at10', 'auc3', 'auc7']:
                    delta = m[metric]-values[metric]
                    pr.update({f'{metric}_delta_{stat}': val for stat, val in summarize(delta).items()})
                    pr[f'{metric}_p_weighted_better'] = float(np.mean(delta > 1e-10))
                    pr[f'{metric}_p_tied'] = float(np.mean(np.abs(delta) <= 1e-10))
                paired.append(pr)
        if n in complementary:
            complement = np.concatenate(complementary[n])
            other_in = membership(complement, 10)
            other_m = evaluation(complement, positive, strong)
            dr = dict(samples=n, complement_samples=50-n,
                      p_all3_both_top10=float(np.mean((m['hits3_at10'] == 3) & (other_m['hits3_at10'] == 3))))
            for key, value in dict(overlap10=(included*other_in).sum(axis=1),
                                   rank_rho=row_correlation(ranks, rankdata(-complement, axis=-1))).items():
                dr.update({f'{key}_{stat}': val for stat, val in summarize(value).items()})
            disjoint.append(dr)
        if n < 50:
            # Retrospectively complete the initially retained enzymes to 50
            # using the remaining poses, then select 10 by their full-50 score.
            # No activity labels enter either selection stage.
            for keep in [15, 20, 25]:
                stage1 = membership(score, keep)
                final_score = np.where(stage1 == 1, full_score[None], -np.inf)
                final_m = evaluation(final_score, positive, strong)
                final_in = membership(final_score, 10)
                cost = len(ids)*n + keep*(50-n)
                recovered_all3 = np.isclose(final_m['hits3_at10'], 3)
                assert np.array_equal(recovered_all3, (stage1[:, strong] == 1).all(axis=1))
                two_stage.append(dict(
                    initial_samples_per_enzyme=n, enzymes_screened=len(ids), retained_for_50_samples=keep,
                    final_shortlist=10, total_structures=cost, full50_total_structures=50*len(ids),
                    sample_reduction=1-cost/(50*len(ids)), ideal_sampling_throughput_multiplier=50*len(ids)/cost,
                    p_all3_final_top10=float(recovered_all3.mean()),
                    hits7_final_top10_mean=float(final_m['hits7_at10'].mean()),
                    p_exact_full50_top10=float(np.mean((final_in*full_top10).sum(axis=1) == 10))))
    write_csv(out/'sample_budget_summary.csv', summary)
    write_csv(out/'per_enzyme_subsampling.csv', candidate_rows)
    write_csv(out/'proxy_comparison.csv', comparison)
    write_csv(out/'paired_proxy_differences.csv', paired)
    write_csv(out/'disjoint_complement_checks.csv', disjoint)
    write_csv(out/'two_stage_budget.csv', two_stage)
    np.savez_compressed(out/'subsampling_draws.npz', ids=np.array(ids), **draws)

    # Separate with-replacement sensitivity: no finite-population collapse at
    # n=50. This empirical bootstrap still cannot discover unsampled pose modes.
    boot = {n: [] for n in COUNTS}
    bootstrap_rng = np.random.default_rng(args.seed+1)
    for offset in range(0, args.bootstrap_repetitions, args.batch_size):
        size = min(args.batch_size, args.bootstrap_repetitions-offset)
        ix = bootstrap_rng.integers(0, 50, size=(size, 48, 50))
        stats = permuted_statistics({'distance': arrays['distance'], 'angle': arrays['angle']}, ix)
        for n in COUNTS:
            scores = stats['angle_weight'][:, :, n-1]/stats['weight'][:, :, n-1]
            boot[n].append(evaluation(scores, positive, strong))
    bootstrap_rows = []
    for n, pieces in boot.items():
        if not pieces:
            continue
        m = {key: np.concatenate([part[key] for part in pieces]) for key in pieces[0]}
        row = dict(samples=n, repetitions=args.bootstrap_repetitions,
                   p_all3_at10=float(np.mean(m['hits3_at10'] == 3)),
                   p_at_least4_of7_at10=float(np.mean(m['hits7_at10'] >= 4)))
        for key, value in m.items():
            row.update({f'{key}_{stat}': val for stat, val in summarize(value).items()})
        bootstrap_rows.append(row)
    if bootstrap_rows:
        write_csv(out/'with_replacement_sensitivity.csv', bootstrap_rows)
    draw_plots(out, summary, candidate_rows, baseline)
    decisions = {
        'minimum_samples_all3_at10_probability_ge_090': minimum_count(summary, 'p_all3_at10', .9),
        'minimum_samples_all3_at10_probability_ge_095': minimum_count(summary, 'p_all3_at10', .95),
        'minimum_samples_all3_at10_probability_ge_099': minimum_count(summary, 'p_all3_at10', .99),
        'minimum_samples_both_original_shortlist_endpoints_probability_ge_095': minimum_count(summary, 'p_same_signal', .95),
        'minimum_samples_exact_full50_top10_probability_ge_095': minimum_count(summary, 'p_exact_top10', .95),
        'minimum_samples_all3_at15_probability_ge_095': minimum_count(summary, 'p_all3_at15', .95),
        'minimum_samples_all3_at15_probability_ge_099': minimum_count(summary, 'p_all3_at15', .99),
    }
    results = dict(
        panel='48 homologs, 50 one-Mg unrestrained Boltz-2 step-1 poses each; SfaB excluded from ranking.',
        metric='A=sum(angle/distance)/sum(1/distance), larger ranks higher; harmonic distance retained alongside A.',
        endpoints=dict(strong=['AS9', 'AS17', 'AS23'], above_sfab=list(np.array(ids)[positive]),
                       sfab_mean_yield_ugml=reference, shortlist_budget=10),
        method=dict(subsampling='Independent uniform within-enzyme subsets without replacement; nested across counts; identical draws across descriptors.',
                    counts=COUNTS, repetitions=args.repetitions, seed=args.seed,
                    bootstrap_repetitions=args.bootstrap_repetitions, bootstrap_seed=args.seed+1,
                    tie_handling='Fractional expected inclusion at exact boundary ties; undefined comparator scores deferred below finite scores.',
                    intervals='Central 95% ranges across pose subsets, conditional on this fixed experimental panel; not biological confidence intervals.',
                    decision_thresholds='Exploratory operational thresholds of 90%, 95%, 99% recovery; not a preregistered hypothesis test.'),
        baseline=baseline, decisions=decisions, summary=summary, bootstrap_sensitivity=bootstrap_rows,
        two_stage_retrospective=two_stage,
        source_sha256=source_hashes, concatenated_confidence_sha256=confidence_hash,
        derived_pose_features_sha256=digest(out/'matched_pose_features.csv'),
        script_sha256=digest(Path(__file__)),
        resources=dict(python=platform.python_version(), numpy=np.__version__,
                       cpu_affinity=sorted(os.sched_getaffinity(0)), online_cpus=os.cpu_count(),
                       host_load=os.getloadavg(), thread_limits={k: os.environ[k] for k in THREAD_KEYS},
                       gpu_used=False, new_structures_generated=0,
                       total_analysis_wall_seconds=time.perf_counter()-start),
        limitations=[
            'Full-50 baseline overlaps every subsample; variance necessarily vanishes at 50. Complement comparisons and empirical bootstrap sensitivity are reported separately.',
            'Random generation does not establish statistical independence between poses; all are conditional on the same model, MSA and molecular setup.',
            'Only 48 enzymes and three strongest producers on a repeatedly inspected panel; 10,000 pose draws do not create additional experimental replication.',
            'The frozen weighted angle tied alternatives on shortlist recovery in the previous panel; this study does not establish unique superiority.',
            'Exact rank and shortlist reproduction is stricter than retaining experimental enrichment.',
            'The one-Mg panel does not validate sample budgets for other molecular setups, substrates, model versions, or two-Mg production protocols.',
            '50/n is an ideal sampling-stage throughput multiplier, not an observed end-to-end speedup; featurization and startup costs remain.',
            'Assay labels are fixed at the existing mean-yield definitions; assay measurement uncertainty and expression effects are not resampled.',
        ])
    (out/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(dict(baseline=baseline, decisions=decisions,
                          summary=[{k: r[k] for k in ['samples', 'p_all3_at10', 'p_at_least4_of7_at10',
                                                     'hits7_at10_mean', 'overlap10_mean', 'rank_rho_mean']} for r in summary]), indent=2))


if __name__ == '__main__':
    main()
