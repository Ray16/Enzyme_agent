import csv, collections, sys, json
import numpy as np
from scipy import stats
S = sys.argv[1]

def A_table(path):
    g = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        g[r["target"].replace("step1_", "")].append((float(r["d_A"]), float(r["angle_deg"])))
    out = {}
    for t, v in g.items():
        inv = sum(1/d for d, _ in v)
        out[t] = dict(A=sum(a/d for d, a in v)/inv, Dh=len(v)/inv,
                      ng=sum(1 for d, _ in v if d < 4),
                      nac=sum(1 for d, a in v if d < 4 and a > 150))
    return out

ef = A_table(f"{S}/pose_geometry_esmfold2_stock.csv")
bz = A_table(f"{S}/pose_geometry_boltz2_1mg.csv")
assert set(ef) == set(bz), set(ef) ^ set(bz)

act = {}
for r in csv.DictReader(open("exp_data/amide_synthetase/assay_data/panel_activity.csv")):
    b1 = [float(r[k]) for k in r if k.startswith("rep_1/24/2025")]
    b2 = [float(r[k]) for k in r if k.startswith("rep_01/09/2025")]
    act[r["id"]] = (float(r["activity_N66L1_ugml_avg"]), float(np.mean(b1)), float(np.mean(b2)))

tg = sorted(set(ef) & set(act))
print("targets with both geometry and assay:", len(tg), "| geometry but no assay:", sorted(set(ef)-set(act)))
y = {n: np.array([act[t][i] for t in tg]) for i, n in enumerate(["avg", "batch_1/24", "batch_01/09"])}
rr = stats.spearmanr(y['batch_1/24'], y['batch_01/09'])
print(f"assay batch-vs-batch rho={rr.statistic:+.3f} p={rr.pvalue:.3f}; nonzero avg = {int(np.sum(y['avg']>0))}/{len(tg)}")

print("\n=== Spearman rho vs activity (n=%d) ===" % len(tg))
print(f"{'metric':8} {'endpoint':13} {'ESMFold2':>22} {'Boltz-2':>22}")
rows = {}
for metric in ["A", "Dh", "ng", "nac"]:
    for en in y:
        xe = np.array([ef[t][metric] for t in tg], float)
        xb = np.array([bz[t][metric] for t in tg], float)
        a = stats.spearmanr(xe, y[en]); b = stats.spearmanr(xb, y[en])
        rows[(metric, en)] = (xe, xb, y[en])
        print(f"{metric:8} {en:13} {a.statistic:+.3f} (p={a.pvalue:.3f}){'':>6} {b.statistic:+.3f} (p={b.pvalue:.3f})")

rng = np.random.default_rng(0)
print()
for metric in ["A", "ng"]:
    for en in y:
        xe, xb, yy = rows[(metric, en)]
        d = []
        for _ in range(20000):
            i = rng.integers(0, len(tg), len(tg))
            if len(set(yy[i])) < 3 or len(set(xe[i])) < 3 or len(set(xb[i])) < 3: continue
            d.append(stats.spearmanr(xe[i], yy[i]).statistic - stats.spearmanr(xb[i], yy[i]).statistic)
        d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2*min((d <= 0).mean(), (d >= 0).mean())
        print(f"{metric:3} rho(ESMFold2)-rho(Boltz2) on {en:12}: {d.mean():+.3f} 95%CI [{lo:+.3f},{hi:+.3f}] p={p:.3f}")

ae = np.array([ef[t]["A"] for t in tg]); ab = np.array([bz[t]["A"] for t in tg])
r = stats.spearmanr(ae, ab)
print(f"\nA agreement between models: rho={r.statistic:+.3f} p={r.pvalue:.2g}; "
      f"median A {np.median(ae):.1f} (ESMFold2) vs {np.median(ab):.1f} (Boltz-2)")
nge = np.array([ef[t]["ng"] for t in tg]); ngb = np.array([bz[t]["ng"] for t in tg])
print(f"n_gated: ESMFold2 median {np.median(nge):.0f}/50, zero-dock {int(np.sum(nge==0))}; "
      f"Boltz-2 median {np.median(ngb):.0f}/50, zero-dock {int(np.sum(ngb==0))}")
print(f"SfaB: ESMFold2 A={ef['SfaB']['A']:.1f} Dh={ef['SfaB']['Dh']:.2f} ng={ef['SfaB']['ng']} nac={ef['SfaB']['nac']}"
      f" | Boltz-2 A={bz['SfaB']['A']:.1f} Dh={bz['SfaB']['Dh']:.2f} ng={bz['SfaB']['ng']} nac={bz['SfaB']['nac']}")
json.dump({"esmfold2": ef, "boltz2": bz, "activity": act}, open(f"{S}/A_both_models.json", "w"), indent=1)
