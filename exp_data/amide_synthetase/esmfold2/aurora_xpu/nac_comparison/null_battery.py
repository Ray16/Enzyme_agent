"""Selection-corrected null for the model-vs-model battery (24 correlations)."""
import json, sys
import numpy as np
from scipy import stats
S = sys.argv[1]
D = json.load(open(f"{S}/A_both_models.json"))
tg = sorted(D["esmfold2"])
Y = {n: np.array([D["activity"][t][i] for t in tg]) for i, n in
     enumerate(["avg", "batch_1/24", "batch_01/09"])}
X = {(m, met): np.array([D[m][t][met] for t in tg], float)
     for m in ("esmfold2", "boltz2") for met in ("A", "Dh", "ng", "nac")}

obs = {}
for k, x in X.items():
    for en, y in Y.items():
        obs[(k, en)] = stats.spearmanr(x, y).statistic
best = max(obs, key=lambda k: abs(obs[k]))
print(f"observed max |rho| over the {len(obs)}-test battery: {obs[best]:+.3f} at {best}")

rng = np.random.default_rng(1)
N = 20000
mx = np.empty(N)
for i in range(N):
    perm = {en: rng.permutation(y) for en, y in Y.items()}
    mx[i] = max(abs(stats.spearmanr(x, perm[en]).statistic)
                for x in X.values() for en in Y)
p = (mx >= abs(obs[best])).mean()
print(f"null 95th pct of max |rho| = {np.percentile(mx, 95):.3f}; "
      f"selection-corrected p = {p:.3f}")

# pre-registered single test: A vs avg activity, each model
for m in ("esmfold2", "boltz2"):
    r = stats.spearmanr(X[(m, "A")], Y["avg"])
    print(f"pre-registered A vs avg, {m:9}: rho={r.statistic:+.3f} p={r.pvalue:.3f}")
print("\nSfaB measured activity (avg, batch1, batch2):", D["activity"]["SfaB"])
print("panel activity avg quantiles:",
      np.round(np.percentile([D["activity"][t][0] for t in tg], [50, 75, 90, 100]), 4))
