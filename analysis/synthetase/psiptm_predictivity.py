#!/usr/bin/env python3
"""Compute enzyme<->diacid ps_iptm per enzyme (cofactor ATP/Mg excluded) and test
predictivity vs measured N66-L1 activity. Chain map: 0=enzyme, 1=ATP, 2/3=Mg, 4=diacid."""
import glob, json, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
AS = f"{ROOT}/exp_data/amide_synthetase"
CFG = f"{AS}/boltz2/step1_1mg/results"   # corrected 1-Mg panel
ENZ, DIACID = "0", "3"                    # 1-Mg chain map: 0=enzyme 1=ATP 2=Mg 3=diacid

rows = []
for cfg in sorted(glob.glob(f"{CFG}/*")):
    if not os.path.isdir(cfg):
        continue
    eid = os.path.basename(cfg)                    # 1-Mg dirs are 'AS9'/'SfaB' (no prefix)
    preds = glob.glob(f"{cfg}/boltz_results_*/predictions/*")
    if not preds:
        continue
    es, se = [], []
    for cj in glob.glob(f"{preds[0]}/confidence_*model_*.json"):
        p = json.load(open(cj))["pair_chains_iptm"]
        es.append(p[ENZ][DIACID])   # enzyme-anchored -> substrate (~0.92)
        se.append(p[DIACID][ENZ])   # substrate-anchored -> enzyme (~0.44)
    es, se = np.array(es), np.array(se)
    rows.append(dict(id=eid, n=len(es),
                     es_mean=es.mean(), se_mean=se.mean(),
                     sym_mean=((es+se)/2).mean(),
                     min_mean=np.minimum(es, se).mean(),
                     max_mean=np.maximum(es, se).mean(),
                     es_max=es.max(), se_max=se.max()))
psi = pd.DataFrame(rows)

act = pd.read_csv(f"{AS}/assay_data/panel_activity.csv")[
    ["id", "activity_N66L1_ugml_avg"]].rename(columns={"activity_N66L1_ugml_avg": "act"})
df = act.merge(psi, on="id", how="inner")
df["is_active"] = (df["act"] > 0.2).astype(int)
df.to_csv(f"{AS}/analysis/psiptm_enzyme_diacid.csv", index=False)

print(f"n={len(df)}  actives={df.is_active.sum()}  ({df[df.is_active==1].id.tolist()})")
print(f"{'variant':10} {'Spearman':>9} {'p':>7} {'AUROC':>7}   ranks of actives")
for col in ["se_mean", "es_mean", "sym_mean", "min_mean", "max_mean", "es_max", "se_max"]:
    rho, pv = spearmanr(df[col], np.log10(df["act"] + 1e-3))
    au = roc_auc_score(df["is_active"], df[col])
    d = df.sort_values(col, ascending=False).reset_index(drop=True)
    ranks = [int(d.index[d.id == a][0]) + 1 for a in df[df.is_active == 1].id]
    print(f"{col:10} {rho:+9.3f} {pv:7.3f} {au:7.3f}   {sorted(ranks)}")
print("\nwrote", f"{AS}/analysis/psiptm_enzyme_diacid.csv")
