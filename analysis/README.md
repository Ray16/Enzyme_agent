# Cofold analysis

Analysis of Boltz-2 cofold panels: **amide synthetases** (SfaB diversity panel, 49; adipic-acid
adenylation + aminolysis) and **nylon hydrolases** (96). Structures/results live in `../exp_data/`,
figures in `../figures/`. Scripts use absolute data paths, so they can be run from anywhere.

## Layout
```
analysis/
  common/
    geometry.py        shared PDB parse / bond angle / carboxylate-oxygen helpers
    proxy_eval.py      AUROC + bootstrap-CI + permutation-p + precision@k (vectorised)
  synthetase/
    build_step2_inputs.py            build step-2 aminolysis YAMLs (acyl-AMP + HMDA + Mg + pocket)
    inject_step2_pocket.py           inject pocket restraint from step-1 structures (alt. method)
    step1_nac.py                     step-1 adenylation NAC (carboxylate→ATP Pα, <4Å & >150°)
    step1_nac_reliability.py         test-retest / sample-count / cutoff / leave-one-out reliability
    step1_enrichment.py              active-vs-inactive enrichment of step-1 NAC
    step1_ps_iptm.py                 protein↔adipate pair-ipTM proxy
    step2_attack_distance.py         step-2 amine→acyl-AMP-carbonyl attack distance (loose run)
    step2_reactive_accommodation.py  step-2 reactive run: accommodation (dist/BD/confidence)
  hydrolase/
    build_inputs.py    build αβ-heterodimer + tetrad YAMLs (autocleavage split at catalytic Thr)
    nac.py             hydrolysis NAC (Thr-Oγ → scissile amide carbonyl)
  evaluate_all_proxies.py   run the standard eval block on every proxy -> proxy_evaluation_summary.csv
  make_figures.py           regenerate all figures in ../figures/ from result CSVs
```

## Key result
Step-1 adenylation **NAC fraction** is the one geometric proxy that predicts synthetase activity
(AUROC 0.81, perm-p 0.036; reliable, adenylation-specific) — usable as a recall/enrichment filter
(NAC≥0.5 keeps all actives, ~2.7× enrichment), not a precise picker. Hydrolase NAC is weaker but
significant (AUROC 0.61, p=0.03, n=38). Step-2 aminolysis geometry is **not accessible** via cofolding
(free amine floats ~8Å; a reactive constraint can't impose the near-attack — off Boltz's distribution).

## Typical order
1. `synthetase/build_step2_inputs.py`, `hydrolase/build_inputs.py`  (input prep; run on HPC separately)
2. `synthetase/step1_nac.py`, `step2_attack_distance.py`, `step2_reactive_accommodation.py`, `hydrolase/nac.py`  (per-panel geometry → CSVs in exp_data/*_results/)
3. `synthetase/step1_ps_iptm.py`, `step1_enrichment.py`, `step1_nac_reliability.py`  (extra synthetase analyses)
4. `evaluate_all_proxies.py`  (consistent AUROC/CI/perm-p/precision@k across proxies)
5. `make_figures.py`  (all figures → ../figures/)

Evaluation convention: judge proxies with AUROC + bootstrap CI + permutation-p + precision@k
(not Spearman — activity is heavily skewed). See `common/proxy_eval.py`.
