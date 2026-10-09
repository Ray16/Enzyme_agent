# Results — selectivity structural-proxy test (2026-10-09)

**Verdict: NULL. Tunnel asymmetry does not predict product selectivity (frac_L1); if anything it is
weakly anti-correlated, opposite the pre-registered mechanistic direction.** The selectivity line is
closed per PROTOCOL.md §5.

## What was tested
- Endpoint: `frac_L1 = L1/(L1+L2)`, exp65 Powder (the confound-free selectivity ratio).
- Descriptor: ligand-axis tunnel-clearance asymmetry on the holo cofolds (amendment 2), 85 enzymes
  with usable geometry, 7,898 QC-passing poses scored. Direction fixed a priori: higher asymmetry
  (exo-like, short distal tract) → higher frac_L1.

## Numbers (one-sided in the pre-registered direction)
| cohort | descriptor | ρ | p1 | partial ρ \| α-len | selection-corrected p |
|---|---|---|---|---|---|
| n=37 (L2-significant, primary) | asymmetry | **−0.251** | 0.933 | −0.245 (p=0.14) | **0.990** |
| n=37 | distal_capacity | −0.101 | 0.723 | −0.103 | — |
| n=26 (total≥1.0, sensitivity) | asymmetry | −0.249 | 0.890 | −0.242 (p=0.23) | 0.969 |

- The observed ρ is **negative** (wrong sign); the one-sided p in the registered direction is ~0.9.
- Selection-corrected max-null p ≈ 0.97–0.99 — nowhere near significance.
- Consistent across both cohorts; the α-length covariate does not change the picture.

## Decision (per §5)
Not a single condition of the support rule is met (sign is wrong, p≫0.05). **Selectivity is NOT
predicted by tunnel topology as operationalized here.** This was pre-registered as the last
structural-proxy test on the panel, so the structural-proxy search for the hydrolase is closed.

## Caveats (do not over-read the null)
- Two operationalizations were needed: the beam-search threading descriptor (amendment 1) saturated
  on ground-truth crystals (capped 4/6 sites) and could not be validly run; the ligand-axis measure
  (amendment 2) has dynamic range but relies on the capped ligand lying along the tunnel (PCA axis),
  which fails when it curls (6/91 enzymes dropped; Nyl50/Nyl47 gave ~0 extent).
- So this is "no *large* effect from this tunnel-asymmetry construct," not proof that selectivity has
  no structural basis. But combined with six null rate-proxy families and the saturation of the more
  physical threading measure, the weight of evidence is that soluble-oligomer cofold geometry does
  not encode this panel's activity or selectivity.

## What holds up (the reportable deliverable)
- The **dimer modeling is correct** (verified this session): all 70 QC cofolds pass the A/D
  interface (Jaccard median 0.82, ligand_iptm 0.89); the Nyl50 cofold ligand sits 2.58–2.60 Å from
  both catalytic Thr Oγ1; apo AF2 models are 96/96 split-site validated. The nulls are computed on
  sound structures.
- The hydrolase story is a **mechanistically-grounded negative** + the wrong-dimer methods finding.
See [[project_proxy_quality_reaudit_oct8]].
