# Pre-registration: does tunnel topology predict product *selectivity* (not rate)?

**Written 2026-10-09, BEFORE any structural descriptor is computed against the endpoint.**
Commit this file first; compute only after sign-off. The point of writing it down is that this
project (including this author, this week) has repeatedly manufactured significance by choosing a
threshold, a direction, or a feature after seeing the data. None of those choices may move after
this file is committed.

---

## 0. Why this is a new test, not a re-run

Six proxy families are null **against hydrolytic rate** (near-attack, tunnel *radius*,
thermostability, adsorption, oligomerisation, self-activation; broad battery best feature
selection-corrected p ≈ 1.0). The leading explanation is three confounds: (a) rate is
mass-transfer-limited on crystalline PA66 (<1% conversion), (b) catalytic geometry is ~invariant
across one conserved fold, (c) the lysate label is confounded with expression.

This test changes the **endpoint**, not the fold. The response is the **product-selectivity ratio**
`frac_L1 = L1 / (L1 + L2)`. A ratio divides out (a) and (c) by construction: mass transfer and
enzyme abundance scale both products equally and cancel. What remains is a clean probe of (b) for
one specific structural property — **tunnel topology downstream of the catalytic Thr** — which lives
in tunnel-lining loops that can vary even when the catalytic core does not, and which prior work
never measured (it measured bottleneck *radius* against *rate*).

**Either outcome is publishable.** Positive → the first validated structural proxy on this panel.
Clean null → confound (b) holds even for tunnel topology, which closes the mechanistic story.

---

## 1. Endpoint (frozen)

- **Primary response:** `frac_L1 = L1_ugml / (L1_ugml + L2_ugml)`, from
  `amide_hydrolase_biorep_activity.csv`, rows `exp == 65` and `type == Powder`.
- **Scale:** analysed as-is (bounded 0–1). Spearman ρ is the primary statistic, so no transform.
- **Rationale frozen:** paper is titled "substrate- and *product-selective* nylon hydrolases";
  selectivity is the quantity the paper is actually about. See [[project_hydrolase_fig3_product_length_oct7]].

### Endpoint integrity checks (report, do not gate the hypothesis on them)
- Cross-condition reproducibility already measured: `frac_L1` Powder vs Rinsate ρ = 0.94 (n=22
  producers). ⚠ **Must confirm from the paper's Methods that Powder (exp65) and Rinsate (exp68) are
  independent assays, not two fractions of one run.** If they are not independent, ρ=0.94 is not a
  reproducibility claim and will be reported as such.
- `frac_L1` vs total product ρ = 0.30 (p=0.13): endpoint is weakly, not strongly, coupled to
  turnover — report this so the proxy is not secretly a turnover readout.

### 🔴 What this endpoint may NOT claim
- **No cross-platform (lysate→purified) selectivity claim.** Lysate `frac_L1` ≈ 0.005; purified
  ≈ 0.88–0.97; different regimes; the lysate→purified ordering was tested and is null (n=6,
  ρ=−0.20). This test is confined to the lysate Powder panel.
- **The "recovers the 5 purified enzymes, p=9e-5" statistic is RETIRED as evidence.** It is
  circular: the authors likely purified those five *because* they were selective in this same
  screen, so it recovers their selection rule, not a structural fact. It will not appear in the
  writeup as support.

---

## 2. Cohort (frozen, with a pre-declared sensitivity band)

`frac_L1` is undefined/noise when total product is at the floor, so inclusion is on **total
activity significance**, a criterion orthogonal to the *ratio*:

- **Primary cohort:** `L2_significant == 1` → **n = 38**. Chosen because it is the paper's own
  significance call and is defined on the denominator, not on `frac_L1`.
- **Sensitivity cohort (pre-declared):** `L1_ugml + L2_ugml ≥ 1.0 µg/mL` → **n = 26** (a strict
  subset of the 38). Reported alongside; the conclusion must hold in *both* or be reported as
  cohort-dependent.
- No other cohorts will be introduced after seeing results. No per-enzyme exclusions except the
  structural-interpretability checks in §3.

---

## 3. Structural substrate and descriptors (frozen)

### Structures
- **96 deposited apo AF2 tetramers:** `paper/final_af2_dp95_zenodo/nyl*.pdb` (Zenodo 14976747),
  split-site-validated 96/96 vs authors, core Cα 0.4–0.5 Å. **No new folding, no GPU.**
- **Axis ground truth:** 9DYS crystal has PG4 bound in the tunnel (3.58 Å from Thr227 Oγ1); used
  only to validate the threading tool, not as a panel member.
- Interpretability gate per model: catalytic Thr Oγ1 identifiable; threading tool returns a path
  covering ≥ 60% of PG4 atoms on the 9DYS positive control (the tool's existing validation).
  Models failing the Thr identification are "missing", never encoded as inactive.

### Tool
`analysis/fig3_product_length_20261007/thread_chain.py` — beam-search growth of a clash-free
polyamide chain from the nucleophile through the tunnel; already validated against PG4 in 9DYS
(path covers the bound ligand). Canonical channel helper:
`analysis/broad_proxy_search_20261008/scripts/channel_geometry.py`.

### PRIMARY descriptor — tunnel threading asymmetry
Mechanism: monomer (L1) release needs a short, occluded distal (leaving-group) subsite tract —
exo-like; dimer (L2) release needs a through-going tunnel — endo-like. On an apo model the two
tunnel directions from the nucleophile report this without a bound substrate.

- Seed chain growth from Thr Oγ1 over a sphere of directions (existing `capacity(n_seed=…)`).
- `L_long` = length of the single best clash-free threaded path.
- `L_short` = length of the best path whose initial step lies in the hemisphere **opposing**
  `L_long`'s end-to-end vector (operationalisation frozen here: opposition by sign of the dot
  product of first-step direction with the `L_long` axis).
- **`asymmetry = (L_long − L_short) / (L_long + L_short)`**, in [0,1].
- **Direction fixed by mechanism BEFORE computing: higher `asymmetry` → higher `frac_L1`**
  (one short side = exo = monomer). This sign is locked; it will not be read off the data.

### Secondary descriptors (pre-declared set of 2; BH across all 3)
1. **`distal_capacity` = `L_short`** alone (absolute short-side threading length). Direction:
   **lower → higher `frac_L1`**.
2. **`bottleneck_distal`** = minimum surface clearance along the `L_short` path
   (`channel_geometry.py`). Direction: **tighter (smaller) → higher `frac_L1`**.

No fourth descriptor will be added. Composite/learned scores are **excluded** (see
[[feedback_no_tanimoto_no_composites]]); each descriptor is reported on its own.

---

## 4. Statistics (frozen)

- **Primary:** Spearman ρ(`asymmetry`, `frac_L1`) on the n=38 cohort, one-sided in the
  pre-registered direction.
- **Multiplicity:** Benjamini–Hochberg across the 3 descriptors × 2 cohorts = 6 tests; report all 6
  with q-values. The primary test (asymmetry, n=38) is flagged as primary regardless of BH.
- **Selection-corrected null:** permute `frac_L1` labels within the cohort (≥ 20,000 perms),
  recompute the **max** |ρ| over all 3 descriptors, report the observed primary ρ against that
  max-null distribution. This charges the penalty for having 3 descriptors.
- **🔴 α-length covariate (mandatory):** every prior "signal" on this panel died when α-subunit
  length entered as a covariate. Report partial Spearman ρ(`asymmetry`, `frac_L1` | α_len). The
  proxy is only credible if the partial ρ survives. α_len source: same as prior work.
- **LOO fragility:** report the leave-one-out ρ range; a result that depends on one enzyme is noise.
- **Power honesty:** n=38, continuous endpoint. A true ρ≈0.45 gives p≈0.005; a true ρ≈0.30 is
  underpowered (p≈0.07). State the detectable effect size up front so a null is interpreted as
  "no large effect", not "no effect".

## 5. Decision rule (frozen)
- **Proxy supported** iff: primary ρ in the pre-registered direction, selection-corrected p < 0.05,
  partial ρ | α_len survives (same sign, p < 0.05), consistent across both cohorts, and not
  LOO-fragile. All five, or it is reported as null/ambiguous.
- **This is the last structural-proxy test on this panel** regardless of outcome. A null closes the
  line; a positive moves to a prospective shortlist.

## 6. Leakage firewall
Descriptor extraction reads only structures, never the activity CSV. Ranking/validation reads
activity. No descriptor direction, cohort, or threshold defined in §§1–5 may change after this file
is committed. See [[feedback_data_leakage]].

---

## AMENDMENT 1 (2026-10-09, before any asymmetry-vs-label computation)

**Change:** the descriptor is computed on the **holo (αβ)₂ dimer cofolds**, not the apo AF2
tetramers. **Empirical reason, not outcome-driven:** the apo tunnel is closed — at the catalytic
Thr Oγ1 of apo `nyl50.pdb` there are **0 clash-free seed positions** at a 1.4 Å probe (9 at 1.0 Å),
so no tunnel exists to thread. The holo cofold, whose tunnel is opened by the bound substrate
(induced fit), gives **120–129 clash-free seeds** at 1.4 Å. Confirmed the cofold ligand sits
2.58–2.60 Å from both catalytic Thr Oγ1 (near-attack), so the open tunnel is the physiologically
relevant one. No asymmetry value has been correlated with frac_L1 at the time of this amendment.

**Consequent frozen choices:**
- Structures: `dimer_panel_cofold/capped/<id>/.../<id>_model_*.pdb`, QC-passing poses only
  (`proxy_verification_20261008/qc/model_qc.csv`, `qc_pass == True`).
- Catalytic Thr = β-subunit N-terminus (min resseq of each β chain), atom Oγ1. Two catalytic
  sites per dimer (β chains B and D).
- **Ligand (LIG) excluded** from the clash coordinate set (trap #6): the probe measures the
  protein tunnel, not the bound substrate.
- Threading params: `clash=1.4` (primary; 1.2 reported as sensitivity), `n_seed=8`, `n_second=2`,
  `beam=60`, `max_len=40`. Reduced from tool defaults for runtime; stability vs defaults spot-checked.
- Aggregation: per enzyme = mean of `asymmetry` over the first **up to 8** QC-passing poses × 2
  β-sites. `distal_capacity = mean L_short` likewise. (`bottleneck_distal` deferred unless the
  primary shows signal, to keep the multiplicity count honest — if computed it enters the BH set.)
- Cohort: the n=38 (`L2_significant==1`) and n=26 (total≥1.0) cohorts of §2, **intersected with the
  QC-passing cofold panel**; the realized n is reported.

Everything else in §§1,2,4,5,6 is unchanged. Direction still fixed a priori: higher asymmetry
→ higher frac_L1.
