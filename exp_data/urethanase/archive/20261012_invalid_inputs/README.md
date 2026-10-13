# Archived urethanase panel inputs — 2026-10-12

Moved here because the inputs are invalid or superseded. **Nothing in this
directory may be used to compute a descriptor or enter a proxy test.** Files are
moved, never deleted, per project policy.

Panel location: `exp_data/urethanase/panel/`

---

## `tag_contaminated_sequence/` — 10 systems, 500 poses

Folded with **non-native expression-tag residues left on the C-terminus**. The
Bendtsen constructs end `…native… + ENLYFQ (TEV site) + GLEHHHHHH (His tag)`.
TEV cleaves **ENLYFQ↓G**, so the native protein ends *before* `ENLYFQ`.
`build_cofold_inputs.py` trimmed only the substring `QGLEHHHHHH`, leaving
`ENLYF` behind.

| system | labelled | modeled C-term (last 12) | leftover |
|---|---|---|---|
| CCPUR1 | yes | `VPPGRSDENLYF` | `ENLYF` + 3 extra N-term (`AMG`) |
| CCPUR2 | yes | `VPPGRNAENLYF` | `ENLYF` |
| MPPUR3 | yes | `SFTLAAGENLYF` | `ENLYF` |
| BLPUR4 | yes | `ENLYFQGENLYF` | 12 residues; TEV site appears **twice** |
| AXPUR5 | yes | `RPETAKSSGENL` | `ENL` |
| AMPUR6 | yes | `PPGCSSGENLYF` | `ENLYF` |
| CCPUR7 | yes | `GASSSSGENLYF` | `ENLYF` |
| PMGPUR8 | yes | `GFRPPPGENLYF` | `ENLYF` |
| PMGPUR9 | yes | `APAAFAAENLYF` | `ENLYF` |
| PYNYL1 | yes | `LPPVCALGSENL` | `ENL` |

CCPUR1 additionally carries 3 extra N-terminal residues: modeled as
`AMGMDLLTQP…` against its own crystal **9TW8** (100 % identity,
`mutation_count: 0`) which begins `MDLLTQP…`.

Why this is not cosmetic: the remnant is **non-uniform** across the panel
(`ENLYF` ×8, `ENL` ×2, clean ×2), so it is a per-enzyme variable tracking
construct bookkeeping rather than biology, sitting inside a cross-enzyme
comparison. It is also hydrophobic and aromatic (Phe/Leu/Tyr) on a flexible
tail, and the documented failure mode on this panel is a hydrophobic group
parking in the catalytic site. Not proven causal — but an uncontrolled
alternative explanation.

**Supersede by:** re-folding with TEV-aware trimming (cut at `ENLYFQ↓G`).
These 10 are all labelled, i.e. the entire labelled set except UMG-SP2/SP3.

`pQR3139/3141/3144` are **not** affected — their tag is `LEHHHHHH` with native
residues before it, no TEV remnant. They remain in the live panel.

## `needs_assembly/` — 4 systems, 200 poses

Folded as monomers, but `modeling_unit_resolution.json` resolves them to
`unit=assembly`: the catalytic site is **6.5–9.4 Å from the nearest other
chain**, so a monomer omits part of the site.

| system | distance to neighbouring chain | reference |
|---|---|---|
| NylB | 9.3 Å | 1WYC |
| NylB_prime | 9.3 Å | 1WYB |
| NylBp2_Hyb24_D1_mutant | 9.4 Å | 2E8I |
| NylC-TS_D36A_D122G_H130Y_E263Q | 6.5 Å | 5Y0M |

All four are **unlabelled**, so the proxy test is unaffected. Same error class
as the nylon-hydrolase heterodimer incident.

**Supersede by:** re-folding the resolved assembly, templated if needed.

## `zero_structures/` — CpUrethanase

3 attempts, all reached `processed/` and emitted **no predictions**. Longest
chain in the panel at 551 aa. Same signature as the glnA3 OOM failure: Boltz
exits 0, no breaker trips, the arm looks merely "pending". Diagnose by grepping
logs for `ran out of memory`, not by counting result dirs.

Unlabelled. Note it is the closest relative to reviewed EC 3.5.1.75 urethanase,
so it is worth recovering on a larger card.

## `failed_empty_attempts/` — 11 retry dirs

Empty retries (0 `*_model_*.cif`) from systems that remain live. Verified empty
before moving, so no partial-pose contamination existed. Retained for the
failure record only. AMPUR6 needed 11 attempts and BLPUR4 10 before one
succeeded — worth understanding before re-folding.

## `superseded_pilot/` — pilot_5samples

5-pose pilot, superseded by the 50-pose production runs.

## `stale_descriptors/` — descriptors.json

Computed from the geometry above, so every row for the 10 labelled enzymes
derives from a contaminated sequence. The `preregistered_test.py` result that
read it (selection-corrected **p = 0.0527**) is therefore **not a result about
structural proxies** and must not be cited as one.

---

## What remains live in `panel/cofold_out/` — 11 systems, 50 poses each

`UMG-SP2`, `UMG-SP3`, `UMG-SP1`, `Aes72`, `NylA`, `NylC`, `TvgC`,
`SaAmd_candidate_unverified`, `pQR3139`, `pQR3141`, `pQR3144`.

Only **UMG-SP2 and UMG-SP3** are both clean-sequence *and* crystal-validated
(backbone RMSD 0.82 Å vs 8WDW, 0.71 Å vs 9FVF; byte-identical sequences).
They are also the only two labelled enzymes left in the live set — and UMG-SP2
has **no activity value**, so the live labelled set supporting a correlation is
**n = 1**.

⚠ 15 of the original 26 carry `unit=undetermined` ("no evidence available") in
`modeling_unit_resolution.json`, including several still live. Undetermined is
not the same as validated-monomer; treat their oligomeric state as an open
question.

⚠ Confidence does not separate good from bad here. UMG-SP2 sits at pLDDT 0.938 /
ligand_iptm 0.881 while its carbamate is 7.5–13.9 Å from the position the
crystal **8XTB** assigns it (2.87 Å from Ser190 OG). Never gate this panel on
pLDDT or ligand_iptm.
