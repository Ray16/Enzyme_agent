# Mechanism basis for the cofold setup — what is validated vs inferred

Everything in the step-1 and step-2 models (which atom is the electrophile, how many Mg²⁺, what the
leaving group is) is fixed here, from evidence, before any fold. Determined 2026-10-21.

---

## 1. Regiochemistry: Pα or Pγ?

A carboxylate activated by ATP can attack either phosphorus, and the two give different
intermediates, different leaving groups, and therefore a **different attack axis for `A`**:

| Route | Attack atom | Leaving group | Axis atom for `A` | Intermediate |
|---|---|---|---|---|
| **Adenylation** | **Pα** | PPi | **O3A** | acyl-AMP |
| **Phosphorylation** | **Pγ** | ADP | **O3B** | acyl-phosphate |

`attack_angle.py` already supports both (`phosphate="PA"` → bridge `O3A`; `"PG"` → `O3B`), so this
is a setup decision, not a code change. Getting it wrong silently measures the angle to the wrong
axis.

### Verdict for this panel: **all six are Pα (adenylating)**

| Enzyme | Family | Pα evidence | Status |
|---|---|---|---|
| DdaG | ANL superfamily | adenylate-forming by family definition; prior campaign | validated (family) |
| SfaB | ANL superfamily | as above | validated (family) |
| **DesD** | NIS type C | **PDB 7TGM** binds `LMS` + `I3U` | **validated (direct)** |
| AcsD *(not assayed; DesD's structural homolog)* | NIS type B | **PDB 2W02** = AcsD·ATP·Mg | validated (direct) |
| AcsA | NIS type B | NIS family chemistry; no structure of AcsA itself checked | **inferred** |
| AsbA | NIS type A | as above | **inferred** |
| AsbB | NIS type C | as above | **inferred** |

**The 7TGM evidence is decisive and worth spelling out.** Its two ligands resolve as:

- `LMS` = *[(2R,3S,4R,5R)-5-(6-amino-9H-purin-9-yl)-3,4-dihydroxytetrahydro-2-furanyl]methyl
  sulfamate* (C10H14N6O6S) — i.e. **5′-O-sulfamoyladenosine (AMS)**
- `I3U` = *4-[(5-aminopentyl)(hydroxy)amino]-4-oxobutanoic acid* (C9H18N2O4) — i.e. **HSC**,
  hydroxy-N-succinyl-cadaverine, DesD's native substrate

Together these are the paper's "HSC-AMS" inhibitor. **AMS is the standard non-hydrolysable mimic of
an acyl-adenylate** — the sulfamate replaces the acyl–phosphate linkage. A structure of DesD bound
to an acyl-adenylate mimic is direct evidence that DesD runs through acyl-AMP, hence **Pα**.

> Contrast with this repo's GS-fold panel (PubC, PuuA, glnA3): those *are* the Pγ /
> acyl-phosphate enzymes. None of them is in this panel. Do not carry a Pγ setup over.

## 2. Mg²⁺ stoichiometry — counted, not assumed

The expectation was 1 Mg²⁺ for the ANL members and possibly **2** for DesD. Counted directly in the
deposited structures:

| PDB | Protein | Chains | Mg²⁺ atoms | **Mg per active site** |
|---|---|---|---|---|
| 7TGM | DesD (NIS type C) + HSC-AMS | A–E (5) | 5 | **1** |
| 2W02 | AcsD (NIS type B) + **ATP** | A, B (2) | 2 | **1** |

Mg²⁺ coordination in **2W02**, the one true ATP ground state available:

```
MG (chain A): O3G 2.17 Å, O1A 2.47 Å
MG (chain B): O3G 2.04 Å, O1A 2.45 Å, O3A 2.51 Å
```

A single Mg²⁺ **bridging the α- and γ-phosphate oxygens** — the canonical α,γ-chelation that
organises PPi departure for an **α-attack**. This is itself independent support for §1.

**Decision: 1 Mg²⁺ for all six enzymes**, matching the existing campaign (and therefore keeping
scores comparable across campaigns).

**Caveat, recorded honestly:** 7TGM is an *inhibitor* complex with **no ATP**, so its single Mg²⁺ is
not an ATP ground state and is weak evidence on stoichiometry by itself. The load-bearing
observation is 2W02 (AcsD·ATP·Mg, 1 Mg bridging α/γ) — AcsD is DesD's closest structural homolog
and the one the paper itself uses for comparison. No deposited structure of AcsA, AsbA or AsbB with
ATP was examined, so for those three "1 Mg" is a **family-level inference**.

**Pre-declared sensitivity test:** because this was raised as a real possibility and the evidence
for the NIS members is homology-based, DesD will additionally be folded with **2 Mg²⁺** on all three
step-1 acid donors (3 extra cofolds). If `A`(2 Mg) and `A`(1 Mg) differ by more than the measured
ΔA noise floor (**SD 2.67°**, from `project_deltaA_noise_floor_measured_oct10`), Mg stoichiometry is
reported as a setup confound and the 1-Mg panel is re-examined. This is a **pre-registered
sensitivity check with a fixed threshold**, not a search for a better-scoring setup.

## 3. Step 2 (aminolysis) — the mechanism that the model must encode

Step 2 is where mechanism matters most, because the electrophile changes identity:

- Nucleophile: the acceptor **amine nitrogen, neutral `-NH2`** (the protonated ammonium is not
  nucleophilic). Distal functional groups are kept in their pH-7 state: a second amine as `[NH3+]`,
  an ω-acid as `[O-]`.
- Electrophile: the **carbonyl carbon of the acyl-adenylate** (not a phosphorus).
- Leaving group: **AMP** (the ester oxygen).
- Geometry: attack is along the **Bürgi–Dunitz** trajectory, N→C=O ideally ≈ **105°**, *not* the
  ~180° in-line geometry of step 1. Scoring step 2 with the step-1 180° ideal would be a category
  error.

So step 1 and step 2 are **not the same metric on different atoms** — they have different ideal
angles. Reported separately; never averaged into a composite.

**Known risk, carried from the previous campaign:** step 2 produced no signal there, with the
leading explanation that the cofold returns the **adenylation** domain state rather than the
**aminolysis** one (ANL enzymes rotate the C-terminal domain ~140° between the two half-reactions,
so the step-2 pocket modelled on an adenylation-state fold is simply the wrong pocket). That is an
**ANL-specific** pathology; four of these six are NIS-fold, which does not perform ANL's
domain alternation. Step 2 therefore has a better prior here than it did last time — but the
domain-state check is run first and reported regardless of outcome.

## 4. Consequence for which figures get which step

| Panel | acid donor | amine acceptor | enzyme differences attributable to |
|---|---|---|---|
| 2A | A | X | either step |
| 2B | Z | M | either step |
| 2C / 2D | **H** (shared) | H / M | **step 1 only** — shared acid, and C≡D in the data |
| 4A | **G** (small, universal) | MA diad | **step 2 only** |
| 4B | MA diad | X | step 1 |
| 5A | **A** (all 6 activate it in 2A) | MAM triad | **step 2 only** |
| 5B | AMA triad | M | step 1 |
| 5C | MA | MA | either step |
| 7 | MA / XA, DesD only | — | no enzyme contrast (regioselectivity) |

**Figs 5A and 4A are the clean step-2 tests**: their acid donors (adipate, glutarate) are activated
by every enzyme per Fig 2A, so any enzyme difference there cannot be step 1.
**Figs 2C/2D are degenerate under step 1 by construction** and are not evidence either way about
the amine.
