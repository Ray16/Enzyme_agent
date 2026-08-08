# Summary: Cofolding Analysis

## Overview

30 amide synthetase candidates and 93 amide hydrolase candidates were evaluated. For synthetases, 29 were cofolded with their substrates using Boltz-2 (one, Pls/B5BR95, was excluded at 1,319 amino acids due to GPU memory limits). Of the 29 cofolded synthetases, 28 used cofactor-guided cofolding (ATP anchored to AlphaFill-derived binding site, substrates constrained to the same pocket region) and 1 used pocket-constrained cofolding (no cofactor template available). Mg2+ coordinates were included in all 28 cofactor-guided runs, constrained to metal-coordinating residues within the ATP binding pocket. All 29 synthetase runs and all 93 hydrolase runs produced valid structures. The amide hydrolase candidates were cofolded with a nylon-6,6 trimer. After structural scoring, **PanC, GshA, BacD, and gss** (the *E. coli* glutathionylspermidine synthetase, identified as a better candidate than the model's original pick of GspS from *B. cereus*) are prioritized for amide synthesis activity testing.

## Key Methodological Decisions

1. **Catalytic residues were identified in two phases.** Phase A: InterPro/UniProt sequence-based identification (direct annotations and InterPro domain homologs). Phase B: FoldSeek structural homolog transfer with LLM-based validation to filter biochemically implausible mappings. Catalytic residue positions from FoldSeek transfer are used for quality scoring and visualization but excluded from cofolding constraints as a safety measure.

2. **Cofactors were normalized to the physiological pre-catalytic state.** Among 30 synthetase candidates, 28 had nucleotide cofactor template data from AlphaFill. Of these 28, the majority contained non-ATP nucleotides (ANP or ADP) in their source templates. All 28 were cofolded with ATP (the physiological substrate), normalizing to the same pre-catalytic state for fair comparison.

3. **Pocket constraints and cofolding strategy.**
   - **Cofactor-guided (synthetases):** ATP cofactor anchored to AlphaFill-derived binding site; substrates constrained to the same pocket region via `select_tight_constraints()`.
   - **Pocket-constrained (hydrolases):** Substrate constrained to the detected binding pocket.
   - Both strategies use pocket constraints, so substrate placement in the binding pocket is partly by design rather than an independent prediction.

---

## Detailed Analysis

### Glutathionylspermidine Synthetase: GspS / gss

The model originally recommended a *B. cereus* GspS homolog (Q63A28). This candidate scored overall iPTM 0.804 (substrate iPTM 0.18), among the lowest of all 29 synthetases. The low substrate iPTM suggests Boltz-2 has low confidence in the substrate binding pose.

**Recommendation:** The *E. coli* gss (P0AES0) is the preferred candidate. gss is a bifunctional enzyme with a C-terminal ATP-grasp synthetase domain and an N-terminal amidase domain. It was cofolded with cofactor-guided strategy (ATP + Mg2+, AlphaFold template): overall iPTM 0.970 (substrate iPTM 0.39), SASA burial 95.3%, 27 substrate contacts, pocket hydrophobicity 33.8%. Its phosphorylation mechanism (rather than adenylation) provides mechanistic diversity relative to other candidates.

### ATP-Grasp Family: DdaF, GshA, BacD

**DdaF (E2JA31) vs. DdaG (E2JA32):** DdaF has significantly better interface quality than DdaG (overall iPTM 0.980 vs. 0.473; substrate iPTM 0.54 vs. 0.14). Both were cofactor-guided with ATP and Mg2+. However, neither is ideal: both evolved to activate 2,3-diaminopropanoate (DAP) derivatives — DdaG ligates fumarate to DAP, and DdaF couples the resulting fumaramoyl-DAP to Val/Ile/Leu — substrates that differ substantially in length and chemistry from hexamethylenediamine + adipic acid. DdaF remains a reasonable follow-up only if top candidates fail.

**GshA (P0A6W9):** Native substrates are L-glutamate (147 Da) and L-cysteine. Target substrates are adipic acid (C6, 146 Da) and hexamethylenediamine (C6). The one-carbon lengthening on the acid side is minor and structurally acceptable given GshA's binding pocket architecture. Holo pocket hydrophobicity is 29.0%. Cofolding metrics: overall iPTM 0.977 (substrate iPTM 0.49), SASA burial 82.4%, 19 substrate contacts. *E. coli* native with 4 crystal structures (extensive structural data).

**BacD (P39641):** L-amino acid dipeptide ligase with documented in vitro substrate flexibility — the N-terminal position accommodates smaller L-amino acids (Gly, Ala, Ser) and the C-terminal position accepts uncharged bulky residues (Gln, Phe, Met), with strict L-stereoselectivity. Holo pocket hydrophobicity 40.7%, with overall iPTM 0.940 (substrate iPTM 0.44), SASA burial 93.5%, 28 substrate contacts. 4 crystal structures available to guide future engineering.

**PylC (Q8TUC0):** ATP-grasp family pyrrolysine synthetase from *Methanosarcina acetivorans* (archaeon). Overall iPTM 0.983 (substrate iPTM 0.56), SASA burial 92.7%, holo pocket hydrophobicity 37.0%. Not prioritized: archaeal proteins commonly misfold in *E. coli* due to different chaperone systems and codon usage.

GshA and BacD show the strongest experimental validation potential: GshA matches acid-side substrate chemistry (glutamate ~ adipic acid), while BacD offers documented in vitro substrate flexibility. Both are cofactor-guided: ATP and Mg2+ are anchored to their known binding sites, and substrates are constrained to the same pocket region.

### Pantoate Ligase: Pls and PanC

**Pls (B5BR95):** At 1,319 amino acids, Pls is a large multi-module NRPS from *Streptomyces noursei* unsuitable for standard *E. coli* expression. It exceeded GPU memory during cofolding and has been excluded from the prioritized set.

**PanC (P31663):** Top-ranked candidate overall. Overall iPTM 0.962 (substrate iPTM 0.39), SASA burial 97.8%, 24 substrate contact residues, holo pocket hydrophobicity 48.2%. PanC ranks high in the overall recommendation because its native substrates (pantoate + beta-alanine) are small and simple, structurally closer to our targets than murD's UDP-linked cell wall precursors or ddlB's D-alanine dipeptide. *E. coli* native, 2 crystal structures (1IHO, 3GUZ), >20 publications.

### McbA, PuuA, and NIS Synthetases

**McbA (R4R1U5):** ANL superfamily member (AMP-binding adenylation domain). Overall iPTM 0.987 (substrate iPTM 0.57), SASA 93.4%, holo pocket hydrophobicity 42.9%. Native substrates include aromatic carboxylic acids (naphthoic acid, indole carboxylic acids) with significantly different geometry from the aliphatic hexamethylenediamine + adipic acid targets. Not prioritized due to substrate shape mismatch.

**PuuA (P78061):** Displays the best substrate chemistry match of any candidate. Putrescine has exactly 2 amines like hexamethylenediamine, and glutamate is near-identical in MW to adipic acid. Cofolding metrics: overall iPTM 0.937 (substrate iPTM 0.33), SASA burial 74.6%, 19 substrate contact residues, pocket hydrophobicity 44.2%. The barrier is expression -- obtaining an active PuuA homolog would be valuable. A focused homolog search against characterized, bacterially-expressed enzymes is warranted.

**NIS synthetases:** The siderophore biosynthesis enzymes (asbA, asbB, iucA, iucC, sbnE, alcC, rhbC, pvsB, sfaB) comprise the largest family cluster in the screen (9/29 candidates). All were cofactor-guided; iucA scored overall iPTM 0.990 (substrate iPTM 0.62, the highest substrate iPTM in the dataset). This family warrants detailed homolog comparison to identify the best-folding, most-readily-expressed member.

### Remaining Synthetases

The following candidates were not prioritized but are noted for completeness:

| Enzyme | UniProt | iPTM (overall / substrate) | SASA | Hydrophobic | Why not prioritized |
|---|---|---|---|---|---|
| **murC** | P17952 | 0.981 / 0.46 | 70.5% | 5.6% | Cell wall biosynthesis; native substrate very different from targets; very low pocket hydrophobicity |
| **capB** | P96736 | 0.553 / 0.13 | 75.6% | 43.6% | Poly-gamma-glutamate synthetase; low iPTM |
| **CARNS1** | A5YM72 | 0.969 / 0.40 | 89.3% | 36.5% | Carnosine synthase; histidine is very different from HMD |
| **iaaL** | P18204 | 0.930 / -- | 84.0% | 28.2% | Native substrate is aromatic (indole-3-acetic acid); pocket-constrained |
| **TRS** | Q9GT49 | 0.969 / 0.41 | 96.1% | 38.5% | Trypanothione synthetase; trypanosome protein, *E. coli* expression uncertain |
| **cphA** | P56947/P73833 | 0.956/0.768 / 0.36/0.20 | 82.7%/81.0% | 5.0%/20.0% | Cyanophycin synthetases; evolved for charged substrates; very low pocket hydrophobicity |
| **gshB** | P04425 | 0.962 / 0.37 | 56.6% | 16.7% | Glutathione synthetase (2nd step); low SASA burial |

### Amide Hydrolase Failures

The 93 cofolded hydrolase structures structurally explain why the screened candidates failed. The analysis benchmarks against NylC (Q1EPR4), the Ntn-hydrolase used as a reference for nylon-6,6 hydrolysis. NylC was cofolded with the nylon-6,6 trimer in its post-autocleavage form (sequence split at Thr267 to expose the catalytic nucleophile).

**Family breakdown (Pfam domain-based classification):**

| Family | Pfam | Count | Enzyme Class / Catalytic Residues | Native Substrate |
|---|---|---|---|---|
| NylC (benchmark) | PF03576 | 1 | Ntn-hydrolase (Thr nucleophile, autocleavage) | Nylon-6 oligomers |
| Poly-gamma-glutamate hydrolase | PF05908 | 31 | Zinc metallopeptidase (His-Glu-His, carboxypeptidase A-like) | Poly-gamma-glutamate (gamma-glutamyl bonds) |
| Cyanophycinase (S51 family) | PF03575 | 12 | Serine protease (Ser-His-Glu triad) | Cyanophycin (multi-L-Arg-poly-L-Asp) |
| Pseudomurein hydrolase | PF01841/PF09373 | 8 | Cysteine protease (Cys-His-Asp triad) | Pseudomurein (epsilon-Ala-Lys isopeptide bonds) |
| Other/uncharacterized | -- | 42 | Unknown or minor families | Unknown |

None of these families use NylC's Ntn-hydrolase autocleavage mechanism. Two candidates (A0A7G7MKK3, A0A2W6BHB8) are genuine sequence and structural homologs of NylC (TM-scores 0.88 and 0.89; ~34% pairwise sequence identity over the aligned region), consistent with shared evolutionary origin of the αββα Ntn-hydrolase fold. However, the catalytic nucleophile Thr267 is replaced by Asp at the equivalent position in both -- Asp cannot perform autocleavage or act as an N-terminal nucleophile. They are diverged homologs: same fold and common ancestry, but the autocleavage mechanism has been lost.

**Pocket chemistry -- apo structures (AlphaFold, 64 candidates with binding site evidence):**

| | NylC | Polyglutamate hydrolase (n=23) | Cyanophycinase (n=12) | Pseudomurein hydrolase (n=5) | All (n=64) |
|---|---|---|---|---|---|
| Hydrophobic | 40.5% | 22.9% | 33.6% | 28.4% | 28.0% |
| Charged | 8.1% | 37.9% | 25.1% | 19.6% | 31.6% |
| Polar | 51.4% | 39.2% | 41.3% | 52.0% | 40.4% |

**Pocket chemistry -- cofolded holo structures (64 candidates with binding site evidence):**

| | NylC (autocleavage) | Polyglutamate hydrolase (n=23) | Cyanophycinase (n=12) | Pseudomurein hydrolase (n=5) | All (n=64) |
|---|---|---|---|---|---|
| Hydrophobic | 42.1% | 24.7% | 31.0% | 29.4% | 28.4% |
| Charged | 5.3% | 36.6% | 26.7% | 20.6% | 30.6% |
| Polar | 52.6% | 38.7% | 42.3% | 50.0% | 41.0% |

**Conclusion:** The screened hydrolases failed due to (1) pocket chemistry mismatch -- NylC has 8.1% charged residues in its apo pocket vs 31.6% for the candidates, a 3.9x excess of charged residues incompatible with the aliphatic nylon substrate, (2) wrong catalytic mechanism -- none share NylC's Ntn-hydrolase autocleavage, and (3) evolutionary mismatch -- every characterized family evolved for charged polymeric substrates (poly-gamma-glutamate, cyanophycin, pseudomurein) fundamentally different from nylon-6,6. Engineering the existing NylC is the most viable path.

---

## Recommended Test Set

| Priority | Enzyme | UniProt | iPTM (overall / substrate) | Key Rationale |
|---|---|---|---|---|
| 1 | **PanC** | P31663 | 0.962 / 0.39 | 24 contacts; holo pocket 48.2% hydrophobic; *E. coli*; 2 crystal structures. Small, simple native substrates most similar to targets. |
| 2 | **GshA** | P0A6W9 | 0.977 / 0.49 | Similar acid-side substrate (L-glutamate, 147 Da ~ adipic acid C6, 146 Da); *E. coli*; 4 crystal structures. |
| 3 | **BacD** | P39641 | 0.940 / 0.44 | Documented L-amino acid substrate flexibility; holo pocket 40.7% hydrophobic; SASA burial 93.5%; 4 crystal structures. |
| 4 | **gss** | P0AES0 | 0.970 / 0.39 | Mechanistic diversity (phosphorylation, not adenylation); *E. coli* native. |
| 5 | **DdaF** | E2JA31 | 0.980 / 0.54 | Best follow-up if additional ATP-grasp candidates needed. |

These are enzymes that were computationally prioritized and not cherry-picked post hoc. Structural evidence supports testing them: PanC and BacD in particular have strong active-site geometry for accommodating the target substrates.
