# Cofolding Analysis: Concise Summary

## Setup

| | Synthetases | Hydrolases |
|---|---|---|
| Total candidates | 30 | 93 |
| Cofolded (Boltz-2) | 29 (Pls excluded, too large) | 93 |
| Strategy: cofactor-guided | 28 (ATP anchored, substrates free) | 0 |
| Strategy: pocket-constrained | 1 (no cofactor template) | 89 |
| Strategy: blind | 0 | 4 |
| With Mg2+ | 28/28 cofactor-guided | N/A |
| With known catalytic residues (UniProt annotations) | 6/30 | 23/93 |
| Cofolded with | ATP (cofactor) + HMD + adipic acid (substrates) | nylon-6,6 trimer (substrate) |

Boltz-2 folds the protein from sequence and docks ligands simultaneously -- it does not use the AlphaFold apo structure. The apo structures are only used upstream to identify where to constrain the cofactor (from AlphaFill crystallographic homologs) or which pocket to target (from Foldseek/P2Rank).

**Cofactor normalization:** AlphaFill often returns crystallographic analogs (ANP, ACP) or product states (ADP, AMP). All 28 cofactor-guided candidates were normalized to ATP (the physiological pre-catalytic form) so every candidate is at the same point in the catalytic cycle.

**Catalytic Center Score (CCS):** Fraction of known catalytic residues (from UniProt "Active site" annotations) that overlap with the predicted binding pocket. 100% means all catalytic residues are correctly positioned around the substrate. This metric is independently meaningful for cofactor-guided candidates (substrates docked freely) but circular for pocket-constrained candidates (substrates were forced to the pocket).

---

## Synthetase vs Hydrolase Comparison

| Metric | Synthetases (n=29) | Hydrolases (n=93) |
|---|---|---|
| Mean iPTM | 0.903 | 0.748 |
| Mean SASA burial | 87.0% | 61.5% |
| With known catalytic residues (UniProt annotations) | 6/30 | 23/93 |
| Of those, 100% Catalytic Center Score (cofactor-guided, non-circular) | 5 | N/A (all pocket-constrained) |

**Why hydrolases failed:** These isopeptidases (cyanophycinases, polyglutamate hydrolases, pseudomurein hydrolases) evolved for charged substrates (cyanophycin = Asp-Arg polymer; polyglutamate = polyanionic). Their holo pockets average 29.2% charged and 30.5% hydrophobic residues. The nylon-6,6 trimer is predominantly aliphatic (long methylene chains between amide bonds), incompatible with highly charged active sites. If hydrolase activity is desired, target the NylC/Ntn-hydrolase family instead.

---

## Recommended Test Set

| Priority | Enzyme | UniProt | iPTM | SASA | Strategy | Key Rationale |
|---|---|---|---|---|---|---|
| 1 | **PanC** | P31663 | 0.962 | 97.8% | cofactor-guided | 100% Catalytic Center Score; native substrates (pantoate + beta-alanine) most similar to targets; holo pocket 48.2% hydrophobic; *E. coli*; 2 structures |
| 2 | **GshA** | P0A6W9 | 0.977 | 82.4% | cofactor-guided | Glutamate (147 Da) ~ adipic acid (146 Da); holo pocket 29.0% hydrophobic; *E. coli*; 4 structures |
| 3 | **BacD** | P39641 | 0.940 | 93.5% | cofactor-guided | Proven broad specificity; holo pocket 40.7% hydrophobic; 4 structures, 8 mutants |
| 4 | **gss** | P0AES0 | 0.970 | 95.3% | cofactor-guided | Phosphorylation mechanism (diversity); *E. coli*; replaces model's GspS (Q63A28, iPTM 0.804). |
| 5 | **DdaF** | E2JA31 | 0.980 | 78.6% | cofactor-guided | Follow-up if top candidates fail; native substrates less similar |

**PanC** ranks #1 despite not having the highest iPTM because the top-iPTM candidates (asbA at 0.990, iucA at 0.990, rhbC at 0.989) are NIS siderophore synthetases whose native substrates are structurally dissimilar to our targets, and their pockets are among the least hydrophobic (3.8--8.8%). PanC's native substrates (pantoate + beta-alanine) are small and simple, structurally closer to hexamethylenediamine + adipic acid, and its holo pocket hydrophobicity (48.2%) is among the highest. For PanC and BacD, ATP and Mg2+ are anchored to their known binding sites while the two substrates (HMD + adipic acid) are unconstrained. The anchored cofactor and metal define the active-site neighborhood, but substrate placement is still an independent prediction.

**GshA** has a similar acid-side substrate in chain length and MW (glutamate: C5, 147 Da vs adipic acid: C6, 146 Da; both dicarboxylic/dicarboxylate) but no catalytic residue data was found in UniProt (NO-CAT-DATA), so we cannot verify active-site positioning.

**gss (P0AES0)** was identified by the experimentalist team as the preferred candidate over the model's original pick GspS (Q63A28, *B. cereus*). The *B. cereus* enzyme scored iPTM 0.804, one of the lowest of all synthetases despite cofactor-guided cofolding. The *E. coli* gss scored iPTM 0.970 with full cofactor guidance (ATP + Mg2+). Its phosphorylation mechanism (vs adenylation for other candidates) provides mechanistic diversity.

---

## Previously Tested Enzymes

| Enzyme | UniProt | Result | Structural Explanation |
|---|---|---|---|
| **McbA** | R4R1U5 | Expressed, not active | High iPTM (0.987) and SASA (93.4%), but native substrates are aromatic/complex carboxylic acids -- geometry mismatch with aliphatic HMD + adipic acid targets. |
| **PylC** | Q8TUC0 | Didn't express | Archaeal protein (*M. acetivorans*) -- *E. coli* misfolding expected (different chaperones, codon usage). iPTM 0.983, SASA burial 92.7%, holo pocket hydrophobicity 37.0%. |

---

## Notable Candidates Not in Test Set

| Enzyme | UniProt | iPTM | Why notable | Why not prioritized |
|---|---|---|---|---|
| **asbA** | A0A0F7RJ52 | 0.990 | Highest iPTM overall (tied with iucA) | NIS siderophore synthetase; pocket hydrophobicity only 7.7%; native substrates structurally dissimilar |
| **iucA** | Q47316 | 0.990 | Highest iPTM overall (tied with asbA) | NIS siderophore synthetase; pocket hydrophobicity only 8.8% |
| **McbA** | R4R1U5 | 0.987 | Very high iPTM and SASA | Experimentally tested -- expressed but not active; geometry mismatch |
| **ddlB** | P07862 | 0.986 | Very high iPTM | Native substrates (D-Ala dipeptide) very different; pocket hydrophobicity 20.0% |
| **murD** | P14900 | 0.975 | High iPTM | Native substrates (UDP-linked cell wall precursors) very different from targets |
| **PuuA** | P78061 | 0.937 | Best substrate chemistry match (putrescine ~ HMD, glutamate ~ adipic acid) | Expression barrier; homolog search warranted |
| **DdaF vs DdaG** | E2JA31/E2JA32 | 0.980/0.473 | DdaF >> DdaG on iPTM | Both cofactor-guided with ATP + Mg2+. Both have non-ideal native substrates (2,3-diaminopropanoate derivatives). |
| **murC** | P17952 | 0.981 | High iPTM | Cell wall biosynthesis (like murD); native substrate very different; pocket hydrophobicity 5.6% |
| **CARNS1** | A5YM72 | 0.969 | Good iPTM + 89.3% SASA | Carnosine synthase (beta-alanine + histidine); histidine very different from HMD |
| **TRS** | Q9GT49 | 0.969 | Native substrate includes spermidine (triamine, similar to HMD diamine) | Trypanosome protein; *E. coli* expression uncertain |
| **cphA** | P56947/P73833 | 0.956/0.768 | Cyanophycin synthetases | Evolved for charged substrates (Asp-Arg polymer); very low pocket hydrophobicity (5.0%/20.0%) |
| **gshB** | P04425 | 0.962 | Glutathione synthetase (2nd step) | Low SASA burial (56.6%); pocket hydrophobicity 16.7% |

---

## Key Caveats

- **Catalytic Center Score circularity:** For cofactor-guided candidates, substrates dock freely so overlap with catalytic residues is independent validation. For pocket-constrained candidates (hydrolases), the pocket was selected partly based on catalytic residue overlap -- the score is partly self-fulfilling.
- **Catalytic data coverage:** Only 6/30 synthetases and 23/93 hydrolases have UniProt active/binding site annotations. The remaining candidates have no known catalytic residues, so we cannot verify whether the substrate landed at the active site.
