# Regioselectivity reproduction — Nyl10 / Nyl12 / Nyl50 on poly(X,AMA)

Self-contained attempt to reproduce the structural regioselectivity claim of
**Dishner, Bourgery et al., *JACS* 2026, 148, 18285** ("Circularity in
Sequence-Controlled Copolyamides…") on the **same enzymes / same panel** as ours
(their refs 39/40 = Drufva 2025, Bourgery 2025 = source of our 96-member panel).

## The claim we are testing (n = 3, ~8 REU gaps)

Enzyme–substrate `interface_delta` (Rosetta REF2015, after MM minimisation) should
track which sequence-defined tetrad each enzyme preferentially releases:

| enzyme | Δ(XAMA) | Δ(MAXA) | predicted pref | observed product |
|--------|--------:|--------:|----------------|------------------|
| Nyl50  | −52.3   | −44.6   | **XAMA**       | XAMA |
| Nyl12  | −39.8   | −47.7   | **MAXA**       | MAXA |
| Nyl10  | −40.4   | −44.3   | ~tie           | no preference |

## Pipeline

1. **`build_substrates.py`** *(done, no GPU)* — two neutrally-capped poly(X,AMA)
   tetrads (`substrates/`), constitutional isomers C29H47N5O5 (MW 545.7, 5 amides).
   The scissile amide differs by framing; the aromatic **X** ring sits in a
   different active-site subsite in each, which is the structural discriminant:
   - `MAXA_precursor` — scissile **A–M** amide (scissile C=`C57`, O=`O52`)
   - `XAMA_precursor` — scissile **A–X** amide (scissile C=`C55`, O=`O50`)
   Atom names reproduce Boltz-2's own `<ELEM><CanonicalRankAtoms(AddHs)+1>` scheme,
   so the contact restraints target the correct carbonyl. **Run under the boltz-2
   env's RDKit** (canonical ranks are version-sensitive).

2. **`build_inputs.py`** *(done, no GPU)* — 6 restrained YAMLs in `inputs/`.
   Faithful to the paper's Methods:
   - **tetramer of heterodimers**: 4 α (chains A,C,E,G) + 4 β (B,D,F,H); β residue 1
     = catalytic N-terminal Thr (autocleaved/mature form). α/β MSAs reused from the
     panel `frag_msa/`.
   - two **contact restraints** into heterodimer-1: scissile carbonyl **C → β1 Thr**
     `[B,1]`; scissile carbonyl **O → α1 oxyanion Asn** `[A,{180/186/180}]`.
   Verified against the paper: Thr @ split 211/233/227 ✓, Asn 180/186/180 ✓ (all α),
   Phe/Tyr 106/113/105 ✓.

3. **`run_folds.sh <gpu_idx>`** *(GPU — pending go-ahead)* — Boltz-2, **10 diffusion
   samples** per system (paper protocol, not our usual 50). Goes through
   `gpu_reserve`. Output → `results/<enz>__<framing>/`.

4. **score** *(GPU/CPU — to build after folds)* — geometry-filter the 10 models
   (d1<3.2 Å Thr Oγ1→C; d2,d3<3.8 Å oxyanion; αBD 80–120°; αFL 70–110°), pick top by
   geometry+confidence, OpenMM FF19SB/GAFF minimise (`openmm` env), PyRosetta REF2015
   `interface_delta` (`pyrosetta` env). Compare Δ(MAXA) vs Δ(XAMA) per enzyme.

## Honest caveats

- The paper's signal is **n=3, ~8 REU** (Nyl10 only ~4). Boltz sampling + minimisation
  noise can flip a gap that small; a correct reproduction may still miss a sign, and
  matching 3 points is weak validation regardless.
- Substrate is a **faithful chemical analog** of the paper's Fig S68 tetrads (exact
  SI geometry not visible to us); the discriminant (X-ring position vs the scissile
  bond) is correctly represented.
- We select the top model **objectively** (geometry+confidence); the paper selected
  by hand from 10.

## Env
boltz-2 (fold), openmm 8.2 (minimise), pyrosetta (interface_delta), palm/boltz-2 (rdkit).
