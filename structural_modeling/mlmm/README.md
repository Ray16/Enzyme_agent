# ML/MM reaction energetics

Agent-facing gating and ingestion around [`mlmm-toolkit`][mlmm] (Ohmura, Inoue &
Terada, ChemRxiv 2025, [doi:10.26434/chemrxiv-2025-jft1k][doi], GPL-3.0). This
package **does not implement an ML/MM engine** — the engine already provides
3-layer ONIOM, link-atom coupling, microiteration, path search, saddle
optimization, IRC and thermochemistry.

[mlmm]: https://github.com/t-0hmura/mlmm_toolkit
[doi]: https://doi.org/10.26434/chemrxiv-2025-jft1k

What this package owns is the part the engine cannot know: whether a structure
deserves to be computed, and whether a finished number deserves to be reported.

| module | question it answers |
|---|---|
| `spec.py` | what does the agent *claim* about this step — roles, charges, metals, changing coordinates? |
| `gate.py` | is this structure fit to compute, **before** any GPU time? |
| `verdict.py` | did this finished profile earn the right to be called a barrier? |
| `ingest.py` | read real engine output into a profile, without inheriting its defects |
| `structure.py` | a PDB reader that keeps hydrogens, elements and virtual sites, and guesses nothing |

Division of labour mirrors `proxies`: the agent supplies the system-specific
declaration, this package supplies everything derivable from it, and nothing here
knows about any particular enzyme.

## Why gates, and why they are fatal

In the previous campaign the solver was not what failed. Roughly twenty of thirty
recorded barrier-style outputs fail the single-imaginary-mode criterion, and
**every one of them still wrote a number** — including the one result that
"agreed with experiment".

The checks mostly *existed*. `ts_valid` was computed and never used to gate
anything; a reaction-mode overlap of 0.04 was printed as `SPURIOUS off-pathway
mode` and the number was reported anyway. **An advisory flag eventually gets
ignored.** So here the number is unreachable except through the verdict:

```python
profile.barrier()        # raises NotReportable, with the full verdict
diagram_values(profile)  # the plot path cannot bypass it either
```

There is no barrier attribute to read.

## Design rules

1. **Fatal, not advisory.** `barrier()`, `reaction_energy()` and
   `diagram_values()` raise rather than return a doubtful number.
2. **Fail closed.** `ABSTAIN` blocks exactly like `FAIL`. An unchecked
   precondition is not a satisfied one. `strict=False` exists only for triaging
   data you already know is incomplete.
3. **One threshold, one count.** `IMAGINARY_THRESHOLD_CM = 50.0` everywhere.
   The previous engine counted above 50 cm⁻¹ in one place and above 1e-6 in
   another, so one run could report `ts_valid: true` beside `n_imag_raw: 2`.
   The floor is measured, not chosen: a pure rigid-body **rotation** of an
   input moved a near-zero mode from 0.2i to 27.3i cm⁻¹, so a count taken at
   ~0 depends on how the molecule sits in the file.
4. **Recount, never inherit.** `ingest` recomputes imaginary modes from the raw
   spectrum. Where a count cannot be recomputed it is left `None`, which
   abstains.
5. **Label by geometry, never energy.** Calling the lower minimum "reactant"
   reports the *reverse* barrier for every exothermic step and forces every
   reaction endothermic.
6. **Units explicit.** `kcal(value, unit)` raises on anything unrecognized.
7. **Say what the number is.** `comparable_to_experiment()` refuses to set a
   bare ΔE beside a kcat-derived ΔG.

## Usage

```python
from structural_modeling.mlmm import (
    AtomRef, BondChange, LigandSpec, MLMMSpec, MetalSite, ReactionStep,
    check_structure, read_pdb, profile_from_freq_dirs,
)

spec = MLMMSpec(
    system_id="DesD", net_charge=-3,
    ligands=(LigandSpec("ATP", -4), LigandSpec("HSC", 0)),
    metals=(MetalSite(AtomRef("MG", resname="MG"), expected_coordination=6,
                      nucleophile_must_be_free=True),),
    step=ReactionStep(
        step_id="adenylation",
        nucleophile=AtomRef("O25", resname="HSC"),
        electrophile=AtomRef("PA", resname="ATP"),
        leaving_group=AtomRef("O3A", resname="ATP"),
        bond_changes=(
            BondChange("form", AtomRef("O25", resname="HSC"),
                       AtomRef("PA", resname="ATP"), 1.8),
            BondChange("break", AtomRef("PA", resname="ATP"),
                       AtomRef("O3A", resname="ATP"), 2.6),
        ),
        ideal_attack_angle=180.0, proton_coupled=False,
    ),
)

report = check_structure(read_pdb("system.pdb"), spec)   # BEFORE any GPU time
if not report.ok:
    raise SystemExit(report.render())

# ... run mlmm scan / tsopt / irc / freq ...

profile = profile_from_freq_dirs("results/adenylation_MA", system_id="DesD",
                                 step_id="adenylation", net_charge=-3)
print(profile.barrier())        # raises unless the profile earned it
```

Comparing variants that will sit on the same plot:

```python
from structural_modeling.mlmm import check_freeze_consistency

check_freeze_consistency({
    "MA": (structure_ma, spec_ma, frozen_serials_ma),
    "XA": (structure_xa, spec_xa, frozen_serials_xa),
})
```

This catches the largest recorded artifact in the previous campaign: one
variant's metal-coordinating sidechain was free to relax while the other's was
frozen, so the frozen one's metal shell under-relaxed and its barrier came out
inflated. The rule is not "freeze the same residue numbers" — different variants
coordinate through different residues — it is "each variant's *actual*
coordinating sidechains must be free".

## Back-test

The gate was validated against the recorded corpus rather than against intuition.
Of **31** cluster-route result files and **10** ONIOM-route result directories,
**none is reportable**:

- 11 files disagree with *themselves* on imaginary-mode count; three are stamped
  `ts_valid: true` while carrying `n_imag_raw` of 2, 3 and 4.
- Cluster-route failures are substantive: `ts_single_imaginary`,
  `endpoints_distinct` (product geometrically identical to reactant),
  `endpoint_mechanism`, `endpoint_labelling`.
- The ten ONIOM DesD runs have clean spectra (TS 1 imaginary, reactant 0,
  mode counts consistent) but are blocked on **genuine information gaps**: no
  product state was ever computed, so there is no reaction energy diagram, and
  neither reaction-mode overlap nor per-state max|F| was recorded.

The distinction matters: `FAIL` means known-bad, `ABSTAIN` means unknown. Most
historical runs are the latter — not refuted, just not evidenced.

## Known engine-level issues to respect

Found by audit of the surrounding code; none is fixed by this package, and two
affect what you can trust:

- **ML-region exclusion loss.** The ONIOM model system is built by a bare ParmEd
  slice, so ML-atom pairs whose 1-2/1-3/1-4 path runs through an MM atom lose
  their exclusion. Measured on a real 373-atom ML region: a spurious
  **+54.3 kcal/mol** inside `E_model_low` (subtracted, so an *attraction* in the
  total) and **113–130 kcal/mol/Å** of unphysical force on the catalytic
  histidine. Region-dependent — two other regions showed zero lost exclusions.
  Check region closure before trusting a number.
- **Amide-bond boundary cuts.** C–N boundary bonds are whitelisted, so a
  catalytic residue can be handed to the MLIP as an aldehyde plus free amine.
  Prefer non-polar Cα–Cβ cuts, or include the whole residue.
- **`embedcharge` is unsound** — it double-counts ML–MM electrostatics and feeds
  the MLIP's region to xTB uncapped. It is off by default. Keep it off; discard
  any number produced with it.
- **Mechanical embedding only.** The ML region never feels the protein field, and
  its MM charges are frozen at reactant-state values across the whole coordinate.
  For chemistry whose catalysis *is* electrostatic preorganization, this is the
  dominant physical approximation.
- **No MLIP applicability check exists** anywhere, and no DFT reference has ever
  been run in this tree. UMA on phosphorus is the least-validated regime; a
  33 kcal/mol pose-dependent spread was recorded on one system from pose and
  freeze choices alone.

## Not yet built

- Boltz-2 → prepared-system bridge (protonation, metal hydration, Amber
  parameterization, layer assignment).
- ML-region closure validation (exclusion graph + boundary-bond chemistry).
- A stage driver that runs `mlmm` and records the provenance the verdict needs
  (per-state max|F|, mode overlap, IRC endpoints).
