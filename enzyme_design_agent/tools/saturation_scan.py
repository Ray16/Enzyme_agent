"""Agent-invoked tools: saturation mutagenesis of an existing scaffold.

The *retrieval* tools in the discovery agent ask whether a protein can
accommodate a chemistry.  The generative tools build a protein when none can.
These sit between: the scaffold exists and already binds, and the question is
which single substitutions in its pocket change the chemistry.

Two tools, matching the two decisions the agent actually makes:

``scan_design_pocket``
    Where is the pocket?  Defined around the WHOLE bound complex -- substrate or
    intermediate plus every cofactor and catalytic metal -- because a pocket
    defined around one ligand cannot contain a residue that only touches
    another, and will report its own blind spot as a finding.

``build_saturation_library``
    Which variants to make.  Every non-WT amino acid at every selected
    position, including positions believed essential: those are the campaign's
    negative controls, and masking them means the data never exists.

Both are CPU-only and fast; neither reserves a GPU or folds anything.  Folding
the library is a separate, deliberate step the agent takes afterwards.
"""
from __future__ import annotations

import json
from pathlib import Path

from enzyme_discovery_agent.tool_registry import tool

from ..saturation.pocket import define_pocket
from ..saturation.variants import AA20, build_variants, job_counts
from ..saturation.variants import write_variants_fasta, write_variants_tsv
from ..structure import load_structure


def _out_dir(run_dir: str | None, name: str) -> Path:
    base = Path(run_dir).expanduser() if run_dir else Path.cwd()
    d = base / "design" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


@tool(
    description=(
        "Find the residues lining a bound complex in a structure, as the mutable set "
        "for a design campaign. The pocket is defined around EVERYTHING bound at once "
        "-- substrate or reaction intermediate plus any cofactor and catalytic metal -- "
        "not around a single ligand, so cofactor-contacting positions are not excluded "
        "by construction. Components far from the site (a surface-bound ion) are "
        "reported and dropped rather than importing a shell defined by a misplaced "
        "atom. CPU-only and fast."
    ),
    category="structure",
    parallel_safe=True,
    params={
        "structure_path": {
            "type": "string",
            "description": "PDB file holding the protein with its ligands/cofactors bound.",
        },
        "cutoff": {
            "type": "number",
            "description": "Heavy-atom distance from residue to complex, in angstrom.",
            "default": 5.0,
        },
        "components": {
            "type": "string",
            "description": (
                "Comma-separated component ids (usually ligand chain ids, e.g. 'L,T,M,N') "
                "to treat as the bound complex. Leave empty to use every non-water "
                "non-protein entity in the file."
            ),
            "default": "",
        },
        "anchor": {
            "type": "string",
            "description": (
                "Component the site is grown outward from, normally the substrate or "
                "intermediate. Empty picks the largest component."
            ),
            "default": "",
        },
        "link_cutoff": {
            "type": "number",
            "description": (
                "How close a component must come to the growing site to count as part "
                "of it, in angstrom. Guards against a stray modelled ion."
            ),
            "default": 5.0,
        },
        "protein_chain": {
            "type": "string",
            "description": "Restrict positions to one protein chain. Empty scans all.",
            "default": "",
        },
        "run_dir": {"type": "string", "description": "Agent run directory (injected)."},
    },
    injected_params=["run_dir"],
)
def scan_design_pocket(
    structure_path: str,
    cutoff: float = 5.0,
    components: str = "",
    anchor: str = "",
    link_cutoff: float = 5.0,
    protein_chain: str = "",
    run_dir: str | None = None,
) -> dict:
    """Residues lining the bound complex, with per-component distances."""
    path = Path(structure_path).expanduser()
    if not path.is_file():
        return {
            "status": "error",
            "summary": f"No such structure file: {path}",
            "data": None,
            "suggestions": ["Fold or download the holo structure first."],
        }
    try:
        st = load_structure(path)
        pocket = define_pocket(
            st,
            cutoff=cutoff,
            components=[c.strip() for c in components.split(",") if c.strip()] or None,
            anchor=anchor or None,
            link_cutoff=link_cutoff,
            protein_chain=protein_chain or None,
        )
    except (ValueError, OSError) as exc:
        return {
            "status": "error",
            "summary": f"Could not define a pocket: {exc}",
            "data": {"available_components": [c.id for c in load_structure(path).components]}
                    if path.is_file() else None,
            "suggestions": [
                "Check the ligand chain ids with the available_components list.",
                "If the ligand is recorded as ATOM rather than HETATM, name it explicitly.",
            ],
        }

    out = _out_dir(run_dir, "pocket")
    rows = pocket.to_rows()
    tsv = out / f"{path.stem}_positions.tsv"
    import csv as _csv
    with tsv.open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["chain"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    by_membership: dict[str, int] = {}
    for p in pocket.positions:
        by_membership[p.membership] = by_membership.get(p.membership, 0) + 1

    excluded = [
        {"component": e.component, "resname": e.resname, "n_atoms": e.n_atoms,
         "d_to_site": e.d_to_site, "reason": e.reason}
        for e in pocket.excluded
    ]
    summary = (
        f"{len(pocket.positions)} positions within {cutoff} A of the complex "
        f"({'+'.join(pocket.admitted)}; anchor {pocket.anchor})"
    )
    if excluded:
        summary += f"; {len(excluded)} component(s) excluded as off-site"

    return {
        "status": "success",
        "summary": summary,
        "data": {
            "structure_path": str(path),
            "cutoff": cutoff,
            "anchor": pocket.anchor,
            "admitted_components": list(pocket.admitted),
            "excluded_components": excluded,
            "n_positions": len(pocket.positions),
            "positions_by_membership": by_membership,
            "positions": rows,
            "positions_tsv": str(tsv),
        },
        "suggestions": [
            "Pass positions_tsv to build_saturation_library to make the variant set.",
            "Check excluded_components: an unexpectedly excluded cofactor means the "
            "pose is wrong, not that the cofactor is irrelevant.",
        ],
    }


@tool(
    description=(
        "Build a saturation-mutagenesis library from a set of positions: every non-"
        "wild-type amino acid at every position, with stable ids of the form "
        "<label>_<WT><resnum><MUT> so results merge with earlier panels. Positions "
        "believed catalytically essential are INCLUDED and annotated, not masked -- "
        "they are the campaign's internal negative controls. Writes a variant table "
        "and a FASTA ready to fold. CPU-only and fast."
    ),
    category="structure",
    parallel_safe=True,
    params={
        "structure_path": {
            "type": "string",
            "description": "PDB the wild-type sequence and numbering are read from.",
        },
        "positions": {
            "type": "string",
            "description": (
                "Comma-separated residue numbers to saturate (e.g. '20,143,190'), or "
                "the path to a positions TSV from scan_design_pocket."
            ),
        },
        "label": {
            "type": "string",
            "description": "Prefix for variant ids, e.g. the enzyme name. May be empty.",
            "default": "",
        },
        "modes": {
            "type": "string",
            "description": (
                "Comma-separated condition labels every variant is built for, e.g. "
                "'apo,holo' for two ligand compositions. Empty means one condition."
            ),
            "default": "",
        },
        "alphabet": {
            "type": "string",
            "description": "Substitutions to try. Default is all 20 amino acids.",
            "default": AA20,
        },
        "annotate_positions": {
            "type": "string",
            "description": (
                "Optional JSON object mapping residue number to a note, e.g. "
                "'{\"143\": \"catalytic\"}'. Recorded as an annotation column; it "
                "never excludes a position."
            ),
            "default": "",
        },
        "protein_chain": {
            "type": "string",
            "description": "Chain the sequence is taken from. Empty uses the first.",
            "default": "",
        },
        "n_samples": {
            "type": "integer",
            "description": "Predicted models per (variant, mode), for the job estimate.",
            "default": 1,
        },
        "run_dir": {"type": "string", "description": "Agent run directory (injected)."},
    },
    injected_params=["run_dir"],
)
def build_saturation_library(
    structure_path: str,
    positions: str,
    label: str = "",
    modes: str = "",
    alphabet: str = AA20,
    annotate_positions: str = "",
    protein_chain: str = "",
    n_samples: int = 1,
    run_dir: str | None = None,
) -> dict:
    """Saturate the given positions and write the variant table + FASTA."""
    path = Path(structure_path).expanduser()
    if not path.is_file():
        return {"status": "error", "summary": f"No such structure file: {path}",
                "data": None, "suggestions": ["Provide the scaffold PDB."]}

    try:
        st = load_structure(path)
        seq = st.sequence(protein_chain or None)
    except (ValueError, OSError) as exc:
        return {
            "status": "error",
            "summary": f"Could not read a usable sequence: {exc}",
            "data": None,
            "suggestions": [
                "Mutations are applied by residue index, so the numbering must be "
                "gap-free and 1-based. Renumber the structure before designing."
            ],
        }

    pos_pairs = _parse_positions(positions, seq)
    if isinstance(pos_pairs, dict):
        return pos_pairs

    notes: dict[int, str] = {}
    if annotate_positions.strip():
        try:
            notes = {int(k): str(v) for k, v in json.loads(annotate_positions).items()}
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return {"status": "error", "summary": f"annotate_positions is not valid JSON: {exc}",
                    "data": None, "suggestions": ["Pass an object like {\"143\": \"catalytic\"}."]}

    mode_tuple = tuple(m.strip() for m in modes.split(",") if m.strip())
    try:
        variants = build_variants(
            seq, pos_pairs, label=label, alphabet=alphabet, modes=mode_tuple,
            annotate=lambda rn, wt: notes.get(rn, ""),
        )
    except ValueError as exc:
        return {
            "status": "error",
            "summary": f"Variant construction failed: {exc}",
            "data": None,
            "suggestions": ["A wild-type mismatch means the structure and the position "
                            "list disagree on numbering. Re-derive the positions."],
        }

    out = _out_dir(run_dir, "variants")
    stem = label or path.stem
    tsv = write_variants_tsv(variants, out / f"{stem}_variants.tsv", modes=mode_tuple)
    fasta = write_variants_fasta(variants, out / f"{stem}_variants.fasta")
    counts = job_counts(variants, mode_tuple or ("",), n_samples=n_samples)
    annotated = sorted({v.resnum for v in variants if v.annotation and v.resnum})

    return {
        "status": "success",
        "summary": (
            f"{counts['n_variants']} variants over {counts['n_positions']} positions "
            f"-> {counts['n_jobs']} jobs, {counts['n_models']} models"
        ),
        "data": {
            "label": label,
            "sequence_length": len(seq),
            "modes": list(mode_tuple),
            "annotated_positions": annotated,
            "variants_tsv": str(tsv),
            "variants_fasta": str(fasta),
            **counts,
        },
        "suggestions": [
            "Generate MSAs in BATCH for the FASTA before cofolding; never transplant "
            "the wild-type MSA onto a mutant.",
            "Keep the annotated positions in the fold set: they are the negative "
            "controls that calibrate everything else.",
        ],
    }


def _parse_positions(spec: str, seq: str):
    """Residue numbers from a comma list or a positions TSV -> [(resnum, wt_aa)]."""
    spec = spec.strip()
    if not spec:
        return {"status": "error", "summary": "No positions given.", "data": None,
                "suggestions": ["Run scan_design_pocket first, or list residue numbers."]}

    p = Path(spec).expanduser()
    nums: list[int] = []
    if p.is_file():
        import csv as _csv
        with p.open() as fh:
            for row in _csv.DictReader(fh, delimiter="\t"):
                if "resnum" in row and row["resnum"]:
                    nums.append(int(row["resnum"]))
        if not nums:
            return {"status": "error", "summary": f"No 'resnum' column in {p}",
                    "data": None, "suggestions": ["Pass a positions TSV from scan_design_pocket."]}
    else:
        try:
            nums = [int(x) for x in spec.replace(" ", "").split(",") if x]
        except ValueError:
            return {"status": "error", "summary": f"Could not read positions from {spec!r}",
                    "data": None, "suggestions": ["Give '20,143,190' or a positions TSV path."]}

    bad = [n for n in nums if not 1 <= n <= len(seq)]
    if bad:
        return {"status": "error",
                "summary": f"Positions outside the sequence (length {len(seq)}): {bad[:5]}",
                "data": None,
                "suggestions": ["The positions and the structure disagree on numbering."]}
    return [(n, seq[n - 1]) for n in sorted(set(nums))]
