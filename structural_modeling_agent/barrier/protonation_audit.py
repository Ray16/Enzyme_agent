"""Protonation audit for a prepared holo system — pipeline gate.

WHY THIS EXISTS: `prepare.py` assigns protonation from pdb4amber/tleap defaults (every His -> HIE,
every Asp/Glu deprotonated, every Lys +1). Those defaults are usually right but are NEVER checked, and
a wrong charge state is *structurally invisible* — the MD runs beautifully and looks normal while the
active-site electrostatics are wrong. Distances tolerate that; barriers and salt-bridge/register
conclusions do not.

WHAT IT DOES: runs PROPKA3 on the prepared protein, compares predicted states at the build pH against
the states actually built, and — critically — classifies each discrepancy by its ligand environment.

THE ARTIFACT PROBLEM (do not skip): PROPKA sees protein only. A holo metalloenzyme active site has
Mg(2+) and ATP(4-) that PROPKA never saw, so it mis-predicts exactly the residues we care about, in
BOTH directions:
  * carboxylate coordinating a metal -> PROPKA (no metal) sees a repulsive carboxylate cluster and
    raises its pKa, predicting protonation. WRONG: a metal-coordinating carboxylate is deprotonated.
  * cation salt-bridged to ATP phosphate -> PROPKA (no ATP) sees an unpaired cation and lowers its
    pKa, predicting neutral. WRONG: a phosphate-bound Lys/Arg stays charged.
So a bare "PROPKA says X" is not actionable. Each flag is classified as:
  artifact_metal_ligand / artifact_anionic_ligand  -> built state is correct, no action
  real_active_site                                  -> genuine error near the reaction centre: FIX
  real_remote                                       -> genuine error far from the site: fix for a
                                                       clean net charge, low impact on geometry
  uncertain                                         -> buried/ambiguous, needs a human look

Verdict feeds `preparation.json` so the gate can refuse to hand a mis-protonated system to a barrier.

CLI:
  python -m structural_modeling_agent.barrier.protonation_audit \
      --prep <prep_dir> [--ph 8.0] [--json out.json]
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

PROPKA = "/homes/rzhu/miniforge3/envs/pypka/bin/propka3"

# residue -> (built protonation at default pH, charge when "protonated form" dominates)
ACIDIC = {"ASP", "ASH", "GLU", "GLH"}
BASIC_HIS = {"HIS", "HIE", "HID", "HIP"}
CATIONIC = {"LYS", "LYN", "ARG"}

# distance thresholds (A)
METAL_COORD = 2.8      # carboxylate O / His N within this of a metal = coordinating
LIGAND_SALT = 3.5      # cation N within this of an anionic ligand atom = salt bridge
ACTIVE_SITE = 8.0      # within this of substrate/reaction centre = active site


def run_propka(pdb: Path, workdir: Path) -> Path:
    """Run PROPKA3 on a protein-only PDB; return the .pka file."""
    local = workdir / pdb.name
    shutil.copy(pdb, local)
    subprocess.run([PROPKA, local.name], cwd=workdir, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    pka = local.with_suffix(".pka")
    if not pka.is_file():
        raise RuntimeError(f"PROPKA produced no .pka for {pdb}")
    return pka


def parse_pka(pka: Path):
    """Parse the SUMMARY block -> [(resname, resid, pKa)]."""
    rows, inside = [], False
    for line in pka.read_text().splitlines():
        if "SUMMARY OF THIS PREDICTION" in line:
            inside = True
            continue
        if inside:
            if line.strip().startswith("-" * 10):
                break
            p = line.split()
            if len(p) >= 4:
                try:
                    rows.append((p[0], int(p[1]), float(p[3])))
                except ValueError:
                    continue
    return rows


def expected_state(resname: str, pka: float, ph: float) -> str:
    """What the pKa implies at this pH."""
    if resname in ACIDIC:
        return "protonated_neutral" if pka > ph else "deprotonated_anion"
    if resname in BASIC_HIS:
        return "protonated_cation" if pka > ph else "neutral"
    if resname in CATIONIC:
        return "protonated_cation" if pka > ph else "neutral"
    if resname in {"CYS", "CYM"}:
        return "thiolate_anion" if pka < ph else "neutral"
    if resname in {"TYR"}:
        return "tyrosinate_anion" if pka < ph else "neutral"
    return "unknown"


def built_state(resname: str) -> str:
    return {
        "ASP": "deprotonated_anion", "GLU": "deprotonated_anion",
        "ASH": "protonated_neutral", "GLH": "protonated_neutral",
        "HIE": "neutral", "HID": "neutral", "HIS": "neutral", "HIP": "protonated_cation",
        "LYS": "protonated_cation", "LYN": "neutral", "ARG": "protonated_cation",
        "CYS": "neutral", "CYM": "thiolate_anion", "CYX": "disulfide", "TYR": "neutral",
    }.get(resname, "unknown")


# Atoms that actually carry/accept the proton. Distances MUST be measured from these, not from the
# whole residue: a side chain can have its CB near a metal while the titratable N/O points away, which
# silently mis-classifies a real error as a metal-ligand artifact.
TITRATABLE_ATOMS = {
    "ASP": "OD1 OD2", "ASH": "OD1 OD2", "GLU": "OE1 OE2", "GLH": "OE1 OE2",
    "HIS": "ND1 NE2", "HIE": "ND1 NE2", "HID": "ND1 NE2", "HIP": "ND1 NE2",
    "LYS": "NZ", "LYN": "NZ", "ARG": "NE NH1 NH2",
    "CYS": "SG", "CYM": "SG", "TYR": "OH",
}


def classify(resname, resid, universe, cofactor_resn, substrate_resn, metal_resn):
    """Environment-based classification; returns (kind, detail dict)."""
    import numpy as np
    names = TITRATABLE_ATOMS.get(resname)
    sel = universe.select_atoms(f"resid {resid} and name {names}") if names else universe.atoms[[]]
    if not len(sel):
        sel = universe.select_atoms(f"resid {resid}")
    if not len(sel):
        return "uncertain", {"note": "residue absent from topology"}
    def mind(selstr):
        t = universe.select_atoms(selstr)
        if not len(t):
            return None
        return float(np.linalg.norm(sel.positions[:, None] - t.positions[None], axis=2).min())
    d_metal = mind(f"resname {metal_resn}")
    d_cof = mind(f"resname {cofactor_resn}")
    d_sub = mind(f"resname {substrate_resn}")
    waters = len(universe.select_atoms(f"resname WAT and name O and around 4 resid {resid}"))
    burial = len(universe.select_atoms(f"(protein and not name H*) and around 6 resid {resid}"))
    detail = {"min_dist_metal_A": d_metal, "min_dist_cofactor_A": d_cof,
              "min_dist_substrate_A": d_sub, "waters_4A": waters, "burial_6A": burial}

    if resname in ACIDIC and d_metal is not None and d_metal <= METAL_COORD:
        return "artifact_metal_ligand", detail
    if resname in BASIC_HIS and d_metal is not None and d_metal <= METAL_COORD:
        return "artifact_metal_ligand", detail
    if resname in CATIONIC and d_cof is not None and d_cof <= LIGAND_SALT:
        return "artifact_anionic_ligand", detail
    near = [d for d in (d_sub, d_cof, d_metal) if d is not None]
    if near and min(near) <= ACTIVE_SITE:
        return "real_active_site", detail
    if burial >= 100 and waters == 0:
        return "uncertain", detail
    return "real_remote", detail


def audit(prep_dir, ph=8.0, protein_pdb=None, cofactor_resn="ATP",
          substrate_resn="SUB", metal_resn="MG", coords=None):
    import MDAnalysis as mda
    prep = Path(prep_dir)
    protein_pdb = Path(protein_pdb) if protein_pdb else prep / "protein_clean.pdb"
    prmtop = prep / "complex_1264.prmtop"
    if not prmtop.is_file():
        prmtop = prep / "complex.prmtop"
    # coordinates are required: classification is geometric (metal / ligand contacts)
    crd = Path(coords) if coords else prep / "complex.inpcrd"
    if not crd.is_file():
        raise FileNotFoundError(f"need coordinates for geometric classification: {crd}")
    u = mda.Universe(str(prmtop), str(crd))
    built = {r.resid: r.resname for r in u.residues}

    with tempfile.TemporaryDirectory() as td:
        rows = parse_pka(run_propka(protein_pdb, Path(td)))

    findings = []
    for resname, resid, pka in rows:
        b = built.get(resid)
        if b is None:
            continue
        exp, got = expected_state(resname, pka, ph), built_state(b)
        if exp == "unknown" or got == "unknown" or exp == got or got == "disulfide":
            continue
        kind, detail = classify(b, resid, u, cofactor_resn, substrate_resn, metal_resn)
        findings.append({"resid": resid, "resname_propka": resname, "resname_built": b,
                         "pKa": pka, "ph": ph, "built_state": got, "propka_state": exp,
                         "classification": kind, **detail})

    counts = {}
    for f in findings:
        counts[f["classification"]] = counts.get(f["classification"], 0) + 1
    real_as = [f for f in findings if f["classification"] == "real_active_site"]
    verdict = ("protonation_ok" if not real_as and not counts.get("uncertain")
               else "protonation_REVIEW_REQUIRED" if real_as
               else "protonation_ok_with_uncertain")
    return {"verdict": verdict, "ph": ph, "counts": counts,
            "n_active_site_errors": len(real_as), "findings": findings}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prep", required=True)
    p.add_argument("--ph", type=float, default=8.0)
    p.add_argument("--protein-pdb", default=None)
    p.add_argument("--cofactor-resn", default="ATP")
    p.add_argument("--substrate-resn", default="SUB")
    p.add_argument("--metal-resn", default="MG")
    p.add_argument("--coords", default=None, help="coordinates (default <prep>/complex.inpcrd)")
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    rep = audit(a.prep, a.ph, a.protein_pdb, a.cofactor_resn, a.substrate_resn, a.metal_resn, a.coords)
    text = json.dumps(rep, indent=2)
    if a.json:
        Path(a.json).write_text(text)
    print(text)
    return 0 if rep["verdict"] != "protonation_REVIEW_REQUIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
