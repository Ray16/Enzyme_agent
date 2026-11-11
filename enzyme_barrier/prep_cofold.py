"""Stage-1 transfer prep: Boltz cofold pose (protein + ATP + metal + acid substrate,
heavy-atom-only) -> Amber-prepped, protonated, hydrated holo system. Reuses the validated
ATP library + Li-Merz 12-6-4 metal params + the AS9 leap recipe; the substrate (adipate,
glutarate, ...) is parameterized per-pose (obabel protonation at pH 7 + antechamber GAFF2/
AM1-BCC at the spec's formal charge). GENERAL: ligands are detected by composition (ATP by
its PA atom, metal by a lone metal atom, substrate = the remaining ligand), not by fixed
chains, so it transfers to any enzyme cofold of the same reaction family.
"""
from __future__ import annotations
import subprocess, shutil
from pathlib import Path
import numpy as np

ASSET = Path("/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/structural_modeling_agent/runs/"
             "amide_sampling_pilot_20261105/preparation/AS9/manual_prep_v1/3_solvation")
OBABEL = "/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/boltz-2/bin/obabel"
METALS = {"MG", "MN", "ZN", "CA", "FE", "NI", "CO"}


def _read_pdb_atoms(pdb):
    rows = []
    for ln in Path(pdb).read_text().splitlines():
        if ln[:6] in ("ATOM  ", "HETATM"):
            el = (ln[76:78].strip() or ln[12:16].strip()[0]).upper()
            rows.append(dict(rec=ln[:6], name=ln[12:16].strip(), chain=ln[21],
                             x=float(ln[30:38]), y=float(ln[38:46]), z=float(ln[46:54]),
                             elem=el, raw=ln))
    return rows


def split_cofold(cofold_pdb, out_dir):
    """Split into protein / ATP / metal / substrate by COMPOSITION (general)."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = _read_pdb_atoms(cofold_pdb)
    prot = [r for r in rows if r["rec"] == "ATOM  "]
    het = [r for r in rows if r["rec"] == "HETATM"]
    # group HETATM by (chain,) -- cofold puts each ligand on its own chain
    groups = {}
    for r in het:
        groups.setdefault(r["chain"], []).append(r)
    atp = sub = None; metals = []
    for ch, g in groups.items():
        names = {a["name"] for a in g}
        if "PA" in names and "PG" in names:
            atp = g
        elif len(g) == 1 and g[0]["elem"] in METALS:
            metals.append(g[0])          # keep ALL metals (e.g. the 2 Mg of a two-metal site)
        else:
            sub = g if sub is None else sub + g
    assert atp and metals and sub, f"could not classify ligands: ATP={bool(atp)} metals={len(metals)} sub={bool(sub)}"

    raw_prot = out / "protein_raw.pdb"
    raw_prot.write_text("\n".join(r["raw"] for r in prot) + "\nTER\nEND\n")
    subprocess.run(["pdb4amber", "-i", str(raw_prot), "-o", str(out / "protein_clean.pdb"), "--dry"],
                   check=True, capture_output=True, text=True)
    with open(out / "atp_heavy.pdb", "w") as f:
        for i, r in enumerate(atp, 1):
            f.write(f"ATOM  {i:5d} {r['name']:<4s} ATP A   1    "
                    f"{r['x']:8.3f}{r['y']:8.3f}{r['z']:8.3f}  1.00  0.00          {r['elem']:>2s}\n")
        f.write("TER\nEND\n")
    with open(out / "metal.pdb", "w") as f:
        for i, m in enumerate(metals, 1):          # one residue per metal ion
            f.write(f"ATOM  {i:5d} {m['elem']:<4s} {m['elem']:<3s} A{i:4d}    "
                    f"{m['x']:8.3f}{m['y']:8.3f}{m['z']:8.3f}  1.00  0.00          {m['elem']:>2s}\n")
        f.write("TER\nEND\n")
    return out, atp, metals, sub


def param_substrate(sub_rows, atp_rows, out_dir, resname, net_charge, ph=8.0):
    """Protonate the acid substrate (obabel at `ph`; 8.0 = experimental assay pH) +
    antechamber GAFF2/AM1-BCC at the spec's formal charge. Returns the cofold atom name of
    the carboxylate O nearest ATP PA (the nucleophile; resolved geometrically at carve)."""
    out = Path(out_dir)
    co = {r["name"]: np.array([r["x"], r["y"], r["z"]]) for r in sub_rows}
    pa = next(np.array([r["x"], r["y"], r["z"]]) for r in atp_rows if r["name"] == "PA")
    o_names = [r["name"] for r in sub_rows if r["elem"] == "O"]
    nuc_name = min(o_names, key=lambda n: np.linalg.norm(co[n] - pa))
    with open(out / "sub_heavy.pdb", "w") as f:
        for i, r in enumerate(sub_rows, 1):
            f.write(f"HETATM{i:5d} {r['name']:<4s} {resname:<3s} A   1    "
                    f"{r['x']:8.3f}{r['y']:8.3f}{r['z']:8.3f}  1.00  0.00          {r['elem']:>2s}\n")
        f.write("END\n")
    # protonate at pH 7 AND let obabel write explicit connectivity (mol2) -- its bond
    # perception is reliable on the clean protonated molecule; antechamber then KEEPS
    # those bonds (-j 1 = atom types only) instead of re-perceiving from the (possibly
    # distorted) cofold geometry, which previously invented a spurious ring bond.
    subprocess.run([OBABEL, "sub_heavy.pdb", "-O", "sub_h.mol2", "-p", str(ph)],
                   cwd=out, capture_output=True, text=True, check=True)
    ac = subprocess.run(["antechamber", "-i", "sub_h.mol2", "-fi", "mol2", "-o", "sub.mol2",
                         "-fo", "mol2", "-c", "bcc", "-nc", str(net_charge), "-at", "gaff2",
                         "-rn", resname, "-dr", "no", "-j", "1"], cwd=out, capture_output=True, text=True)
    (out / "antechamber.log").write_text(ac.stdout + "\n" + ac.stderr)
    ac.check_returncode()
    if (out / "sub.mol2").exists():
        subprocess.run(["parmchk2", "-i", "sub.mol2", "-f", "mol2", "-o", "sub.frcmod", "-s", "gaff2"],
                       cwd=out, capture_output=True, text=True, check=True)
    return nuc_name


def build_tleap(out_dir, resname, ion_set="1264", *, polarizabilities=None, pol_overrides=None):
    out = Path(out_dir)
    for a in ("ATP.lib", "ATP.frcmod"):
        shutil.copy(ASSET / a, out / a)
    frc = "frcmod.ions234lm_1264_tip3p" if ion_set == "1264" else "frcmod.ions234lm_126_tip3p"
    leap = f"""source leaprc.protein.ff14SB
source leaprc.gaff2
source leaprc.water.tip3p
loadamberparams {frc}
loadoff ATP.lib
loadamberparams ATP.frcmod
SUB = loadmol2 sub.mol2
loadamberparams sub.frcmod
prot = loadpdb protein_clean.pdb
atp = loadpdb atp_heavy.pdb
mgu = loadpdb metal.pdb
cmplx = combine {{prot atp mgu SUB}}
solvateOct cmplx TIP3PBOX 12.0
addIonsRand cmplx Na+ 0
addIonsRand cmplx Cl- 0
charge cmplx
saveamberparm cmplx complex.prmtop complex.inpcrd
quit
"""
    (out / "build.leap").write_text(leap)
    r = subprocess.run(["tleap", "-f", "build.leap"], cwd=out, capture_output=True, text=True)
    (out / "build.log").write_text(r.stdout + "\n" + r.stderr)
    r.check_returncode()
    if not (out / "complex.prmtop").exists() or not (out / "complex.inpcrd").exists():
        raise RuntimeError("tleap did not write both topology and coordinates; inspect build.log")
    topology = out / "complex.prmtop"
    if ion_set == "1264":
        from .parameters import install_c4
        if polarizabilities is None:
            import os
            polarizabilities = Path(os.environ.get('AMBERHOME', '')) / 'dat/leap/parm/lj_1264_pol.dat'
        topology = out / 'complex_1264.prmtop'
        install_c4(out / 'complex.prmtop', topology, polarizabilities, overrides=pol_overrides)
    return True, topology, out / "complex.inpcrd"
