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


def _load_heavy_pts(pdb):
    """(element, xyz) for every heavy atom in a PDB (H dropped)."""
    pts = []
    for ln in Path(pdb).read_text().splitlines():
        if ln[:6] in ("ATOM  ", "HETATM"):
            el = (ln[76:78].strip() or ln[12:16].strip()[0]).upper()
            if el == "H":
                continue
            pts.append((el, np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])))
    return pts


def _make_water(O, M, dOH=0.9572, aHOH=104.52):
    """TIP3P water at O, dipole-negative (lone-pair) end facing metal M: the two H's point
    AWAY from the metal so the O partial charge coordinates it. Returns (O, H1, H2)."""
    z = O - M; z = z / np.linalg.norm(z)                      # O->away-from-metal (H side)
    t = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(z, t); x = x / np.linalg.norm(x)
    half = np.radians(aHOH / 2.0)
    H1 = O + dOH * (np.cos(half) * z + np.sin(half) * x)
    H2 = O + dOH * (np.cos(half) * z - np.sin(half) * x)
    return O, H1, H2


def seed_metal_waters(out_dir, target_coord=6, mo_dist=2.10, exist_cut=2.6, clash=2.25,
                      min_sep_deg=55.0):
    """Complete each metal's first coordination shell to `target_coord` (octahedral) with
    explicit TIP3P waters placed in the VACANT sites, so MD starts from a physical metal site
    instead of the under-coordinated geometry a ligand-only Boltz pose leaves.

    A nonbonded (Li--Merz 12-6-4) metal has no bonds to hold its ligands; if the inner shell
    is left empty, bulk solvation will NOT pack water at ~2.1 A into a buried pocket and the
    ion rearranges during equilibration, dragging the reactive geometry with it. For each
    metal we count donors (O/N) already within `exist_cut`, then drop (target_coord - n_have)
    waters along sphere directions that are >=`min_sep_deg` from every occupied direction and
    whose O does not clash (< `clash` A) with any heavy atom, other metal, or prior shell
    water. If a site is too buried to place a non-clashing water, it is left (and flagged by
    the post-build coordination check) rather than forced into a clash. GENERAL across 1- and
    2-metal sites. Writes mg_shell_waters.pdb; returns the number of waters placed."""
    out = Path(out_dir)
    heavy = []
    for f in ("protein_clean.pdb", "atp_heavy.pdb", "sub_heavy.pdb"):
        if (out / f).exists():
            heavy += _load_heavy_pts(out / f)
    all_xyz = np.array([x for _, x in heavy]) if heavy else np.zeros((0, 3))
    donor_xyz = np.array([x for el, x in heavy if el in ("O", "N")]) if heavy else np.zeros((0, 3))
    metals = [np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
              for ln in (out / "metal.pdb").read_text().splitlines()
              if ln[:6] in ("ATOM  ", "HETATM")]
    # candidate directions: Fibonacci sphere (even coverage, no axis bias)
    N = 600
    i = np.arange(N); phi = np.arccos(1 - 2 * (i + 0.5) / N); theta = np.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    cos_sep = np.cos(np.radians(min_sep_deg))
    waters = []
    placed_O = []                       # shell-water O's already placed (clash partners)
    for M in metals:
        occ = []                        # unit vectors to current inner-shell ligands
        if len(donor_xyz):
            for v in donor_xyz[np.linalg.norm(donor_xyz - M, axis=1) < exist_cut]:
                occ.append((v - M) / np.linalg.norm(v - M))
        others = np.array([m for m in metals if not np.allclose(m, M)] + placed_O) \
            if (len(metals) > 1 or placed_O) else np.zeros((0, 3))
        for _ in range(max(0, target_coord - len(occ))):
            best, best_score = None, -1.0
            for d in dirs:
                if occ and max(float(np.dot(d, od)) for od in occ) > cos_sep:
                    continue            # too close to an occupied direction
                p = M + mo_dist * d
                if len(all_xyz) and np.min(np.linalg.norm(all_xyz - p, axis=1)) < clash:
                    continue            # clashes a solute heavy atom
                if len(others) and np.min(np.linalg.norm(others - p, axis=1)) < clash:
                    continue            # clashes another metal / prior shell water
                score = min((1 - float(np.dot(d, od)) for od in occ), default=2.0)
                if score > best_score:
                    best_score, best = score, (d, p)
            if best is None:
                break                   # buried site: leave it, the coord check will flag it
            d, p = best
            occ.append(d); placed_O.append(p); waters.append(_make_water(p, M))
    if waters:
        with open(out / "mg_shell_waters.pdb", "w") as fh:
            ai = ri = 0
            for (O, H1, H2) in waters:
                ri += 1
                for nm, co in (("O", O), ("H1", H1), ("H2", H2)):
                    ai += 1
                    fh.write("ATOM  %5d %-4s WAT A%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n"
                             % (ai, nm, ri, co[0], co[1], co[2], nm[0]))
                fh.write("TER\n")
            fh.write("END\n")
    return len(waters)


def build_tleap(out_dir, resname, ion_set="1264", *, polarizabilities=None, pol_overrides=None,
                seed_waters=True):
    out = Path(out_dir)
    for a in ("ATP.lib", "ATP.frcmod"):
        shutil.copy(ASSET / a, out / a)
    n_shell = seed_metal_waters(out) if seed_waters else 0
    frc = "frcmod.ions234lm_1264_tip3p" if ion_set == "1264" else "frcmod.ions234lm_126_tip3p"
    shell_load = "shell = loadpdb mg_shell_waters.pdb\n" if n_shell else ""
    combine = "combine {{prot atp mgu SUB{shell}}}".format(shell=" shell" if n_shell else "")
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
{shell_load}cmplx = {combine}
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


def _lattice_from_box(box6):
    """Triclinic lattice matrix (rows = a,b,c vectors) from Amber's [a,b,c,alpha,beta,gamma].
    Orientation is arbitrary; only the metric matters for minimum-image distances, so this is
    valid for the truncated-octahedron cell solvateOct writes (alpha=beta=gamma=109.47)."""
    a, b, c, al, be, ga = box6
    al, be, ga = np.radians([al, be, ga])
    va = [a, 0.0, 0.0]
    vb = [b * np.cos(ga), b * np.sin(ga), 0.0]
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return np.array([va, vb, [cx, cy, cz]])


def validate_prepared(prmtop, inpcrd, spec_path, *, coord_cut=2.7, min_metal_coord=5):
    """Self-check a freshly built system BEFORE it costs any GPU time. Uses the correct
    triclinic minimum image (truncated-octahedron safe) for every distance, so it reports:
      - per-metal first-shell coordination number (N/O/S donors within coord_cut), and whether
        every metal reached at least `min_metal_coord` (octahedral-ish), and
      - the attack geometry (forming nuc->electrophile distance + attack angle) as actually
        built, to confirm assembly/solvation did not move the reactive pose.
    Returns a dict folded into preparation.json; never raises on an under-coordinated metal
    (a legitimately buried site may not reach 6) but flags it loudly via metal_octahedral_ok."""
    import parmed as pmd
    parm = pmd.load_file(str(prmtop), str(inpcrd))
    x = parm.coordinates
    L = _lattice_from_box(parm.box) if parm.box is not None else None
    inv = np.linalg.inv(L) if L is not None else None

    def mi(v):
        return v if inv is None else (v @ inv - np.round(v @ inv)) @ L

    metals = [a.idx for a in parm.atoms if a.atomic_number == 12]
    donors = [a.idx for a in parm.atoms if a.atomic_number in (7, 8, 16)]
    dx = x[donors]
    coordination = [int(np.count_nonzero(np.linalg.norm(mi(dx - x[m]), axis=1) < coord_cut))
                    for m in metals]

    from .md import resolve_attack_geometry
    from .spec import ReactionSpec
    spec = ReactionSpec.load(spec_path)
    i, j, k, angle_atom, angle_kind = resolve_attack_geometry(parm, spec)
    v1 = mi(x[i] - x[j]); v2 = mi(x[angle_atom] - x[j])
    forming = float(np.linalg.norm(v1))
    angle = float(np.degrees(np.arccos(np.clip(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))))
    return dict(metal_coordination=coordination,
                metal_octahedral_ok=all(c >= min_metal_coord for c in coordination),
                coordination_cutoff_A=coord_cut, min_metal_coordination=min_metal_coord,
                forming_distance_A=round(forming, 2), attack_angle_deg=round(angle, 1),
                attack_angle_definition=angle_kind)
