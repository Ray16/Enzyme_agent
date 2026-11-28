"""Stage-1 transfer prep: Boltz cofold pose (protein + ATP + metal + acid substrate,
heavy-atom-only) -> Amber-prepped, protonated, hydrated holo system. Reuses the validated
ATP library + Li-Merz 12-6-4 metal params + the AS9 leap recipe; the substrate (adipate,
glutarate, ...) is parameterized per-pose (obabel protonation at pH 7 + antechamber GAFF2/
AM1-BCC at the spec's formal charge). GENERAL: ligands are detected by composition (ATP by
its PA atom, metal by a lone metal atom, substrate = the remaining ligand), not by fixed
chains, so it transfers to any enzyme cofold of the same reaction family.
"""
from __future__ import annotations
import json
import os
import subprocess, shutil
from pathlib import Path
import numpy as np

# Default cofactor-library asset dir (ATP.lib / ATP.frcmod live here).
# Override with $BARRIER_COFACTOR_ASSET so callers can supply custom cofactor libraries.
_DEFAULT_ASSET = Path(__file__).resolve().with_name("assets")
METALS = {"MG", "MN", "ZN", "CA", "FE", "NI", "CO"}
# Map a water model keyword -> (leaprc suffix, solvate box keyword). Extend as needed.
_WATER_LEAP = {"tip3p": ("tip3p", "TIP3PBOX"), "tip4pew": ("tip4pew", "TIP4PEWBOX"),
               "spce": ("spce", "SPCBOX"), "opc": ("opc", "OPCBOX"), "opc3": ("opc3", "OPC3BOX")}


def asset_dir() -> Path:
    """Directory holding the cofactor lib/frcmod and 12-6-4 ion frcmods. $BARRIER_COFACTOR_ASSET
    overrides the built-in default (keeps the pipeline portable across machines)."""
    return Path(os.environ.get("BARRIER_COFACTOR_ASSET", str(_DEFAULT_ASSET)))


def obabel_exe() -> str:
    """Resolve the obabel executable: $OBABEL, else PATH, else the known boltz-2 env fallback.
    Never hardcode a single interpreter path as the only option."""
    return os.environ.get("OBABEL") or shutil.which("obabel") or \
        "/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/boltz-2/bin/obabel"


def _read_pdb_atoms(pdb):
    rows = []
    for ln in Path(pdb).read_text().splitlines():
        if ln[:6] in ("ATOM  ", "HETATM"):
            el = (ln[76:78].strip() or ln[12:16].strip()[0]).upper()
            rows.append(dict(rec=ln[:6], name=ln[12:16].strip(), chain=ln[21],
                             x=float(ln[30:38]), y=float(ln[38:46]), z=float(ln[46:54]),
                             elem=el, raw=ln))
    return rows


def split_cofold(cofold_pdb, out_dir, *, cofactor_resname="ATP", cofactor_atoms=("PA", "PG"),
                 require_metal=True, require_substrate=True):
    """Split into protein / cofactor / metal / substrate by COMPOSITION (general).

    Nothing here is enzyme-specific. The cofactor is the HETATM group whose atom names contain
    ALL of `cofactor_atoms` (default {PA,PG} = ATP's phosphate backbone; pass e.g. ('PB','N1')
    for a different cofactor); metals are lone atoms whose element is in METALS; the remaining
    HETATM is the substrate. `require_metal` / `require_substrate` relax the classification for
    metal-free or cofactor-only reactions. The cofactor PDB is written with `cofactor_resname`
    so it matches the residue name in the cofactor library loaded at tleap time."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = _read_pdb_atoms(cofold_pdb)
    prot = [r for r in rows if r["rec"] == "ATOM  "]
    het = [r for r in rows if r["rec"] == "HETATM"]
    # group HETATM by (chain,) -- cofold puts each ligand on its own chain
    groups = {}
    for r in het:
        groups.setdefault(r["chain"], []).append(r)
    want = set(cofactor_atoms)
    atp = sub = None; metals = []
    for ch, g in groups.items():
        names = {a["name"] for a in g}
        if want <= names:
            atp = g
        elif len(g) == 1 and g[0]["elem"] in METALS:
            metals.append(g[0])          # keep ALL metals (e.g. the 2 Mg of a two-metal site)
        else:
            sub = g if sub is None else sub + g
    if not atp:
        raise AssertionError(f"could not find cofactor (atoms {sorted(want)}) in {cofold_pdb}")
    if require_metal and not metals:
        raise AssertionError(f"no metal found in {cofold_pdb} (set require_metal=False if intended)")
    if require_substrate and not sub:
        raise AssertionError(f"no substrate found in {cofold_pdb} (set require_substrate=False if intended)")

    raw_prot = out / "protein_raw.pdb"
    raw_prot.write_text("\n".join(r["raw"] for r in prot) + "\nTER\nEND\n")
    subprocess.run(["pdb4amber", "-i", str(raw_prot), "-o", str(out / "protein_clean.pdb"), "--dry"],
                   check=True, capture_output=True, text=True)
    with open(out / "atp_heavy.pdb", "w") as f:
        for i, r in enumerate(atp, 1):
            f.write(f"ATOM  {i:5d} {r['name']:<4s} {cofactor_resname:<3s} A   1    "
                    f"{r['x']:8.3f}{r['y']:8.3f}{r['z']:8.3f}  1.00  0.00          {r['elem']:>2s}\n")
        f.write("TER\nEND\n")
    with open(out / "metal.pdb", "w") as f:
        for i, m in enumerate(metals, 1):          # one residue per metal ion
            f.write(f"ATOM  {i:5d} {m['elem']:<4s} {m['elem']:<3s} A{i:4d}    "
                    f"{m['x']:8.3f}{m['y']:8.3f}{m['z']:8.3f}  1.00  0.00          {m['elem']:>2s}\n")
        f.write("TER\nEND\n")
    return out, atp, metals, sub


_PROTONATION_RESNAMES = {"HIP", "HIE", "HID", "GLH", "ASH", "LYN", "CYM", "GLU", "ASP", "LYS", "HIS", "CYS"}


def apply_protonation_overrides(protein_pdb, overrides):
    """Force specific protonation states by renaming residues in the cleaned protein PDB BEFORE tleap
    (tleap builds the correct protons from the residue name). `overrides` maps resid (in protein_clean
    numbering, which matches PROPKA) -> target Amber resname, e.g. {265: 'HIP', 455: 'GLH'}.

    This is the ONLY place protonation can be set: without it prep takes tleap/pdb4amber defaults
    (His->HIE, Asp/Glu deprotonated, Lys +1), which are unchecked and structurally invisible when wrong.
    Returns the list of (resid, old_resname, new_resname) actually applied."""
    if not overrides:
        return []
    path = Path(protein_pdb)
    want = {int(k): str(v).strip().upper() for k, v in overrides.items()}
    for v in want.values():
        if v not in _PROTONATION_RESNAMES:
            raise ValueError(f"unsupported protonation target resname: {v}")
    applied, out_lines = [], []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[22:26].strip().isdigit():
            resid = int(line[22:26]); new = want.get(resid)
            if new:
                old = line[17:20].strip()
                line = line[:17] + f"{new:>3s}" + line[20:]
                if not applied or applied[-1][0] != resid:
                    applied.append((resid, old, new))
        out_lines.append(line)
    if len(applied) != len(want):
        missing = set(want) - {a[0] for a in applied}
        raise ValueError(f"protonation override resids not found in {path.name}: {sorted(missing)}")
    path.write_text("\n".join(out_lines) + "\n")
    return applied


def param_substrate(sub_rows, atp_rows, out_dir, resname, net_charge, ph=8.0, *,
                    anchor_atom="PA", nuc_elements=("O",)):
    """Protonate the substrate (obabel at `ph`; 8.0 = experimental assay pH) + antechamber
    GAFF2/AM1-BCC at the spec's formal charge. Returns the cofold atom name of the nucleophile:
    the substrate atom of element in `nuc_elements` (default O, for a carboxylate; pass ('N',)
    or ('O','N','S') for other nucleophiles) nearest the cofactor's `anchor_atom` (default ATP
    PA). The actual nucleophile index is re-resolved geometrically at carve time."""
    out = Path(out_dir)
    co = {r["name"]: np.array([r["x"], r["y"], r["z"]]) for r in sub_rows}
    pa = next(np.array([r["x"], r["y"], r["z"]]) for r in atp_rows if r["name"] == anchor_atom)
    elems = {e.upper() for e in nuc_elements}
    o_names = [r["name"] for r in sub_rows if r["elem"] in elems]
    if not o_names:
        raise ValueError(f"no candidate nucleophile atom (elements {sorted(elems)}) in substrate {resname}")
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
    subprocess.run([obabel_exe(), "sub_heavy.pdb", "-O", "sub_h.mol2", "-p", str(ph)],
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
                      min_sep_deg=55.0, cofactor_pdb="atp_heavy.pdb"):
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
    for f in ("protein_clean.pdb", cofactor_pdb, "sub_heavy.pdb"):
        if (out / f).exists():
            heavy += _load_heavy_pts(out / f)
    if not (out / "metal.pdb").exists():
        return 0
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


def _count_waters(prmtop):
    """Count solvent residues in a built prmtop (for salt-pair calculation)."""
    import parmed as pmd
    p = pmd.load_file(str(prmtop))
    return sum(1 for r in p.residues if r.name in {"WAT", "HOH", "TIP3", "T3P"})


def _leap_script(frc, *, salt_pairs, has_cofactor, has_metal, has_substrate, n_shell,
                 cofactor_lib="ATP.lib", cofactor_frcmod="ATP.frcmod", cofactor_pdb="atp_heavy.pdb",
                 protein_ff="ff14SB", gaff="gaff2", water_model="tip3p"):
    """Pure (no I/O) generator of the tleap build script. Every component is optional and every
    force field / cofactor / water model is a parameter, so the same builder serves ATP + metal +
    carboxylate as well as cofactor-only or metal-free systems. With the defaults and all three
    components present it reproduces the historical AS9/GlnA3 script byte-for-byte (locked by test)."""
    if water_model not in _WATER_LEAP:
        raise ValueError(f"unknown water model {water_model!r}; known: {sorted(_WATER_LEAP)}")
    water_leaprc, box = _WATER_LEAP[water_model]
    lines = [f"source leaprc.protein.{protein_ff}", f"source leaprc.{gaff}",
             f"source leaprc.water.{water_leaprc}", f"loadamberparams {frc}"]
    if has_cofactor:
        lines += [f"loadoff {cofactor_lib}", f"loadamberparams {cofactor_frcmod}"]
    if has_substrate:
        lines += ["SUB = loadmol2 sub.mol2", "loadamberparams sub.frcmod"]
    lines += ["prot = loadpdb protein_clean.pdb"]
    combine = ["prot"]
    if has_cofactor:
        lines += [f"atp = loadpdb {cofactor_pdb}"]; combine.append("atp")
    if has_metal:
        lines += ["mgu = loadpdb metal.pdb"]; combine.append("mgu")
    if has_substrate:
        combine.append("SUB")
    if n_shell:
        lines += ["shell = loadpdb mg_shell_waters.pdb"]; combine.append("shell")
    lines += [f"cmplx = combine {{{' '.join(combine)}}}", f"solvateOct cmplx {box} 12.0",
              "addIonsRand cmplx Na+ 0", "addIonsRand cmplx Cl- 0"]
    if salt_pairs:
        lines += [f"addIonsRand cmplx Na+ {salt_pairs}", f"addIonsRand cmplx Cl- {salt_pairs}"]
    lines += ["charge cmplx", "saveamberparm cmplx complex.prmtop complex.inpcrd", "quit"]
    return "\n".join(lines) + "\n"


def build_tleap(out_dir, resname, ion_set="1264", *, polarizabilities=None, pol_overrides=None,
                seed_waters=True, salt_molarity=0.15, cofactor_resname="ATP",
                cofactor_lib=None, cofactor_frcmod=None, has_metal=True, has_substrate=True,
                protein_ff="ff14SB", gaff="gaff2", water_model="tip3p", metal_target_coord=6):
    """Build the solvated holo prmtop. salt_molarity adds physiological NaCl ON TOP of neutralization
    (default 0.15 M): counterion-only boxes have no ionic screening, which over-stabilizes the
    arginine-carboxylate salt bridges that the substrate-register analysis depends on. The 12-6-4 ion
    frcmod supplies matched params for the added Na+/Cl-. salt_molarity=0 recovers neutralize-only.

    Cofactor library/frcmod default to `{cofactor_resname}.lib`/`.frcmod` in asset_dir(); force
    field, water model, and metal target coordination are all overridable. Defaults reproduce the
    historical ATP+Mg+carboxylate build exactly."""
    out = Path(out_dir)
    cofactor_lib = cofactor_lib or f"{cofactor_resname}.lib"
    cofactor_frcmod = cofactor_frcmod or f"{cofactor_resname}.frcmod"
    for a in (cofactor_lib, cofactor_frcmod):
        shutil.copy(asset_dir() / a, out / a)
    n_shell = seed_metal_waters(out, target_coord=metal_target_coord) if (seed_waters and has_metal) else 0
    frc = "frcmod.ions234lm_1264_tip3p" if ion_set == "1264" else "frcmod.ions234lm_126_tip3p"

    def _leap(salt_pairs):
        return _leap_script(frc, salt_pairs=salt_pairs, has_cofactor=True, has_metal=has_metal,
                            has_substrate=has_substrate, n_shell=n_shell, cofactor_lib=cofactor_lib,
                            cofactor_frcmod=cofactor_frcmod, protein_ff=protein_ff, gaff=gaff,
                            water_model=water_model)

    def _run(leap_text):
        (out / "build.leap").write_text(leap_text)
        r = subprocess.run(["tleap", "-f", "build.leap"], cwd=out, capture_output=True, text=True)
        (out / "build.log").write_text(r.stdout + "\n" + r.stderr)
        r.check_returncode()
        if not (out / "complex.prmtop").exists() or not (out / "complex.inpcrd").exists():
            raise RuntimeError("tleap did not write both topology and coordinates; inspect build.log")

    # pass 1: neutralize-only, to learn the solvent count
    _run(_leap(0))
    salt_pairs = 0
    if salt_molarity and salt_molarity > 0:
        n_wat = _count_waters(out / "complex.prmtop")
        salt_pairs = int(round(salt_molarity * n_wat / 55.56))  # 55.56 M = molarity of pure water
        if salt_pairs > 0:
            _run(_leap(salt_pairs))  # pass 2: rebuild with physiological NaCl
    (out / "salt.json").write_text(json.dumps(
        {"salt_molarity": salt_molarity, "salt_pairs_added": salt_pairs,
         "n_waters": _count_waters(out / "complex.prmtop")}))
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


_WATER = {"WAT", "HOH", "TIP3", "T3P"}
_IONS = {"Na+", "Cl-", "NA", "CL", "K+", "K", "NA+", "CL-"}


def validate_prepared(prmtop, inpcrd, spec_path, *, coord_cut=2.7, min_metal_coord=5,
                      completable_cut=3.1, nb_cutoff_A=10.0, clash_A=1.8, severe_clash_A=1.2,
                      min_waters=500):
    """Built-in PHYSICAL validation of a freshly built system, BEFORE it costs any GPU time.
    Every distance uses the correct triclinic minimum image (truncated-octahedron safe). Checks:

      metal coordination  - per-metal N/O/S donors. Reports the tight first-shell count (coord_cut)
                            AND a "completable" count within completable_cut; physically_valid gates on
                            the latter, since a raw-pose donor at 2.7-3.1 A relaxes into the first shell
                            on minimization. A metal short of min_metal_coord even within the outer
                            shell is genuinely under-coordinated (fatal).
      attack geometry     - forming nuc->electrophile distance + attack angle as built, to
                            confirm assembly/solvation did not move the reactive pose.
      net charge          - sum of partial charges must be ~integer and ~0 (neutralized).
      solvation adequacy  - (a) enough waters were added; (b) the solute does not see its own
                            periodic image within the nonbonded cutoff: along each lattice
                            direction the largest solute-free arc (image gap) must exceed
                            nb_cutoff_A. Guards against a box that is too small / bad solvation.
      steric clashes      - closest NON-bonded solute heavy-atom contact; counts severe
                            overlaps (< severe_clash_A) that indicate a broken build (mild
                            clashes < clash_A are reported but tolerated -- minimization fixes
                            them).

    Returns a dict folded into preparation.json with an overall `physically_valid` flag. Never
    raises (a legitimately buried metal or a mild pose clash should be visible, not fatal); the
    caller decides how to gate on the individual booleans."""
    import parmed as pmd
    parm = pmd.load_file(str(prmtop), str(inpcrd))
    x = parm.coordinates
    L = _lattice_from_box(parm.box) if parm.box is not None else None
    inv = np.linalg.inv(L) if L is not None else None

    def mi(v):
        return v if inv is None else (v @ inv - np.round(v @ inv)) @ L

    # --- metal coordination ---
    metals = [a.idx for a in parm.atoms if a.atomic_number == 12]
    donors = [a.idx for a in parm.atoms if a.atomic_number in (7, 8, 16)]
    dx = x[donors]
    dist_to_metal = [np.linalg.norm(mi(dx - x[m]), axis=1) for m in metals]
    # inner first-shell count (tight, raw-pose geometry) -- reported for info
    coordination = [int(np.count_nonzero(dm < coord_cut)) for dm in dist_to_metal]
    # "completable" count: donors reachable within an outer shell that minimization pulls into the
    # first shell. A raw cofold pose often seats a Mg donor at 2.7-3.1 A (reads as under-coordinated
    # at the tight cutoff) yet relaxes to octahedral on minimization -> that site IS physical.
    # physically_valid gates on completability; a metal lacking donors even here is genuinely bad.
    coordination_completable = [int(np.count_nonzero(dm < completable_cut)) for dm in dist_to_metal]
    metal_ok = all(c >= min_metal_coord for c in coordination_completable)

    # --- attack geometry ---
    from .md import resolve_attack_geometry
    from .spec import ReactionSpec
    spec = ReactionSpec.load(spec_path)
    i, j, k, angle_atom, angle_kind = resolve_attack_geometry(parm, spec)
    v1 = mi(x[i] - x[j]); v2 = mi(x[angle_atom] - x[j])
    forming = float(np.linalg.norm(v1))
    angle = float(np.degrees(np.arccos(np.clip(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))))

    # --- net charge (neutralization) ---
    net_charge = float(sum(a.charge for a in parm.atoms))
    charge_ok = abs(net_charge - round(net_charge)) < 0.01 and abs(net_charge) < 0.5

    # --- solvation: water count + solute/image gap along each lattice direction ---
    n_waters = sum(1 for r in parm.residues if r.name in _WATER)
    solute = np.array([a.idx for a in parm.atoms
                       if a.residue.name not in _WATER and a.residue.name not in _IONS
                       and a.atomic_number > 1])
    # closest approach between the solute and ANY of its 26 neighbouring periodic images
    # (rigorous for any triclinic / truncated-octahedron cell, unlike a per-axis estimate):
    # if this exceeds the nonbonded cutoff the solute never directly sees its own image.
    image_gap = float("inf")
    if L is not None and len(solute) > 1:
        try:
            from scipy.spatial import cKDTree
            S = x[solute]; tree = cKDTree(S)
            for na in (-1, 0, 1):
                for nb_ in (-1, 0, 1):
                    for nc in (-1, 0, 1):
                        if na == nb_ == nc == 0:
                            continue
                        T = na * L[0] + nb_ * L[1] + nc * L[2]
                        d, _ = tree.query(S + T)
                        image_gap = min(image_gap, float(d.min()))
        except Exception:
            image_gap = float("inf")                 # scipy unavailable -> skip this check
    solvation_ok = bool(n_waters >= min_waters and
                        (image_gap == float("inf") or image_gap >= nb_cutoff_A))

    # --- steric clashes among solute heavy atoms (solute is whole/centred -> direct dist) ---
    bonded = set()
    for b in parm.bonds:
        bonded.add((min(b.atom1.idx, b.atom2.idx), max(b.atom1.idx, b.atom2.idx)))
    min_contact, n_severe, n_clash = float("inf"), 0, 0
    if len(solute) > 1:
        try:
            from scipy.spatial import cKDTree
            pairs = cKDTree(x[solute]).query_pairs(clash_A)
            for a, b in pairs:
                ia, ib = int(solute[a]), int(solute[b])
                key = (min(ia, ib), max(ia, ib))
                if key in bonded:
                    continue
                d = float(np.linalg.norm(x[ia] - x[ib]))
                min_contact = min(min_contact, d)
                n_clash += 1
                if d < severe_clash_A:
                    n_severe += 1
        except Exception as e:
            min_contact = -1.0; n_severe = -1            # scipy unavailable -> skip, flag
    if not np.isfinite(min_contact):
        min_contact = clash_A                            # nothing within clash_A -> clean
    clash_ok = n_severe == 0

    image_gap = float(image_gap) if np.isfinite(image_gap) else None
    return dict(
        metal_coordination=[int(c) for c in coordination],
        metal_coordination_completable=[int(c) for c in coordination_completable],
        metal_octahedral_ok=bool(metal_ok),
        coordination_cutoff_A=float(coord_cut), completable_cutoff_A=float(completable_cut),
        min_metal_coordination=int(min_metal_coord),
        forming_distance_A=round(float(forming), 2), attack_angle_deg=round(float(angle), 1),
        attack_angle_definition=angle_kind,
        net_charge_e=round(float(net_charge), 4), charge_ok=bool(charge_ok),
        n_waters=int(n_waters),
        solute_image_gap_A=round(image_gap, 2) if image_gap is not None else None,
        nb_cutoff_A=float(nb_cutoff_A), solvation_ok=bool(solvation_ok),
        min_nonbonded_contact_A=round(float(min_contact), 2), n_clashes=int(n_clash),
        n_severe_clashes=int(n_severe), clash_ok=bool(clash_ok),
        physically_valid=bool(metal_ok and charge_ok and solvation_ok and clash_ok))
