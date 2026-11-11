"""Brief OpenMM minimization of a prepped (protonated+hydrated) cofold system -> a
NAC-conditioned frame for carving. Starting from a Boltz cofold pose, the substrate must
be driven into a NEAR-ATTACK conformation (NAC) before the barrier scan, so every enzyme's
barrier is measured from the SAME poised reactant (nucleophile ~3.3 A from the electrophile)
rather than from whatever loose distance the pose/minimize happened to settle at -- the
latter makes cross-enzyme barriers incomparable and, if the substrate drifts out entirely
(e.g. AS9 cofold O(nuc)-Palpha 3.6 A raw -> 5.3 A unrestrained min), breaks the scan.

Recipe: hold the active-site FRAME (ATP + catalytic metal) on positional restraints, let
protein/water/substrate relax, and add a harmonic distance restraint pulling the nucleophile
O -> electrophile to the NAC target. The result is a consistent NAC reactant; the scan then
drives the forming bond NAC -> TS. Run in an OpenMM env (processivity-md)."""
from pathlib import Path
from .spec import ReactionSpec

_METALS = {"MG", "MN", "ZN", "CA", "FE", "NI", "CO"}

# Bondi van der Waals radii (A) -- NAC distance = vdW(nuc)+vdW(elec) = contact, poised, no bond.
_VDW = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 15: 1.80, 16: 1.80, 17: 1.75,
        5: 1.92, 14: 2.10, 35: 1.85, 53: 1.98}


def _angle(xyz, i, j, k):
    """Angle (rad) i-j-k with j as vertex."""
    import numpy as np
    a = xyz[i] - xyz[j]; b = xyz[k] - xyz[j]
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.arccos(max(-1.0, min(1.0, c))))


def _resolve_nac_atoms(parm, spec):
    """Resolve (i_nuc, i_elec, i_leav) with parmed only (no ASE dep): electrophile + leaving by
    resname+atom; nucleophile = the atom of the substrate residue nearest the electrophile
    (geometric AUTO -- restricted to the nucleophile element if the spec names one, else any
    heavy atom; defaults to O for a carboxylate/hydroxyl nucleophile)."""
    import numpy as np
    xyz = parm.coordinates

    def by_name(ref):
        return next(a.idx for res in parm.residues if res.name == ref.resname
                    for a in res.atoms if a.name == ref.atom)
    i_elec = by_name(spec.electrophile)
    i_leav = by_name(spec.leaving)
    sub = spec.nucleophile.resname
    # AUTO nucleophile element: O by default (carboxylate); honour an explicit element hint
    if spec.nucleophile.atom.upper() not in {'AUTO','AUTO_NEAREST_O'}:
        return by_name(spec.nucleophile),i_elec,i_leav
    nuc_Z = 8
    cand = [a.idx for res in parm.residues if res.name == sub
            for a in res.atoms if a.atomic_number == nuc_Z]
    if not cand:
        raise KeyError(f"no element-{nuc_Z} atom in substrate residue {sub} for NAC nucleophile")
    i_nuc = int(min(cand, key=lambda i: np.linalg.norm(xyz[i] - xyz[i_elec])))
    return i_nuc, i_elec, i_leav


def minimize(prmtop, inpcrd, out_dir, steps=5000, temp=303.15, restrain_k=100.0,
             spec_path=None, nac_target=None, nac_k=None):
    """Minimize with (a) positional restraints on the ATP+metal frame and (b) -- when a
    ReactionSpec is given -- a harmonic NAC restraint conditioning the substrate to a
    near-attack geometry: distance d(nuc-elec) -> nac_target (A, default = vdW(nuc)+vdW(elec),
    i.e. van der Waals contact) and, if spec.nac_angle is set, the attack angle
    nuc-elec-leaving -> that angle. Without a spec it falls back to holding ALL ligand heavy
    atoms (legacy behaviour)."""
    import math
    import parmed as pmd, openmm as mm, openmm.app as app
    from openmm import unit
    import numpy as np
    p = pmd.load_file(str(prmtop), str(inpcrd))
    from .parameters import require_c4
    require_c4(p)

    spec = ReactionSpec.load(spec_path) if spec_path else None
    # the substrate must be FREE so it can move into NAC; hold only the ATP+metal frame.
    sub_resname = spec.nucleophile.resname if spec else None
    held = set()
    if spec:
        held = {r for r, _ in spec.ligands if r != sub_resname} | set(spec.metals)
    _AA = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","HID","HIE","HIP","ILE",
           "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","CYX","ASH","GLH","LYN"}
    _FREE = _AA | {"WAT","HOH","TIP3","T3P"} | {"Na+","Cl-","NA","CL","K+","K","NA+","CL-"}

    sysm = p.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer,
                          constraints=app.HBonds)

    # (a) positional restraint on the active-site frame (ATP + metal; or all ligands if no spec)
    restr = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restr.addGlobalParameter("k", restrain_k * 4.184 * 100.0 * unit.kilojoule_per_mole / unit.nanometer**2)
    for nm in ("x0", "y0", "z0"):
        restr.addPerParticleParameter(nm)
    pos_nm = p.positions.value_in_unit(unit.nanometer)

    def is_frame(resn):
        if spec:
            return resn in held
        return resn not in _FREE          # legacy: hold all non-free (incl. substrate)

    n_restr = 0
    for atom in p.atoms:
        if is_frame(atom.residue.name) and atom.atomic_number > 1:
            x0, y0, z0 = pos_nm[atom.idx]
            restr.addParticle(atom.idx, [x0, y0, z0])
            n_restr += 1
    sysm.addForce(restr)

    # (b) NAC conditioning: distance d(nuc-elec) -> vdW contact, and optional attack angle
    nac_msg = ""
    if spec:
        i_nuc, i_elec, i_leav = _resolve_nac_atoms(p, spec)
        # target distance: spec/arg override, else vdW(nuc)+vdW(elec)
        tgt = nac_target if nac_target is not None else spec.nac_target
        if tgt is None:
            znuc, zelec = p.atoms[i_nuc].atomic_number, p.atoms[i_elec].atomic_number
            tgt = _VDW.get(znuc, 1.6) + _VDW.get(zelec, 1.8)
        kd = (nac_k if nac_k is not None else spec.nac_k)
        d0 = float(np.linalg.norm(p.coordinates[i_nuc] - p.coordinates[i_elec]))
        nacf = mm.CustomBondForce("0.5*k*(r-r0)^2")
        nacf.addPerBondParameter("k"); nacf.addPerBondParameter("r0")
        nacf.addBond(i_nuc, i_elec, [kd * 4.184 * 100.0, tgt * 0.1])   # kcal/mol/A^2->kJ/mol/nm^2; A->nm
        sysm.addForce(nacf)
        nac_msg = (f"; NAC d(nuc={i_nuc},elec={i_elec}) {d0:.2f} -> {tgt:.2f} A (k={kd})")
        # optional attack-angle restraint nuc-elec-leaving
        if spec.nac_angle is not None:
            a0 = float(np.degrees(_angle(p.coordinates, i_nuc, i_elec, i_leav)))
            angf = mm.CustomAngleForce("0.5*k*(theta-theta0)^2")
            angf.addPerAngleParameter("k"); angf.addPerAngleParameter("theta0")
            angf.addAngle(i_nuc, i_elec, i_leav, [spec.nac_k_angle * 4.184, math.radians(spec.nac_angle)])
            sysm.addForce(angf)
            nac_msg += f"; angle(nuc-elec-leav={i_leav}) {a0:.0f} -> {spec.nac_angle:.0f} deg"

    print(f"[minimize] held {n_restr} frame atoms{nac_msg}", flush=True)
    integ = mm.LangevinMiddleIntegrator(temp * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    try:
        plat = mm.Platform.getPlatformByName("CUDA"); props = {"Precision": "mixed"}
    except Exception:
        plat, props = mm.Platform.getPlatformByName("CPU"), {}
    sim = app.Simulation(p.topology, sysm, integ, plat, props)
    sim.context.setPositions(p.positions)
    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    sim.minimizeEnergy(maxIterations=steps)
    st = sim.context.getState(getPositions=True, getEnergy=True)
    e1 = st.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    # report the achieved NAC distance
    dmsg = ""
    if spec:
        fin = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        dfin = float(np.linalg.norm(fin[i_nuc] - fin[i_elec]))
        dmsg = f" ; achieved d(nuc-elec)={dfin:.2f} A"
        if spec.nac_angle is not None:
            dmsg += f", angle={np.degrees(_angle(fin, i_nuc, i_elec, i_leav)):.0f} deg"
    out = Path(out_dir) / "minimized.pdb"
    with open(out, "w") as f:
        app.PDBFile.writeFile(p.topology, st.getPositions(), f)
    print(f"[minimize] E {e0:.0f} -> {e1:.0f} kcal/mol{dmsg} ; wrote {out}", flush=True)
    return out
