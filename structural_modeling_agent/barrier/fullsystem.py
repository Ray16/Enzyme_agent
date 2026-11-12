"""Full-system MLIP barrier: UMA over every atom, one consistent environment.

An alternative to the embedded DFT/MM backend (embedded.py) for barriers. There
is no QM/MM partition, no link atoms, no point-charge embedding, hence no
boundary or MM-remainder artifact: a single UMA potential (uma-s-1p2, the OMol25
= wB97M-V/def2-TZVPD target level) is evaluated over the whole solvated system.

Model conventions:
  * Each point keeps its own active-site geometry and shares the reference frame's
    distant bulk. Mobile-environment terms still change along the path; fixed bulk
    does not guarantee exact energy cancellation or long-range convergence.
  * Every frame is structurally validated (jobs.validate_geometry) before any GPU
    time: integer charge, intact covalent bonds (except the reacting bond), no
    steric clashes.

Caveat: this model does not provide explicit long-range electrostatic embedding.
It is a learned reference-potential energy, not a DFT evaluation or a validated
activation free energy. Water trimming needs separate sensitivity checks.

CLI: python -m enzyme_barrier.fullsystem <path_dir> <out.json> [model]
"""
import json
import time
from pathlib import Path
import numpy as np

def _uma():
    from . import mlip
    return mlip


WATER_RESNAMES = {'WAT', 'HOH', 'TIP3', 'TIP3P', 'TIP4P', 'TIP5P', 'SOL', 'T3P', 'SPC'}


def _water_shell_keep(parm, xref, qm, radius):
    """Indices to keep: every non-water atom (protein/ligand/metal/ion intact)
    plus whole water molecules within `radius` of the QM region. Dropping whole
    neutral waters cuts no covalent bonds. Energy-gap convergence with radius
    must be checked; model receptive fields do not prove exact cancellation."""
    qxyz = xref[qm]
    keep = []
    for r in parm.residues:
        idx = [a.idx for a in r.atoms]
        if r.name in WATER_RESNAMES:
            d = np.linalg.norm(xref[idx][:, None, :] - qxyz[None, :, :], axis=2).min()
            if d > radius:
                continue
        keep.extend(idx)
    return np.array(sorted(keep))


def full_system_barrier(path_dir, out, *, model="uma-s-1p2", inference_settings="default",
                        order=("R", "path_max", "TS", "P"), validate=True, water_radius=None):
    """Build consistent frames, validate each, run full-system UMA single points.

    If `water_radius` is set, bulk water beyond that distance from the QM region
    is dropped before the energy evaluation (whole molecules only; protein,
    ligands, metals and ions are always kept). This is only to fit large systems
    in GPU memory. Trimming changes the evaluated model and requires explicit
    energy-gap sensitivity checks. Returns the result dict (also written to `out`).
    """
    from ase import Atoms
    from .jobs import consistent_path_positions, validate_geometry
    uma = _uma()
    EV2KCAL = uma.KCAL_MOL_PER_EV

    parm, region, ref, frames = consistent_path_positions(path_dir)
    qm = np.array(sorted(set(region["indices"])))
    N = len(parm.atoms)
    transplant = set(region["consistent_env_transplant"])
    frozen = [i for i in range(N) if i not in transplant]
    reaction_atoms = region.get("reaction_atoms", ())

    reports = {}
    xref = frames[ref]
    for nm, x in frames.items():
        moved = int((np.linalg.norm(x - xref, axis=1)[frozen] > 1e-9).sum()) if frozen else 0
        if moved:
            raise ValueError(f"Frame {nm!r}: {moved} frozen-bulk atoms moved -- "
                             "environment is not consistent; aborting.")
        if validate:
            reports[nm] = validate_geometry(parm, x, reaction_atoms=reaction_atoms)
        print(f"  {nm}: QM max move={np.linalg.norm(x-xref,axis=1)[qm].max():.3f} A  "
              f"frozen-bulk moved=0  geometry_ok={reports.get(nm,{}).get('passed','skipped')}",
              flush=True)

    keep = _water_shell_keep(parm, xref, qm, water_radius) if water_radius else np.arange(N)
    if water_radius:
        # Never delete atoms whose coordinates are part of the local response.
        required={parm.atoms[i].residue.idx for i in transplant}
        keep=np.array(sorted(set(keep)|{a.idx for r in parm.residues if r.idx in required for a in r.atoms}))
    qtot = int(round(sum(parm.atoms[i].charge for i in keep)))
    numbers = [parm.atoms[i].atomic_number for i in keep]
    if water_radius:
        print(f"water-shell trim: {N} -> {len(keep)} atoms (radius {water_radius} A), "
              f"net charge {qtot}", flush=True)
    frames = {nm: x[keep] for nm, x in frames.items()}
    print(f"full system: N={N} total_charge={qtot} spin=1 model={model} ({inference_settings})",
          flush=True)
    calc = uma.get_calculator("omol", model, inference_settings=inference_settings)

    # Evaluate EVERY image in the path's native order (never a label subset --
    # skipping intermediates can miss the true maximum). consistent_path_positions
    # preserves path.json order; reference is first/'R', product is last.
    present = list(frames)
    E = {}
    for nm in dict.fromkeys([ref,*present]):
        a = Atoms(numbers=numbers, positions=frames[nm])
        a.info.update(charge=qtot, spin=1)
        a.calc = calc
        t = time.monotonic()
        E[nm] = float(a.get_potential_energy())
        print(f"  {nm}: E={E[nm]:.6f} eV  dE={ (E[nm]-E[ref])*EV2KCAL:+.1f} kcal  ({time.monotonic()-t:.1f}s)", flush=True)

    top = max(E, key=lambda k: E[k])
    last = present[-1]
    dEa = (E[top] - E[ref]) * EV2KCAL
    dErxn = (E[last] - E[ref]) * EV2KCAL
    res = dict(path_dir=str(path_dir), method=f"full-system MLIP ({model}, omol)",
               level="UMA OMol25 learned approximation to its DFT training target", n_atoms=len(keep),n_full_system_atoms=N,
               inference_settings=uma.PROFILE,water_radius_A=water_radius,
               reference=ref, sampled_path_max_label=top, product_label=last,
               ts_is_certified=False,
               n_transplant=len(transplant), total_charge=qtot,
               geometry_reports=reports, E_eV=E,
               dE_act_kcal=dEa, dE_rxn_kcal=dErxn,
               quantity="reference_potential_energy_gaps_on_scan_geometries",
               validated_activation_free_energy=False,
               note="Full-system MLIP on scan geometries with one fixed bulk environment; "
                    "no certified saddle or activation free energy. Water trimming, if used, "
                    "requires energy-gap sensitivity checks.")
    Path(out).write_text(json.dumps(res, indent=2) + "\n")
    if dEa is not None:
        print(f"\nΔE‡ = {dEa:.1f} kcal/mol   ΔE_rxn = {dErxn:+.1f} kcal/mol  -> {out}", flush=True)
    return res


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Full-system MLIP barrier on a consistent environment")
    p.add_argument("path_dir")
    p.add_argument("out")
    p.add_argument("model", nargs="?", default="uma-s-1p2")
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--water-radius", type=float, default=None,
                   help="drop whole water molecules beyond this A from the QM region (memory only)")
    a = p.parse_args(argv)
    full_system_barrier(a.path_dir, a.out, model=a.model, validate=not a.no_validate,
                        water_radius=a.water_radius)


if __name__ == "__main__":
    main()
