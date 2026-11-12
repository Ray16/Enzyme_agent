"""Electrostatically embedded DFT/MM barrier on ONE consistent environment.

Companion to fullsystem.py (MLIP). Here the QM region is treated with DFT
(wB97M-V) while the surrounding enzyme+solvent enters as fixed point charges in
the one-electron Hamiltonian (embedded.EmbeddedDFTMM). Two robustness guarantees
beyond the raw backend:

  * jobs.consistent_path_positions uses one fixed bulk environment and a checked
    common partition. Mobile MM atoms and their embedding field may still move;
    their energy contributions do not generally cancel.
  * A fixed whole-residue embedding shell (or the full cell) is selected once.
    Optional cap-charge deletion defines a separate, unvalidated boundary model.
  * jobs.validate_geometry gates every frame before SCF.

Reference-potential energy gaps, NOT validated activation free energies.
CLI: python -m enzyme_barrier.ee_barrier <path_dir> <out.json> [basis] [scf]
"""
import json
import time
from pathlib import Path
import numpy as np

EV2KCAL = 23.060548254


def full_cell_embedding(parm, xref, region, *, cap_A=None, embedding_radius=None):
    """Augment a region with electrostatic embedding + link-cap charge deletion,
    computed once on the (frozen) reference environment.

    `embedding_radius=None` retains every nonzero MM charge in the primary cell.
    A finite radius selects whole MM residues near any real QM atom. Neither
    neutrality nor fixed bulk proves barrier convergence: the reacting QM
    density and mobile MM coordinates change along the path.
    """
    from .region import redistribute_cap_charges
    if embedding_radius is not None and (not np.isfinite(embedding_radius) or embedding_radius <= 0):
        raise ValueError('Embedding radius must be positive and finite, or None for the full cell')
    charges = np.array([a.charge for a in parm.atoms], float)
    for k, v in region.get("mm_charge_updates", {}).items():
        charges[int(k)] = float(v)
    selected = set(region["indices"])
    if not np.allclose(charges[region['indices']], 0, atol=1e-12, rtol=0):
        raise ValueError('QM classical charges must be zero before embedding selection')
    if any(abs(charges[link['mm']]) > 1e-12 for link in region['links']):
        raise ValueError('Link-parent charges must be zero before embedding selection')
    nonqm = np.array([i for i in range(len(parm.atoms)) if i not in selected], dtype=int)
    charges, cap_deleted = redistribute_cap_charges(charges, xref, region, cap_A)
    residual = float(charges.sum()) + region['charge'] - sum(a.charge for a in parm.atoms)
    if abs(residual) > 0.005:
        raise ValueError('Embedding charge conservation failed')
    cand = nonqm
    if embedding_radius is not None:
        from scipy.spatial import cKDTree
        tree = cKDTree(xref[np.array(sorted(selected))])
        kept = []
        for residue in parm.residues:
            ids = [a.idx for a in residue.atoms if a.idx not in selected]
            if ids and tree.query(xref[ids])[0].min() <= embedding_radius:
                kept.extend(ids)
        cand = np.array(kept, dtype=int)
    emb = sorted(int(i) for i in cand if abs(charges[i]) > 1e-12)
    reg = dict(region)
    reg["embedding_indices"] = emb
    reg["embedding_charges"] = charges[emb].tolist()
    reg["embedding_radius_A"] = embedding_radius if embedding_radius is not None else 1e9
    reg["cap_charge_delete_A"] = cap_A
    reg["cap_deleted_mm_indices"] = cap_deleted
    reg["mm_charge_updates"] = {str(i): float(charges[i]) for i, a in enumerate(parm.atoms)
                                if abs(charges[i] - a.charge) > 1e-12}
    reg["net_mm_charge"] = float(charges.sum())
    reg["embedded_total_charge"] = float(sum(reg["embedding_charges"])) + reg["charge"]
    reg['charge_rounding_residual_e'] = residual
    reg['embedding_selection'] = 'full_cell' if embedding_radius is None else 'whole_residue_shell'
    return reg


def ee_dft_barrier(path_dir, out, *, xc="wb97m-v", basis="def2-svp", scf="soscf",
                   cap_A=None, embedding_radius=22.0, conv_tol=1e-9, max_cycle=250,
                   require_magnesium_c4=True, order=("R", "path_max", "TS", "P"), validate=True,
                   engine='gpu', platform=None, grid_response=True):
    from ase import Atoms
    from .jobs import consistent_path_positions, validate_geometry
    from .embedded import EmbeddedDFTMM

    if scf not in {'soscf','diis'}:
        raise ValueError('SCF must be soscf or diis')
    parm, region, ref, frames = consistent_path_positions(path_dir)
    reg = full_cell_embedding(parm, frames[ref], region, cap_A=cap_A,
                              embedding_radius=embedding_radius)
    reaction_atoms = region.get("reaction_atoms", ())
    N = len(parm.atoms)
    print(f"EE-DFT: nQM={len(reg['indices'])} QMcharge={reg['charge']} "
          f"n_emb={len(reg['embedding_indices'])} cap_deleted={len(reg['cap_deleted_mm_indices'])} "
          f"embedded_total_charge={reg['embedded_total_charge']:+.4f}e basis={basis} scf={scf}", flush=True)

    reports = {}
    for nm, x in frames.items():
        if validate:
            reports[nm] = validate_geometry(parm, x, reaction_atoms=reaction_atoms)

    numbers = [a.atomic_number for a in parm.atoms]
    calc = EmbeddedDFTMM(parm, reg, xc=xc, basis=basis, engine=engine, platform=platform,
                         grid_response=grid_response, require_magnesium_c4=require_magnesium_c4,
                         conv_tol=conv_tol, max_cycle=max_cycle,
                         soscf=(scf == "soscf"),
                         level_shift=(0.3 if scf == "diis" else 0.0),
                         damp=(0.3 if scf == "diis" else 0.0))
    atoms = Atoms(numbers=numbers, positions=frames[ref])
    atoms.calc = calc

    present = list(frames)  # every image, in path order (never a label subset)
    E, comp = {}, {}
    for nm in dict.fromkeys([ref, *present]):
        atoms.positions[:] = frames[nm]
        t = time.monotonic()
        E[nm] = float(atoms.get_potential_energy())
        comp[nm] = dict(high_embedded_eV=calc.last_components.get("high_embedded_eV"),
                        mm_remaining_eV=calc.last_components.get("mm_remaining_eV"))
        print(f"  {nm}: E={E[nm]:.6f} eV  dE={(E[nm]-E[ref])*EV2KCAL:+.1f} kcal "
              f"(QM_emb={comp[nm]['high_embedded_eV']:.2f}, MM={comp[nm]['mm_remaining_eV']:.2f})  "
              f"({time.monotonic()-t:.0f}s)", flush=True)

    top = max(E, key=lambda k: E[k]); last = present[-1]
    dEa = (E[top] - E[ref]) * EV2KCAL
    dErxn = (E[last] - E[ref]) * EV2KCAL
    res = dict(path_dir=str(path_dir), method=f"EE-DFT/MM ({xc}/{basis}, {reg['embedding_selection']})",
               n_atoms=N, reference=ref, sampled_path_max_label=top,
               product_label=last, ts_is_certified=False,
               n_qm=len(reg["indices"]), qm_charge=reg["charge"],
               n_embedding=len(reg["embedding_indices"]),
               embedding_radius_A=reg["embedding_radius_A"],
               embedded_total_charge_e=reg["embedded_total_charge"],
               cap_charge_delete_A=cap_A, basis=basis, xc=xc, scf=scf,
               engine=engine, mm_platform=calc.mm_platform, grid_response=grid_response,
               scf_conv_tol=conv_tol, scf_max_cycle=max_cycle,
               geometry_reports=reports, E_eV=E, components_eV=comp,
               dE_act_kcal=dEa, dE_rxn_kcal=dErxn,
               quantity="embedded_DFT_MM_energy_gaps_on_scan_geometries",
               validated_activation_free_energy=False,
               limitations=['Finite nonperiodic embedding; radius and QM-size convergence required',
                            'Mobile MM and QM-MM terms need not cancel between images',
                            'Scan geometries have no DFT stationarity or connectivity certificate',
                            'No thermal or ensemble free energy'],
               note="EE-DFT//scan geometries with fixed bulk and a common partition.")
    Path(out).write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nΔE‡ = {dEa:.1f} kcal/mol   ΔE_rxn = {dErxn:+.1f} kcal/mol  -> {out}", flush=True)
    return res


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="EE-DFT/MM barrier on a consistent environment")
    p.add_argument("path_dir"); p.add_argument("out")
    p.add_argument("basis", nargs="?", default="def2-svp")
    p.add_argument("scf", nargs="?", default="soscf", choices=['soscf','diis'])
    p.add_argument("--embedding-radius", type=lambda s: None if s.lower()=='full' else float(s), default=22.0,
                   help="whole-residue shell radius in A, or 'full' for the primary cell")
    p.add_argument('--cap-charge-delete-A', type=float, default=None,
                   help='optional alternative boundary model; requires separate validation')
    p.add_argument('--engine', choices=['gpu','cpu'], default='gpu')
    p.add_argument('--platform', choices=['CUDA','CPU','Reference'])
    p.add_argument("--no-validate", action="store_true")
    a = p.parse_args(argv)
    ee_dft_barrier(a.path_dir, a.out, basis=a.basis, scf=a.scf,
                   embedding_radius=a.embedding_radius, cap_A=a.cap_charge_delete_A,
                   engine=a.engine, platform=a.platform, validate=not a.no_validate)


if __name__ == "__main__":
    main()
