"""Stage 3: relaxed-scan electronic barrier (DeltaE-dagger) -- the per-enzyme metric,
DEFAULT. Reads a carved cluster (meta + xyz), scans the forming bond with UMA, locates
the TS as the scan maximum, and gates on the mechanism (right bonds form/break,
associative in-line TS). Optional dg=True adds the qRRHO free energy at the scan-max TS
(slow). System-agnostic: reaction atoms come from the cluster meta (i_nuc/i_elec/i_leav).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from ase.io import read
from ase.constraints import FixAtoms, FixInternals
from ase.optimize import LBFGS
from . import mlip as uma_helper
from .scan_geometry import reaction_groups, set_group_distance

KCAL = uma_helper.KCAL_MOL_PER_EV


def _relax_held(atoms, q, s, calc, scan_bond, dval, held_bonds, frozen, fmax, steps,
                moving_group=None,deadline=None):
    """Relax with the scanned bond fixed at dval AND any `held_bonds` [(i,j,val),...] fixed;
    `frozen` atoms via FixAtoms. Returns (energy_eV, relaxed_atoms)."""
    a = atoms
    i, j = scan_bond
    set_group_distance(a,i,j,dval,[i] if moving_group is None else moving_group,frozen)
    a.info["charge"] = int(q); a.info["spin"] = int(s); a.calc = calc
    bonds = [[float(dval), [i, j]]] + [[float(v), [p, r]] for (p, r, v) in held_bonds]
    cons = [FixInternals(bonds=bonds)]
    if frozen:
        cons.append(FixAtoms(indices=frozen))
    a.set_constraint(cons)
    optimizer=LBFGS(a,logfile=None)
    if deadline is not None:
        def budget():
            if time.monotonic()>deadline:raise TimeoutError('Cluster path wall budget reached')
        optimizer.attach(budget,interval=1)
    converged = optimizer.run(fmax=fmax, steps=steps)
    e = a.get_potential_energy()
    if not converged or not np.isfinite(e):
        raise RuntimeError('Cluster scan image did not converge to a finite energy')
    a.set_constraint(FixAtoms(indices=frozen) if frozen else [])
    return e, a.copy()


def _two_leg(atoms, q, s, model, i_nuc, i_elec, i_leav, d0, fmax, steps, traj_path, frozen,
             form_end=1.6, break_end=3.0, moving_groups=None,images=None,deadline=None):
    """Form-then-break reaction path to a UNIFORM product (PPi fully departed), so barriers
    are comparable across enzymes regardless of concerted vs stepwise timing.
    Leg 1: drive forming bond d(nuc-elec) NAC->~1.6 (breaking bond free).
    Leg 2: hold forming bond at ~1.6, drive breaking bond d(elec-leav) out to ~3.0.
    Returns dists1, d2_1, E1, G1, (bond_form_hold), d2_grid, E2, G2."""
    from ase.io import Trajectory
    calc = uma_helper.get_calculator("omol", model)
    if moving_groups is None:raise ValueError('Topology-defined moving groups are required')
    traj = Trajectory(traj_path, "w") if traj_path else None
    d2 = lambda g: float(np.linalg.norm(g.positions[i_elec] - g.positions[i_leav]))
    a = atoms.copy(); a.info.update(charge=q, spin=s)
    # Leg 1: forming bond
    nform=max(2,images//2) if images is not None else max(2,int(np.ceil(abs(d0-form_end)/.2))+1)
    dists1 = np.linspace(d0, form_end,nform).tolist()
    E1, G1 = [], []
    for d in dists1:
        try:e, g = _relax_held(a, q, s, calc, (i_nuc, i_elec), d, [], frozen, fmax, steps,moving_groups[0],deadline)
        except TimeoutError:
            if traj_path:np.save(str(traj_path)+'.partial.npy',a.positions)
            raise
        E1.append(e); G1.append(g); a = g.copy()
        if traj: traj.write(g)
        print(f"  L1 d(nuc-elec)={d:.2f} d(elec-leav)={d2(g):.2f} E={(e-E1[0])*KCAL:+.2f}", flush=True)
    # Leg 2: hold forming bond at its leg-1 end value, drive breaking bond out
    bf = dists1[-1]                      # ~1.6
    r2_start = d2(G1[-1])
    nbreak=images-nform+1 if images is not None else max(2,int(np.ceil(abs(break_end-r2_start)/.2))+1)
    d2_grid = np.linspace(r2_start,max(r2_start,break_end),nbreak).tolist()
    if images is not None:d2_grid=d2_grid[1:]
    a = G1[-1].copy(); E2, G2 = [], []
    for d in d2_grid:
        try:e, g = _relax_held(a, q, s, calc, (i_elec, i_leav), d, [(i_nuc, i_elec, bf)], frozen, fmax, steps,moving_groups[1],deadline)
        except TimeoutError:
            if traj_path:np.save(str(traj_path)+'.partial.npy',a.positions)
            raise
        E2.append(e); G2.append(g); a = g.copy()
        if traj: traj.write(g)
        print(f"  L2 d(elec-leav)={d:.2f} d(nuc-elec)={float(np.linalg.norm(g.positions[i_nuc]-g.positions[i_elec])):.2f} "
              f"E={(e-E1[0])*KCAL:+.2f}", flush=True)
    if traj: traj.close()
    return dists1, E1, G1, d2_grid, E2, G2


def _concerted(atoms, q, s, model, i_nuc, i_elec, i_leav, d0_form, d0_break, fmax, steps,
               traj_path, frozen, form_end=1.65, break_end=3.0, n=13,moving_groups=None,deadline=None):
    """CONCERTED antisymmetric path: drive the forming bond d(nuc-elec) DOWN and the breaking
    bond d(elec-leav) UP simultaneously along a single progress variable lam (reactant->product),
    both held each step, everything else relaxed. Avoids the pentacoordinate-phosphorane trap the
    form-then-break two-leg produces for in-line phosphoryl transfer. Returns Df, Db, E, G."""
    from ase.io import Trajectory
    calc = uma_helper.get_calculator("omol", model)
    if moving_groups is None:raise ValueError('Topology-defined moving groups are required')
    traj = Trajectory(traj_path, "w") if traj_path else None
    a = atoms.copy(); a.info.update(charge=q, spin=s)
    Df, Db, E, G = [], [], [], []
    for L in np.linspace(0.0, 1.0, n):
        if deadline is not None and time.monotonic()>deadline:raise TimeoutError('Cluster path wall budget reached')
        df = float(d0_form + (form_end - d0_form) * L)
        db = float(d0_break + (break_end - d0_break) * L)
        # physically move atoms to BOTH target distances (as the two-leg scan does), else a hard
        # FixInternals constraint on an unmoved geometry does nothing and the energy stays flat
        set_group_distance(a,i_nuc,i_elec,df,moving_groups[0],frozen)
        set_group_distance(a,i_elec,i_leav,db,moving_groups[1],frozen)
        a.info.update(charge=q, spin=s); a.calc = calc
        cons = [FixInternals(bonds=[[df, [i_nuc, i_elec]], [db, [i_elec, i_leav]]])]
        if frozen:
            cons.append(FixAtoms(indices=frozen))
        a.set_constraint(cons)
        optimizer=LBFGS(a,logfile=None)
        if deadline is not None:
            def budget():
                if traj_path:np.save(str(traj_path)+'.partial.npy',a.positions)
                if time.monotonic()>deadline:raise TimeoutError('Cluster path wall budget reached')
            optimizer.attach(budget,interval=1)
        converged = optimizer.run(fmax=fmax, steps=steps)
        e = a.get_potential_energy()
        if not converged or not np.isfinite(e):
            raise RuntimeError('Cluster scan image did not converge to a finite energy')
        a.set_constraint(FixAtoms(indices=frozen) if frozen else [])
        g = a.copy(); Df.append(df); Db.append(db); E.append(e); G.append(g)
        adf = float(np.linalg.norm(g.positions[i_nuc] - g.positions[i_elec]))
        adb = float(np.linalg.norm(g.positions[i_elec] - g.positions[i_leav]))
        if traj: traj.write(g)
        print(f"  C lam={L:.2f} d(nuc-elec)={adf:.2f} d(elec-leav)={adb:.2f} E={(e-E[0])*KCAL:+.2f}", flush=True)
        a = g.copy()
    if traj: traj.close()
    return Df, Db, E, G


def scan_barrier(meta_path: str | Path, model: str = "uma-s-1p2", fast: bool = True,
                 dg: bool = False, wt: float | None = None, T: float = 303.15,
                 concerted: bool = False,images=None,max_steps=None,wall_seconds=None) -> dict:
    if images is not None and (type(images) is not int or images<3):raise ValueError('At least three scan images required')
    if max_steps is not None and (type(max_steps) is not int or max_steps<1):raise ValueError('Positive optimizer step budget required')
    if wall_seconds is not None and (not np.isfinite(wall_seconds) or wall_seconds<=0):raise ValueError('Positive wall budget required')
    meta_path = Path(meta_path)
    meta = json.load(open(meta_path))
    base = meta_path.parent
    tag = meta["enzyme_id"]
    prefix = str(base / f"barrier_{tag}")
    atoms = read(str(base / f"cluster_{tag}.xyz"))
    q, s, frozen = int(meta["net_charge"]), int(meta.get("multiplicity", 1)), list(meta["frozen"])
    i, j, k = meta["i_nuc"], meta["i_elec"], meta["i_leav"]
    atoms.info.update(charge=q, spin=s)
    if 'cluster_bonds' not in meta:
        raise ValueError('Cluster lacks authoritative bonds; rebuild it with carve before scanning')
    moving_groups=reaction_groups(len(atoms),meta['cluster_bonds'],[i,j,k])

    d0 = float(np.linalg.norm(atoms.positions[i] - atoms.positions[j]))
    definition=meta.get('path_definition')
    if definition is None:
        if atoms.numbers[j]!=15:
            raise ValueError('Non-phosphate cluster scans require path_definition; rebuild the cluster from its spec')
        definition=dict(forming_distance_A=1.65,breaking_distance_A=3.0,
                        reactant_forming_A=d0,reactant_breaking_A=float(atoms.get_distance(j,k)))
    print(f"[{tag}] {meta['reaction_id']}: {len(atoms)} atoms ({meta['n_free']} free) q={q}; "
          f"d(nuc-elec)={d0:.2f}", flush=True)

    fmax, steps = (0.10, 80) if fast else (0.08, 120)
    if max_steps is not None:steps=max_steps
    deadline=time.monotonic()+wall_seconds if wall_seconds is not None else None
    t0 = time.time()
    coordinate = "concerted_antisym" if concerted else "form_then_break"
    dne = lambda g: float(np.linalg.norm(g.positions[i] - g.positions[j]))
    d2_at = lambda g: float(np.linalg.norm(g.positions[j] - g.positions[k]))
    if concerted:
        # single concerted path: forming bond down & breaking bond up together (no phosphorane trap)
        d0b = float(np.linalg.norm(atoms.positions[j] - atoms.positions[k]))
        Df, Db, E, G = _concerted(atoms, q, s, model, i, j, k, d0, d0b, fmax, steps,
                                  f"{prefix}_scan.traj", frozen,
                                  form_end=definition['forming_distance_A'],break_end=definition['breaking_distance_A'],moving_groups=moving_groups,
                                  n=images or 13,deadline=deadline)
        geoms = list(G); energies = np.asarray(E)
        dne_all = [dne(g) for g in geoms]; d2_all = [d2_at(g) for g in geoms]
        n1 = 0
        i_react = 0; i_prod = len(geoms) - 1
        i_ts = i_react + int(np.argmax(energies[i_react:i_prod + 1]))
    else:
        # two-leg form-then-break path to a UNIFORM product (PPi departed) -> comparable barriers
        dists1, E1, G1, d2grid, E2, G2 = _two_leg(atoms, q, s, model, i, j, k, d0, fmax, steps,
                                                  f"{prefix}_scan.traj", frozen,
                                                  form_end=definition['forming_distance_A'],break_end=definition['breaking_distance_A'],moving_groups=moving_groups,
                                                  images=images,deadline=deadline)
        geoms = list(G1) + list(G2); energies = np.asarray(list(E1) + list(E2))
        dne_all = [dne(g) for g in geoms]; d2_all = [d2_at(g) for g in geoms]
        n1 = len(G1)
        # reactant = min energy on leg1 where forming bond still long (>=2.4); product = end of leg2
        r_cand = [n for n in range(n1) if dists1[n] >
                  (definition['reactant_forming_A']+definition['forming_distance_A'])/2]
        i_react = min(r_cand, key=lambda n: energies[n])
        i_prod = len(geoms) - 1
        i_ts = i_react + int(np.argmax(energies[i_react:i_prod + 1]))
    xi = [d2_all[n] - dne_all[n] for n in range(len(geoms))]       # monotone reactant(-)->product(+)
    barrier = float((energies[i_ts] - energies[i_react]) * KCAL)
    rxn = float((energies[i_prod] - energies[i_react]) * KCAL)

    # gate: product has forming bond made AND leaving group fully departed;
    # associative = at TS both bonds partial / not yet a free phosphorane blow-up
    form_cut=(definition['reactant_forming_A']+definition['forming_distance_A'])/2
    break_cut=(definition['reactant_breaking_A']+definition['breaking_distance_A'])/2
    forms = dne_all[i_prod] < form_cut
    breaks = d2_all[i_prod] > break_cut
    ts_s = dne_all[i_ts] + d2_all[i_ts]
    assoc = dne_all[i_ts] < 2.8 and d2_all[i_ts] < 2.8
    ts_leg = None if concerted else (1 if i_ts < n1 else 2)
    gate_ok = bool(forms and breaks)

    e0 = float(energies[i_react])
    curve = [dict(xi=round(xi[n], 3), d_nuc_elec=round(dne_all[n], 3), d_elec_leav=round(d2_all[n], 3),
                  leg=(0 if concerted else (1 if n < n1 else 2)), dE_kcal=round(float((energies[n] - e0) * KCAL), 2))
             for n in range(len(geoms))]
    out = dict(reaction_id=meta["reaction_id"], enzyme_id=tag, model=model, T=T, coordinate=coordinate,
               inference_settings=uma_helper.PROFILE,
               requested_images=images,optimizer_max_steps=steps,wall_budget_seconds=wall_seconds,
               path_definition=definition,
               quantity="constrained_cluster_electronic_energy_gap",
               all_optimizations_converged=True, transition_state_certified=False,
               validation_status="exploratory", validated_activation_free_energy=False,
               limitations=["isolated frozen-boundary cluster", "imposed reaction path",
                            "no transition-state connectivity validation", "no ensemble averaging"],
               n_atoms=len(atoms), n_free=meta["n_free"], charge=q,
               scan_reactant_d=round(dne_all[i_react], 2), scan_ts_d=round(dne_all[i_ts], 2),
               scan_ts_d_break=round(d2_all[i_ts], 2), ts_leg=ts_leg, scan_product_dbreak=round(d2_all[i_prod], 2),
               i_reactant=i_react, i_ts=i_ts, i_product=i_prod,
               dE_act_kcal=round(barrier, 1), dE_rxn_kcal=round(rxn, 1),
               mechanism_gate=dict(associative=bool(assoc), forming_bond_made=bool(forms),
                                   breaking_bond_broken=bool(breaks), ts_s_A=round(ts_s, 2),
                                   ts_leg=ts_leg, ok=gate_ok),
               scan_curve=curve, scan_wall_s=round(time.time() - t0), dG_act_kcal=None)
    if wt is not None:
        out["wt_dE_act_kcal"] = wt; out["ddE_vs_wt_kcal"] = round(barrier - wt, 1)
    json.dump(out, open(f"{prefix}.json", "w"), indent=2)
    print(f"[{tag}] DeltaE-dagger = {barrier:+.1f} kcal/mol ({coordinate}"
          + (f" leg{ts_leg}" if ts_leg else "") + f") ; rxn = {rxn:+.1f} ; "
          f"gate {'PASS' if gate_ok else 'FAIL'}"
          + (f" ; ddE vs WT = {barrier-wt:+.1f}" if wt is not None else ""), flush=True)

    if dg:
        try:
            out["dG_act_kcal"] = _qrrho_dg(geoms, i_react, i_ts, i_prod, q, s, model, frozen, T, prefix)
        except (ValueError,RuntimeError) as exc:
            out.update(thermochemistry_status='rejected',thermochemistry_error=str(exc))
            Path(f'{prefix}.json').write_text(json.dumps(out,indent=2))
            raise
        out["quantity"] = "cluster_scan_gap_with_qrrho_correction"
        json.dump(out, open(f"{prefix}.json", "w"), indent=2)
    return out


def _qrrho_dg(geoms, i_react, i_ts, i_prod, q, s, model, frozen, T, prefix):
    """Refuse thermochemistry without stationary, correctly indexed seed points."""
    from .thermochemistry import partial_frequencies,qrrho_correction
    if i_ts in {i_react,i_prod}:raise ValueError('qRRHO requires an interior saddle candidate')
    frozen=np.asarray(frozen,int)
    if len(frozen)<3 or np.linalg.matrix_rank(geoms[i_react].positions[frozen]-geoms[i_react].positions[frozen[0]])<2:
        raise ValueError('Frozen-cluster qRRHO requires a noncollinear pinned boundary')
    free=sorted(set(range(len(geoms[0])))-set(frozen));rows=[]
    calc=uma_helper.get_calculator('omol',model)
    for label,index,n_imag in [('R',i_react,0),('scan_max',i_ts,1),('P',i_prod,0)]:
        a=geoms[index].copy();a.info.update(charge=q,spin=s)
        a.set_constraint(FixAtoms(indices=frozen));a.calc=calc
        force=float(np.linalg.norm(a.get_forces(apply_constraint=False)[free],axis=1).max())
        if force>.08:raise ValueError(f'{label} is not stationary on the thermochemistry potential')
        frequencies,diagnostics=partial_frequencies(a,free)
        row=qrrho_correction(frequencies,T)
        if row['significant_imaginary_modes']!=n_imag:
            raise ValueError(f'{label} has the wrong Hessian index for qRRHO')
        row.update(label=label,hessian=diagnostics,energy_eV=float(a.get_potential_energy()),max_free_force_eV_A=force)
        row['corrected_energy_eV']=row['energy_eV']+row['correction_eV'];rows.append(row)
    Path(f'{prefix}_qrrho.json').write_text(json.dumps(dict(points=rows,inference_settings=uma_helper.PROFILE,
        transition_state_certified=False,validated_activation_free_energy=False),indent=2))
    gR,gT,gP=(row['corrected_energy_eV'] for row in rows)
    return round(float((gT - gR) * KCAL), 1)
