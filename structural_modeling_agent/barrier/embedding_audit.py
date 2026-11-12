"""Export the actual EE atoms/charges and audit the constructed MM Hamiltonian."""
import argparse
import csv
import itertools
from pathlib import Path
import numpy as np
from .jobs import atomic_json,sha256,load_snapshot
from .region import capped_coordinates


def audit_snapshot(snapshot,out):
    import openmm as mm
    from openmm import unit
    from .embedded import build_mm_system
    p,r,a,manifest=load_snapshot(snapshot);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    system=build_mm_system(p,r);qm=set(r['indices']);emb=r['embedding_indices'];checks={}
    nb=next(f for f in system.getForces() if isinstance(f,mm.NonbondedForce))
    charges=np.array([nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) for i in range(len(a))])
    original=sum(at.charge for at in p.atoms)
    checks['unique_disjoint_QM_and_embedding']=len(qm)==len(r['indices']) and len(set(emb))==len(emb) and not qm.intersection(emb)
    checks['MM_QM_particle_charges_zero']=bool(np.all(charges[r['indices']]==0))
    checks['embedding_matches_remaining_MM_charges']=bool(np.allclose(charges[emb],r['embedding_charges'],atol=1e-12,rtol=0))
    residual=float(charges.sum()+r['charge']-original)
    checks['total_charge_conserved_within_library_rounding']=abs(residual)<=.005
    checks['declared_integer_QM_charge']=r['charge']==sum(e['charge'] for e in r['entries'])
    electrons=sum(p.atoms[i].atomic_number for i in qm)+len(r['links'])-r['charge']
    checks['closed_shell_electron_parity']=electrons%2==0 and r.get('multiplicity',1)==1
    expected_cuts=set()
    for bond in p.bonds:
        i,j=bond.atom1.idx,bond.atom2.idx
        if (i in qm)!=(j in qm):expected_cuts.add(tuple(sorted((i,j))))
    actual_cuts={tuple(sorted((link['qm'],link['mm']))) for link in r['links']}
    checks['every_covalent_boundary_has_one_link']=actual_cuts==expected_cuts and len(actual_cuts)==len(r['links'])
    checks['boundary_charge_zero']=all(charges[link['mm']]==0 for link in r['links'])
    checks['cuts_are_CA_CB']=all(p.atoms[l['qm']].name=='CB' and p.atoms[l['mm']].name=='CA' for l in r['links'])
    exception_pairs=set();bad_qm_coulomb=0;bad_qm_lj=0
    for k in range(nb.getNumExceptions()):
        i,j,q,s,e=nb.getExceptionParameters(k);exception_pairs.add(tuple(sorted((i,j))))
        if (i in qm or j in qm) and q.value_in_unit(unit.elementary_charge**2)!=0:bad_qm_coulomb+=1
        if i in qm and j in qm and e.value_in_unit(unit.kilojoule_per_mole)!=0:bad_qm_lj+=1
    pairs=set(itertools.combinations(sorted(qm),2))
    checks['QM_internal_LJ_pairs_excluded']=pairs<=exception_pairs and bad_qm_lj==0
    checks['QM_classical_Coulomb_exceptions_zero']=bad_qm_coulomb==0
    checks['LJ_switch_enabled']=nb.getUseSwitchingFunction()
    checks['C4_QM_internal_pairs_excluded']=True
    checks['C4_switch_enabled']=True
    checks['QM_internal_bonded_terms_removed']=True
    c4_count=0
    for force in system.getForces():
        if isinstance(force,mm.CustomNonbondedForce):
            c4_count+=1
            checks['C4_switch_enabled'] &= force.getUseSwitchingFunction()
            exclusions={tuple(sorted(force.getExclusionParticles(k))) for k in range(force.getNumExclusions())}
            checks['C4_QM_internal_pairs_excluded'] &= pairs<=exclusions
        if isinstance(force,mm.HarmonicBondForce):
            parameters=[force.getBondParameters(k) for k in range(force.getNumBonds())];n=2
        elif isinstance(force,mm.HarmonicAngleForce):
            parameters=[force.getAngleParameters(k) for k in range(force.getNumAngles())];n=3
        elif isinstance(force,mm.PeriodicTorsionForce):
            parameters=[force.getTorsionParameters(k) for k in range(force.getNumTorsions())];n=4
        else:continue
        checks['QM_internal_bonded_terms_removed'] &= all(params[-1]._value==0 for params in parameters if set(params[:n])<=qm)
    metals=r.get('metal_indices',[at.idx for at in p.atoms if at.atomic_number==12]);shells=[]
    checks['C4_force_present_for_Mg']=not any(p.atoms[i].atomic_number==12 for i in metals) or c4_count>0
    donors=np.array([at.idx for at in p.atoms if at.atomic_number in (7,8,16)])
    for metal in metals:
        selected=[int(i) for i in donors[np.linalg.norm(a.positions[donors]-a.positions[metal],axis=1)<r.get('metal_shell_A',2.7)]]
        shells.append(dict(metal_index_zero_based=metal,donors_zero_based=selected,all_in_QM=all(i in qm for i in selected)))
    checks['all_Mg_and_first_shell_donors_in_QM']=all(m in qm for m in metals) and all(s['all_in_QM'] for s in shells)
    checks['all_declared_metals_and_first_shell_donors_in_QM']=checks['all_Mg_and_first_shell_donors_in_QM']
    qpos=capped_coordinates(a.positions,r);closest=(float('inf'),None,None)
    for start in range(0,len(emb),1000):
        ids=emb[start:start+1000];dist=np.linalg.norm(a.positions[ids,None,:]-qpos[None,:,:],axis=2)
        i,j=np.unravel_index(dist.argmin(),dist.shape)
        if dist[i,j]<closest[0]:closest=(float(dist[i,j]),ids[i],int(j))
    checks['no_point_charge_at_QM_or_link_position']=closest[0]>.1
    with (out/'qm_atoms.csv').open('w') as file:
        writer=csv.writer(file);writer.writerow(['index_zero_based','amber_index_one_based','residue_zero_based','resname','atom','element','x_A','y_A','z_A'])
        for i in r['indices']:
            at=p.atoms[i];writer.writerow([i,i+1,at.residue.idx,at.residue.name,at.name,at.element_name,*a.positions[i]])
    with (out/'embedding_charges.csv').open('w') as file:
        writer=csv.writer(file);writer.writerow(['index_zero_based','residue_zero_based','resname','atom','charge_e','x_A','y_A','z_A'])
        for i in emb:
            at=p.atoms[i];writer.writerow([i,at.residue.idx,at.residue.name,at.name,charges[i],*a.positions[i]])
    atomic_json(out/'links.json',dict(policy='fixed initial CA-CB interpolation ratio; cap at 1.09 A in snapshot geometry',links=r['links']))
    result=dict(status='structural_invariants_passed_requires_scientific_review' if all(checks.values()) else 'embedding_audit_failed',
        passed=all(checks.values()),checks=checks,snapshot=str(Path(snapshot).resolve()),snapshot_sha256=sha256(Path(snapshot)/'input.json'),
        input_sha256=manifest['input_sha256'],n_real_QM=len(qm),n_links=len(r['links']),n_embedding_charges=len(emb),
        QM_charge=r['charge'],QM_electrons_with_links=electrons,MM_charge_e=float(charges.sum()),charge_rounding_residual_e=residual,
        embedding_charge_sum_e=float(charges[emb].sum()),magnesium_first_shells=shells,entries=r['entries'],
        embedding_policy=dict(cutoff_A=r['embedding_radius_A'],selection='fixed union of whole MM residues within cutoff of any real QM atom at snapshot creation',
            charge_units='elementary charge',coordinate_units='angstrom',polarization='QM density responds to fixed MM charges; no mutual MM polarization',
            periodic=False,MM_cutoff_A=12,MM_Coulomb='OpenMM CutoffNonPeriodic reaction field',
            MM_LJ_C4_switch_A=nb.getSwitchingDistance().value_in_unit(unit.angstrom),
            MM_reaction_field_dielectric=nb.getReactionFieldDielectric(),mobile_radius_A=r['mobile_radius_A']),
        nearest_point_charge_to_QM_or_cap=dict(distance_A=closest[0],MM_index_zero_based=closest[1],QM_or_cap_local_index=closest[2]),
        validated_activation_free_energy=False,
        pending=['real-enzyme DFT force finite differences','QM and embedding radius sensitivity','boundary charge/link policy sensitivity',
            'SCF solution stability and basis convergence','match native periodic and finite DFT Hamiltonians before correction'])
    atomic_json(out/'audit.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--snapshot',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();result=audit_snapshot(args.snapshot,args.out)
    print(result['status'],result['checks'],flush=True)
    if not result['passed']:raise SystemExit(1)
