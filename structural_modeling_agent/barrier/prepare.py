"""Step-1 cofold preparation with explicit chemistry, C4 verification, and a protonation audit.

This stage builds inputs; it does not validate a binding pose. It DOES now audit the protonation
model: tleap/pdb4amber assign every His -> HIE, every Asp/Glu deprotonated and every Lys +1 by
default, and a wrong charge state is structurally invisible (the MD runs fine and looks normal while
the active-site electrostatics are wrong). `protonation_audit` runs PROPKA and classifies each
discrepancy against the ligand environment, so metal/anion-coupled false positives are separated
from genuine errors. Active-site errors escalate the status rather than silently passing.
"""
import argparse
import json
from pathlib import Path
from .jobs import atomic_json,sha256


def _protonation_audit(out,ph,substrate,top,coords):
    """Best-effort protonation audit. Never aborts a build: a missing PROPKA must not destroy a
    finished parameterization, but the failure is recorded so the gate cannot mistake it for a pass."""
    try:
        from .protonation_audit import audit
        return audit(out,ph=ph,substrate_resn=substrate,coords=str(coords))
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        return {'verdict':'protonation_audit_FAILED','error':f'{type(exc).__name__}: {exc}'}


def prepare(cofold,out,spec_path,polarizabilities,*,ph=8.0,overrides=None,audit_protonation=True,
            protonation_overrides=None,salt_molarity=0.15):
    from .prep_cofold import split_cofold,param_substrate,build_tleap,validate_prepared,apply_protonation_overrides
    from .spec import ReactionSpec
    spec=ReactionSpec.load(spec_path);ligands=dict(spec.ligands)
    substrate=spec.nucleophile.resname
    if substrate not in ligands:raise ValueError('Substrate formal charge missing')
    for file in [cofold,spec_path,polarizabilities]:
        if not Path(file).is_file():raise ValueError(f'Missing preparation input: {file}')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'inputs.json',dict(source=str(Path(cofold).resolve()),source_sha256=sha256(cofold),
        spec=str(Path(spec_path).resolve()),spec_sha256=sha256(spec_path),ph=ph,
        polarizability_overrides=overrides or {},protonation_overrides=protonation_overrides or {}))
    _,atp,metals,sub=split_cofold(cofold,out)
    applied_prot=apply_protonation_overrides(out/'protein_clean.pdb',protonation_overrides)
    param_substrate(sub,atp,out,substrate,ligands[substrate],ph=ph)
    _,top,coords=build_tleap(out,substrate,polarizabilities=polarizabilities,pol_overrides=overrides,salt_molarity=salt_molarity)
    check=validate_prepared(top,coords,spec_path)
    prot=_protonation_audit(out,ph,substrate,top,coords) if audit_protonation else None
    if not check['physically_valid']:
        status='parameterized_PHYSICAL_CHECK_FAILED_review'
    elif prot and prot.get('verdict')=='protonation_REVIEW_REQUIRED':
        status='parameterized_physically_valid_PROTONATION_REVIEW_REQUIRED'
    elif prot and prot.get('verdict')=='protonation_audit_FAILED':
        status='parameterized_physically_valid_PROTONATION_UNVERIFIED'
    else:
        status='parameterized_physically_valid_requires_equilibration_review'
    result=dict(source=str(Path(cofold).resolve()),source_sha256=sha256(cofold),
      spec=str(Path(spec_path).resolve()),spec_sha256=sha256(spec_path),ph=ph,
      magnesium_count=len(metals),substrate_charge=ligands[substrate],
      prmtop=str(top.resolve()),coords=str(coords.resolve()),
      prmtop_sha256=sha256(top),coords_sha256=sha256(coords),
      status=status,validation=check,protonation_audit=prot,
      protonation_overrides_applied=[{'resid':r,'from':o,'to':n} for r,o,n in applied_prot],
      salt=json.loads((out/'salt.json').read_text()) if (out/'salt.json').is_file() else None,
      polarizability_overrides=overrides or {})
    atomic_json(out/'preparation.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cofold',required=True);p.add_argument('--out',required=True);p.add_argument('--spec',required=True)
    p.add_argument('--polarizabilities',required=True);p.add_argument('--overrides-json');p.add_argument('--ph',type=float,default=8)
    p.add_argument('--no-protonation-audit',action='store_true',
                   help='skip the PROPKA protonation audit (NOT recommended: it is the only check on '
                        'tleap default charge states, which are structurally invisible when wrong)')
    p.add_argument('--protonation-overrides-json',
                   help='JSON mapping resid (protein_clean/PROPKA numbering) -> Amber resname, e.g. '
                        '{"265":"HIP","455":"GLH"}; forces protonation states before tleap')
    p.add_argument('--salt-molarity',type=float,default=0.15,
                   help='physiological NaCl added on top of neutralization (default 0.15 M; 0 = neutralize only)')
    a=p.parse_args(argv);overrides=json.loads(Path(a.overrides_json).read_text()) if a.overrides_json else None
    prot_ov=json.loads(Path(a.protonation_overrides_json).read_text()) if a.protonation_overrides_json else None
    print(json.dumps(prepare(a.cofold,a.out,a.spec,a.polarizabilities,ph=a.ph,overrides=overrides,
                             audit_protonation=not a.no_protonation_audit,protonation_overrides=prot_ov,
                             salt_molarity=a.salt_molarity),indent=2))


if __name__=='__main__':main()
