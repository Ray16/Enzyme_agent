"""Step-1 cofold preparation with explicit chemistry and C4 verification.

This stage builds inputs, not validation of a binding pose or protonation model.
"""
import argparse
import json
from pathlib import Path
from .jobs import atomic_json,sha256


def prepare(cofold,out,spec_path,polarizabilities,*,ph=8.0,overrides=None):
    from .prep_cofold import split_cofold,param_substrate,build_tleap
    from .spec import ReactionSpec
    spec=ReactionSpec.load(spec_path);ligands=dict(spec.ligands)
    substrate=spec.nucleophile.resname
    if substrate not in ligands:raise ValueError('Substrate formal charge missing')
    for file in [cofold,spec_path,polarizabilities]:
        if not Path(file).is_file():raise ValueError(f'Missing preparation input: {file}')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'inputs.json',dict(source=str(Path(cofold).resolve()),source_sha256=sha256(cofold),
        spec=str(Path(spec_path).resolve()),spec_sha256=sha256(spec_path),ph=ph,
        polarizability_overrides=overrides or {}))
    _,atp,metals,sub=split_cofold(cofold,out)
    param_substrate(sub,atp,out,substrate,ligands[substrate],ph=ph)
    _,top,coords=build_tleap(out,substrate,polarizabilities=polarizabilities,pol_overrides=overrides)
    result=dict(source=str(Path(cofold).resolve()),source_sha256=sha256(cofold),
      spec=str(Path(spec_path).resolve()),spec_sha256=sha256(spec_path),ph=ph,
      magnesium_count=len(metals),substrate_charge=ligands[substrate],
      prmtop=str(top.resolve()),coords=str(coords.resolve()),
      prmtop_sha256=sha256(top),coords_sha256=sha256(coords),
      status='parameterized_requires_protonation_assembly_equilibration_review',
      polarizability_overrides=overrides or {})
    atomic_json(out/'preparation.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cofold',required=True);p.add_argument('--out',required=True);p.add_argument('--spec',required=True)
    p.add_argument('--polarizabilities',required=True);p.add_argument('--overrides-json');p.add_argument('--ph',type=float,default=8)
    a=p.parse_args(argv);overrides=json.loads(Path(a.overrides_json).read_text()) if a.overrides_json else None
    print(json.dumps(prepare(a.cofold,a.out,a.spec,a.polarizabilities,ph=a.ph,overrides=overrides),indent=2))


if __name__=='__main__':main()
