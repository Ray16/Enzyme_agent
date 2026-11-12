"""Explicit, audited Li--Merz C4 installation. Never infer C4 from a filename."""
from pathlib import Path
import json


def install_c4(prmtop, output, polarizabilities, *, water_model="TIP3P", overrides=None):
    import parmed as pmd
    from parmed.tools.actions import add12_6_4
    from parmed.tools.argumentlist import ArgumentList
    source, dest = Path(prmtop), Path(output)
    if source.resolve() == dest.resolve():
        raise ValueError("Write a new topology; preserve the original")
    parm = pmd.load_file(str(source))
    params = {}
    for line in Path(polarizabilities).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            name, value, *_ = line.split(); params[name] = float(value)
    params.update(overrides or {})
    missing = set(parm.parm_data['AMBER_ATOM_TYPE']) - params.keys()
    if missing:
        raise ValueError(f"Missing atom-type polarizabilities: {sorted(missing)}")
    metals = [a for a in parm.atoms if a.atomic_number == 12]
    if not metals:
        raise ValueError("No magnesium in topology")
    pol = dest.with_suffix('.polarizabilities.dat')
    pol.write_text(''.join(f'{k} {v:.8f}\n' for k, v in sorted(params.items())))
    # An explicit mask and ArgumentList avoid interpreting the mask as a parm name.
    mask = '@' + ','.join(str(a.idx + 1) for a in metals)
    action = add12_6_4(parm, ArgumentList(f'{mask} watermodel {water_model} polfile {pol}'))
    action.execute()
    c4 = parm.parm_data.get('LENNARD_JONES_CCOEF', [])
    if not any(float(v) != 0 for v in c4):
        raise ValueError("C4 installation produced no nonzero coefficients")
    parm.save(str(dest), overwrite=False)
    reread = pmd.load_file(str(dest))
    if list(reread.parm_data['LENNARD_JONES_CCOEF']) != list(c4):
        # Amber's decimal serialization can round the final digits.
        import numpy as np
        if not np.allclose(reread.parm_data['LENNARD_JONES_CCOEF'], c4, rtol=1e-7):
            raise ValueError("C4 serialization mismatch")
    result = dict(source=str(source.resolve()), output=str(dest.resolve()),
                  water_model=water_model, magnesium_count=len(metals),
                  nonzero_c4=sum(float(x) != 0 for x in c4),
                  polarizability_overrides=overrides or {})
    dest.with_suffix('.parameters.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def require_c4(parm):
    if any(a.atomic_number == 12 for a in parm.atoms):
        if not any(parm.parm_data.get('LENNARD_JONES_CCOEF', [])):
            raise ValueError("Mg topology lacks nonzero C4 coefficients; run install-c4 first")
