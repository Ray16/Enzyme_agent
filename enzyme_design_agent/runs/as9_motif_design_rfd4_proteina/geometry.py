"""Geometry from explicitly selected atoms in predicted protein-ligand structures."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def select_atoms(atoms, spec):
    """Select by chain, optional residue, and atom name(s) or element.

    Optional res_id must be supplied when a chain contains repeated atom names.
    """
    mask=np.asarray(atoms.chain_id)==str(spec['chain'])
    if 'res_id' in spec:mask &= np.asarray(atoms.res_id)==int(spec['res_id'])
    if 'atom' in spec:mask &= np.asarray(atoms.atom_name)==str(spec['atom'])
    if 'atoms' in spec:mask &= np.isin(atoms.atom_name,spec['atoms'])
    if 'element' in spec:mask &= np.char.upper(np.asarray(atoms.element).astype(str))==str(spec['element']).upper()
    if not ('atom' in spec or 'atoms' in spec or 'element' in spec):raise ValueError('Selection requires atom, atoms, or element')
    return np.flatnonzero(mask)

def evaluate(atoms,spec):
    center=select_atoms(atoms,spec['center']);axis=select_atoms(atoms,spec['angle_reference']);attack=select_atoms(atoms,spec['attack_candidates'])
    if len(center)!=1 or len(axis)!=1:raise ValueError('Center and angle reference must each select exactly one atom')
    expected=spec.get('expected_attack_candidates')
    if not len(attack) or (expected is not None and len(attack)!=expected):raise ValueError(f'Unexpected attack candidate count: {len(attack)}')
    coords=np.asarray(atoms.coord,float)
    if not np.isfinite(coords[np.r_[center,axis,attack]]).all():raise ValueError('Nonfinite reactive coordinates')
    center_xyz=coords[center[0]];dist=np.linalg.norm(coords[attack]-center_xyz,axis=1);j=int(np.argmin(dist));a=int(attack[j])
    u=coords[a]-center_xyz;v=coords[axis[0]]-center_xyz
    denom=np.linalg.norm(u)*np.linalg.norm(v)
    if denom==0:raise ValueError('Zero-length angle vector')
    return dict(attack_chain=str(atoms.chain_id[a]),attack_residue=int(atoms.res_id[a]),attack_atom=str(atoms.atom_name[a]),
                distance_A=float(dist[j]),angle_deg=float(np.degrees(np.arccos(np.clip(u@v/denom,-1,1)))))

def read_structure(path):
    from biotite.structure.io import pdbx
    cif=pdbx.CIFFile.read(str(path));return pdbx.get_structure(cif,model=1)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('structure',type=Path);p.add_argument('--geometry',type=Path,required=True);a=p.parse_args()
    spec=json.loads(a.geometry.read_text());print(json.dumps(evaluate(read_structure(a.structure),spec),indent=2))
if __name__=='__main__':main()
