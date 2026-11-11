"""Load coordinates against an authoritative topology without PDB bond inference."""
from pathlib import Path
import numpy as np


def load_system(prmtop,coordinates):
    import parmed as pmd
    path=Path(coordinates);parm=pmd.load_file(str(prmtop))
    if path.suffix.lower()=='.npz':
        with np.load(path) as data:
            parm.coordinates=np.asarray(data['positions_A'],float)
            if 'box_lengths_angles' in data:parm.box=np.asarray(data['box_lengths_angles'],float)
        if parm.box is not None and parm.pointers.get('IFBOX')==2 and np.allclose(parm.box[3:],109.47122,atol=.001):
            parm.box[3:]=np.degrees(np.arccos(-1/3))
    elif path.suffix.lower()=='.pdb':
        lines=path.read_text().splitlines()
        rows=[l for l in lines if l.startswith(('ATOM  ','HETATM'))]
        if len(rows)!=len(parm.atoms):raise ValueError('Coordinate/topology atom count mismatch')
        # The external topology supplies all bonds. OpenMM's >99999 atom PDB
        # serial encoding is not understood by ParmEd's CONECT parser.
        parm.coordinates=np.array([[float(l[30:38]),float(l[38:46]),float(l[46:54])] for l in rows])
        cell=next((l for l in lines if l.startswith('CRYST1')),None)
        if cell:
            parm.box=np.array([float(cell[a:b]) for a,b in [(6,15),(15,24),(24,33),(33,40),(40,47),(47,54)]])
            if parm.pointers.get('IFBOX')==2:parm.box[3:]=np.degrees(np.arccos(-1/3))
    else:parm=pmd.load_file(str(prmtop),str(coordinates))
    if parm.coordinates.shape!=(len(parm.atoms),3) or not np.isfinite(parm.coordinates).all():raise ValueError('Invalid coordinates')
    return parm


def reimage_molecules(parm,positions,anchor):
    """Translate complete bonded molecules to the closest image of an anchor.

    Required for nonperiodic embedding of a periodic frame. Does not infer bonds
    or repair internally broken molecules; topology and atom ordering are fixed.
    """
    x=np.asarray(positions,float).copy()
    if parm.box is None:return x
    from parmed.geometry import box_lengths_and_angles_to_vectors
    from parmed import unit
    box=np.array([v.value_in_unit(unit.angstrom) for v in box_lengths_and_angles_to_vectors(*parm.box)],float)
    inv=np.linalg.inv(box);reference=x[anchor].copy();graph=[[] for _ in parm.atoms]
    for b in parm.bonds:
        i,j=b.atom1.idx,b.atom2.idx;graph[i].append(j);graph[j].append(i)
    seen=np.zeros(len(x),bool)
    for first in range(len(x)):
        if seen[first]:continue
        seen[first]=True;queue=[first];ids=[]
        while queue:
            i=queue.pop();ids.append(i)
            for j in graph[i]:
                if not seen[j]:seen[j]=True;queue.append(j)
        heavy=[i for i in ids if parm.atoms[i].atomic_number>1] or ids
        fractional=(x[heavy]-reference)@inv;translations=np.round(fractional)
        distance=np.linalg.norm((fractional-translations)@box,axis=1)
        shift=translations[int(np.argmin(distance))]@box;x[ids]-=shift
    return x
