"""Small deterministic validation systems, independent of pytest."""
import numpy as np


def water_pair():
    import parmed as pmd
    from parmed.topologyobjects import Atom,AtomType,Bond,BondType,Angle,AngleType
    p=pmd.Structure();ot=AtomType('OW',1,16,8);ot.set_lj_params(0.152,1.7682)
    ht=AtomType('HW',2,1,1);ht.set_lj_params(0,0)
    for r in range(2):
        for nm,z,q,m,t in [('O',8,-0.834,16,ot),('H1',1,0.417,1,ht),('H2',1,0.417,1,ht)]:
            a=Atom(name=nm,type=t.name,atomic_number=z,charge=q,mass=m);a.atom_type=t;p.add_atom(a,'WAT',r+1)
        base=3*r;bt=BondType(553,0.9572);p.bond_types.append(bt)
        for j in (1,2):p.bonds.append(Bond(p.atoms[base],p.atoms[base+j],type=bt))
        at=AngleType(100,104.52);p.angle_types.append(at)
        p.angles.append(Angle(p.atoms[base+1],p.atoms[base],p.atoms[base+2],type=at))
    p.coordinates=np.array([[0,0,0],[0.9572,0,0],[-0.24,0.927,0],[0,0,3.2],[0.9572,0,3.2],[-0.24,0.927,3.2]])
    return p
