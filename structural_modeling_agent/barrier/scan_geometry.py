"""Topology-defined rigid initializations for exploratory cluster paths."""
import numpy as np


def reaction_groups(natoms,bonds,reaction_atoms):
    """Resolve substrate and leaving fragments after deleting the breaking bond."""
    i,j,k=reaction_atoms
    graph=[[] for _ in range(natoms)]
    for a,b in bonds:
        if {a,b}=={j,k}:continue
        graph[a].append(b);graph[b].append(a)
    def component(start):
        seen={start};stack=[start]
        while stack:
            for neighbor in graph[stack.pop()]:
                if neighbor not in seen:
                    seen.add(neighbor);stack.append(neighbor)
        return sorted(seen)
    forming=component(i);leaving=component(k)
    if j in forming or j in leaving or set(forming)&set(leaving):
        raise ValueError('Reaction fragments are not separable under the declared bond exchange; review the mechanism')
    return forming,leaving


def set_group_distance(atoms,i,j,target,moving_group, frozen=()):
    """Move a declared fragment only; never infer bonds or move frozen atoms."""
    group=set(moving_group)
    if (i in group)==(j in group):
        raise ValueError('Exactly one distance endpoint must belong to the moving group')
    if group&set(frozen):
        raise ValueError('Reactive fragment contains frozen atoms; enlarge the free region or review the path')
    moving,anchor=(i,j) if i in group else (j,i)
    vector=atoms.positions[moving]-atoms.positions[anchor]
    length=np.linalg.norm(vector)
    if length<1e-10 or not np.isfinite(target) or target<=0:
        raise ValueError('Invalid distance initialization')
    # Direct assignment bypasses constraints from the preceding image; only the
    # specified mobile fragment is translated, preserving all its internal bonds.
    atoms.positions[sorted(group)]+=(target/length-1)*vector
