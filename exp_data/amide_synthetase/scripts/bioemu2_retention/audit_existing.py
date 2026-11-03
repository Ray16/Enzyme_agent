"""Measure the original AS9 propagation without changing its saved outputs."""
import argparse
from pathlib import Path
import mdtraj as md
from bioemu2 import System, map_cif_atom_order
from core import frame_metrics, write_json, digest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    base=Path(__file__).resolve().parents[2]
    root=base/'results/bioemu2_as9_cofold_20261026'
    system=System.from_yaml(root/'system.yaml')
    start=map_cif_atom_order(system,root/'starting_cofold/topology.cif')
    frames=md.load_xtc(str(root/'ensemble/samples.xtc'),top=start.topology)
    result=dict(source=str(root),baseline=frame_metrics(start,start),
        endpoints=[frame_metrics(start,frames[i:i+1]) for i in range(frames.n_frames)],
        files=[dict(path=str(p),sha256=digest(p)) for p in [root/'system.yaml',
            root/'starting_cofold/topology.cif',root/'ensemble/samples.xtc',root/'mapping.json']],
        interpretation='Four independent 1 ns endpoints; not a continuous trajectory or residence-time measurement.')
    write_json(args.out,result)
    print(result,flush=True)


if __name__=='__main__':
    main()
