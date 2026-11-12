"""CLI: python -m enzyme_barrier <carve|scan|run> ...
Default barrier is the electronic DeltaE-dagger (fast scan); --dg adds qRRHO free energy."""
import argparse, sys
from pathlib import Path

def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    operators={'qmmm':'jobs','prepare':'prepare','analyze':'analysis','md':'md','path':'path','worker':'worker','status':'status','queue':'queue',
               'embedded-path':'ee_barrier','full-system':'fullsystem'}
    if argv and argv[0] in operators:
        import importlib
        module=importlib.import_module('.'+operators[argv[0]],__package__)
        return module.main(argv[1:])
    p = argparse.ArgumentParser(prog="enzyme_barrier",description="Reviewed chemical setups, embedded QM/MM pilots, and gated free energies. Legacy cluster scans remain exploratory.")
    sub = p.add_subparsers(dest="cmd", required=True)
    for operator,module_name in operators.items():
        sub.add_parser(operator,help=f"{module_name} operator; use {operator} --help for options")

    c = sub.add_parser("carve", help="build active-site cluster from a spec + enzyme")
    c.add_argument("--spec", required=True); c.add_argument("--prmtop", required=True)
    c.add_argument("--coords", required=True); c.add_argument("--id", required=True, dest="enzyme_id")
    c.add_argument("--out", required=True); c.add_argument("--r_lig", type=float); c.add_argument("--free_r", type=float)

    s = sub.add_parser("scan", help="relaxed-scan barrier from a carved cluster meta")
    s.add_argument("--meta", required=True); s.add_argument("--model", default="uma-s-1p2")
    s.add_argument("--full", action="store_true", help="dense grid (default: fast coarse)")
    s.add_argument("--dg", action="store_true", help="also qRRHO DeltaG (slow)")
    s.add_argument("--wt", type=float, help="WT reference DeltaE for DeltaDelta")
    s.add_argument("--concerted", action="store_true",
                   help="concerted antisymmetric coordinate (form & break together); for in-line phosphoryl transfer")

    r = sub.add_parser("run", help="carve + scan end-to-end")
    for a in ("--spec", "--prmtop", "--coords"): r.add_argument(a, required=True)
    r.add_argument("--id", required=True, dest="enzyme_id"); r.add_argument("--out", required=True)
    r.add_argument("--r_lig", type=float); r.add_argument("--free_r", type=float)
    r.add_argument("--model", default="uma-s-1p2"); r.add_argument("--full", action="store_true")
    r.add_argument("--dg", action="store_true"); r.add_argument("--wt", type=float)

    a = p.parse_args(argv)
    from .spec import ReactionSpec
    from .carve import carve
    if a.cmd == "carve":
        m = carve(ReactionSpec.load(a.spec), a.prmtop, a.coords, a.enzyme_id, a.out,
                  r_lig=a.r_lig, free_r=a.free_r)
        print(f"carved {a.enzyme_id}: {m['n_atoms']} atoms, q={m['net_charge']}, free={m['n_free']} "
              f"-> {Path(a.out)/('cluster_'+a.enzyme_id+'.meta.json')}")
    elif a.cmd == "scan":
        from .scan import scan_barrier
        scan_barrier(a.meta, model=a.model, fast=not a.full, dg=a.dg, wt=a.wt, concerted=a.concerted)
    elif a.cmd == "run":
        m = carve(ReactionSpec.load(a.spec), a.prmtop, a.coords, a.enzyme_id, a.out,
                  r_lig=a.r_lig, free_r=a.free_r)
        from .scan import scan_barrier
        meta = Path(a.out)/f"cluster_{a.enzyme_id}.meta.json"
        scan_barrier(meta, model=a.model, fast=not a.full, dg=a.dg, wt=a.wt)

if __name__ == "__main__":
    main()
