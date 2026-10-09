"""Run the threading-asymmetry descriptor over the QC-passing holo cofold panel, in parallel.
Writes per-(enzyme,pose,site) rows and per-enzyme aggregates. Reads only structures + QC table.
"""
import sys, csv, json, os
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path("/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent")
HERE = ROOT / "exp_data/nylon_hydrolase/analysis/selectivity_proxy_20261009"
QC = ROOT / "exp_data/nylon_hydrolase/analysis/proxy_verification_20261008/qc/model_qc.csv"
MAX_POSES = 4
CLASH = float(sys.argv[1]) if len(sys.argv) > 1 else 1.4

sys.path.insert(0, str(HERE))
# extract_asymmetry.arm_lengths uses single-threaded (workers=1) KD-tree queries, so
# each worker process uses exactly one core. MAX_WORKERS caps our total core use on a
# shared node (courtesy: never oversubscribe).
import extract_asymmetry as E  # noqa
import thread_chain as tc  # noqa

MAX_WORKERS = int(os.environ.get("SEL_WORKERS", "8"))


def beta_sites(chains):
    """Return [(chain, nuc_xyz)] for each beta subunit (short chain), nuc = N-term Thr OG1."""
    out = []
    for ch, atoms in chains.items():
        res = {rs for (rs, rn, an, el, xyz) in atoms}
        if len(res) >= 150:
            continue  # alpha subunit
        first = min(res)
        og = [xyz for (rs, rn, an, el, xyz) in atoms
              if rs == first and an == "OG1" and rn == "THR"]
        if og:
            out.append((ch, og[0]))
    return out


def one_pose(args):
    enz, path = args
    try:
        chains = E.load_chains(ROOT / path)
        prot = E.prot_tuples(chains)
        rows = []
        for ch, nuc in beta_sites(chains):
            lf, lb = E.arm_lengths(prot, nuc, clash=CLASH)
            if lf + lb == 0:
                continue
            La, Ls = max(lf, lb), min(lf, lb)
            rows.append(dict(id=enz, path=path, site=ch, lf=lf, lb=lb,
                             L_long=La, L_short=Ls,
                             asymmetry=(La - Ls) / (La + Ls)))
        return rows
    except Exception as e:  # noqa
        return [dict(id=enz, path=path, site="ERR", error=str(e))]


def main():
    # cap internal threads so workers don't oversubscribe
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    qc = [r for r in csv.DictReader(open(QC)) if r["qc_pass"] == "True"]
    by_enz = {}
    for r in qc:
        by_enz.setdefault(r["id"], []).append(r["path"])
    jobs = []
    for enz, paths in by_enz.items():
        for p in sorted(paths)[:MAX_POSES]:
            jobs.append((enz, p))
    print(f"{len(by_enz)} enzymes, {len(jobs)} pose-jobs (<= {MAX_POSES} poses each), clash={CLASH}",
          flush=True)
    rows = []
    print(f"using {MAX_WORKERS} single-threaded workers (courtesy cap)", flush=True)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(one_pose, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            rows.extend(f.result())
            if n % 50 == 0:
                print(f"  {n}/{len(jobs)} done", flush=True)
    out = HERE / f"per_pose_clash{CLASH}.csv"
    keys = ["id", "path", "site", "lf", "lb", "L_long", "L_short", "asymmetry", "error"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    # aggregate per enzyme
    agg = {}
    for r in rows:
        if r.get("site") in (None, "ERR") or "asymmetry" not in r:
            continue
        agg.setdefault(r["id"], {"asym": [], "Lshort": []})
        agg[r["id"]]["asym"].append(r["asymmetry"])
        agg[r["id"]]["Lshort"].append(r["L_short"])
    aggout = HERE / f"per_enzyme_clash{CLASH}.csv"
    with open(aggout, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["id", "n_obs", "asymmetry", "distal_capacity"])
        for enz, d in sorted(agg.items()):
            w.writerow([enz, len(d["asym"]), np.mean(d["asym"]), np.mean(d["Lshort"])])
    print("wrote", out, "and", aggout, "| enzymes with data:", len(agg), flush=True)


if __name__ == "__main__":
    main()
