#!/usr/bin/env python3
"""Poll the Polaris regio-repro PBS job via the Globus Compute endpoint until the
DONE flag appears (or the job leaves the queue). Prints per-system exit status."""
import time
from globus_compute_sdk import Executor

EP = "4d01f9b2-5d45-46eb-8bc8-840733c4253f"
BASE = "/grand/projects/FRAME-IDP/rzhu/Project/enzyme_agent/regioselectivity_repro"
JOB = "7649614.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov"


def status(base, job):
    import subprocess, os, glob, re
    q = subprocess.run(["/opt/pbs/bin/qstat", "-x", job.split(".")[0]],
                       capture_output=True, text=True)
    state = "?"
    for ln in q.stdout.splitlines():
        if job.split(".")[0] in ln:
            state = ln.split()[-2]
    done = os.path.exists(f"{base}/DONE")
    exits = {}
    for lg in sorted(glob.glob(f"{base}/logs/*.log")):
        m = re.findall(r"EXIT rc=(\d+)", open(lg).read())
        exits[os.path.basename(lg)[:-4]] = m[-1] if m else "running"
    nmodels = {}
    for d in sorted(glob.glob(f"{base}/results/*")):
        nmodels[os.path.basename(d)] = len(glob.glob(f"{d}/boltz_results_*/predictions/*/*_model_*.pdb"))
    return {"state": state, "done": done, "exits": exits, "nmodels": nmodels}


with Executor(endpoint_id=EP) as ex:
    for _ in range(60):                       # up to ~60 min
        r = ex.submit(status, BASE, JOB).result(timeout=120)
        print(f"[{time.strftime('%H:%M:%S')}] state={r['state']} done={r['done']} "
              f"exits={r['exits']} models={r['nmodels']}", flush=True)
        if r["done"] or r["state"] in ("F", "?") and any(r["exits"].values()):
            print("FINISHED", flush=True); break
        time.sleep(90)
