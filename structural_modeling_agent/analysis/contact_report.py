"""ONE entry point for substrate-contact analysis of an enzyme, across Boltz-2 cofold poses AND/OR MD.

Call `analyze_enzyme(...)` (or run a registry JSON) per enzyme; it:
  1. detects + aggregates contact FREQUENCY over the Boltz-pose ensemble (if given),
  2. detects + aggregates contact FREQUENCY over the MD trajectory/replicates (if given),
  3. renders the paired structure + frequency-bar figure using whatever is available.
Everything for an enzyme lands in one directory: <out_root>/<name>/.

Two conda envs are used under the hood (kept separate on purpose):
  - analysis  -> processivity-md env   (MDAnalysis)
  - rendering -> boltz-2 env + pymol    (matplotlib / PIL / pymol)
so this orchestrator just shells each stage to its env and can itself run under any python.

Registry JSON (list of enzymes), e.g.:
  [{"name": "GlnA3_Q9KZC7", "title": "GlnA3 (Q9KZC7) - glutarate step-1",
    "boltz_poses_json": "....json",            # list of cofold pose PDB paths  (optional)
    "representative_pose": "model_28.pdb",       # structure for the 3D panel     (optional)
    "md_top": "complex_1264.prmtop", "md_trajs": ["s17/production.dcd", ...],      # (optional)
    "hetero_sel": "resname ATP SUB MG", "mech": "adenylate", "stride": 3}]

CLI:
  python -m structural_modeling_agent.analysis.contact_report --registry enzymes.json --out-root <dir>
  python -m structural_modeling_agent.analysis.contact_report --name X --md-top T --md-traj d1 d2 \
      --representative-pose eq.pdb --hetero-sel "resname ATP SUB MG" --out-root <dir>
"""
from __future__ import annotations
import os, json, subprocess, argparse, threading

from ..env_config import MD_PYTHON, VIZ_PYTHON
from ..io_utils import SINGLE_THREAD_ENV

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MDPY = MD_PYTHON   # MDAnalysis
VIZPY = VIZ_PYTHON  # mpl + PIL
SC = "structural_modeling_agent.analysis.substrate_contacts"
RENDER = os.path.join(HERE, "render_contacts.py")


def _run(cmd):
    env = {**os.environ, **SINGLE_THREAD_ENV}   # single-threaded workers: courteous on a shared node
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{r.stderr[-1500:]}")
    return r.stdout


def analyze_enzyme(name, out_root, boltz_poses=None, representative_pose=None,
                   md_top=None, md_trajs=None, hetero_sel=None, mech=None,
                   reaction_center_sel=None, stride=3, title=None, render_sem=None):
    """Analyze one enzyme's substrate contacts (Boltz and/or MD) and render the figure. Returns paths."""
    out = os.path.join(out_root, name); os.makedirs(out, exist_ok=True)
    boltz_json = md_json = None
    common = []
    if mech: common += ["--mech", mech]
    if reaction_center_sel: common += ["--reaction-center-sel", reaction_center_sel]

    if boltz_poses:
        poses_json = os.path.join(out, f"{name}_boltz_poses.json")
        json.dump(list(boltz_poses), open(poses_json, "w"))
        boltz_json = os.path.join(out, f"{name}_boltz_freq.json")
        _run([MDPY, "-m", SC, "ensemble", "--poses-json", poses_json, "--json", boltz_json] + common)

    if md_top and md_trajs:
        md_json = os.path.join(out, f"{name}_md_freq.json")
        cmd = [MDPY, "-m", SC, "traj", "--top", md_top, "--traj", *md_trajs,
               "--stride", str(stride), "--json", md_json] + common
        if hetero_sel:
            cmd += ["--hetero-sel", hetero_sel]
        _run(cmd)

    # representative structure for the 3D panel: explicit > best Boltz pose > (MD needs one)
    rep = representative_pose or (boltz_poses[0] if boltz_poses else None)
    if rep is None:
        raise ValueError(f"{name}: need representative_pose (or boltz_poses) for the structure panel")
    primary = boltz_json or md_json          # solid bars
    overlay = md_json if (boltz_json and md_json) else None   # hatched MD bars only if both exist
    fig = os.path.join(out, f"{name}_contact_frequency.png")
    cmd = [VIZPY, RENDER, "--pose", rep, "--freq-json", primary,
           "--title", title or name, "--out", fig]
    if overlay:
        cmd += ["--md-freq-json", overlay]
    if render_sem is not None:
        with render_sem:
            print(_run(cmd).strip())
    else:
        print(_run(cmd).strip())
    return {"name": name, "dir": out, "boltz_freq": boltz_json, "md_freq": md_json, "figure": fig}


def _one(e, out_root, render_sem=None):
    try:
        r = analyze_enzyme(
            e["name"], out_root, boltz_poses=_load_poses(e), representative_pose=e.get("representative_pose"),
            md_top=e.get("md_top"), md_trajs=e.get("md_trajs"), hetero_sel=e.get("hetero_sel"),
            mech=e.get("mech"), reaction_center_sel=e.get("reaction_center_sel"),
            stride=e.get("stride", 3), title=e.get("title"), render_sem=render_sem)
        print(f"[contact_report] OK {e['name']}", flush=True)
        return r
    except Exception as ex:
        print(f"[contact_report] FAILED {e['name']}: {str(ex)[:200]}", flush=True)
        return {"name": e["name"], "error": str(ex)}


def run_registry(registry_path, out_root, workers=8, render_workers=2):
    """Analyze every enzyme in the registry across `workers` threads while bounding concurrent
    headless PyMOL renders via `render_workers` to avoid cluster ray-trace stalls."""
    from concurrent.futures import ThreadPoolExecutor
    with open(registry_path) as fh:
        reg = json.load(fh)
    render_sem = threading.Semaphore(max(1, int(render_workers)))
    print(f"[contact_report] {len(reg)} enzymes, {workers} parallel workers (render cap {render_workers})", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda e: _one(e, out_root, render_sem=render_sem), reg))
    with open(os.path.join(out_root, "_summary.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    done = [r for r in results if "figure" in r]
    print(f"[contact_report] {len(done)}/{len(reg)} figures -> {out_root}")
    return results


def _load_poses(e):
    if e.get("boltz_poses"):
        return e["boltz_poses"]
    if e.get("boltz_poses_json"):
        return json.load(open(e["boltz_poses_json"]))
    return None


def _cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--name"); ap.add_argument("--title")
    ap.add_argument("--boltz-poses-json"); ap.add_argument("--representative-pose")
    ap.add_argument("--md-top"); ap.add_argument("--md-traj", nargs="+")
    ap.add_argument("--hetero-sel"); ap.add_argument("--mech"); ap.add_argument("--stride", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out_root, exist_ok=True)
    if a.registry:
        run_registry(a.registry, a.out_root, workers=a.workers)
    else:
        poses = json.load(open(a.boltz_poses_json)) if a.boltz_poses_json else None
        r = analyze_enzyme(a.name, a.out_root, boltz_poses=poses, representative_pose=a.representative_pose,
                           md_top=a.md_top, md_trajs=a.md_traj, hetero_sel=a.hetero_sel, mech=a.mech,
                           stride=a.stride, title=a.title)
        print(json.dumps(r, indent=1))


if __name__ == "__main__":
    _cli()
