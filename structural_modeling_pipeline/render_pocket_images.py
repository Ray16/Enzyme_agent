#!/usr/bin/env python3
"""
Batch-render pocket visualization images in parallel.

Renders both apo and cofolded structure scripts into annotated PNGs
with title bars showing enzyme name, pocket volume, hydrophobicity, and organism.

Output: all_structures/pocket_images/
          ├── amide_synthetases_candidates/
          │     ├── {enzyme}_{uid}_apo_pocket.png
          │     ├── {enzyme}_{uid}_apo_overlay.png
          │     └── {enzyme}_{uid}_cofold.png
          └── amide_hydrolase_candidates/
                └── ...
"""

import csv
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BASE = Path(__file__).resolve().parent
OUT_BASE = BASE / "all_structures" / "pocket_images"
CSV_PATH = BASE / "analysis_results" / "comprehensive_enzyme_table.csv"

WIDTH = 2400
HEIGHT = 1800
MAX_WORKERS = min(16, os.cpu_count() or 4)


def load_csv_data() -> dict:
    data = {}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            uid = r["uniprot_id"].strip()
            data[uid] = {
                "enzyme_name": r.get("enzyme_name", ""),
                "apo_volume": r.get("apo_pocket_volume_A3", ""),
                "apo_hydro_pct": r.get("apo_frac_hydrophobic", ""),
                "holo_volume": r.get("holo_pocket_volume_A3", ""),
                "holo_hydro_pct": r.get("holo_frac_hydrophobic", ""),
                "volume_change_pct": r.get("volume_change_pct", ""),
                "organism": r.get("organism", ""),
            }
    return data


def extract_uid(filename: str) -> str:
    return filename.replace(".pdb", "").replace("_pocket_analysis", "").rsplit("_", 1)[-1]


def add_title_bar(raw_png: Path, final_png: Path, title: str, subtitle: str):
    img = mpimg.imread(str(raw_png))
    fig_w = 10
    fig_h = fig_w * img.shape[0] / img.shape[1] + 0.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(img, aspect="equal")
    ax.axis("off")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98, va="top")
    fig.text(0.5, 0.95, subtitle, fontsize=11, ha="center", va="top", color="gray")
    fig.subplots_adjust(top=0.92, bottom=0.01, left=0.01, right=0.99)
    fig.savefig(str(final_png), dpi=200, bbox_inches="tight", pad_inches=0.15,
                facecolor="white", edgecolor="none")
    plt.close(fig)


def render_job(job: dict) -> tuple[bool, str]:
    """Render a single job (one PML → one or two PNGs)."""
    pml_path = Path(job["pml"]).resolve()
    out_dir = Path(job["out_dir"]).resolve()
    meta = job["meta"]
    job_type = job["type"]  # "apo" or "cofold"

    stem = pml_path.stem.replace("_pocket_analysis", "")
    uid = extract_uid(pml_path.name)
    enzyme = meta.get("enzyme_name", uid)
    organism = meta.get("organism", "")

    if job_type == "apo":
        vol = meta.get("apo_volume", "")
        hydro = meta.get("apo_hydro_pct", "")
        vol_str = f"{vol} A\u00b3" if vol else "N/A"
        hydro_str = hydro if hydro else "N/A"
        subtitle = f"Apo pocket: {vol_str}  |  Hydrophobic: {hydro_str}  |  {organism}"
    else:
        holo_vol = meta.get("holo_volume", "")
        holo_hydro = meta.get("holo_hydro_pct", "")
        vol_change = meta.get("volume_change_pct", "")
        holo_vol_str = f"{holo_vol} A\u00b3" if holo_vol else "N/A"
        holo_hydro_str = holo_hydro if holo_hydro else "N/A"
        change_str = f"  |  \u0394vol: {vol_change}%" if vol_change else ""
        subtitle = f"Holo pocket: {holo_vol_str}  |  Hydrophobic: {holo_hydro_str}{change_str}  |  {organism}"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            parts = []

            if job_type == "apo":
                raw_apo = Path(tmpdir) / "apo.png"
                raw_overlay = Path(tmpdir) / "overlay.png"
                cmds = "; ".join([
                    "scene 001_apo_pocket, recall",
                    f"draw {WIDTH}, {HEIGHT}; png {raw_apo}",
                    "scene 002_overlay, recall",
                    f"draw {WIDTH}, {HEIGHT}; png {raw_overlay}",
                ])
                subprocess.run(
                    ["pymol", "-cq", str(pml_path), "-d", cmds],
                    capture_output=True, text=True, timeout=300,
                    cwd=str(pml_path.parent),
                )
                if raw_apo.exists():
                    add_title_bar(raw_apo, out_dir / f"{stem}_apo_pocket.png",
                                  f"{enzyme} ({uid})", subtitle)
                    parts.append("apo OK")
                else:
                    parts.append("apo FAILED")

                if raw_overlay.exists():
                    add_title_bar(raw_overlay, out_dir / f"{stem}_apo_overlay.png",
                                  f"{enzyme} ({uid}) \u2014 Substrate Overlay", subtitle)
                    parts.append("overlay OK")
                else:
                    parts.append("overlay skipped")

            elif job_type == "cofold":
                raw = Path(tmpdir) / "cofold.png"
                cmds = "; ".join([
                    "scene 001_cofold, recall",
                    f"draw {WIDTH}, {HEIGHT}; png {raw}",
                ])
                subprocess.run(
                    ["pymol", "-cq", str(pml_path), "-d", cmds],
                    capture_output=True, text=True, timeout=300,
                    cwd=str(pml_path.parent),
                )
                if raw.exists():
                    add_title_bar(raw, out_dir / f"{stem}_cofold.png",
                                  f"{enzyme} ({uid}) \u2014 Cofolded Complex", subtitle)
                    parts.append("cofold OK")
                else:
                    parts.append("cofold FAILED")

            ok = any("OK" in p for p in parts)
            return ok, f"{stem} [{job_type}]: {', '.join(parts)}"

    except subprocess.TimeoutExpired:
        return False, f"{stem} [{job_type}]: TIMEOUT"
    except Exception as e:
        return False, f"{stem} [{job_type}]: {e}"


def main():
    csv_data = load_csv_data()

    DIRS = [
        {
            "name": "synthetase",
            "apo_dir": BASE / "all_structures" / "amide_synthetases_candidates" / "alphafold_apo_structure",
            "cofold_dir": BASE / "all_structures" / "amide_synthetases_candidates" / "cofolded_structure",
            "out_dir": OUT_BASE / "amide_synthetases_candidates",
        },
        {
            "name": "hydrolase",
            "apo_dir": BASE / "all_structures" / "amide_hydrolase_candidates" / "alphafold_apo_structure",
            "cofold_dir": BASE / "all_structures" / "amide_hydrolase_candidates" / "cofolded_structure",
            "out_dir": OUT_BASE / "amide_hydrolase_candidates",
        },
    ]

    # Collect all rendering jobs
    jobs = []
    for d in DIRS:
        d["out_dir"].mkdir(parents=True, exist_ok=True)

        for pml in sorted(d["apo_dir"].glob("*_pocket_analysis.pml")):
            uid = extract_uid(pml.name)
            jobs.append({
                "pml": str(pml), "out_dir": str(d["out_dir"]),
                "meta": csv_data.get(uid, {}), "type": "apo",
            })

        for pml in sorted(d["cofold_dir"].glob("*_pocket_analysis.pml")):
            uid = extract_uid(pml.name)
            jobs.append({
                "pml": str(pml), "out_dir": str(d["out_dir"]),
                "meta": csv_data.get(uid, {}), "type": "cofold",
            })

    print(f"Rendering {len(jobs)} jobs with {MAX_WORKERS} parallel workers...\n")

    success = 0
    failed = []
    done = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(render_job, job): job for job in jobs}

        for future in as_completed(futures):
            done += 1
            ok, msg = future.result()
            if ok:
                success += 1
            else:
                failed.append(msg)
            tag = "OK" if ok else "FAIL"
            print(f"  [{done:3d}/{len(jobs)}] {msg}")

    print(f"\n{'='*60}")
    print(f"  RENDERING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total jobs:  {len(jobs)}")
    print(f"  Success:     {success}")
    print(f"  Failed:      {len(failed)}")
    if failed:
        print(f"\n  Failures:")
        for f_msg in failed:
            print(f"    - {f_msg}")

    for d in DIRS:
        n = len(list(d["out_dir"].glob("*.png")))
        print(f"\n  {d['name']}: {n} images")

    print(f"\n  Output: {OUT_BASE}")


if __name__ == "__main__":
    main()
