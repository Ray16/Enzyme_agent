"""A run-local sequence map with stable coordinates and auditable revisions."""

import colorsys
import hashlib
import json
from pathlib import Path
import re
import textwrap

import numpy as np

from enzyme_discovery_agent.models.candidate_registry import discovery_status, family_display, load_candidate_registry
from enzyme_discovery_agent.models.embedding_store import load_store
from enzyme_discovery_agent.run_layout import run_file
from enzyme_discovery_agent.state_store import atomic_write_json, state_lock

_RENDER_STYLE = "18pt-separated-legend-v1"


def _retrieved_family_id(metadata):
    """Use an explicit family assignment or an exactly supported database claim."""
    assigned = metadata.get("interpro_family_id") or metadata.get("screening_family_id")
    if assigned:
        return assigned
    for claim in (metadata.get("claim_validation") or {}).get("claims", []):
        family_id = claim.get("asserted_value")
        evidence = claim.get("evidence") or {}
        observed = evidence.get("observed_value")
        if (claim.get("claim_type") == "family_membership"
                and claim.get("verdict") == "supported"
                and isinstance(family_id, str) and re.fullmatch(r"IPR\d{6}", family_id)
                and evidence.get("source") == "UniProtKB"
                and isinstance(observed, list) and family_id in observed):
            return family_id
    return None


def _color(family):
    hue = int(hashlib.sha256(family.encode()).hexdigest()[:8], 16) / 0xffffffff
    rgb = colorsys.hsv_to_rgb(hue, .65, .82)
    return "#" + "".join(f"{round(v * 255):02x}" for v in rgb)


def stable_projection(run_dir, store):
    """Fit PCA once, then place additional embeddings in the same basis."""
    root = Path(run_dir) / "artifacts" / "candidate_map"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "projection.npz"
    matrix = store.matrix
    if matrix is None or len(store) < 3:
        return None
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["model"]) != store.model or saved["basis"].shape[1] != matrix.shape[1]:
                raise ValueError("Embedding model changed; create a new projection explicitly")
            center, basis = saved["center"], saved["basis"]
    else:
        center = matrix.mean(axis=0)
        _, _, basis = np.linalg.svd(matrix - center, full_matrices=False)
        basis = basis[:2]
        temporary = path.with_name("projection.tmp.npz")
        np.savez(temporary, center=center, basis=basis, model=store.model)
        temporary.replace(path)
    return (matrix - center) @ basis.T


def update_candidate_map(run_dir, iteration=None, log_fn=None):
    # Discovery and embedding workers can refresh the same run concurrently.
    with state_lock(Path(run_dir) / "artifacts" / "candidate_map" / "current.json"):
        return _update_candidate_map(run_dir, iteration, log_fn)


def _update_candidate_map(run_dir, iteration=None, log_fn=None):
    root = Path(run_dir)
    registry = load_candidate_registry(root, strict=True).get("candidates", {})
    if not registry:
        return None
    store = load_store(root)
    xy = stable_projection(root, store)
    family_policy_path = run_file(root, "family_learning.json")
    policy = json.loads(family_policy_path.read_text()) if family_policy_path.exists() else {}
    ranking_path = run_file(root, "pass1_ranking.json")
    top = set()
    if ranking_path.exists():
        ranking = json.loads(ranking_path.read_text())
        top = {row.get("accession") for row in ranking[:10]}
    family_names = {key.removeprefix("ipr:"): value.get("name")
                    for key, value in policy.get("families", {}).items() if key.startswith("ipr:")}
    for entry in registry.values():
        meta = entry.get("metadata") or {}
        family_id = meta.get("interpro_family_id") or meta.get("screening_family_id")
        if family_id and meta.get("interpro_family"):
            family_names[family_id] = meta["interpro_family"]
    points = []
    for acc, entry in registry.items():
        meta = entry.get("metadata") or {}
        family_id = _retrieved_family_id(meta)
        display_meta = {"screening_family_id": family_id, "interpro_family": family_names.get(family_id)}
        family = family_display(display_meta, acc) if family_id else "Unresolved family"
        row = store.acc_to_row.get(acc)
        position = xy[row].tolist() if xy is not None and row is not None else None
        points.append({"accession": acc, "name": meta.get("protein_name") or entry.get("enzyme_name") or acc,
                       "family": family, "family_id": family_id, "color": _color(family_id) if family_id else "#999999",
                       "status": discovery_status(entry) or "discovered", "top_ranked": acc in top,
                       "family_priority": (policy.get("families", {}).get(f"ipr:{family_id}") or {}).get("state", "unassessed"),
                       "xy": position})
    payload = {"model": store.model, "projection": "PCA in a fixed basis", "points": points,
               "registered": len(points), "embedded": sum(p["xy"] is not None for p in points)}
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    directory = root / "artifacts" / "candidate_map"
    directory.mkdir(parents=True, exist_ok=True)
    current = directory / "current.json"
    previous = json.loads(current.read_text()) if current.exists() else {}
    if previous.get("signature") == signature:
        rendered = directory / "rendering.json"
        render_state = json.loads(rendered.read_text()) if rendered.exists() else {}
        if (render_state.get("signature") != signature
                or render_state.get("style") != _RENDER_STYLE
                or not (root / "candidate_map.png").exists()):
            _render_png(root, previous)
        return root / "candidate_map.png" if (root / "candidate_map.png").exists() else None
    revision = previous.get("revision", 0) + 1
    payload.update(signature=signature, revision=revision, iteration=iteration)
    atomic_write_json(current, payload)
    atomic_write_json(directory / "history" / f"revision_{revision:05d}.json", payload)
    _render_png(root, payload)
    if log_fn:
        log_fn(f"  Candidate map revision {revision}: {payload['embedded']}/{payload['registered']} embedded candidates")
    return root / "candidate_map.png" if (root / "candidate_map.png").exists() else None


def _render_png(root, payload):
    if payload["embedded"] < 3:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        return
    families = {point["family"]: point["color"] for point in payload["points"] if point["xy"] is not None}
    labels = {family: textwrap.fill(family, width=52, break_long_words=True, break_on_hyphens=False)
              for family in families}
    line_count = sum(label.count("\n") + 1 for label in labels.values())
    # Size the legend panel for its full text instead of shrinking its font.
    height = max(9, 1.5 + .33 * line_count + .15 * len(families))
    style = {"font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
             "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18,
             "figure.titlesize": 18}
    with plt.rc_context(style):
        fig, (ax, legend_ax) = plt.subplots(1, 2, figsize=(22, height),
                                           gridspec_kw={"width_ratios": [1.2, 1]}, layout="constrained")
        ax.set_box_aspect(1)
        ax.set_anchor("N")
        legend_ax.set_axis_off()
    for point in payload["points"]:
        if point["xy"] is None:
            continue
        x, y = point["xy"]
        ax.scatter(x, y, c=[point["color"]], s=45 if point["top_ranked"] else 12,
                   marker="x" if point["status"] == "rejected" else "o",
                   alpha=.35 if point["status"] == "discovered" else .8,
                   edgecolors="black" if point["top_ranked"] else None)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", color="none", markerfacecolor=color,
                      markersize=9, label=labels[family])
               for family, color in sorted(families.items())]
    if handles:
        legend_ax.legend(handles=handles, loc="upper left", fontsize=18,
                         borderaxespad=0, labelspacing=.65, handletextpad=.6, frameon=False)
    ax.set_xlabel("Fixed PCA 1", fontsize=18, labelpad=12)
    ax.set_ylabel("Fixed PCA 2", fontsize=18, labelpad=12)
    ax.tick_params(labelsize=18)
    fig.suptitle(f"ESM sequence map: {payload['embedded']}/{payload['registered']} candidates · revision {payload['revision']}",
                 fontsize=18)
    temporary = root / ".candidate_map.png.tmp"
    fig.savefig(temporary, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    temporary.replace(root / "candidate_map.png")
    atomic_write_json(root / "artifacts" / "candidate_map" / "rendering.json", {
        "signature": payload["signature"], "revision": payload["revision"],
        "style": _RENDER_STYLE, "font_size_pt": 18,
    })
