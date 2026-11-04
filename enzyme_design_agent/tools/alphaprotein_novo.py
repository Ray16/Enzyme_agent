"""Agent-invoked tool: design an enzyme *de novo* around a catalytic motif with
AlphaProtein Novo (AP Novo).

Where this sits in the agent's toolkit
--------------------------------------
Every other structural tool here is *retrieval*-shaped: it finds an existing
protein (UniProt/Foldseek/ESM-neighbour expansion) and asks whether that protein
can accommodate the target chemistry.  AP Novo is the first *generative* tool --
when retrieval turns up no scaffold that holds the geometry, this builds one.
The input is therefore not a sequence or an accession but a **motif**: a 3D
arrangement of catalytic sidechain atoms plus ligand, representing the
transition state or covalent intermediate of the target reaction.

Pipeline (DeepMind, 2026; see ``APNOVO_DIR/README.md``):
  1. generation  -- diffusion co-generates backbone + sequence scaffolding the motif (GPU)
  2. resequence  -- optional LigandMPNN sequence redesign in the ligand context (CPU)
  3. folding     -- AlphaFold 3 (leaving-atom checkpoint) refolds each design (GPU)
  4. evaluation  -- catalytic-geometry recovery + confidence metrics (CPU)

Design notes
------------
  - The heavy JAX/CUDA stack lives in its OWN venv outside this repo
    (``config.APNOVO_PYTHON``) and is invoked as a subprocess, so a normal agent
    run never imports jax or touches a GPU.  This module is stdlib-only.
  - **GPU courtesy.** AP Novo auto-detects GPUs via ``nvidia-smi`` when
    ``CUDA_VISIBLE_DEVICES`` is unset and spawns one worker per *visible* device
    -- on a shared 8-GPU node that silently seizes the whole node.  This module
    therefore ALWAYS reserves explicit devices through the cluster gate
    (``gpu_reserve``) and pins ``CUDA_VISIBLE_DEVICES`` to exactly the reserved
    set, releasing them on every exit path.  It never sets the variable without
    a reservation behind it.
  - GPU count is a pure throughput knob: stages 1 and 3 shard *over designs*
    with no cross-device communication, so N GPUs is ~N x faster on a campaign
    of many designs and 1 GPU is always sufficient to make progress.
  - All numeric knobs are fallback defaults the agent overrides per call.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

from enzyme_discovery_agent import config
from enzyme_discovery_agent.tool_registry import tool

_STAGES = ("generation", "resequence", "folding", "evaluation")
# Stages 2 and 4 are pure CPU (LigandMPNN runs with CUDA_VISIBLE_DEVICES="" and
# the metrics are numpy): running one of them alone must not reserve a GPU.
_CPU_ONLY_STAGES = frozenset({"resequence", "evaluation"})


# --------------------------------------------------------------------------
# Readiness
# --------------------------------------------------------------------------
def _readiness() -> dict:
    """Inspect the AP Novo install without importing or running it.

    Returns ``{"ready": bool, "blocking": [...], "optional": [...], "paths": {...}}``.
    ``blocking`` prevents any run; ``optional`` only disables a stage.
    """
    apnovo_dir = Path(config.APNOVO_DIR)
    paths = {
        "apnovo_dir": str(apnovo_dir),
        "python": config.APNOVO_PYTHON,
        "run_pipeline": str(apnovo_dir / "run_pipeline.py"),
        "generator_weights": str(Path(config.APNOVO_GENERATOR_MODEL_DIR) / "generator.bin.zst"),
        "af3_weights": str(Path(config.APNOVO_AF3_MODEL_DIR) / "af3_leaving_atom.bin.zst"),
        "gpu_reserve": config.CANDIDATE_MAP_GPU_RESERVE,
        "ligandmpnn_dir": config.LIGANDMPNN_DIR,
        "ligandmpnn_python": config.LIGANDMPNN_PYTHON,
    }
    blocking, optional = [], []
    for key in ("python", "run_pipeline", "generator_weights", "gpu_reserve"):
        if not Path(paths[key]).exists():
            blocking.append(f"{key} missing at {paths[key]}")
    if not Path(paths["af3_weights"]).exists():
        optional.append(
            f"AF3 leaving-atom weights missing at {paths['af3_weights']} -- "
            "folding and evaluation stages unavailable"
        )
    if not (Path(paths["ligandmpnn_python"]).exists() and Path(paths["ligandmpnn_dir"]).exists()):
        optional.append(
            "LigandMPNN env missing -- run with resequence=false (the generated "
            "sequence is folded directly)"
        )
    return {
        "ready": not blocking,
        "blocking": blocking,
        "optional": optional,
        "paths": paths,
    }


# --------------------------------------------------------------------------
# GPU reservation (acquire-and-hold K devices for one self-sharding process)
# --------------------------------------------------------------------------
def _free_gpu_indices(max_n: int) -> list[str]:
    """Free GPU indices ON THIS HOST, via the cluster reservation gate."""
    host = socket.gethostname().split(".")[0]
    try:
        r = subprocess.run(
            [config.CANDIDATE_MAP_GPU_RESERVE, "list"],
            capture_output=True, text=True, timeout=120,
        )
    except Exception:  # noqa: BLE001
        return []
    out = []
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == host and parts[1].isdigit():
            out.append(parts[1])
    return out[:max_n]


def _subprocess_env(gpus: list[str]) -> dict:
    """Environment for the AP Novo subprocess.

    Two things have to be right here:

    1. ``CUDA_VISIBLE_DEVICES`` is pinned to exactly the reserved devices (empty
       for a CPU-only stage).  Without it AP Novo enumerates every GPU on the
       node via ``nvidia-smi`` and forks a worker onto each one.
    2. A system CUDA on ``LD_LIBRARY_PATH`` is stripped.  Some nodes' login
       profiles export e.g. ``/usr/local/cuda-12.4/lib64``, whose cuSPARSE
       shadows the CUDA wheels pinned inside the AP Novo venv; JAX then fails
       with "Unable to load cuSPARSE", falls back to CPU, and the run dies deep
       inside the folding stage with "Not supported on cpu" instead of saying
       anything about libraries.  Measured on lambda13: with the login profile
       JAX saw ``[CpuDevice]``, without it all 8 ``CudaDevice``s.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    threads = str(config.APNOVO_THREADS_PER_WORKER)
    env.update({
        "OMP_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
    })
    ld = env.get("LD_LIBRARY_PATH", "")
    if ld:
        kept = [p for p in ld.split(":") if p and "cuda" not in p.lower()]
        if kept:
            env["LD_LIBRARY_PATH"] = ":".join(kept)
        else:
            env.pop("LD_LIBRARY_PATH", None)
    return env


def _local_gpu_capability() -> float | None:
    """Highest CUDA compute capability on this host, or None if undetectable."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception:  # noqa: BLE001
        return None
    caps = []
    for line in r.stdout.splitlines():
        try:
            caps.append(float(line.strip()))
        except ValueError:
            continue
    return max(caps) if caps else None


def _folding_in_scope(only_stage: str, fold_apo_monomer: bool, folding_states: list[dict]) -> bool:
    """Would this call run the AF3 folding stage (the sm80+ one)?"""
    if only_stage:
        # Evaluation reads folded structures but is itself numpy on CPU.
        return only_stage == "folding"
    return bool(fold_apo_monomer or folding_states)


def _acquire_gpus(n: int, label: str) -> list[str]:
    """Reserve up to ``n`` GPUs on this host, recording THIS process as owner.

    ``gpu_reserve acquire --pid`` ties the lock's lifetime to our pid, so the
    reservation survives for as long as this tool call is running and goes stale
    (rather than leaking forever) if we are killed.  Re-checks each candidate at
    the moment of claim -- another session may have taken it since ``list``.
    """
    reserved: list[str] = []
    for idx in _free_gpu_indices(n * 2):  # over-fetch: some claims lose the race
        if len(reserved) >= n:
            break
        try:
            r = subprocess.run(
                [config.CANDIDATE_MAP_GPU_RESERVE, "acquire", idx,
                 "--label", label, "--pid", str(os.getpid())],
                capture_output=True, text=True, timeout=120,
            )
        except Exception:  # noqa: BLE001
            continue
        if r.returncode == 0:
            reserved.append(idx)
    return reserved


def _release_gpus(idxs: list[str]) -> None:
    """Release reservations.  Spawned directly from the owning process so the
    gate sees ``opid == $PPID`` and releases without ``--force``."""
    for idx in idxs:
        try:
            subprocess.run(
                [config.CANDIDATE_MAP_GPU_RESERVE, "release", idx],
                capture_output=True, text=True, timeout=120,
            )
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------
# Manifest construction
# --------------------------------------------------------------------------
def _build_manifest(
    *,
    motif_cif_path: str,
    motif_str: str,
    motif_atoms: str,
    seq_length: str,
    unindexed_motif_residues: str,
    num_designs: int,
    num_sampling_steps: int,
    seed_start: int,
    is_author_naming: bool,
    design_name: str,
    resequence: bool,
    resequence_temperature: float,
    num_sequences: int,
    folding_states: list[dict],
    fold_apo_monomer: bool,
    fold_seeds: int,
    evaluation_suite: str,
    partial_diffusion_input_file: str,
    partial_diffusion_num_steps: int,
) -> dict:
    """Translate tool arguments into an AP Novo manifest dict.

    Paths are written ABSOLUTE: AP Novo resolves relative manifest paths against
    the manifest's own directory, and we write the manifest into the run dir
    rather than next to the user's motif CIF.
    """
    design: dict = {
        "name": design_name,
        "output_prefix": design_name,
        "input_file": str(Path(motif_cif_path).resolve()),
        "motif_str": motif_str,
        "num_designs": int(num_designs),
        "is_author_naming": bool(is_author_naming),
    }
    if motif_atoms:
        design["motif_atoms"] = motif_atoms
    if seq_length:
        design["seq_length"] = seq_length
    if unindexed_motif_residues:
        design["unindexed_motif_residues"] = unindexed_motif_residues
    if partial_diffusion_input_file:
        design["partial_diffusion_input_file"] = str(Path(partial_diffusion_input_file).resolve())
        design["partial_diffusion_num_steps"] = int(partial_diffusion_num_steps)

    manifest: dict = {
        "settings": {"model_dir": config.APNOVO_GENERATOR_MODEL_DIR},
        "defaults": {
            "num_sampling_steps": int(num_sampling_steps),
            "seed_start": int(seed_start),
        },
        "designs": [design],
        "resequence": {"enabled": bool(resequence)},
    }
    if resequence:
        manifest["resequence"].update({
            "temperature": float(resequence_temperature),
            "num_sequences": int(num_sequences),
            "use_side_chain_context": False,
        })

    states: list[dict] = []
    if fold_apo_monomer:
        states.append({"name": "monomer", "ligands": [], "covalent_bonds": []})
    for i, state in enumerate(folding_states):
        # Each ligand's `id` must be a ligand chain declared in motif_str
        # (".../B1/C1"), and those chains must exist in the motif CIF. Separate
        # states may put DIFFERENT chemistry on the same chain id -- that is how
        # a reaction coordinate is expressed (e.g. ATP in one state, ADP in the
        # next), and it is what makes cross-state consistency filtering possible.
        states.append({
            "name": state.get("name") or f"state{i + 1}",
            "ligands": [
                {k: v for k, v in lig.items()
                 if k in ("id", "ccd_code", "smiles", "name") and v}
                for lig in state.get("ligands") or []
            ],
            "covalent_bonds": state.get("covalent_bonds") or [],
        })
    if states:
        manifest["folding"] = {
            "inputs": ["resequenced" if resequence else "generated"],
            "seeds": list(range(int(fold_seeds))),
            "states": states,
        }
    if evaluation_suite:
        manifest["evaluation"] = {"suite": evaluation_suite}
    return manifest


# --------------------------------------------------------------------------
# Result parsing
# --------------------------------------------------------------------------
def _read_eval_summary(campaign_dir: Path, max_rows: int = 50) -> tuple[list[dict], int]:
    """Read ``04_eval/evaluation_summary.csv`` -> (rows, total_rows).

    Numeric-looking cells are coerced to float so the agent can rank on them.
    """
    csv_path = campaign_dir / "04_eval" / "evaluation_summary.csv"
    if not csv_path.exists():
        return [], 0
    rows: list[dict] = []
    with csv_path.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            row: dict = {}
            for k, v in raw.items():
                if v in (None, ""):
                    continue
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    row[k] = v
            rows.append(row)
    return rows[:max_rows], len(rows)


def _collect_designs(campaign_dir: Path) -> dict:
    """Count the artifacts actually on disk, per stage."""
    def _n(subdir: str, pattern: str, exclude: str = "") -> int:
        d = campaign_dir / subdir
        if not d.is_dir():
            return 0
        return sum(
            1 for p in d.glob(pattern) if not (exclude and p.name.endswith(exclude))
        )

    return {
        # Stage 1 writes <prefix>.cif AND <prefix>_motif.cif (the re-indexed
        # ground-truth motif) side by side -- counting both double-counts designs.
        "generated": _n("01_generation", "*.cif", exclude="_motif.cif"),
        "resequenced": _n("02_resequence", "*_reseq.cif"),
        "folded": _n("03_folded", "*_folded_seed*.cif"),
        "evaluated": _n("04_eval", "*_evaluation.json"),
    }


def _stage_log_tail(campaign_dir: Path, n_chars: int = 600) -> dict:
    """Tail of each stage log -- the useful part when a stage fails."""
    tails = {}
    logs = campaign_dir / "logs"
    if not logs.is_dir():
        return tails
    for stage in _STAGES:
        f = logs / f"{stage}.log"
        if f.exists():
            try:
                text = f.read_text(errors="replace").strip()
            except OSError:
                continue
            if text:
                tails[stage] = text[-n_chars:]
    return tails


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
@tool(
    description=(
        "Check whether the de novo enzyme design backend (AlphaProtein Novo) is "
        "installed and how many GPUs are currently free for it. Call this before "
        "committing to a design campaign, which is a long GPU job."
    ),
    category="structure",
    parallel_safe=True,
    params={},
)
def check_de_novo_design_readiness() -> dict:
    """Report AP Novo install state and current GPU availability."""
    info = _readiness()
    free = _free_gpu_indices(config.APNOVO_MAX_GPUS)
    host = socket.gethostname().split(".")[0]
    cap = _local_gpu_capability()
    folding_ok = cap is None or cap >= config.APNOVO_MIN_FOLDING_CAPABILITY
    if info["ready"]:
        summary = (
            f"AP Novo ready; {len(free)} GPU(s) free on {host}"
            f"{' (' + ','.join(free) + ')' if free else ''}"
        )
        if not folding_ok:
            summary += (
                f"; GPUs here are sm{cap:.1f} so generation works but AF3 folding "
                f"needs sm{config.APNOVO_MIN_FOLDING_CAPABILITY:.1f}+"
            )
        if info["optional"]:
            summary += f"; {len(info['optional'])} optional component(s) missing"
    else:
        summary = f"AP Novo NOT ready: {'; '.join(info['blocking'])}"
    return {
        "status": "success" if info["ready"] else "error",
        "summary": summary,
        "data": {
            "ready": info["ready"],
            "blocking": info["blocking"],
            "optional_missing": info["optional"],
            "paths": info["paths"],
            "host": host,
            "free_gpus": free,
            "max_gpus": config.APNOVO_MAX_GPUS,
            "local_compute_capability": cap,
            "folding_supported_here": folding_ok,
            "evaluation_suites": list(config.APNOVO_EVAL_SUITES),
        },
        "suggestions": (
            []
            if info["ready"]
            else ["See APNOVO_DIR/README.md for install steps, or set "
                  "ENZYME_AGENT_APNOVO_DIR to an existing install."]
        ),
    }


@tool(
    description=(
        "Design an enzyme DE NOVO around a catalytic motif using AlphaProtein Novo: "
        "a diffusion model generates protein backbones+sequences that scaffold a "
        "fixed 3D arrangement of catalytic sidechain atoms and ligand (the reaction "
        "transition state or covalent intermediate), then optionally redesigns the "
        "sequence with LigandMPNN, refolds with AlphaFold 3, and scores how well "
        "each design recovers the intended catalytic geometry. "
        "Use this when retrieval found NO natural scaffold that can hold the target "
        "chemistry -- it builds a scaffold instead of finding one. "
        "Requires a motif CIF file. Runs on reserved GPUs; cost scales with "
        "num_designs x num_sampling_steps, so scout with a small num_designs and "
        "num_sampling_steps=50 before a full campaign (1000 steps)."
    ),
    category="structure",
    parallel_safe=False,
    injected_params=["run_dir"],
    params={
        "motif_cif_path": {
            "type": "string",
            "description": (
                "Path to a CIF file containing the catalytic motif: the catalytic "
                "residues and any ligand, positioned as the enzyme should hold them."
            ),
        },
        "motif_str": {
            "type": "string",
            "description": (
                "Motif layout string. Chains are delimited by '/'; the first chain is "
                "comma-separated segments of either fixed input residues (e.g. 'A1', "
                "'A5-6') or designable length ranges (e.g. '10-100'); later chains are "
                "single fixed ligand residues (e.g. 'B1'). Segment order can be sampled "
                "by writing the fixed residues before a '|' and '{}' placeholders after, "
                "e.g. 'A1,A2|10-100,{},2-80,{},10-100/B1'."
            ),
        },
        "motif_atoms": {
            "type": "string",
            "description": (
                "Optional 'tip atom' conditioning: space-separated 'residue:atom,atom' "
                "groups, e.g. 'A1:OE2,OE1,CD,CG A2:OG,CB,CA'. Only these atoms are fixed "
                "and the rest of each sidechain is generated (more design freedom). "
                "Default: every atom of every motif residue is fixed."
            ),
            "default": "",
        },
        "seq_length": {
            "type": "string",
            "description": (
                "Optional total length of the designed chain: exact ('150') or an "
                "inclusive range ('120-160')."
            ),
            "default": "",
        },
        "unindexed_motif_residues": {
            "type": "string",
            "description": (
                "Optional comma-separated residues (e.g. 'A1,A2,A3') to place by "
                "UNINDEXED conditioning, letting the model choose their sequence "
                "positions. Requires motif_str to hold a single designable segment "
                "plus ligand chains (e.g. '10-250/B1'). Indexed conditioning (the "
                "default) performs slightly better in practice."
            ),
            "default": "",
        },
        "num_designs": {
            "type": "integer",
            "description": "Number of designs to generate (default 2; a real campaign uses 100s).",
            "default": config.APNOVO_NUM_DESIGNS,
        },
        "num_sampling_steps": {
            "type": "integer",
            "description": (
                "Reverse diffusion steps. 1000 is the full-quality trajectory; 50 is a "
                "fast smoke setting for verifying a motif spec before spending GPU time."
            ),
            "default": config.APNOVO_SAMPLING_STEPS,
        },
        "seed_start": {
            "type": "integer",
            "description": "First RNG seed; designs use seed_start..seed_start+num_designs-1.",
            "default": 0,
        },
        "is_author_naming": {
            "type": "boolean",
            "description": (
                "True if motif_str uses author-assigned residue numbers (what PyMOL and "
                "the literature show); False for PDB internal numbering."
            ),
            "default": True,
        },
        "resequence": {
            "type": "boolean",
            "description": (
                "Run LigandMPNN sequence redesign on the generated backbone (stage 2). "
                "If false, the diffusion model's own sequence is folded directly."
            ),
            "default": True,
        },
        "resequence_temperature": {
            "type": "number",
            "description": "LigandMPNN sampling temperature; lower is more conservative.",
            "default": 0.1,
        },
        "num_sequences": {
            "type": "integer",
            "description": "LigandMPNN sequences sampled per generated backbone.",
            "default": 2,
        },
        "folding_states": {
            "type": "array",
            "description": (
                "Catalytic states to refold each design in, as objects with 'name' and "
                "'ligands'. Each ligand has: 'id' = the ligand chain letter declared in "
                "motif_str and present in the motif CIF; 'ccd_code' = a PDB "
                "chemical-component code (e.g. 'ATP', 'MG') OR, for a species with no CCD "
                "entry, the custom code 'pd00'; and 'smiles' = required whenever ccd_code "
                "is 'pd00' or otherwise not a real CCD code. One ligand entry per chain, "
                "so two metal ions are two entries on two chains. "
                "Giving SEVERAL states walks the reaction coordinate (substrate complex, "
                "then intermediate) and is the basis of the strongest published filter: "
                "keep designs whose predicted geometry is CONSISTENT across states. "
                "Different states may place different chemistry on the same chain id. "
                "Example: [{'name':'substrate','ligands':[{'id':'B','ccd_code':'ATP'},"
                "{'id':'C','ccd_code':'MG'},{'id':'D','ccd_code':'MG'},"
                "{'id':'E','ccd_code':'pd00','smiles':'<substrate>'}]}, "
                "{'name':'intermediate','ligands':[{'id':'B','ccd_code':'ADP'},"
                "{'id':'C','ccd_code':'MG'},{'id':'D','ccd_code':'MG'},"
                "{'id':'E','ccd_code':'pd00','smiles':'<intermediate>'}]}]"
            ),
            "items": {"type": "object"},
            "default": [],
        },
        "fold_apo_monomer": {
            "type": "boolean",
            "description": "Also fold each design as an apo monomer (no ligand).",
            "default": True,
        },
        "fold_seeds": {
            "type": "integer",
            "description": "AlphaFold 3 seeds per design per state; more seeds = more robust metrics.",
            "default": 1,
        },
        "evaluation_suite": {
            "type": "string",
            "description": (
                "Problem-specific metric suite. Use one that matches the target "
                "chemistry; leave empty to skip reaction-specific scoring and keep only "
                "generic structural/confidence metrics."
            ),
            "enum": ["", *config.APNOVO_EVAL_SUITES],
            "default": "",
        },
        "partial_diffusion_input_file": {
            "type": "string",
            "description": (
                "Optional CIF of an existing protein (e.g. a parent design) to diversify "
                "by partial diffusion instead of generating from scratch."
            ),
            "default": "",
        },
        "partial_diffusion_num_steps": {
            "type": "integer",
            "description": (
                "Reverse diffusion steps to unroll from the partial-diffusion parent "
                "(of 1000): ~600 keeps TM-score ~95 to the parent, ~900 gives ~80."
            ),
            "default": 600,
        },
        "num_gpus": {
            "type": "integer",
            "description": (
                "GPUs to reserve (1-8). Purely a throughput knob: designs are sharded "
                "across GPUs with no cross-device communication, so N GPUs is ~Nx faster "
                "on a multi-design campaign. Never exceeds the GPUs actually free."
            ),
            "default": 1,
        },
        "only_stage": {
            "type": "string",
            "description": (
                "Run a single stage against existing outputs instead of the whole "
                "pipeline (e.g. refold or re-evaluate a campaign already on disk). "
                "'resequence' and 'evaluation' are CPU-only and reserve no GPU."
            ),
            "enum": ["", *_STAGES],
            "default": "",
        },
        "campaign_name": {
            "type": "string",
            "description": (
                "Short slug naming this campaign; also the output subdirectory and "
                "design-file prefix. Re-using a name RESUMES that campaign, skipping "
                "designs that already finished."
            ),
            "default": "denovo",
        },
        "timeout_s": {
            "type": "integer",
            "description": "Wall-clock budget for the whole pipeline, in seconds.",
            "default": config.APNOVO_TIMEOUT,
        },
        "run_dir": {"type": "string", "description": "Agent run directory (injected)."},
    },
)
def design_de_novo_enzyme(
    motif_cif_path: str,
    motif_str: str,
    motif_atoms: str = "",
    seq_length: str = "",
    unindexed_motif_residues: str = "",
    num_designs: int = config.APNOVO_NUM_DESIGNS,
    num_sampling_steps: int = config.APNOVO_SAMPLING_STEPS,
    seed_start: int = 0,
    is_author_naming: bool = True,
    resequence: bool = True,
    resequence_temperature: float = 0.1,
    num_sequences: int = 2,
    folding_states: list[dict] | None = None,
    fold_apo_monomer: bool = True,
    fold_seeds: int = 1,
    evaluation_suite: str = "",
    partial_diffusion_input_file: str = "",
    partial_diffusion_num_steps: int = 600,
    num_gpus: int = 1,
    only_stage: str = "",
    campaign_name: str = "denovo",
    timeout_s: int = config.APNOVO_TIMEOUT,
    run_dir: str | None = None,
) -> dict:
    """Run an AP Novo design campaign on reserved GPUs and summarize the designs."""
    folding_states = list(folding_states or [])
    info = _readiness()
    if not info["ready"]:
        return {
            "status": "error",
            "summary": f"De novo design backend unavailable: {'; '.join(info['blocking'])}",
            "data": None,
            "suggestions": ["Call check_de_novo_design_readiness for the full install state."],
        }

    motif = Path(motif_cif_path).expanduser()
    if not motif.exists():
        return {
            "status": "error",
            "summary": f"Motif CIF not found: {motif}",
            "data": None,
            "suggestions": [
                "Provide a CIF holding the catalytic residues and ligand in the "
                "geometry the designed enzyme should enforce."
            ],
        }
    if not motif_str.strip():
        return {
            "status": "error",
            "summary": "motif_str is required (it describes how the motif is laid out in the design).",
            "data": None,
            "suggestions": ["Example: 'A1,A2|10-100,{},2-80,{},10-100/B1'."],
        }
    if resequence and any("LigandMPNN" in m for m in info["optional"]):
        return {
            "status": "error",
            "summary": "resequence=true but the LigandMPNN environment is not installed.",
            "data": {"optional_missing": info["optional"]},
            "suggestions": ["Re-run with resequence=false, or install LigandMPNN."],
        }

    # Hardware gate: AF3 folding needs Ampere+. Refuse up front rather than
    # burning a long generation stage and dying at stage 3.
    if _folding_in_scope(only_stage, fold_apo_monomer, folding_states):
        cap = _local_gpu_capability()
        floor = config.APNOVO_MIN_FOLDING_CAPABILITY
        if cap is not None and cap < floor:
            return {
                "status": "error",
                "summary": (
                    f"This host's GPUs are compute capability {cap:.1f}; the AlphaFold 3 "
                    f"folding stage requires >= {floor:.1f} (Ampere or newer) and would "
                    "fail partway through."
                ),
                "data": {"local_compute_capability": cap, "required": floor},
                "suggestions": [
                    "Run with fold_apo_monomer=false and no folding_states to generate "
                    "designs here, then run only_stage='folding' on a newer-GPU host.",
                    "Or run the whole campaign on a host with Ampere+ GPUs.",
                ],
            }

    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in campaign_name) or "denovo"
    base = Path(run_dir) / "artifacts" / "alphaprotein_novo" if run_dir else Path.cwd() / "alphaprotein_novo"
    campaign_dir = base / slug
    campaign_dir.mkdir(parents=True, exist_ok=True)

    manifest = _build_manifest(
        motif_cif_path=str(motif),
        motif_str=motif_str,
        motif_atoms=motif_atoms,
        seq_length=seq_length,
        unindexed_motif_residues=unindexed_motif_residues,
        num_designs=num_designs,
        num_sampling_steps=num_sampling_steps,
        seed_start=seed_start,
        is_author_naming=is_author_naming,
        design_name=slug,
        resequence=resequence,
        resequence_temperature=resequence_temperature,
        num_sequences=num_sequences,
        folding_states=folding_states,
        fold_apo_monomer=fold_apo_monomer,
        fold_seeds=fold_seeds,
        evaluation_suite=evaluation_suite,
        partial_diffusion_input_file=partial_diffusion_input_file,
        partial_diffusion_num_steps=partial_diffusion_num_steps,
    )
    manifest_path = campaign_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    cpu_only = only_stage in _CPU_ONLY_STAGES
    want = max(1, min(int(num_gpus), config.APNOVO_MAX_GPUS))
    gpus = [] if cpu_only else _acquire_gpus(want, label=f"apnovo_{slug}")
    if not gpus and not cpu_only:
        return {
            "status": "error",
            "summary": (
                f"No free GPU on {socket.gethostname().split('.')[0]} -- design not started "
                "(the reservation gate found every device busy or reserved)."
            ),
            "data": {"manifest": str(manifest_path), "campaign_dir": str(campaign_dir)},
            "suggestions": [
                "Retry later, or check `gpu_reserve list` for a host with free devices.",
                "The manifest is already written, so the retry costs nothing extra.",
            ],
        }

    cmd = [
        config.APNOVO_PYTHON,
        str(Path(config.APNOVO_DIR) / "run_pipeline.py"),
        f"--manifest={manifest_path}",
        f"--output_dir={campaign_dir}",
        f"--apn_model_dir={config.APNOVO_GENERATOR_MODEL_DIR}",
        f"--af3_model_dir={config.APNOVO_AF3_MODEL_DIR}",
    ]
    if resequence:
        cmd += [
            f"--ligandmpnn_dir={config.LIGANDMPNN_DIR}",
            f"--ligandmpnn_python={config.LIGANDMPNN_PYTHON}",
        ]
    # Which stages to actually run. `run_pipeline.py` has --only_stage and
    # --from_stage but NO "stop after" flag, and its `_should_run` returns True for
    # FOLDING on any default invocation -- so asking for generation+resequencing
    # only (fold_apo_monomer=False, folding_states=[]) still ran AF3, which dies on
    # a pre-Ampere GPU and returned rc=1 on an otherwise complete campaign
    # (observed 2026-10-06, campaign pmd_clamp6_v2: 24/24 generated and 96/96
    # resequenced, then NotImplementedError on a V100). Drive the stages
    # explicitly in that case instead of letting the default run them all.
    if only_stage:
        stage_plan = [only_stage]
    elif _folding_in_scope(only_stage, fold_apo_monomer, folding_states):
        stage_plan = [""]                      # default: let the pipeline run everything
    else:
        stage_plan = ["generation"] + (["resequence"] if resequence else [])
    cmds = [cmd + ([f"--only_stage={st}"] if st else []) for st in stage_plan]

    env = _subprocess_env(gpus)
    started = time.time()
    rc, stdout, stderr, timed_out = 0, "", "", False
    try:
        for sub_cmd in cmds:
            remaining = timeout_s - (time.time() - started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(sub_cmd, timeout_s)
            proc = subprocess.run(
                sub_cmd, capture_output=True, text=True, timeout=remaining,
                env=env, cwd=config.APNOVO_DIR,
            )
            rc, stdout, stderr = proc.returncode, stdout + proc.stdout, stderr + proc.stderr
            if rc != 0:
                break
    except subprocess.TimeoutExpired as exc:
        rc, timed_out = None, True
        got = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        stdout, stderr = stdout + got, stderr + err
    except Exception as exc:  # noqa: BLE001 -- `finally` below does the release
        return {
            "status": "error",
            "summary": f"Failed to launch the design pipeline: {exc}",
            "data": {"campaign_dir": str(campaign_dir), "cmds": cmds},
            "suggestions": ["Check that APNOVO_PYTHON points at the AP Novo venv."],
        }
    finally:
        _release_gpus(gpus)
    elapsed = time.time() - started

    counts = _collect_designs(campaign_dir)
    rows, n_rows = _read_eval_summary(campaign_dir)
    data = {
        "campaign_dir": str(campaign_dir),
        "manifest": str(manifest_path),
        "artifact_counts": counts,
        "evaluation_rows": rows,
        "n_evaluated": n_rows,
        "elapsed_s": round(elapsed, 1),
        "gpus_used": gpus,
        "host": socket.gethostname().split(".")[0],
        "requested": {
            "num_designs": num_designs,
            "num_sampling_steps": num_sampling_steps,
            "resequence": resequence,
            "evaluation_suite": evaluation_suite or None,
            "only_stage": only_stage or None,
            "stages_run": stage_plan,
        },
    }
    if counts["generated"]:
        data["seconds_per_generated_design"] = round(elapsed / counts["generated"], 1)

    if timed_out:
        data["stage_logs"] = _stage_log_tail(campaign_dir)
        return {
            "status": "error",
            "summary": (
                f"Design campaign '{slug}' hit the {timeout_s}s budget after "
                f"{elapsed / 60:.0f} min; {counts['generated']} design(s) generated "
                "(partial results kept on disk)."
            ),
            "data": data,
            "suggestions": [
                "Re-call with the same campaign_name to RESUME from where it stopped.",
                "Or lower num_designs / num_sampling_steps, or raise num_gpus.",
            ],
        }

    if rc != 0:
        data["stage_logs"] = _stage_log_tail(campaign_dir)
        data["stderr_tail"] = (stderr or "").strip()[-1500:]
        return {
            "status": "error",
            "summary": f"Design pipeline failed (rc={rc}) for campaign '{slug}'.",
            "data": data,
            "suggestions": [
                "Inspect data.stage_logs for the failing stage.",
                "Motif-spec errors usually surface in the generation log -- check that "
                "motif_str residues exist in the CIF and that is_author_naming matches "
                "its numbering scheme.",
            ],
        }

    if not counts["generated"]:
        data["stage_logs"] = _stage_log_tail(campaign_dir)
        return {
            "status": "error",
            "summary": f"Pipeline reported success but produced no designs for '{slug}'.",
            "data": data,
            "suggestions": ["Inspect data.stage_logs; the motif spec may select no residues."],
        }

    parts = [f"{counts['generated']} design(s) generated"]
    if counts["resequenced"]:
        parts.append(f"{counts['resequenced']} resequenced")
    if counts["folded"]:
        parts.append(f"{counts['folded']} folded")
    if n_rows:
        parts.append(f"{n_rows} evaluated")
    return {
        "status": "success",
        "summary": (
            f"De novo campaign '{slug}': " + ", ".join(parts)
            + f" in {elapsed / 60:.1f} min "
            + ("on CPU." if cpu_only else f"on {len(gpus)} GPU(s).")
        ),
        "data": data,
        "suggestions": [
            "Rank designs on the evaluation metrics (catalytic-geometry recovery "
            "first, then fold confidence) before proposing any for synthesis.",
            "These are PREDICTIONS from a generative model -- untested designs, not "
            "validated enzymes.",
        ],
    }


@tool(
    description=(
        "Summarize an existing de novo design campaign on disk: artifact counts per "
        "stage and the per-design evaluation metrics table. Use this to analyze a "
        "campaign as its outputs land, without re-running any GPU work."
    ),
    category="structure",
    parallel_safe=True,
    params={
        "campaign_dir": {
            "type": "string",
            "description": "Campaign output directory (the campaign_dir of a previous design call).",
        },
        "max_rows": {
            "type": "integer",
            "description": "Maximum per-design metric rows to return.",
            "default": 50,
        },
    },
)
def summarize_de_novo_campaign(campaign_dir: str, max_rows: int = 50) -> dict:
    """Read a campaign's artifacts and evaluation table without touching a GPU."""
    d = Path(campaign_dir).expanduser()
    if not d.is_dir():
        return {
            "status": "error",
            "summary": f"No such campaign directory: {d}",
            "data": None,
            "suggestions": ["Pass the campaign_dir returned by design_de_novo_enzyme."],
        }
    counts = _collect_designs(d)
    rows, n_rows = _read_eval_summary(d, max_rows=max_rows)
    index = {}
    idx_path = d / "pipeline_index.json"
    if idx_path.exists():
        try:
            index = json.loads(idx_path.read_text())
        except (OSError, json.JSONDecodeError):
            index = {}
    if not any(counts.values()):
        return {
            "status": "error",
            "summary": f"Campaign '{d.name}' has no design artifacts yet.",
            "data": {"campaign_dir": str(d), "stage_logs": _stage_log_tail(d)},
            "suggestions": ["The campaign may still be running, or stage 1 failed."],
        }
    return {
        "status": "success",
        "summary": (
            f"Campaign '{d.name}': {counts['generated']} generated, "
            f"{counts['resequenced']} resequenced, {counts['folded']} folded, "
            f"{n_rows} evaluated."
        ),
        "data": {
            "campaign_dir": str(d),
            "artifact_counts": counts,
            "evaluation_rows": rows,
            "n_evaluated": n_rows,
            "completed_stages": index.get("completed_stages"),
        },
        "suggestions": [],
    }
