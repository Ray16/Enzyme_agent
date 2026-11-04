"""Agent-invoked tool: design an enzyme DE NOVO around a ligand with DISCO.

Where this sits in the agent's toolkit
--------------------------------------
This is the second generative tool, and the complement of the motif-scaffolding
one (``alphaprotein_novo``).  AP Novo needs a 3D arrangement of catalytic
sidechain atoms -- it scaffolds a *geometry*.  DISCO needs only the ligand(s)
the pocket should complement: it jointly co-designs a protein sequence AND
backbone conditioned on a small molecule (or nucleic acid), with NO
pre-specified catalytic residues.  It therefore covers the case where the
target chemistry is known but a trustworthy catalytic geometry is not, which is
exactly where motif scaffolding cannot start.

Pipeline (DISCO: DIffusion for Sequence-structure CO-design; see
``DISCO_DIR/README.md``): a single diffusion model co-generates backbone +
sequence conditioned on the ligand(s) in one pass per (job, seed).  There is no
separate inverse-folding or refolding stage -- designability is assessed
afterwards by an external structure predictor, which this tool does NOT run.

Design notes
------------
  - The heavy torch/CUDA stack lives in its OWN uv venv outside this repo
    (``config.DISCO_PYTHON``) and is invoked as a subprocess, so a normal agent
    run never imports torch or touches a GPU.  This module is stdlib-only.
  - **GPU courtesy.** DISCO's Lightning Fabric enumerates every visible device.
    This module ALWAYS reserves one explicit device through the cluster gate
    (``gpu_reserve``) and pins ``CUDA_VISIBLE_DEVICES`` to exactly it, releasing
    on every exit path.  It never sets the variable without a reservation.
  - **Attention backend by hardware.** DeepSpeed EvoformerAttention compiles
    CUTLASS kernels that need nvcc + a CUTLASS checkout and an Ampere+ GPU.  The
    tool enables it only when ``config.DISCO_CUTLASS_PATH`` is set AND the host's
    GPUs are sm>=8.0; otherwise it runs with naive attention (correct, but more
    memory -- keep lengths modest on a 32 GB V100).
  - All numeric knobs are fallback defaults the agent overrides per call.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

from enzyme_discovery_agent import config
from enzyme_discovery_agent.tool_registry import tool


# --------------------------------------------------------------------------
# Readiness
# --------------------------------------------------------------------------
def _readiness() -> dict:
    """Inspect the DISCO install without importing or running it."""
    disco_dir = Path(config.DISCO_DIR)
    paths = {
        "disco_dir": str(disco_dir),
        "python": config.DISCO_PYTHON,
        "runner": str(disco_dir / "runner" / "inference.py"),
        "checkpoint": config.DISCO_CHECKPOINT,
        "gpu_reserve": config.CANDIDATE_MAP_GPU_RESERVE,
        "cutlass_path": config.DISCO_CUTLASS_PATH,
    }
    blocking, optional = [], []
    for key in ("python", "runner", "gpu_reserve"):
        if not Path(paths[key]).exists():
            blocking.append(f"{key} missing at {paths[key]}")
    # The checkpoint is optional: the runner pulls it from HuggingFace on first
    # use when the path is empty or absent.  Flag it only as informational.
    if config.DISCO_CHECKPOINT and not Path(config.DISCO_CHECKPOINT).exists():
        optional.append(
            f"checkpoint not on disk at {config.DISCO_CHECKPOINT} -- the runner will "
            "download it from HuggingFace (DISCO-Design/DISCO) on first use"
        )
    return {
        "ready": not blocking,
        "blocking": blocking,
        "optional": optional,
        "paths": paths,
    }


# --------------------------------------------------------------------------
# GPU reservation (acquire-and-hold one device for the single DISCO process)
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


def _acquire_gpus(n: int, label: str) -> list[str]:
    """Reserve up to ``n`` GPUs on this host, recording THIS process as owner."""
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
    """Release reservations from the owning process (no ``--force`` needed)."""
    for idx in idxs:
        try:
            subprocess.run(
                [config.CANDIDATE_MAP_GPU_RESERVE, "release", idx],
                capture_output=True, text=True, timeout=120,
            )
        except Exception:  # noqa: BLE001
            pass


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


def _fast_attention_available(cap: float | None) -> bool:
    """Would DeepSpeed EvoformerAttention run here? Needs CUTLASS + Ampere+."""
    if not config.DISCO_CUTLASS_PATH:
        return False
    if not Path(config.DISCO_CUTLASS_PATH).exists():
        return False
    return cap is not None and cap >= config.DISCO_MIN_DS_ATTENTION_CAPABILITY


def _subprocess_env(gpus: list[str], run_dir: str | None, use_fast_attention: bool) -> dict:
    """Environment for the DISCO subprocess.

    ``CUDA_VISIBLE_DEVICES`` is pinned to exactly the reserved device, a system
    CUDA on ``LD_LIBRARY_PATH`` is stripped (its cuSPARSE can shadow the venv's
    CUDA wheels and silently push the run onto CPU), the Triton autotune cache is
    moved off NFS, and CUTLASS_PATH is exported only when the fast kernel is on.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    threads = str(config.DISCO_THREADS_PER_WORKER)
    env.update({
        "OMP_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
    })
    cache_base = Path(run_dir) if run_dir else Path("/tmp")  # noqa: S108 -- fallback only
    env["TRITON_CACHE_DIR"] = str(cache_base / ".triton_cache")
    ld = env.get("LD_LIBRARY_PATH", "")
    if ld:
        kept = [p for p in ld.split(":") if p and "cuda" not in p.lower()]
        if kept:
            env["LD_LIBRARY_PATH"] = ":".join(kept)
        else:
            env.pop("LD_LIBRARY_PATH", None)
    if use_fast_attention and config.DISCO_CUTLASS_PATH:
        env["CUTLASS_PATH"] = config.DISCO_CUTLASS_PATH
    return env


# --------------------------------------------------------------------------
# Input-JSON construction
# --------------------------------------------------------------------------
def _ligand_spec(lig: dict) -> str | None:
    """Translate one ligand dict into DISCO's ligand string.

    Accepts exactly one of: ``smiles`` (raw SMILES), ``ccd_code`` (a PDB
    chemical-component code, concatenate multi-component with underscores), or
    ``sdf_path`` (a 3D structure file: SDF/MOL/MOL2/PDB).  Returns None if none
    is present.
    """
    smiles = (lig.get("smiles") or "").strip()
    ccd = (lig.get("ccd_code") or "").strip()
    sdf = (lig.get("sdf_path") or "").strip()
    if sdf:
        return "FILE_" + str(Path(sdf).expanduser().resolve())
    if ccd:
        return "CCD_" + ccd
    if smiles:
        return smiles
    return None


def _build_input_jobs(
    *, slug: str, lengths: list[int], ligand_specs: list[str]
) -> list[dict]:
    """One job per chain length; each masks a protein chain and attaches ligands."""
    jobs: list[dict] = []
    for length in lengths:
        sequences: list[dict] = [
            {"proteinChain": {"sequence": "-" * int(length), "count": 1}}
        ]
        for spec in ligand_specs:
            sequences.append({"ligand": {"ligand": spec, "count": 1}})
        jobs.append({"name": f"{slug}_len{int(length)}", "sequences": sequences})
    return jobs


# --------------------------------------------------------------------------
# Result parsing
# --------------------------------------------------------------------------
def _read_sequence(seq_path: Path) -> str:
    """First co-generated amino-acid sequence from a DISCO sequences/*.txt file."""
    if not seq_path.exists():
        return ""
    seq_lines: list[str] = []
    started = False
    for raw in seq_path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if line.startswith(">"):
            if started:  # second record begins -> stop at the first sequence
                break
            started = True
            continue
        if not started or not line or line.startswith("ligand_smiles"):
            continue
        # protein lines look like "<chain> <SEQ>" or just the raw sequence;
        # keep only residue letters.
        token = line.split()[-1]
        if token.isalpha():
            seq_lines.append(token)
    return "".join(seq_lines)


def _collect_designs(dump_dir: Path, max_rows: int = 60) -> tuple[list[dict], int, int]:
    """Gather (rows, n_designs, n_errors) from a DISCO dump directory."""
    pdbs_dir = dump_dir / "pdbs"
    seq_dir = dump_dir / "sequences"
    err_dir = dump_dir / "ERR"
    rows: list[dict] = []
    n = 0
    if pdbs_dir.is_dir():
        for pdb in sorted(pdbs_dir.glob("*_sample_*.pdb")):
            n += 1
            if len(rows) >= max_rows:
                continue
            stem = pdb.stem  # <name>_sample_<seed>
            seq = _read_sequence(seq_dir / f"{stem}.txt")
            lig_file = pdbs_dir / f"{stem}_ligands.txt"
            ligands = []
            if lig_file.exists():
                for line in lig_file.read_text(errors="replace").splitlines():
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2 and parts[0] == "ligand_smiles":
                        ligands.append(parts[1].strip())
            rows.append({
                "design": stem,
                "pdb": str(pdb),
                "sequence_length": len(seq),
                "sequence": seq,
                "ligands": ligands,
            })
    n_errors = sum(1 for _ in err_dir.glob("*.txt")) if err_dir.is_dir() else 0
    return rows, n, n_errors


def _error_tail(dump_dir: Path, n_chars: int = 800) -> dict:
    """Tail of each per-sample error file, if any -- the useful part on failure."""
    err_dir = dump_dir / "ERR"
    tails: dict[str, str] = {}
    if not err_dir.is_dir():
        return tails
    for f in sorted(err_dir.glob("*.txt")):
        try:
            text = f.read_text(errors="replace").strip()
        except OSError:
            continue
        if text:
            tails[f.stem] = text[-n_chars:]
    return tails


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
@tool(
    description=(
        "Check whether the ligand-conditioned de novo design backend (DISCO) is "
        "installed, whether its model checkpoint is present, how many GPUs are "
        "free for it, and whether this host supports the fast attention kernel. "
        "Call this before committing to a design campaign, which is a GPU job."
    ),
    category="structure",
    parallel_safe=True,
    params={},
)
def check_disco_design_readiness() -> dict:
    """Report DISCO install state, GPU availability and attention backend."""
    info = _readiness()
    free = _free_gpu_indices(config.DISCO_MAX_GPUS)
    host = socket.gethostname().split(".")[0]
    cap = _local_gpu_capability()
    fast = _fast_attention_available(cap)
    if info["ready"]:
        summary = (
            f"DISCO ready; {len(free)} GPU(s) free on {host}"
            f"{' (' + ','.join(free) + ')' if free else ''}; "
            + ("fast EvoformerAttention available"
               if fast else "naive attention (keep lengths modest)")
        )
        if info["optional"]:
            summary += f"; {len(info['optional'])} note(s)"
    else:
        summary = f"DISCO NOT ready: {'; '.join(info['blocking'])}"
    return {
        "status": "success" if info["ready"] else "error",
        "summary": summary,
        "data": {
            "ready": info["ready"],
            "blocking": info["blocking"],
            "optional_notes": info["optional"],
            "paths": info["paths"],
            "host": host,
            "free_gpus": free,
            "max_gpus": config.DISCO_MAX_GPUS,
            "local_compute_capability": cap,
            "fast_attention_available": fast,
        },
        "suggestions": (
            []
            if info["ready"]
            else ["See DISCO_DIR/README.md for install steps, or set "
                  "ENZYME_AGENT_DISCO_DIR to an existing install."]
        ),
    }


@tool(
    description=(
        "Design a protein DE NOVO around one or more ligands using DISCO: a "
        "diffusion model jointly co-generates a protein sequence AND 3D backbone "
        "whose pocket complements the given ligand(s), WITHOUT a pre-specified "
        "catalytic motif. Use this generative strategy when the target chemistry "
        "and its reactive species are known but a trustworthy catalytic geometry "
        "is NOT -- it builds a scaffold around the species the site should hold, "
        "rather than around fixed catalytic sidechain atoms (that is the other "
        "generative tool). Give the ligand(s) as SMILES, a PDB chemical-component "
        "code, or a 3D structure file. Runs on one reserved GPU; cost scales with "
        "len(lengths) x num_seeds x diffusion steps, so scout with one length and "
        "a small num_seeds before a full campaign. "
        "Outputs are PREDICTIONS from a generative model, not validated enzymes."
    ),
    category="structure",
    parallel_safe=False,
    injected_params=["run_dir"],
    params={
        "ligands": {
            "type": "array",
            "description": (
                "Ligand(s) the designed pocket should complement, as objects each "
                "with exactly one of: 'smiles' (a SMILES string; a 3D conformer is "
                "generated automatically), 'ccd_code' (a PDB chemical-component "
                "code, e.g. 'ATP'; join a multi-component ligand with underscores "
                "like 'NAG_BMA_BGC'), or 'sdf_path' (path to a 3D SDF/MOL/MOL2/PDB "
                "file containing a conformer). Give several entries for a "
                "multi-ligand site (e.g. a substrate plus a cofactor and metal). "
                "Pass an empty list for unconditional generation (no target). "
                "Example: [{'ccd_code':'ATP'},{'ccd_code':'MG'},{'smiles':'<substrate>'}]."
            ),
            "items": {"type": "object"},
            "default": [],
        },
        "lengths": {
            "type": "array",
            "description": (
                "Chain lengths (residues) to generate, one design job per length. "
                "Longer chains need more memory under naive attention."
            ),
            "items": {"type": "integer"},
            "default": list(config.DISCO_DEFAULT_LENGTHS),
        },
        "num_seeds": {
            "type": "integer",
            "description": (
                "Random seeds per job; each seed yields one independent design. "
                "Total designs = len(lengths) x num_seeds."
            ),
            "default": config.DISCO_NUM_SEEDS,
        },
        "experiment": {
            "type": "string",
            "description": (
                "Sampling preset. 'diverse' samples more freely (the paper's "
                "ligand-conditioned setting); 'designable' steers toward designs "
                "that refold correctly, at the cost of structural variety."
            ),
            "enum": ["diverse", "designable"],
            "default": config.DISCO_EXPERIMENT,
        },
        "effort": {
            "type": "string",
            "description": (
                "Compute preset: 'max' (200 diffusion steps, 4 cycles -- full "
                "quality, recommended for ligand-conditioned design) or 'fast' "
                "(100 steps, 2 cycles -- ~4x faster, for prototyping/screening)."
            ),
            "enum": ["max", "fast"],
            "default": config.DISCO_EFFORT,
        },
        "output_format": {
            "type": "string",
            "description": (
                "Leave empty to use DISCO's default (protein backbone as PDB plus a "
                "ligand-SMILES sidecar). Set 'complex_cif' to instead write the full "
                "co-designed complex (protein + ligand) as a CIF."
            ),
            "enum": ["", "complex_cif"],
            "default": "",
        },
        "campaign_name": {
            "type": "string",
            "description": (
                "Short slug naming this campaign; also the output subdirectory and "
                "job-name prefix. Re-using a name RESUMES: DISCO skips (job, seed) "
                "samples already on disk."
            ),
            "default": "disco",
        },
        "timeout_s": {
            "type": "integer",
            "description": "Wall-clock budget for the whole campaign, in seconds.",
            "default": config.DISCO_TIMEOUT,
        },
        "run_dir": {"type": "string", "description": "Agent run directory (injected)."},
    },
)
def design_enzyme_with_disco(
    ligands: list[dict] | None = None,
    lengths: list[int] | None = None,
    num_seeds: int = config.DISCO_NUM_SEEDS,
    experiment: str = config.DISCO_EXPERIMENT,
    effort: str = config.DISCO_EFFORT,
    output_format: str = "",
    campaign_name: str = "disco",
    timeout_s: int = config.DISCO_TIMEOUT,
    run_dir: str | None = None,
) -> dict:
    """Run a DISCO ligand-conditioned design campaign on one reserved GPU."""
    ligands = list(ligands or [])
    lengths = [int(x) for x in (lengths or config.DISCO_DEFAULT_LENGTHS)]

    info = _readiness()
    if not info["ready"]:
        return {
            "status": "error",
            "summary": f"DISCO backend unavailable: {'; '.join(info['blocking'])}",
            "data": None,
            "suggestions": ["Call check_disco_design_readiness for the full install state."],
        }
    if not lengths:
        return {
            "status": "error",
            "summary": "No chain lengths requested -- nothing to generate.",
            "data": None,
            "suggestions": ["Pass lengths, e.g. [150, 200, 250]."],
        }

    ligand_specs: list[str] = []
    for i, lig in enumerate(ligands):
        spec = _ligand_spec(lig)
        if spec is None:
            return {
                "status": "error",
                "summary": f"Ligand #{i + 1} has none of 'smiles', 'ccd_code', 'sdf_path'.",
                "data": {"ligand": lig},
                "suggestions": ["Give each ligand exactly one of those keys."],
            }
        if spec.startswith("FILE_") and not Path(spec[5:]).exists():
            return {
                "status": "error",
                "summary": f"Ligand #{i + 1} sdf_path not found: {spec[5:]}",
                "data": None,
                "suggestions": ["The file must contain a 3D conformer (2D is rejected)."],
            }
        ligand_specs.append(spec)

    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in campaign_name) or "disco"
    base = Path(run_dir) / "artifacts" / "disco_design" if run_dir else Path.cwd() / "disco_design"
    campaign_dir = base / slug
    dump_dir = campaign_dir / "output"
    campaign_dir.mkdir(parents=True, exist_ok=True)
    dump_dir.mkdir(parents=True, exist_ok=True)

    jobs = _build_input_jobs(slug=slug, lengths=lengths, ligand_specs=ligand_specs)
    input_json = campaign_dir / "input.json"
    input_json.write_text(json.dumps(jobs, indent=2))

    cap = _local_gpu_capability()
    use_fast = _fast_attention_available(cap)

    gpus = _acquire_gpus(1, label=f"disco_{slug}")
    if not gpus:
        return {
            "status": "error",
            "summary": (
                f"No free GPU on {socket.gethostname().split('.')[0]} -- campaign not "
                "started (the reservation gate found every device busy or reserved)."
            ),
            "data": {"input_json": str(input_json), "campaign_dir": str(campaign_dir)},
            "suggestions": [
                "Retry later, or check `gpu_reserve list` for a host with free devices.",
                "The input JSON is already written, so the retry costs nothing extra.",
            ],
        }

    seeds_literal = "[" + ",".join(str(s) for s in range(int(num_seeds))) + "]"
    cmd = [
        config.DISCO_PYTHON,
        str(Path(config.DISCO_DIR) / "runner" / "inference.py"),
        f"experiment={experiment}",
        f"effort={effort}",
        f"input_json_path={input_json}",
        f"dump_dir={dump_dir}",
        f"seeds={seeds_literal}",
        f"use_deepspeed_evo_attention={'true' if use_fast else 'false'}",
    ]
    if config.DISCO_CHECKPOINT and Path(config.DISCO_CHECKPOINT).exists():
        cmd.append(f"load_checkpoint_path={config.DISCO_CHECKPOINT}")
    if output_format == "complex_cif":
        # null disables the protein-only PDB saver, so the dumper writes the full
        # co-designed complex (protein + ligand) as a CIF instead.
        cmd.append("output_format=null")

    env = _subprocess_env(gpus, run_dir, use_fast)
    started = time.time()
    rc, stdout, stderr, timed_out = 0, "", "", False
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
            env=env, cwd=config.DISCO_DIR,
        )
        rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc, timed_out = None, True
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    except Exception as exc:  # noqa: BLE001 -- `finally` does the release
        _release_gpus(gpus)
        return {
            "status": "error",
            "summary": f"Failed to launch DISCO: {exc}",
            "data": {"campaign_dir": str(campaign_dir), "cmd": cmd},
            "suggestions": ["Check that DISCO_PYTHON points at the DISCO venv."],
        }
    finally:
        _release_gpus(gpus)
    elapsed = time.time() - started

    rows, n_designs, n_errors = _collect_designs(dump_dir)
    data = {
        "campaign_dir": str(campaign_dir),
        "dump_dir": str(dump_dir),
        "input_json": str(input_json),
        "n_designs": n_designs,
        "n_errors": n_errors,
        "designs": rows,
        "elapsed_s": round(elapsed, 1),
        "gpu_used": gpus[0],
        "host": socket.gethostname().split(".")[0],
        "requested": {
            "lengths": lengths,
            "num_seeds": num_seeds,
            "experiment": experiment,
            "effort": effort,
            "n_ligands": len(ligand_specs),
            "fast_attention": use_fast,
            "output_format": output_format or "default_pdb",
        },
    }
    if n_designs:
        data["seconds_per_design"] = round(elapsed / n_designs, 1)

    if timed_out:
        data["errors"] = _error_tail(dump_dir)
        return {
            "status": "error",
            "summary": (
                f"DISCO campaign '{slug}' hit the {timeout_s}s budget after "
                f"{elapsed / 60:.0f} min; {n_designs} design(s) generated "
                "(partial results kept on disk)."
            ),
            "data": data,
            "suggestions": [
                "Re-call with the same campaign_name to RESUME (finished samples are skipped).",
                "Or lower num_seeds / len(lengths) / effort.",
            ],
        }
    if rc != 0:
        data["errors"] = _error_tail(dump_dir)
        data["stderr_tail"] = (stderr or "").strip()[-1500:]
        return {
            "status": "error",
            "summary": f"DISCO exited rc={rc} for campaign '{slug}' ({n_designs} design(s) on disk).",
            "data": data,
            "suggestions": [
                "Inspect data.stderr_tail and data.errors for the failing sample.",
                "On a non-Ampere GPU, memory pressure under naive attention is the "
                "usual cause -- lower the chain lengths.",
            ],
        }
    if not n_designs:
        data["errors"] = _error_tail(dump_dir)
        return {
            "status": "error",
            "summary": f"DISCO reported success but produced no designs for '{slug}'.",
            "data": data,
            "suggestions": ["Inspect data.errors; the ligand spec may have been rejected."],
        }

    return {
        "status": "success",
        "summary": (
            f"DISCO campaign '{slug}': {n_designs} design(s) across {len(lengths)} "
            f"length(s) x {num_seeds} seed(s) in {elapsed / 60:.1f} min on 1 GPU"
            + (f" ({n_errors} sample error(s))" if n_errors else "")
            + "."
        ),
        "data": data,
        "suggestions": [
            "Assess designability by refolding each sequence with an external "
            "structure predictor; DISCO does not refold its own outputs.",
            "Score with the structural-modeling proxies before proposing any for "
            "synthesis. These are generative-model PREDICTIONS, not measured activity.",
        ],
    }


@tool(
    description=(
        "Summarize an existing DISCO design campaign on disk: how many designs and "
        "per-sample errors it produced, and the per-design sequence/structure table. "
        "Use this to analyze a campaign as its outputs land, without a GPU."
    ),
    category="structure",
    parallel_safe=True,
    params={
        "campaign_dir": {
            "type": "string",
            "description": "Campaign directory (the campaign_dir of a previous DISCO call).",
        },
        "max_rows": {
            "type": "integer",
            "description": "Maximum per-design rows to return.",
            "default": 60,
        },
    },
)
def summarize_disco_campaign(campaign_dir: str, max_rows: int = 60) -> dict:
    """Read a DISCO campaign's designs and errors without touching a GPU."""
    d = Path(campaign_dir).expanduser()
    dump_dir = d / "output" if (d / "output").is_dir() else d
    if not dump_dir.is_dir():
        return {
            "status": "error",
            "summary": f"No such campaign directory: {d}",
            "data": None,
            "suggestions": ["Pass the campaign_dir returned by design_enzyme_with_disco."],
        }
    rows, n_designs, n_errors = _collect_designs(dump_dir, max_rows=max_rows)
    if not n_designs:
        return {
            "status": "error",
            "summary": f"Campaign '{d.name}' has no designs yet.",
            "data": {"campaign_dir": str(d), "errors": _error_tail(dump_dir)},
            "suggestions": ["The campaign may still be running, or every sample errored."],
        }
    return {
        "status": "success",
        "summary": f"Campaign '{d.name}': {n_designs} design(s), {n_errors} sample error(s).",
        "data": {
            "campaign_dir": str(d),
            "dump_dir": str(dump_dir),
            "n_designs": n_designs,
            "n_errors": n_errors,
            "designs": rows,
        },
        "suggestions": [],
    }
