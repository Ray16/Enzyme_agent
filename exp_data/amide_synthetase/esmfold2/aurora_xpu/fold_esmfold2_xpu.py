#!/usr/bin/env python
"""Stock ESMFold2 cofolding on Intel GPUs (Aurora PVC / torch-xpu).

Device-agnostic port of the Polaris CUDA driver. Three things differ from it:

  * the device is chosen at run time (`xpu` -> `cuda` -> `cpu`), so the same file
    runs on Aurora and on an A100/H100 host;
  * inputs are read in the ESMFold2 *prediction-input JSON* format (the panel's
    own `step1_<target>.json`), MSAs from the `.a3m` each chain names, rather
    than from a hand-built manifest + ColabFold CSV;
  * nothing else -- no flash-attn, no transformer_engine, no fp8. The stock
    model code falls back to `F.scaled_dot_product_attention` by itself when
    flash-attn is absent, which is the numerics the kit calls `off`: the
    accuracy ceiling, at or above its `exact` mode.

One worker owns one GPU and loads the 6B LM once, then pulls the next unclaimed
target from a shared claim queue on the parallel filesystem. Resumable at
per-sample granularity.

Output per target, OUTPUTDIR/<stem>/: sample_XX.cif  sample_XX.json  ranking.tsv
"""
import glob
import json
import os
import random
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTDIR = os.environ.get("ESMFOLD2_INPUTDIR", os.path.join(HERE, "inputs"))
MSADIR = os.environ.get("ESMFOLD2_MSADIR", INPUTDIR)
OUTPUTDIR = os.environ.get("ESMFOLD2_OUTPUTDIR", os.path.join(HERE, "out"))
MODEL_ID = os.environ.get("ESMFOLD2_MODEL", "biohub/ESMFold2")
# Pin the checkpoint. `main` has moved ahead of the transformers 4.57.6 fork the
# model code comes from, and loading it dies in the config with
# `DiffusionStructureHeadConfig.__init__() got an unexpected keyword argument
# 'architectures'`. This is the snapshot in the kit's stock/PINS.json.
REVISION = os.environ.get("ESMFOLD2_REVISION", "1ebf0e3481a5184eb6171d40615c79e384b48796")
SAMPLES = int(os.environ.get("ESMFOLD2_SAMPLES", "5"))
SPC = int(os.environ.get("ESMFOLD2_SPC", "5"))
LOOPS = int(os.environ.get("ESMFOLD2_LOOPS", "20"))
STEPS = int(os.environ.get("ESMFOLD2_STEPS", "200"))
MAXMSA = int(os.environ.get("ESMFOLD2_MAXMSA", "1024"))
DTYPE = os.environ.get("ESMFOLD2_DTYPE", "float32")
DEVICE = os.environ.get("ESMFOLD2_DEVICE", "auto")
SEED0 = int(os.environ.get("ESMFOLD2_SEED", "1"))
STALE_S = float(os.environ.get("ESMFOLD2_STALE", "10800"))
ONLY = os.environ.get("ESMFOLD2_ONLY", "")  # comma-separated stems, "" = all


def rank_size():
    for r, s in (("PALS_RANKID", "PALS_NRANKS"), ("PMI_RANK", "PMI_SIZE"),
                 ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE")):
        if r in os.environ and s in os.environ:
            return int(os.environ[r]), int(os.environ[s])
    return 0, 1


def pick_device(torch):
    """Resolve ESMFOLD2_DEVICE. `auto` prefers an Intel GPU, then CUDA, then CPU."""
    if DEVICE != "auto":
        return DEVICE
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def dev_mem_gb(torch, dev):
    try:
        mod = getattr(torch, dev.split(":")[0], None)
        return round(mod.memory_allocated() / 1e9, 1) if mod else 0.0
    except Exception:
        return 0.0


def load_inputs():
    """Read every ESMFold2 prediction-input JSON in INPUTDIR.

    A file holds either one input dict or a list of them. Each protein chain's
    `msa` is resolved to MSADIR/<basename> -- the panel's JSONs carry absolute
    paths from the machine they were written on, which do not exist here.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(INPUTDIR, "*.json"))):
        try:
            doc = json.load(open(path))
        except Exception:
            continue
        for item in (doc if isinstance(doc, list) else [doc]):
            if not isinstance(item, dict) or "sequences" not in item:
                continue
            item["_stem"] = item.get("id") or os.path.splitext(os.path.basename(path))[0]
            rows.append(item)
    if ONLY:
        keep = {s.strip() for s in ONLY.split(",") if s.strip()}
        rows = [r for r in rows if r["_stem"] in keep]
    return rows


def resolve_msa(spec):
    """Map a JSON `msa` field to a readable a3m path in MSADIR, or None."""
    if not spec:
        return None
    local = os.path.join(MSADIR, os.path.basename(spec))
    if os.path.exists(local):
        return local
    return spec if os.path.exists(spec) else None


def build_spi(row, mods):
    ProteinInput, LigandInput, StructurePredictionInput, MSA = mods
    chains, used_msa = [], False
    for seq in row["sequences"]:
        t = seq.get("type")
        if t == "protein":
            a3m = resolve_msa(seq.get("msa"))
            msa = None
            if a3m:
                msa = MSA.from_a3m(a3m, remove_insertions=True, max_sequences=MAXMSA)
                used_msa = True
            chains.append(ProteinInput(id=seq["id"], sequence=seq["sequence"], msa=msa))
        elif t == "ligand":
            if seq.get("ccd"):
                ccd = seq["ccd"]
                chains.append(LigandInput(id=seq["id"],
                                          ccd=ccd if isinstance(ccd, list) else [ccd]))
            else:
                chains.append(LigandInput(id=seq["id"], smiles=seq["smiles"]))
        else:
            raise ValueError(f"unsupported chain type {t!r} in {row['_stem']}")
    return StructurePredictionInput(sequences=chains), used_msa


def count_done(outdir):
    return sum(1 for p in glob.glob(os.path.join(outdir, "sample_*.cif"))
               if os.path.getsize(p) > 0)


def pair_iptm(r, i=0, j=1):
    """Chain-pair ipTM, read symmetric-min. The matrix is strongly asymmetric."""
    p = getattr(r, "pair_chains_iptm", None)
    if p is None:
        return None
    try:
        if hasattr(p, "detach"):
            p = p.detach().cpu().tolist()
        elif hasattr(p, "tolist"):
            p = p.tolist()
        return round(float(min(p[i][j], p[j][i])), 4)
    except Exception:
        return None


def try_claim(outdir):
    lock = os.path.join(outdir, ".claim")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.uname()[1]}:{os.getpid()}:{time.time()}\n".encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            if (time.time() - os.path.getmtime(lock) > STALE_S
                    and count_done(outdir) < SAMPLES):
                os.utime(lock, None)
                return True
        except FileNotFoundError:
            return try_claim(outdir)
        return False


def fold_target(row, outdir, model, builder, torch, log, mods):
    stem = row["_stem"]
    spi, used_msa = build_spi(row, mods)
    seqlen = sum(len(s["sequence"]) for s in row["sequences"] if s.get("type") == "protein")
    produced = count_done(outdir)
    call = produced // SPC
    while produced < SAMPLES:
        n = min(SPC, SAMPLES - produced)
        seed = SEED0 + call
        try:
            ts = time.time()
            with torch.no_grad():
                res = builder.fold(model, spi, num_loops=LOOPS,
                                   num_sampling_steps=STEPS,
                                   num_diffusion_samples=n, seed=seed,
                                   msa_max_depth=MAXMSA)
            res = res if isinstance(res, list) else [res]
            dt = time.time() - ts
            for k, r in enumerate(res):
                idx = produced + k
                cif = os.path.join(outdir, f"sample_{idx:02d}.cif")
                tmp = cif + ".tmp"
                with open(tmp, "w") as w:
                    w.write(r.complex.to_mmcif())
                os.replace(tmp, cif)
                mp = float(r.plddt.mean())
                if mp <= 1.0:
                    mp *= 100.0
                with open(os.path.join(outdir, f"sample_{idx:02d}.json"), "w") as w:
                    json.dump({"stem": stem, "sample": idx, "seed": seed,
                               "used_msa": used_msa, "seqlen": seqlen,
                               "mean_plddt": round(mp, 3),
                               "ptm": (round(float(r.ptm), 4) if r.ptm is not None else None),
                               "iptm": (round(float(r.iptm), 4) if r.iptm is not None else None),
                               "ps_iptm": pair_iptm(r),
                               "revision": REVISION,
                               "loops": LOOPS, "steps": STEPS, "maxmsa": MAXMSA,
                               "dtype": DTYPE, "device": str(model.device),
                               "runtime_s": round(dt / len(res), 2)}, w)
            produced += len(res)
            try:
                os.utime(os.path.join(outdir, ".claim"), None)
            except OSError:
                pass
            log(f"OK {stem} +{len(res)} -> {produced}/{SAMPLES} "
                f"ps_iptm={pair_iptm(res[0])} iptm={res[0].iptm} ptm={res[0].ptm} {dt:.1f}s")
        except Exception:
            log(f"FAIL {stem} call{call} n={n}\n{traceback.format_exc()}")
            break
        call += 1

    recs = []
    for js in glob.glob(os.path.join(outdir, "sample_*.json")):
        try:
            recs.append(json.load(open(js)))
        except Exception:
            pass
    if recs:
        recs.sort(key=lambda d: (-(d.get("ps_iptm") or -1), -(d.get("iptm") or -1),
                                 -(d.get("ptm") or -1)))
        with open(os.path.join(outdir, "ranking.tsv"), "w") as w:
            w.write("rank\tsample\tps_iptm\tiptm\tptm\tmean_plddt\n")
            for i, d in enumerate(recs):
                w.write(f"{i}\tsample_{d['sample']:02d}\t{d.get('ps_iptm')}\t"
                        f"{d.get('iptm')}\t{d.get('ptm')}\t{d.get('mean_plddt')}\n")


def main():
    rank, size = rank_size()
    log = lambda m: print(f"[rank {rank}/{size} {os.uname()[1]}] {m}", flush=True)

    rows = load_inputs()
    os.makedirs(OUTPUTDIR, exist_ok=True)
    t0 = time.time()

    # A worker whose GPU is unusable must not abort the job: it bows out with
    # exit 0 and the survivors cover every target through the claim queue.
    try:
        import torch
        from esm.models.esmfold2 import ESMFold2InputBuilder
        from esm.utils.msa import MSA
        from esm.utils.structure.input_builder import (
            LigandInput, ProteinInput, StructurePredictionInput)
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
        mods = (ProteinInput, LigandInput, StructurePredictionInput, MSA)

        dev = pick_device(torch)
        dtype = getattr(torch, DTYPE)
        log(f"loading {MODEL_ID}@{REVISION[:8]} device={dev} dtype={DTYPE} "
            f"targets={len(rows)} samples={SAMPLES} spc={SPC} loops={LOOPS} "
            f"steps={STEPS} maxmsa={MAXMSA}")
        model = ESMFold2Model.from_pretrained(
            MODEL_ID, revision=REVISION, dtype=dtype).to(dev).eval()
        _ = (torch.zeros(1, device=dev) + 1).cpu()   # prove the device really works
        builder = ESMFold2InputBuilder()
        log(f"model loaded in {time.time()-t0:.1f}s ({dev} mem {dev_mem_gb(torch, dev)} GB)")
    except Exception:
        log(f"DEVICE INIT FAILED -- bowing out:\n{traceback.format_exc()}")
        return

    order = list(rows)
    random.Random(rank).shuffle(order)
    progressed = True
    while progressed:
        progressed = False
        for row in order:
            od = os.path.join(OUTPUTDIR, row["_stem"])
            if count_done(od) >= SAMPLES:
                continue
            os.makedirs(od, exist_ok=True)
            if not try_claim(od):
                continue
            progressed = True
            fold_target(row, od, model, builder, torch, log, mods)

    log(f"finished wall={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
