"""Standalone GPU worker: embed protein sequences with an ESM2 model (fair-esm).

This is the ONLY module in the package that imports torch/esm, so it is
deliberately NOT registered as an agent tool and NOT listed in bootstrap.py -- the
main network/LLM-bound run never imports it. It is launched as a subprocess through
the cluster GPU reservation gate, e.g.::

    gpu_reserve run <idx> -- <python> -m enzyme_discovery_agent.tools.esm_embed_worker \
        pending.json out_vectors.npy out_meta.json

``gpu_reserve run`` sets ``CUDA_VISIBLE_DEVICES`` for us; this worker never sets it.
It uses ``fair-esm`` (``import esm``), which loads weights via torch.hub and needs
neither ``transformers`` nor ``huggingface_hub`` -- run it with a python that has
fair-esm + a CUDA torch (see config.CANDIDATE_MAP_EMBED_PYTHON).

Input  ``pending.json``  = {"model": str, "items": [{"accession","seq_sha1","sequence"}]}
Output ``out_vectors.npy`` = float32 (N, D), rows L2-normalized, aligned to items order
       ``out_meta.json``    = {"accessions":[...], "seq_sha1":[...], "model":..., "dim":...}

Exit 0 on success; non-zero on any failure (the caller treats that as "skip").
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Sequences longer than the model's positional limit are truncated.
_MAX_LEN = 1022
_BATCH = 8


def _resolve_model_name(model: str) -> str:
    """Map an HF-style id ('facebook/esm2_t33_650M_UR50D') to the fair-esm loader
    name ('esm2_t33_650M_UR50D')."""
    return model.split("/")[-1] if model else "esm2_t33_650M_UR50D"


def embed_sequences(model_name: str, sequences: list[str]):
    """Return an (N, D) float32 numpy array of mean-pooled, L2-normalized embeddings."""
    import numpy as np
    import torch
    import esm

    # Require a GPU: the whole point is to compute the ESM forward pass on the
    # reserved GPU. Fail loudly rather than silently crawling on CPU. The CPU
    # escape hatch exists only for explicit developer testing.
    allow_cpu = os.environ.get("ENZYME_AGENT_EMBED_ALLOW_CPU", "0") == "1"
    if torch.cuda.is_available():
        device = "cuda"
    elif allow_cpu:
        device = "cpu"
    else:
        raise RuntimeError(
            "CUDA is not available; ESM embedding requires a GPU. Launch this worker "
            "via `gpu_reserve run <idx> -- ...` (which sets CUDA_VISIBLE_DEVICES), or "
            "set ENZYME_AGENT_EMBED_ALLOW_CPU=1 to override for testing."
        )

    short = _resolve_model_name(model_name)
    m = re.search(r"_t(\d+)_", short)
    if not m:
        raise ValueError(f"cannot infer representation layer from model '{short}'")
    repr_layer = int(m.group(1))
    sys.stderr.write(f"esm_embed_worker: device={device} model={short} layer={repr_layer}\n")

    loader = getattr(esm.pretrained, short, None)
    if loader is None:
        raise ValueError(f"unknown fair-esm model '{short}'")
    model, alphabet = loader()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    pad_idx, cls_idx, eos_idx = alphabet.padding_idx, alphabet.cls_idx, alphabet.eos_idx

    out_rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(sequences), _BATCH):
            chunk = [s[:_MAX_LEN] for s in sequences[start:start + _BATCH]]
            data = [(f"seq{start + i}", s) for i, s in enumerate(chunk)]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)
            reps = model(tokens, repr_layers=[repr_layer])["representations"][repr_layer]
            # Mask out BOS/EOS/pad, mean-pool over real residues.
            mask = (tokens != pad_idx) & (tokens != cls_idx) & (tokens != eos_idx)
            mask = mask.unsqueeze(-1).float()
            summed = (reps * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / counts
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out_rows.append(pooled.float().cpu().numpy())
    return np.vstack(out_rows).astype("float32")


def main(argv: list[str]) -> int:
    import numpy as np

    if len(argv) != 3:
        sys.stderr.write(
            "usage: esm_embed_worker <pending.json> <out_vectors.npy> <out_meta.json>\n"
        )
        return 2
    pending_path, out_vec_path, out_meta_path = argv
    try:
        pending = json.loads(Path(pending_path).read_text())
        items = pending.get("items", [])
        model_name = pending.get("model", "facebook/esm2_t33_650M_UR50D")
        if not items:
            sys.stderr.write("no items to embed\n")
            return 1
        sequences = [it["sequence"] for it in items]
        vectors = embed_sequences(model_name, sequences)
        np.save(out_vec_path, vectors)
        Path(out_meta_path).write_text(json.dumps({
            "accessions": [it["accession"] for it in items],
            "seq_sha1": [it.get("seq_sha1", "") for it in items],
            "model": model_name,
            "dim": int(vectors.shape[1]),
        }, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - report failure via exit code
        sys.stderr.write(f"esm_embed_worker failed: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
