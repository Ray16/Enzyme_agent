"""Standalone GPU worker: embed protein sequences with an ESM2 model.

This is the ONLY module in the package that imports torch/transformers, so it is
deliberately NOT registered as an agent tool and NOT listed in bootstrap.py -- the
main network/LLM-bound run never imports it. It is launched as a subprocess through
the cluster GPU reservation gate, e.g.::

    gpu_reserve run <idx> -- <python> -m enzyme_discovery_agent.tools.esm_embed_worker \
        pending.json out_vectors.npy out_meta.json

``gpu_reserve run`` sets ``CUDA_VISIBLE_DEVICES`` for us; this worker never sets it.

Input  ``pending.json``  = {"model": str, "items": [{"accession","seq_sha1","sequence"}]}
Output ``out_vectors.npy`` = float32 (N, D), rows L2-normalized, aligned to items order
       ``out_meta.json``    = {"accessions":[...], "seq_sha1":[...], "model":..., "dim":...}

Exit 0 on success; non-zero on any failure (the caller treats that as "skip").
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Sequences longer than the model's positional limit are truncated. 1022 leaves
# room for the ESM CLS/EOS tokens under the common 1024 limit.
_MAX_LEN = 1022
_BATCH = 8


def embed_sequences(model_name: str, sequences: list[str]):
    """Return an (N, D) float32 numpy array of mean-pooled, L2-normalized embeddings."""
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModel

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
    sys.stderr.write(f"esm_embed_worker: device={device} model={model_name}\n")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    out_rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(sequences), _BATCH):
            batch = [s[:_MAX_LEN] for s in sequences[start:start + _BATCH]]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=_MAX_LEN + 2,
            ).to(device)
            hidden = model(**enc).last_hidden_state  # (B, L, D)
            mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / counts  # (B, D) mean over real residues
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
