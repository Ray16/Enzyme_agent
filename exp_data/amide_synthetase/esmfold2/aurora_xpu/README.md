# Stock ESMFold2 on Aurora (Intel PVC / torch-xpu)

Why this exists: the 49-member amide-synthetase panel at
`exp_data/archive/amide_synthetase/esmfold2/archive_5samp_20260923_074123/step1_1mg/`
was folded with the accelerated kit's `--mode fast` (bf16 re-association) rather
than the house standard `--mode exact`. Re-running it in `exact` was blocked at
every CUDA venue: Polaris is **−1,088 node hours** overdrawn, the lambda
V100/T4 fleet is sm70/75 against the kit's sm80 floor, and lambda13's NVML is
broken so the shared-GPU gate cannot verify a free card. Aurora has
**+354,751 node hours**, but the kit is CUDA-only (flash-attn,
transformer_engine, xformers, torch+cu130, an sm_90 ABI).

So this runs **stock** ESMFold2 on Aurora's Intel GPUs instead. That is not a
downgrade from `exact`: with flash-attn absent the model code takes its own
`F.scaled_dot_product_attention` path, which is the configuration the kit calls
`off` — the accuracy ceiling, at or above `exact`, just slower.

## What the port actually changes

Almost nothing, which was the point.

* `fold_esmfold2_xpu.py` resolves its device at run time (`xpu` → `cuda` →
  `cpu`) instead of hard-coding `.cuda()`. Three lines in the Polaris driver.
* It reads the panel's own ESMFold2 prediction-input JSONs and the `.a3m` each
  protein chain names, instead of a hand-built manifest + ColabFold CSV. The
  JSONs carry absolute `/grand/...` MSA paths from Polaris, which do not exist
  on Aurora, so only the basename is used and the file is looked up in
  `ESMFOLD2_MSADIR`.
* **No model-code edits at all.** `modeling_esmfold2_common.py` imports
  flash-attn inside a `try/except ImportError` and already implements the full
  SDPA fallback; `transformer_engine` is only reached for fp8.

## Environment (built 2026-10-10)

`/lus/flare/projects/FRAME-IDP/rzhu/esmfold2_xpu/`

| | |
|---|---|
| `venv/` | `python -m venv --system-site-packages` on the frameworks interpreter |
| | torch `2.13.0a0+gitcf30153` from `frameworks/2026.1.0`, `torch.xpu.is_available() True` |
| | Intel Data Center GPU Max 1550, 64.0 GiB — `device_count 1` per worker even at `accelerators:12` |
| | `esm` 3.3.0 @ `26b0bc2b` and the Biohub `transformers` 4.57.6 fork @ `ef32577f` (the two pins in the kit's `stock/PINS.json`), plus the stock `requirements.lock` minus torch/triton/flash-attn/transformer_engine/xformers/nvidia-\*/cuda-\* |
| `hf/` | 49 GB — `biohub/ESMFold2` + `biohub/ESMC-6B`, pulled straight from huggingface.co (the UAN has internet) |
| `inputs/` | the 49 `step1_*.json` + 49 `.a3m` (depths 10.7k–13.0k) |

Sanity-checked on compute node: `FLASH_ATTN_AVAILABLE False`, all imports OK.

## Running

```bash
source env.sh
export ESMFOLD2_INPUTDIR=$R/inputs ESMFOLD2_OUTPUTDIR=$R/out
export ESMFOLD2_SAMPLES=5          # matches the archived fast-mode run
python fold_esmfold2_xpu.py
```

Defaults are the stock ones: `LOOPS=20`, `STEPS=200`, `MAXMSA=1024`,
`DTYPE=float32`. `ESMFOLD2_ONLY=step1_AS9` restricts to one target. Workers
pull targets from a claim queue on Lustre and resume at per-sample granularity,
so a job that dies mid-panel can simply be resubmitted.

Output per target, `out/<stem>/`: `sample_XX.cif`, `sample_XX.json`
(`mean_plddt`, `ptm`, `iptm`, `ps_iptm`, and the loops/steps/maxmsa/dtype/device
actually used), `ranking.tsv`.

## Gotchas paid for already

* **Run on compute nodes, not a UAN.** `module load frameworks/2026.1.0` works
  on compute nodes but is a silent no-op in a non-interactive Globus Compute
  shell on a UAN — lmod reports "No modules loaded" and torch then cannot
  dlopen `libmkl_intel_lp64.so.3` or `libpti_view.so.0`.
* `ShellFunction` runs `str.format()` on the command, so every literal brace
  must be doubled — a bash `${VAR:-x}` or `{ ...; }` raises `KeyError`.
* bash caps a **single argument at 128 KB**, so base64-in-a-command cannot move
  a file of any size; send a plain Python function with `bytes` instead.
  `scripts/gc_push.py`-style chunking moved the 76 MB input archive in 19
  chunks, sha256-verified.
* Aurora's `debug` queue rejects a 2 h walltime (`Job violates queue and/or
  server resource limits`).
* The `aurora-gpu` endpoint's schema has **no `account` key** (sending one is a
  422 `SEMANTICALLY_INVALID`); `accelerators` is 1–12 **tiles**.
* Aurora mounts neither `/grand` nor `/eagle` — only `/lus/flare` and
  `/lus/gecko`. A `find` over `/lus/flare` (91 PB) times out.
