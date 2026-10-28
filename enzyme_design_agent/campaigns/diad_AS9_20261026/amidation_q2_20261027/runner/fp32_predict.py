#!/usr/bin/env python
"""Stock Boltz-2 prediction forced to FP32 (process-local Trainer precision override).

Matches the prior Aurora assay_activation_r1 precision regime so the refolded WT
controls and designs are internally comparable. No shared env or weight is edited.
"""
import torch
import boltz.main as bm

torch.set_float32_matmul_precision("highest")
_Orig = bm.Trainer


class _FP32Trainer(_Orig):
    def __init__(self, *args, **kwargs):
        kwargs["precision"] = 32
        super().__init__(*args, **kwargs)


bm.Trainer = _FP32Trainer
print("FP32_OVERRIDE precision=32", flush=True)
bm.cli()
