"""Centralized external environment and binary paths with env-var overrides.

Default paths match the established Lambda cluster layout so existing workflows
behave identically without extra configuration.
"""
from __future__ import annotations

import os
from pathlib import Path

NFS_ROOT = Path(os.environ.get("SMA_NFS_ROOT", "/nfs/lambda_stor_01/homes/rzhu"))
GPU_RESERVE_BIN = Path(os.environ.get("SMA_GPU_RESERVE", str(NFS_ROOT / "bin/gpu_reserve")))

MD_PYTHON = os.environ.get(
    "SMA_MD_PYTHON",
    str(NFS_ROOT / "Tool/envs/processivity-md-20261106/bin/python"),
)
VIZ_PYTHON = os.environ.get(
    "SMA_VIZ_PYTHON",
    "/homes/rzhu/miniforge3/envs/boltz-2/bin/python",
)
STOCK_BOLTZ_PYTHON = os.environ.get(
    "SMA_STOCK_BOLTZ_PYTHON",
    "/homes/rzhu/miniforge3/envs/boltz-2/bin/python",
)
STOCK_BOLTZ_LIB = os.environ.get(
    "SMA_STOCK_BOLTZ_LIB",
    "/homes/rzhu/miniforge3/envs/boltz-2/lib",
)
EXACT_BOLTZ_PYTHON = os.environ.get(
    "SMA_EXACT_BOLTZ_PYTHON",
    str(NFS_ROOT / "boltz2-venv/bin/python"),
)
PYMOL_BIN = os.environ.get(
    "SMA_PYMOL_BIN",
    str(NFS_ROOT / "miniforge3/envs/pymol/bin/pymol"),
)
