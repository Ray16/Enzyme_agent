# ESMFold2 stock-on-XPU environment (Aurora PVC). Source before fold_esmfold2_xpu.py.
#
# The venv is layered on the frameworks python with --system-site-packages, so
# torch comes from the module and only the two Biohub packages (esm 3.3.0 and
# the transformers 4.57.6 fork that carries models/esmfold2) plus the stock
# requirements come from the venv. flash-attn / transformer_engine / xformers
# are deliberately absent: the model code then takes its own
# F.scaled_dot_product_attention path, which is the numerics the accelerated
# kit calls `off` -- at or above its `exact` mode.
#
# `module load` works on compute nodes but is a silent no-op in a
# non-interactive Globus Compute shell on a UAN (empty MODULEPATH, lmod reports
# "No modules loaded" and torch then cannot dlopen MKL or libpti_view). Run the
# folding on compute nodes.
source /usr/share/lmod/lmod/init/bash
module load frameworks/2026.1.0

export R=${ESMFOLD2_ROOT:-/lus/flare/projects/FRAME-IDP/rzhu/esmfold2_xpu}
export VIRTUAL_ENV=$R/venv
export PATH=$VIRTUAL_ENV/bin:$PATH
unset PYTHONHOME

export HF_HOME=$R/hf
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export PYTHONHASHSEED=0

# One tile per worker; keep BLAS from oversubscribing the shared node.
: "${OMP_NUM_THREADS:=8}"
export OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
