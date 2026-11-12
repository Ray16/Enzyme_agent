"""One explicit UMA numerical profile, loaded only by compute operators."""
from functools import lru_cache
import os

KCAL_MOL_PER_EV = 23.06054783061903
PROFILE = 'explicit_ieee_fp32_general_no_compile'


@lru_cache(maxsize=4)
def _predictor(model,device):
    import torch
    from fairchem.core import pretrained_mlip
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','1')))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    settings=InferenceSettings(tf32=False,compile=False,merge_mole=False,
        activation_checkpointing=False,execution_mode='general',nitro=False,
        torch_num_threads=None,base_precision_dtype=torch.float32)
    return pretrained_mlip.get_predict_unit(model,device=device,inference_settings=settings)


def get_calculator(task_name='omol',model='uma-s-1p2',*,device='cuda',inference_settings=PROFILE):
    """Historical default/batch labels map to the explicit validated profile.

    Return a fresh ASE calculator around cached weights. There is no silent
    device fallback or dependence on changing upstream named presets.
    """
    if inference_settings not in {PROFILE,'default','batch'}:
        raise ValueError('Unsupported UMA numerical profile; validate an explicit alternative first')
    from fairchem.core import FAIRChemCalculator
    return FAIRChemCalculator(_predictor(model,device),task_name=task_name)
