"""Lazy ASE adapters; model environments and checkpoint files are supplied explicitly.

These adapters require a per-environment import/force smoke test before use.
Keep each model stack isolated; do not upgrade the existing UMA environment.
"""
from pathlib import Path


def bind_electronic_state(atoms, charge, multiplicity):
    """Bind explicit state in both molecular-model metadata conventions.

    UMA/Orb use ``spin`` as multiplicity; SO3LR's ASE reader uses
    ``multiplicity`` and converts it to the number of unpaired electrons.
    Never pass a number of unpaired electrons as the shared ``spin`` field.
    """
    if type(charge) is not int or type(multiplicity) is not int or multiplicity < 1:
        raise ValueError("Require integer charge and positive spin multiplicity")
    electrons = int(atoms.numbers.sum()) - charge
    if electrons < multiplicity - 1 or electrons <= 0 or (electrons - multiplicity + 1) % 2:
        raise ValueError("Charge/spin electron count is inconsistent")
    for key, value in (("charge", charge), ("spin", multiplicity),
                       ("multiplicity", multiplicity)):
        if key in atoms.info and atoms.info[key] != value:
            raise ValueError(f"Structure {key} disagrees with configuration")
        atoms.info[key] = value


def orbmol_v2(checkpoint, device="cuda", precision="float32-highest", compile=False):
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator

    model, adapter = pretrained.orbmol_v2(
        weights_path=str(Path(checkpoint).resolve(strict=True)), device=device,
        precision=precision, compile=compile,
    )
    return ORBCalculator(model, atoms_adapter=adapter, device=device)


def so3lr(model_directory, precision="float64", lr_cutoff_A=12.0):
    import jax
    import numpy as np
    from so3lr import So3lrCalculator

    if precision not in ("float32", "float64"):
        raise ValueError("SO3LR precision must be float32 or float64")
    jax.config.update("jax_enable_x64", precision == "float64")
    if jax.default_backend() != "gpu":
        raise RuntimeError("SO3LR GPU pilot requires a working JAX GPU backend")
    # Explicit neighbor-list ASE calculator. Do not assume native flash MD timings apply.
    return So3lrCalculator(model=str(Path(model_directory).resolve(strict=True)),
                           dtype=getattr(np, precision), lr_cutoff=lr_cutoff_A)


def uma(checkpoint, charge, multiplicity, reference_yaml, device="cuda", precision="float32"):
    import inspect
    import torch
    from omegaconf import OmegaConf
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    # An explicit object works across the old default/turbo and newer Nitro-era
    # setting names. General FP32 is the ordinary portable reference path.
    if precision not in ("float32", "float64"):
        raise ValueError("Select an explicit supported ordinary UMA precision")
    options = dict(tf32=False, activation_checkpointing=True, merge_mole=False,
                   compile=False, base_precision_dtype=getattr(torch, precision),
                   execution_mode="general", torch_num_threads=1)
    if "nitro" in inspect.signature(InferenceSettings).parameters:
        options["nitro"] = False
    settings = InferenceSettings(**options)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    predictor = pretrained_mlip.load_predict_unit(
        str(Path(checkpoint).resolve(strict=True)), device=device, inference_settings=settings,
        atom_refs=OmegaConf.load(Path(reference_yaml).resolve(strict=True)),
    )
    # Molecular charge/spin live in atoms.info, validated by the pilot; no implicit defaults.
    if type(charge) is not int or type(multiplicity) is not int or multiplicity < 1:
        raise ValueError("UMA requires integer charge and spin multiplicity")
    calc = FAIRChemCalculator(predictor, task_name="omol")
    calc.sampling_settings = {k: sorted(v) if isinstance(v, set) else v
                              for k, v in settings.to_omegaconf().items()}
    return calc
