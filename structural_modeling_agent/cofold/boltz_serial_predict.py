"""Explicit FP32 Boltz-2 prediction with guarded process-local sample batching.

For GPUs without BF16 support. This is a separately recorded numerical profile,
not a claim of byte identity to stock mixed-precision predictions. No shared
environment files are modified. Invoke only under an owned GPU reservation.
"""
import hashlib
import inspect
import json
import os
import re
import sys
import textwrap


def serial_sampler_source(source):
    pattern=r'sample_ids\.chunk\(\s*multiplicity % max_parallel_samples \+ 1\s*\)'
    if len(re.findall(pattern,source))==1:
        return re.sub(pattern,'sample_ids.split(max_parallel_samples)',source),True
    if 'sample_ids.split(max_parallel_samples)' in source:
        return source,False
    raise RuntimeError('Unrecognized diffusion batching implementation; no patch applied')


def main():
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('A GPU reservation must own this process')
    import boltz.main as engine
    from boltz.model.modules import diffusionv2
    source=textwrap.dedent(inspect.getsource(diffusionv2.AtomDiffusion.sample))
    fixed,changed=serial_sampler_source(source)
    if changed:
        namespace={}
        exec(compile(fixed,__file__,'exec'),diffusionv2.__dict__,namespace)
        diffusionv2.AtomDiffusion.sample=namespace['sample']
    trainer=engine.Trainer
    def fp32_trainer(*args,**kwargs):
        if kwargs.get('precision') not in (32,'32-true','bf16-mixed'):
            raise RuntimeError('Unrecognized requested precision')
        kwargs['precision']=32
        return trainer(*args,**kwargs)
    engine.Trainer=fp32_trainer
    print(json.dumps({'producer':'boltz_serial_fp32_v1','precision':32,
        'sample_batch_patch':changed,'original_sampler_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'executed_sampler_sha256':hashlib.sha256(fixed.encode()).hexdigest()}),flush=True)
    engine.cli()


if __name__=='__main__':
    main()
