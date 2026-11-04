"""Full-precision Boltz CLI with a guarded, process-local sample-memory cap."""
import hashlib
import inspect
import re
import textwrap


def capped_source(source):
    pattern = r'sample_ids\.chunk\(\s*multiplicity % max_parallel_samples \+ 1\s*\)'
    if len(re.findall(pattern, source)) != 1:
        raise RuntimeError('Unrecognized sampler implementation; preflight required')
    return re.sub(pattern, 'sample_ids.split(max_parallel_samples)', source)


def patch_sampler():
    from boltz.model.modules import diffusionv2
    source = textwrap.dedent(inspect.getsource(diffusionv2.AtomDiffusion.sample))
    fixed = capped_source(source)
    namespace = {}
    exec(compile(fixed, __file__, 'exec'), diffusionv2.__dict__, namespace)
    diffusionv2.AtomDiffusion.sample = namespace['sample']
    print('PROCESS_LOCAL_SAMPLE_CAP source_sha256='+hashlib.sha256(source.encode()).hexdigest(),flush=True)

if __name__ == '__main__':
    patch_sampler()
    import boltz.main as main
    original_trainer = main.Trainer
    def full_precision_trainer(*args, **kwargs):
        # Explicit FP32 also supports GPUs without native bfloat16 arithmetic.
        kwargs['precision'] = '32-true'
        return original_trainer(*args, **kwargs)
    main.Trainer = full_precision_trainer
    print('PREDICTION_PRECISION 32-true', flush=True)
    main.cli()
