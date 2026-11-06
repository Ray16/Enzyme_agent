"""Instrument the existing FP32 CUDA compatibility profile without changing weights."""
from pathlib import Path
import json
import os
import sys
import time

STARTED = time.time()
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'model_source/src'))
import legacy_gpu
legacy_gpu.REPO = HERE/'model_source'
legacy_gpu.prepare_environment()

import torch
import rfproteina.generate as generation
from lightning.pytorch.callbacks import Callback

TIMING = {'process_started':STARTED}
ORIGINAL_TRAINER = generation.L.Trainer
ORIGINAL_RUN = generation.run_generation


class TimingAudit(Callback):
    def on_validation_batch_start(self,trainer,model,batch,batch_idx,dataloader_idx=0):
        p = next(model.parameters())
        if p.device.type!='cuda' or p.dtype!=torch.float32:
            raise RuntimeError('Expected FP32 CUDA execution')
        TIMING.setdefault('first_generation_batch_started',time.time())
        TIMING['actual_device'] = str(p.device)
        TIMING['actual_dtype'] = str(p.dtype)
        print('RFD4_ACTUAL_CUDA_GENERATION',p.device,p.dtype,flush=True)


def trainer(*args,**kwargs):
    kwargs['callbacks'] = [*kwargs.get('callbacks',[]),TimingAudit()]
    return ORIGINAL_TRAINER(*args,**kwargs)


def run(cfg):
    TIMING['generation_entrypoint_started'] = time.time()
    try:
        return ORIGINAL_RUN(cfg)
    finally:
        TIMING['generation_entrypoint_finished'] = time.time()
        TIMING['peak_allocated_device_bytes'] = int(torch.cuda.max_memory_allocated())
        TIMING['peak_reserved_device_bytes'] = int(torch.cuda.max_memory_reserved())
        TIMING['cpu_affinity'] = sorted(os.sched_getaffinity(0))
        TIMING['thread_caps'] = {k:os.environ.get(k) for k in
            ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}
        TIMING['batch_size'] = 1
        TIMING['requested_samples'] = int(cfg.num_replicates)
        (Path(cfg.out_dir)/'timing.json').write_text(json.dumps(TIMING,indent=2)+'\n')


generation.L.Trainer = trainer
generation.run_generation = run
legacy_gpu.main()
