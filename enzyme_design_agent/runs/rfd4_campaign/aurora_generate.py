"""RFD4 FP32 native inference on one scheduler-assigned Aurora Intel XPU."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
from run_logging import summarize_run

PROCESS_STARTED = time.time()

ROOT=Path('/lus/flare/projects/FRAME-IDP/rzhu/rfd4_proteina_20261026')
REPO=ROOT/'source/RFD4-Proteina'


def install_xpu_accelerator(torch, generation, execution):
    from lightning.pytorch.accelerators import Accelerator
    from lightning.pytorch.callbacks import Callback
    from lightning.pytorch.strategies import SingleDeviceStrategy

    class XPUAccelerator(Accelerator):
        def setup_device(self,device):
            if device.type!='xpu':raise RuntimeError('Expected an Intel XPU')
            torch.xpu.set_device(device)
        def get_device_stats(self,device):return torch.xpu.memory_stats(device)
        def teardown(self):torch.xpu.empty_cache()
        @staticmethod
        def parse_devices(devices):
            if devices!=1:raise RuntimeError('Smoke test owns exactly one XPU')
            return 1
        @staticmethod
        def get_parallel_devices(devices):return [torch.device('xpu',0)]
        @staticmethod
        def auto_device_count():return 1
        @staticmethod
        def is_available():return torch.xpu.is_available()
        @staticmethod
        def name():return 'xpu'
        @classmethod
        def register_accelerators(cls,registry):registry.register('xpu',cls)

    class DeviceAudit(Callback):
        def on_validation_batch_start(self,trainer,model,batch,batch_idx,dataloader_idx=0):
            p=next(model.parameters())
            if p.device.type!='xpu' or p.dtype!=torch.float32:
                raise RuntimeError('Generation did not execute with an FP32 XPU model')
            execution['actual_validation_device']=str(p.device)
            execution['actual_model_dtype']=str(p.dtype)
            execution.setdefault('sampling_started', time.time())
            print('RFD4_ACTUAL_XPU_GENERATION',str(p.device),str(p.dtype),flush=True)

    original=generation.L.Trainer
    def trainer(*args,**kwargs):
        if kwargs.get('precision')!='32-true':raise RuntimeError('Expected FP32')
        accelerator=XPUAccelerator()
        kwargs.update(accelerator=accelerator,devices=1,
            strategy=SingleDeviceStrategy(device=torch.device('xpu',0)))
        kwargs['callbacks']=[*kwargs.get('callbacks',[]),DeviceAudit()]
        return original(*args,**kwargs)
    generation.L.Trainer=trainer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-check',action='store_true')
    parser.add_argument('--input',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--batch',type=int,choices=range(1,7),default=6)
    parser.add_argument('--replicates',type=int,required=True)
    parser.add_argument('--seed',type=int,required=True)
    options=parser.parse_args()
    if options.replicates<1:raise ValueError('replicates must be positive')
    for key in ('DISABLE_CUEQUIVARIANCE','TORCH_COMPILE_DISABLE','TORCHDYNAMO_DISABLE'):
        os.environ[key]='1'
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['WANDB_MODE']='disabled'
    os.environ['DATA_PATH']=str(ROOT/'unused_native_checkpoints')
    sys.path.insert(0,str(REPO/'src'))
    os.chdir(REPO)
    import torch
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    from rfproteina.nn import local_latents_transformer as trunk
    from rfproteina.nn.modules import norm
    from rfproteina.utils.hydra import register_resolvers
    import rfproteina.generate as generation
    def math_attention():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    trunk._configure_flash_attention_backends=math_attention
    norm._compiled_rms_norm=norm._rms_norm_preserve_dtype
    math_attention()
    register_resolvers()
    source=options.input or REPO/'data/benchmarks/monomer/monomer-short.cif.zst'
    output=options.out.resolve()
    if output.exists() and any(output.iterdir()):raise RuntimeError('Preserve previous attempt; use a fresh output path')
    output.mkdir(parents=True,exist_ok=True)
    args=['inference_experiment=quickstart','inputs='+str(source),'out_dir='+str(output),
          'hf_ckpt_path='+str(ROOT/'weights'),'precision=32-true','inference_backend=native',
          'compile_lengths=null','nn.attention_type=torch','nn.tri_mult_backend=torch',
          'num_workers=0','batch_size='+str(options.batch),'num_replicates='+str(options.replicates),
          'prevalidate_inputs=true','generation.args.partial_t_backbone=0.0','generation.args.partial_t_latent=0.0',
          'recycle_strategy=medium/midpeak','seed='+str(options.seed),'compress_outputs=false']
    with initialize_config_dir(version_base=None,config_dir=str(REPO/'configs')):
        cfg=compose(config_name='inference',overrides=args)
        OmegaConf.resolve(cfg)
    OmegaConf.save(cfg,output/'effective_config.yaml',resolve=True)
    execution=dict(status='starting',torch=torch.__version__,input=str(source),
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),started=time.time(),
        source_commit='59e5abb1baa46bb10901812b875b738a15aa2d77',
        model_revision='a06e2509c7033b010fc43614e9fd840303588502',
        precision='FP32',backend='native PyTorch; no CUDA emulation',checkpoint_weights_changed=False)
    try:
        if options.cpu_check:
            from unittest.mock import patch
            with patch.object(torch.cuda,'is_available',return_value=False):
                model=generation.load_ckpt_n_configure_inference(cfg)
            params=list(model.parameters())
            if any(p.device.type!='cpu' or p.dtype!=torch.float32 for p in params):
                raise RuntimeError('CPU checkpoint precision/device mismatch')
            execution.update(status='CPU_strict_checkpoint_load_passed',parameters=sum(p.numel() for p in params),XPU_test='not run')
        else:
            if not os.environ.get('PBS_JOBID'):raise RuntimeError('XPU inference requires an allocated PBS job')
            if not os.environ.get('ZE_AFFINITY_MASK'):raise RuntimeError('MPI wrapper must select one physical XPU')
            if not torch.xpu.is_available() or torch.xpu.device_count()!=1:
                raise RuntimeError('Expected exactly one visible scheduler-assigned Intel XPU')
            execution.update(PBS_JOBID=os.environ['PBS_JOBID'],ZE_AFFINITY_MASK=os.environ['ZE_AFFINITY_MASK'],
                device=torch.xpu.get_device_name(0),CPU_affinity=sorted(os.sched_getaffinity(0)),
                device_memory_bytes=getattr(torch.xpu.get_device_properties(0),'total_memory',None),batch=options.batch,num_replicates=options.replicates,seed=options.seed)
            a=torch.arange(256,dtype=torch.float32).reshape(16,16)/256
            actual=(a.to('xpu')@a.to('xpu')).cpu()
            if not torch.allclose(actual,a@a,atol=1e-5,rtol=1e-5):raise RuntimeError('FP32 XPU matmul probe failed')
            execution['XPU_FP32_matmul']='passed'
            install_xpu_accelerator(torch,generation,execution)
            generation.run_generation(cfg)
            execution['sampling_finished'] = time.time()
            if execution.get('actual_validation_device')!='xpu:0':raise RuntimeError('Missing actual XPU execution evidence')
            from biotite.structure.io import pdbx
            import numpy as np
            files=list((output/'inference_samples').rglob('*_generated.cif'))
            if len(files)!=options.replicates:raise RuntimeError(f'Expected {options.replicates} generated CIFs; got {len(files)}')
            designs=[]
            for file in files:
                atoms=pdbx.get_structure(pdbx.CIFFile.read(file),model=1)
                if not np.isfinite(atoms.coord).all():raise RuntimeError('Nonfinite generated coordinates')
                protein=atoms[np.isin(atoms.res_name,list('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split()))]
                residues=set(zip(map(str,protein.chain_id),map(int,protein.res_id)))
                for chain,residue in residues:
                    names=set(protein.atom_name[(protein.chain_id==chain)&(protein.res_id==residue)])
                    if not {'N','CA','C','O'}<=names:raise RuntimeError('Incomplete generated protein backbone')
                designs.append(dict(generated_CIF=str(file),generated_CIF_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),protein_residues=len(residues),atoms=len(atoms)))
            execution.update(status='XPU_generation_and_structure_checks_passed',designs=designs,peak_XPU_memory_bytes=torch.xpu.max_memory_allocated())
    except BaseException as error:
        execution.update(status='failed_preserved',error=repr(error))
        raise
    finally:
        execution['finished']=time.time()
        execution['run_statistics'] = summarize_run(
            process_started=PROCESS_STARTED,
            entrypoint_started=execution['started'],
            sampling_started=execution.get('sampling_started'),
            sampling_finished=execution.get('sampling_finished'),
            finished=execution['finished'],
            protein_lengths=[d['protein_residues'] for d in execution.get('designs', [])],
            requested_samples=options.replicates,
            completed_samples=len(execution.get('designs', [])),
            batch_size=options.batch,
        )
        if not options.cpu_check:
            try:
                execution['peak_XPU_memory_bytes']=int(torch.xpu.max_memory_allocated())
                execution['peak_XPU_reserved_bytes']=int(torch.xpu.max_memory_reserved())
            except Exception as stat_error:
                execution['memory_stats_error']=repr(stat_error)
        (output/'validation.json').write_text(json.dumps(execution,indent=2)+'\n')
        print(json.dumps(execution,indent=2),flush=True)


if __name__=='__main__':main()
