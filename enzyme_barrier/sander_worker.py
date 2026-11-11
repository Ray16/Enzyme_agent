"""Native Amber Python process for periodic electrostatically embedded QM/MM.

Binary IPC is restricted to the launching local process. Fortran output goes to
stderr, never the IPC pipe. One native SANDER context is owned by this process.
"""
import os
import pickle
import struct
import sys
import time
import numpy as np


def send(stream,value):
    data=pickle.dumps(value,protocol=5);stream.write(struct.pack('!Q',len(data)));stream.write(data);stream.flush()


def receive(stream):
    header=stream.read(8)
    if not header:return None
    size=struct.unpack('!Q',header)[0];data=bytearray()
    while len(data)<size:
        chunk=stream.read(size-len(data))
        if not chunk:raise EOFError('Truncated local IPC message')
        data.extend(chunk)
    return pickle.loads(data)


def main():
    out=os.fdopen(os.dup(sys.stdout.fileno()),'wb',buffering=0)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    import sander
    cfg=receive(sys.stdin.buffer)
    options=sander.pme_input() if cfg.get('periodic',True) else sander.gas_input(0)
    options.cut=cfg.get('cutoff_A',10.0);options.lj1264=1
    options.ifqnt=1
    qm=sander.QmInputOptions();qm.qm_theory=cfg.get('method','DFTB3')
    qm.qmmask='@'+','.join(str(i+1) for i in cfg['qm_indices'])
    qm.qmcharge=cfg['charge'];qm.spin=cfg.get('multiplicity',1);qm.qmshake=0
    qm.qmcut=cfg.get('qmcut_A',12.0);qm.qm_ewald=1 if cfg.get('periodic',True) else 0
    qm.qm_pme=1 if cfg.get('periodic',True) else 0
    qm.scfconv=1e-9;qm.tight_p_conv=1;qm.verbosity=1
    if qm.qm_theory=='DFTB3':
        # This field selects an additional diagonal DFTB2 correction/file,
        # NOT the full-DFTB3 switch. qm_theory='DFTB3' selects full DFTB3/3OB.
        qm.dftb_3rd_order='NONE'
        qm.dftb_maxiter=cfg.get('dftb_maxiter',200)
        qm.dftb_telec=cfg.get('dftb_telec',0.0)
    t0=time.monotonic()
    sander.setup(cfg['prmtop'],np.asarray(cfg['positions']).reshape(-1),cfg.get('box'),options,qm)
    send(out,dict(ready=True,setup_wall_s=time.monotonic()-t0,method=qm.qm_theory,
                  third_order=qm.dftb_3rd_order,dftb_maxiter=qm.dftb_maxiter,electronic_temperature_K=qm.dftb_telec))
    try:
        while True:
            request=receive(sys.stdin.buffer)
            if request is None or request.get('command')=='stop':break
            t0=time.monotonic();sander.set_positions(np.asarray(request['positions']).reshape(-1))
            energy,forces=sander.energy_forces()
            force_array=np.asarray(forces).reshape(-1,3)
            if not np.isfinite(force_array).all() or np.max(np.abs(force_array))>1e8 or abs(energy.vdw)>1e10:
                raise ValueError('Catastrophic MM overlap: verify coordinates, topology, cell and periodic conventions')
            components={}
            for key in dir(energy):
                if key.startswith('_'):continue
                value=getattr(energy,key)
                if isinstance(value,(int,float)):components[key]=float(value)
            if not np.isfinite(energy.tot):raise ValueError('Nonfinite SANDER energy')
            send(out,dict(energy_kcal=float(energy.tot),forces_kcal_A=force_array,
                          components_kcal=components,wall_s=time.monotonic()-t0))
    finally:sander.cleanup()


if __name__=='__main__':main()
