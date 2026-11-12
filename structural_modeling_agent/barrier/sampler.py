"""Candidate fast Hamiltonian: UMA(in vacuo) + GFN2 embedding response + MM.

This is a sampling hypothesis that MUST be calibrated against embedded DFT for
each reaction/partition. xTB calls are serial, never two native pools per worker.
There is no fixed QM-charge Coulomb term in MM, hence no double counting.
"""
import os
import tempfile
from pathlib import Path
import numpy as np
from .embedded import EmbeddedDFTMM


class EmbeddedUMAXTB(EmbeddedDFTMM):
    def __init__(self,*args,model='uma-s-1p2',xtb_command=None,**kwargs):
        super().__init__(*args,**kwargs)
        from .mlip import get_calculator
        self.uma=get_calculator('omol',model)
        self.model=model
        self.xtb_command=xtb_command or os.environ.get('BARRIER_XTB_COMMAND','/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/xtb/bin/xtb')
        if not Path(self.xtb_command).is_file():raise ValueError('xTB executable unavailable')
        self.scratch=Path(tempfile.mkdtemp(prefix=f'enzyme_barrier_xtb_{os.getpid()}_',dir='/tmp'))
        self.high_diagnostics={}

    def high_energy_forces(self,symbols,qpos,mmpos,mmcharges,need_forces):
        from ase import Atoms
        from .xtb import run_xtb
        a=Atoms(symbols=symbols,positions=qpos)
        a.info.update(charge=self.region['charge'],spin=self.region['multiplicity']);a.calc=self.uma
        if need_forces:uf=a.get_forces()
        else:uf=None
        ue=float(a.get_potential_energy())
        common=dict(symbols=symbols,positions=qpos,charge=self.region['charge'],multiplicity=self.region['multiplicity'],
            forces=need_forces,command=self.xtb_command,accuracy=.2,
            workdir=str(self.scratch),threads=int(os.environ.get('OMP_NUM_THREADS','1')))
        ve,vf,_=run_xtb(**common)
        ee,ef,mf=run_xtb(mm_positions=mmpos,mm_charges=mmcharges,**common)
        self.high_diagnostics=dict(uma_vacuum_eV=ue,xtb_embedding_correction_eV=float(ee-ve),
            inference_settings='explicit_ieee_fp32_general_no_compile',tf32=False,
            external_charge_hardness=99,xtb_interface='standalone_documented_cli',
            potential=f'{self.model}+GFN2_embedding',method_validation='requires_reaction_specific_DFT_calibration')
        return ue+ee-ve,(uf+ef-vf if need_forces else None),mf
