"""Render AS9's three matched runs with overlays verified against MD geometry logs."""
from pathlib import Path
import json,os,subprocess,sys
import numpy as np
import MDAnalysis as mda
import parmed
from structural_modeling_agent.barrier.md import resolve_attack_geometry
from structural_modeling_agent.barrier.spec import ReactionSpec
from structural_modeling_agent.tools.render_md_movie import build_parser,stage_frames,stage_assemble
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'structural_modeling_agent/runs/as9_step1_multipose_md_20261117'
MOVIES=ROOT/'structural_modeling_agent/runs/glutarate_cadaverine_step1_multipose_md_20261117/movies'
SPEC=ROOT/'structural_modeling_agent/barrier/specs/adenylation_adipate_cofold.json'
for rank in range(1,4):
    label=f'AS9_pose{rank}'; base=RUN/label; top=base/'prep/complex_1264.prmtop'; traj=base/'md_hmr_seed17/production.dcd'
    work=MOVIES/'_work'/f'{label}_matched';work.mkdir(parents=True,exist_ok=True)
    parm=parmed.load_file(str(top),xyz=str(base/'prep/complex.inpcrd'))
    spec=ReactionSpec.load(str(SPEC))
    i,j,k,angle_atom,kind=resolve_attack_geometry(parm,spec)
    nuc=f'resname SUB and name {parm.atoms[i].name}'
    common=['--workdir',str(work),'--prmtop',str(top),'--traj',str(traj),'--nuc',nuc,'--elec','resname ATP and name PA','--stride','1','--zoom','5.0']
    args=build_parser().parse_args(['frames',*common]);stage_frames(args)
    universe=mda.Universe(str(top),str(traj));frames=[json.loads(x) for x in (base/'md_hmr_seed17/geometry.jsonl').read_text().splitlines()]
    times=[];dist=[];angles=[]
    for n,ts in enumerate(universe.trajectory,1):
        t=n*50.0; row=next(x for x in frames if abs(x['time_ps']-t)<.01)
        xyz=universe.atoms.positions;v=xyz[i]-xyz[j];w=xyz[angle_atom]-xyz[j]
        d=float(np.linalg.norm(v));a=float(np.degrees(np.arccos(np.clip(np.dot(v,w)/(np.linalg.norm(v)*np.linalg.norm(w)),-1,1))))
        assert abs(d-row['forming_A'])<.01,(label,t,d,row)
        assert abs(a-row['attack_angle_deg'])<.1,(label,t,a,row)
        times.append(t);dist.append(d);angles.append(a)
    for name,values in [('times',times),('distances',dist),('angles',angles)]: (work/f'{name}.json').write_text(json.dumps(values))
    (work/'provenance.json').write_text(json.dumps(dict(nucleophile=parm.atoms[i].name,electrophile='PA',angle_reference=parm.atoms[angle_atom].name,angle_definition=kind,verified_against=str(base/'md_hmr_seed17/geometry.jsonl'),times_ps=times),indent=2))
    print('Rendering',label,'with four CPU workers',flush=True)
    procs=[]
    for worker in range(4):
        lo=worker*3+1; hi=min(lo+2,len(times))
        if lo>hi: continue
        shim=work/f'render_{worker}.py'
        shim.write_text('import sys\nsys.path.insert(0,'+repr(str(ROOT))+')\nfrom pymol import cmd\ncmd.set("max_threads",1)\nfrom structural_modeling_agent.tools.render_md_movie import stage_render,build_parser\nstage_render(build_parser().parse_args('+repr(['render',*common,'--lo',str(lo),'--hi',str(hi)])+'))\n')
        procs.append(subprocess.Popen(['taskset','-c',str(60+worker),'/homes/rzhu/miniforge3/envs/pymol/bin/pymol','-cq',str(shim)]))
    codes=[p.wait() for p in procs]
    assert all(c==0 for c in codes),codes
    assert len(list((work/'png').glob('frame_*.png')))==len(times)
    out=MOVIES/f'{label}_reactive_geometry.gif'
    stage_assemble(build_parser().parse_args(['assemble',*common,'--out',str(out)]))
    print('DONE',out,flush=True)
