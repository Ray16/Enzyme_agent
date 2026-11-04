"""Isolated GENzyme worker. Run with the GENzyme environment, never the agent's.

Upstream source is loaded without its unbounded main loop. Every request has a
bounded attempt count, stable sample seeds, its own output directory and a
single gpu_reserve-owned device. No benchmark chemistry is embedded here.
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
from pathlib import Path
import random
import runpy
import sys
import textwrap
import traceback
import types


def _save(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def _graph_functions(repo):
    import numpy as np
    import torch
    from rdkit import Chem
    from rdkit.Chem.rdchem import BondType
    tree = ast.parse((repo/'generate.py').read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ('process_smiles', 'process_ligand')]
    if len(funcs) != 2: raise RuntimeError('Unrecognized GENzyme graph interface')
    scope = dict(np=np, torch=torch, Chem=Chem, BONDS={BondType.SINGLE:0,
        BondType.DOUBLE:1, BondType.TRIPLE:2, BondType.AROMATIC:3})
    exec(compile(ast.Module(body=funcs, type_ignores=[]), 'GENzyme_featurizers', 'exec'), scope)
    return scope['process_smiles'], scope['process_ligand']


def validate(request):
    import importlib
    import torch
    from rdkit import Chem
    repo = Path(request['repo']).resolve(); sys.path.insert(0, str(repo))
    if not Path(request['checkpoint']).is_file(): raise FileNotFoundError(request['checkpoint'])
    for module in ('data.data', 'flowmatch.flowmatcher', 'model.main_network',
                   'model.genzyme', 'model.folding_network', 'model.esm3_network'):
        importlib.import_module(module)
    from esm.utils.constants.esm3 import data_root
    esm_root = Path(data_root())
    for name in ('esm3_sm_open_v1.pth','esm3_structure_encoder_v0.pth',
                 'esm3_structure_decoder_v0.pth','esm3_function_decoder_v0.pth'):
        if not (esm_root/'data/weights'/name).is_file(): raise FileNotFoundError(name)
    smiles, ligand = _graph_functions(repo)
    molecules = [Chem.MolFromSmiles(request[k]) for k in ('substrate_smiles', 'product_smiles')]
    if any(m is None or m.GetNumBonds() == 0 for m in molecules):
        raise ValueError('Both SMILES must describe valid molecular graphs containing bonds')
    features = ligand(*[smiles(request[k]) for k in ('substrate_smiles', 'product_smiles')])
    return dict(imports='passed', torch=torch.__version__, esm3_snapshot=str(esm_root),
                canonical_smiles=[Chem.MolToSmiles(m) for m in molecules],
                feature_shapes={k:list(v.shape) for k,v in features.items()})


def _precision_compatibility(net):
    import torch
    if torch.cuda.is_bf16_supported(): return 'native_bfloat16_autocast'
    # ESM 3.0.7 logits hard-codes BF16, unsupported on older devices. Disable
    # that one context in memory while preserving the FP32 parameters.
    method = net.logits.__func__
    tree = ast.parse(textwrap.dedent(inspect.getsource(method))); changed = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'autocast':
            for kw in n.keywords:
                if kw.arg == 'enabled': kw.value = ast.Constant(False); changed += 1
    if changed != 1: raise RuntimeError('Unsupported ESM logits implementation')
    scope = dict(method.__globals__)
    exec(compile(ast.fix_missing_locations(tree), 'esm3_logits_fp32', 'exec'), scope)
    net.logits = types.MethodType(scope['logits'], net)
    return 'fp32_logits_without_bfloat16_autocast'


def _microbatch(sampler, size):
    """Retain all eight inverse-folding trials with bounded model batch memory."""
    def sample(client, proteins, configs):
        results = []
        for start in range(0, len(proteins), size):
            results.extend(sampler(client, proteins=proteins[start:start+size],
                                   configs=configs[start:start+size]))
        return results
    return sample


def run(request, validate_only=False):
    import importlib
    import numpy as np
    import torch
    out = Path(request['output_dir']).resolve(); out.mkdir(parents=True, exist_ok=True)
    validation = validate(request); _save(out/'validation.json', validation)
    if validate_only:
        print(json.dumps(validation), flush=True); return
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count() != 1 or not torch.cuda.is_available():
        raise RuntimeError('GENzyme must run through one exclusive GPU reservation')
    _save(out/'owner.json', dict(pid=os.getpid(), host=os.uname().nodename,
        start_ticks=Path('/proc/self/stat').read_text().split()[21],
        cpu_affinity=sorted(os.sched_getaffinity(0)), gpu=os.environ['CUDA_VISIBLE_DEVICES']))
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
    torch.set_num_interop_threads(1)
    repo = Path(request['repo']).resolve()
    runtime = out/'runtime'; runtime.mkdir()
    (runtime/'genzyme_ckpt').mkdir()
    (runtime/'genzyme_ckpt/genzyme.ckpt').symlink_to(Path(request['checkpoint']).resolve())
    os.chdir(runtime)
    cfg = importlib.import_module('gen_configs')
    for key, value in dict(pdb_name=None, substrate_smiles=request['substrate_smiles'],
        product_smiles=request['product_smiles'], n_pocket_res=request['n_pocket_res'],
        n_protein_res=request['n_protein_res'], num_pocket_design_t=request['pocket_steps'],
        num_inpaint_t=request['inpaint_steps'], n_sample_enzyme=request['num_designs']).items():
        setattr(cfg.Args, key, value)
    random.seed(request['seed']); np.random.seed(request['seed']); torch.manual_seed(request['seed'])
    print('Loading GENzyme and ESM3 weights', flush=True)
    g = runpy.run_path(str(repo/'generate.py'), run_name='genzyme_backend')
    model, args = g['GENzyme'], g['args']; model.eval()
    precision = _precision_compatibility(model.inpainting_model.net)
    # The upstream inverse_folding function resolves globals in its own scope.
    # Replace only its sampling helper; do not modify the installed packages.
    scope = g['inverse_folding'].__globals__
    scope['iterative_sampling_raw'] = _microbatch(scope['iterative_sampling_raw'], request['inverse_fold_microbatch'])
    _save(out/'runtime_settings.json', dict(precision=precision,
        inverse_fold_microbatch=request['inverse_fold_microbatch'], inverse_fold_trials=8,
        gpu_name=torch.cuda.get_device_name(), n_protein_res=args.n_protein_res,
        interpretation='Microbatching preserves eight inverse-folding trials; RNG scheduling differs from upstream batch-eight execution.'))
    args.eval.aa_temp = 0.1; args.eval.aa_noise = 20.0
    features = g['process_ligand'](g['process_smiles'](request['substrate_smiles']),
                                   g['process_smiles'](request['product_smiles']))
    records, designs = [], []
    with torch.no_grad():
        for attempt in range(request['max_attempts']):
            seed = request['seed']+attempt
            np.random.seed(seed); torch.manual_seed(seed)
            print(f'Sampling pocket attempt={attempt} seed={seed}', flush=True)
            feats = g['gen_data_virtual_pocket'](args, g['flow_model'], features)
            feats = {k:v.unsqueeze(0).to(args.device) if torch.is_tensor(v) else v for k,v in feats.items()}
            sampled = g['sampling_inference'](args, init_feats=feats, gen_model=g['flow_model'],
                main_network=model.generation_model, invfold_network=model.inversefold_model,
                min_t=args.min_t, max_t=args.max_t, num_t=args.num_pocket_design_t,
                self_condition=False, center=True, aa_do_purity=True, aa_temp=0.1,
                rot_sample_schedule='exp', trans_sample_schedule='linear')
            coords = sampled['coord_traj'][0][0]
            if not np.isfinite(coords).all(): raise RuntimeError('Non-finite generated pocket coordinates')
            idx = None
            for temp, start in ((1.5,args.n_protein_res//3),(1.5,args.n_protein_res//4),(1.0,1)):
                candidate = g['reindexing'](args, coords, temp=temp, start_idx=start)
                if max(candidate[0]) < args.n_protein_res: idx = candidate; break
            if idx is None:
                records.append(dict(seed=seed, status='pocket_spacing_exceeds_length'))
                _save(out/'progress.json', dict(designs=len(designs), attempts=records)); continue
            feats['residue_index'] = idx
            name = f'design_{len(designs):03d}'
            g['write_pdb_traj'](args, feats_0=feats, feats_1=sampled,
                parent_dir=str(out), pdb_name=name, substrate_name=None, sample_id=0)
            pocket_coords, pocket_seq = g['fill_pocket'](args, prot_feats=feats,
                pocket_feats=sampled, max_len=args.n_protein_res)
            prot = scope['iterative_sampling_raw'](model.inpainting_model.net,
                proteins=[g['ESMProtein'](sequence=pocket_seq, coordinates=pocket_coords)],
                configs=[g['GenerationConfig'](track='structure', num_steps=args.num_inpaint_t, temperature=0.5)])[0]
            ptm, plddt = prot.ptm.mean().item(), prot.plddt.mean().item()
            sequence = g['inverse_folding'](prot, pocket_coord=pocket_coords)
            if len(sequence) != args.n_protein_res or any(a not in 'ACDEFGHIKLMNPQRSTVWY' for a in sequence):
                raise RuntimeError('Incomplete generated protein sequence')
            prot.sequence = sequence
            prot.to_pdb(str(out/name/'protein.pdb'))
            (out/name/'sequence.fasta').write_text(f'>{name} seed={seed}\n{sequence}\n')
            record = dict(id=name, seed=seed, sequence=sequence, fasta=f'{name}/sequence.fasta',
                protein_pdb=f'{name}/protein.pdb', pocket_pdb=f'{name}/pocket_0.pdb',
                ptm=ptm, mean_plddt=plddt,
                upstream_confidence_filter_passed=bool(ptm>request['ptm_filter'] or plddt>request['plddt_filter']))
            designs.append(record); records.append(dict(seed=seed,status='generated'))
            _save(out/'progress.json', dict(designs=len(designs), attempts=records))
            print(f'Generated {name}', flush=True)
            if len(designs) == request['num_designs']: break
    _save(out/'result.json', dict(designs=designs, attempts=records,
        requested_designs=request['num_designs'], activity_validated=False,
        interpretation='Confidence filters are reported, and all generated candidates are retained. Cofold geometry and functional evaluation are separate. Protein coordinates and confidence come from the inpainted model before final sequence assignment, following upstream.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--validate-only', action='store_true')
    options = parser.parse_args()
    request = json.loads(Path(options.request).read_text())
    try: run(request, options.validate_only)
    except Exception:
        out = Path(request['output_dir']); out.mkdir(parents=True, exist_ok=True)
        _save(out/'failure.json', dict(traceback=traceback.format_exc()))
        raise
