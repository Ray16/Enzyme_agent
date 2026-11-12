"""Lightweight structural-agent interface to preparation and evidence review.

This layer does not load a potential or submit a compute job. Numerical work
continues through the barrier CLI and the existing reservation-managed workers.
"""
import json
from pathlib import Path

from .spec import ReactionSpec


def contract():
    return dict(
        owner='structural_modeling_agent.barrier',
        chemistry='Specified bond exchange; nucleophile, electrophile and leaving atom are mapped by residue/atom identity',
        inputs=['Amber topology', 'matching coordinates and actual cell', 'reaction specification'],
        spec_fields=['ligands with integer charges', 'metals with explicit metal_charges except legacy Mg2+',
                     'multiplicity', 'metal_shell_A', 'forming_distance_A', 'breaking_distance_A'],
        preparation_modes=['embedded', 'cluster'],
        stages=['chemical setup review', 'fixed partition or cluster preparation', 'reference path exploration',
                'electronic structure and force validation', 'saddle/connectivity checks or ensemble sampling',
                'separate static-energy and activation-free-energy reporting'],
        cli='python -m structural_modeling_agent.barrier',
        compatibility_cli='python -m enzyme_barrier',
        execution='Use existing reserved workers for GPU/fleet work; preparation does not authorize production',
        limitations=['Multi-step mechanisms need separately reviewed step specifications and references',
                     'Proton relays, redox states and covalent intermediates require explicit molecular setups',
                     'Nonstandard residues and unsupported covalent boundaries require reviewed custom partitions',
                     'New metals, spin states, bases and embedding profiles require method-specific validation',
                     'Inferred endpoint distances are geometric hypotheses, not mechanism evidence'])


def prepare_inputs(prmtop, coordinates, spec_path, out, *, mode='embedded', enzyme_id='enzyme'):
    """Prepare one explicit enzyme/reaction setup, without evaluating a barrier."""
    if mode not in {'embedded','cluster'}:
        raise ValueError('Preparation mode must be embedded or cluster')
    if Path(out).exists():raise FileExistsError(f'Preserve existing barrier output: {out}')
    spec=ReactionSpec.load(spec_path)
    spec.qm_residue_charges()
    if mode=='embedded':
        from .jobs import snapshot
        result=snapshot(prmtop,coordinates,spec_path,out)
    else:
        from .carve import carve
        result=carve(spec,prmtop,coordinates,enzyme_id,out)
    return dict(status='inputs_prepared_requires_chemical_review',mode=mode,
                output_dir=str(Path(out).resolve()),result=result,
                validated_activation_free_energy=False)


def review_result(path):
    """Keep stored numerical output separate from the evidence reporting gate."""
    from .validation import validate_profile
    path=Path(path)
    result=json.loads(path.read_text())
    if not isinstance(result,dict):raise ValueError('Expected a barrier result object')
    # A self-declared validated flag is not a substitute for the reporting gate.
    evidence=result.get('validation_evidence',{})
    validation=validate_profile(evidence)
    return dict(source=str(path.resolve()),quantity=result.get('quantity','unspecified'),
                reported_activation_gap_kcal=result.get('dE_act_kcal'),
                stored_validation_claim=result.get('validated_activation_free_energy',False),
                validation=validation,
                validated_activation_free_energy=validation['validated_activation_free_energy'],
                limitations='The gate checks declared evidence; source integrity and scientific interpretation still require review')


def tool_schemas():
    """Schemas consumed by the structural controller, without a second registry."""
    def schema(name,description,properties,required=()):
        return dict(type='function',function=dict(name=name,description=description,
            parameters=dict(type='object',properties=properties,required=list(required))))
    string=lambda description:dict(type='string',description=description)
    return [
        schema('inspect_barrier_contract','Inspect the generic reaction-barrier submodule, setup requirements and validation limits',{}),
        schema('prepare_barrier_inputs','Prepare one explicit embedded snapshot or isolated cluster in a bounded CPU operator; does not run a potential or submit GPU work',{
            'accession':string('Candidate identifier'),
            'prmtop':string('Authorized Amber topology path'),
            'coordinates':string('Authorized matching PDB, NPZ or Amber coordinates'),
            'spec_path':string('Authorized reaction specification JSON'),
            'python':string('Explicit installed Python with ASE, ParmEd and the barrier package'),
            'mode':dict(type='string',enum=['embedded','cluster']),
            'seconds':dict(type='integer',minimum=1,maximum=60)},
            ('accession','prmtop','coordinates','spec_path','python','mode','seconds')),
        schema('inspect_barrier_result','Inspect an authorized barrier result and its separate evidence-validation gate',{
            'path':string('Authorized numerical result JSON')},('path',))]


def prepare_for_agent(agent,args):
    """Use the controller's existing authorized, CPU-bounded preparation runner."""
    import uuid
    from ..sampling.preparation import run_operator
    from ..validation import file_digest
    accession=args['accession'];agent._candidate(accession)
    if args['mode'] not in {'embedded','cluster'}:raise ValueError('Unknown barrier preparation mode')
    paths={key:agent._authorized_evidence(args[key]) for key in ('prmtop','coordinates','spec_path')}
    coordinate_name='coordinates'+paths['coordinates'].suffix
    names=dict(prmtop='system.prmtop',coordinates=coordinate_name,spec_path='reaction.json')
    inputs=[dict(name=names[key],path=str(path),sha256=file_digest(path)) for key,path in paths.items()]
    package_root=str(Path(__file__).resolve().parents[2])
    code=('import json,sys\nfrom pathlib import Path\n'
          f'sys.path.insert(0,{package_root!r})\n'
          'from structural_modeling_agent.barrier.api import prepare_inputs\n'
          f'result=prepare_inputs("inputs/system.prmtop",{json.dumps("inputs/"+coordinate_name)},'
          f'"inputs/reaction.json","prepared",mode={args["mode"]!r},enzyme_id="candidate")\n'
          'Path("preparation.json").write_text(json.dumps(result,indent=2,allow_nan=False))\n')
    folder=agent.run_dir/'artifacts'/f'barrier_{uuid.uuid4().hex}'
    result=run_operator(folder,args['python'],code,inputs,['preparation.json'],args['seconds'],agent._authorized_evidence)
    agent.state['candidates'][accession].setdefault('barrier_preparations',[]).append(result)
    return result
