"""Record generic, candidate-specific mechanistic roles for native participants."""
from pathlib import Path
from datetime import datetime,timezone
from enzyme_discovery_agent.tool_registry import tool
from enzyme_discovery_agent.models.reaction_roles import validate_roles,apply_roles
from enzyme_discovery_agent.models.reaction_roles import PARTICIPANT_ROLES,TRANSFER_UNITS,SITE_RELATIONS


def _step_inputs(run_dir, accession):
    import json
    import re
    from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry
    from enzyme_discovery_agent.run_layout import run_file
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',accession) or accession in {'.','..'}:
        raise ValueError('Invalid candidate identifier')
    root=Path(run_dir)
    entry=load_candidate_registry(root,strict=True)['candidates'].get(accession)
    if not entry or (entry.get('metadata') or {}).get('is_literature_lead'):
        raise ValueError('A source-resolved candidate is required')
    metadata=entry.get('metadata') or {}
    data={**metadata,**(metadata.get('evaluation_data') or {})}
    context=json.loads(run_file(root,'target_context.json').read_text())
    return root,data,context


@tool(description='Inspect mapped native reactant occurrences and target inputs before comparing actual bond changes and reacting roles.',
      category='reasoning', injected_params=['run_dir'], params={
          'run_dir':{'type':'string'},'accession':{'type':'string'},
          'reaction_index':{'type':'integer','default':-1,'description':'Index in catalytic_activities; omit to use the current selected activity.'}})
def get_reaction_step_context(run_dir, accession, reaction_index=-1):
    from rdkit import Chem
    from enzyme_discovery_agent.tools.native_reactions import substrate_occurrences
    from enzyme_discovery_agent.models.reaction_step import step_summary
    try:
        _,data,context=_step_inputs(run_dir,accession)
        from enzyme_discovery_agent.models.reaction_step import native_step_inputs
        activities=data.get('catalytic_activities') or []
        reaction=(data.get('native_substrate_reaction_scope') or {}).get('reaction')
        if reaction_index>=0:
            if reaction_index>=len(activities):
                raise ValueError('Native reaction index is outside the recorded activities')
            reaction=activities[reaction_index]
        if not reaction and len(data.get('catalytic_activities') or [])==1:
            reaction=data['catalytic_activities'][0]
        native=native_step_inputs(data,reaction)
        if not all(item.get('occurrence_id') for item in native):
            native=substrate_occurrences(reaction or '',native)
        def mapped(inputs):
            output=[]
            next_map=1
            for item in inputs:
                mol=Chem.MolFromSmiles(item.get('smiles') or '')
                if mol is None:
                    raise ValueError('Reactant structures must be resolved before mapping a step')
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(next_map)
                    next_map+=1
                from rdkit.Chem import rdMolDescriptors
                output.append({**item,'mapped_smiles':Chem.MolToSmiles(mol),
                    'molecular_formula':rdMolDescriptors.CalcMolFormula(mol),
                    'carbon_atoms':sum(atom.GetAtomicNum()==6 for atom in mol.GetAtoms())})
            return output
        return {'status':'success','data':{'accession':accession,
            'catalytic_activities':data.get('catalytic_activities'), 'function_comments':data.get('function_comments'),
            'native_occurrences':mapped(native),
            'target_occurrences':mapped([{'smiles':s} for s in context.get('substrate_smiles','').split('.') if s]),
            'target_requirements':context,'previous_step':step_summary(data,context),
            'task':'Compare the reacting atoms and bond changes across distinct occurrences, not whole-molecule group counts. '
                   'Retain all atoms in proposed products, including leaving fragments. Assess remaining reactive ends and activation separately.'}}
    except (ValueError,TypeError,OSError) as exc:
        return {'status':'error','summary':str(exc)}


@tool(description='Record an atom-mapped native versus target reaction-step hypothesis, including activation, leaving fragments and continuation limits.',
      category='reasoning',parallel_safe=False,injected_params=['run_dir'],params={
          'run_dir':{'type':'string'}, 'accession':{'type':'string'},
          'native_reaction':{'type':'string','description':'Atom-mapped reactants>>products using the resolved native occurrences. Keep all atoms, including leaving fragments.'},
          'target_reaction':{'type':'string','description':'Atom-mapped reactants>>products using exactly the target inputs. Preserve each atom identity.'},
          'source':{'type':'object','required':['locator','quote'],'properties':{
              'locator':{'type':'string','pattern':r'^catalytic_activities/[0-9]+$',
                         'description':'Use catalytic_activities/0; replace 0 with the zero-based native reaction index.'},
              'quote':{'type':'string','minLength':20,
                       'description':'Exact substring of that catalytic activity annotation.'}}},
          'interpretation':{'type':'string','description':'Explain which distinct occurrences and reactive ends fill the native and target roles, and what remains uncertain.'},
          'activation_assessment':{'type':'string','description':'Explain activation, intermediates and site access; state unresolved requirements.'},
          'continuation_assessment':{'type':'string','description':'Describe the product and remaining reactive ends. Separate single-step support from repeated extension or processivity.'}})
def record_reaction_step(run_dir, accession, native_reaction, target_reaction, source,
                         interpretation, activation_assessment, continuation_assessment):
    from enzyme_discovery_agent.models.reaction_step import validate_step
    from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
    from enzyme_discovery_agent.state_store import atomic_write_json
    from enzyme_discovery_agent.models.event_log import append_event
    import uuid
    try:
        root,data,context=_step_inputs(run_dir,accession)
        record=validate_step(data,context,native_reaction,target_reaction,source,
                             interpretation,activation_assessment,continuation_assessment)
        record.update(accession=accession,recorded_at=datetime.now(timezone.utc).isoformat())
    except (ValueError,TypeError,OSError) as exc:
        return {'status':'error','summary':str(exc)}
    path=root/'artifacts/reaction_steps'/accession/(uuid.uuid4().hex+'.json')
    atomic_write_json(path,record)
    from enzyme_discovery_agent.models.reasoning_trace import capture_scientific_record
    capture_scientific_record(root,'reaction_step_recorded',path)
    data['reaction_step']=record
    update_candidate_registry(root,accession,None,{'evaluation_data':data,'reaction_step':record,
                                                  'reaction_step_artifact':str(path.relative_to(root))})
    append_event(root,'reaction_step_recorded',accession=accession,artifact=str(path.relative_to(root)),
                 local_bond_changes_match=record['local_bond_changes_match'])
    from enzyme_discovery_agent.output.csv_generator import refresh_ranked_candidates
    refresh_ranked_candidates(root)
    return {'status':'success','data':record,'summary':
            'Reaction-step hypothesis recorded. Structural checks do not establish enzyme acceptance, activation feasibility or repeated extension.'}


@tool(description='Record source-grounded participant roles before treating structural resemblance as mechanistic substrate compatibility.',
      category='reasoning',parallel_safe=False,injected_params=['run_dir'],params={
          'run_dir':{'type':'string'},'accession':{'type':'string'},
          'roles':{'type':'array','minItems':1,'maxItems':20,'items':{'type':'object','required':[
              'role','transferred_unit','site_relation','interpretation','source'],
              'anyOf':[{'required':['native_smiles']},{'required':['participant_name']}],
              'properties':{'native_smiles':{'type':'string'},'participant_name':{'type':'string'},'role':{'type':'string','enum':sorted(PARTICIPANT_ROLES)},
                  'transferred_unit':{'type':'string','enum':sorted(TRANSFER_UNITS)},
                  'site_relation':{'type':'string','enum':sorted(SITE_RELATIONS)},
                  'interpretation':{'type':'string'},'source':{'type':'object','required':['locator','quote'],
                      'properties':{'locator':{'type':'string',
                          'pattern':r'^(function_comments|catalytic_activities)/[0-9]+$',
                          'description':'Use function_comments/0 or catalytic_activities/0; replace 0 with the zero-based source index.'},
                          'quote':{'type':'string','minLength':20,
                                   'description':'Exact substring of the annotation at the given locator.'}}}}},'description':
              'Each record: native_smiles, role, transferred_unit, site_relation, interpretation, and source {locator, quote}. '
              'Roles: direct_reactant, precursor, cofactor, catalyst, carrier, solvent, spectator, unresolved. '
              'Transferred units: whole_molecule, fragment, atom, electrons, proton, unresolved. '
              'Sites: same_site, separate_site, tunneled_intermediate, carrier_tethered, unresolved. '
              'Quote a candidate function_comments/N or catalytic_activities/N passage. Unknown roles remain unresolved.'}})
def record_reaction_roles(run_dir,accession,roles):
    import re
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',accession) or accession in {'.','..'}:
        return {'status':'error','summary':'Invalid candidate identifier'}
    from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry,update_candidate_registry
    from enzyme_discovery_agent.state_store import atomic_write_json
    root=Path(run_dir)
    entry=load_candidate_registry(root,strict=True)['candidates'].get(accession)
    if not entry or (entry.get('metadata') or {}).get('is_literature_lead'):
        return {'status':'error','summary':'A source-resolved evaluated candidate is required before assigning participant roles.'}
    metadata=entry.get('metadata') or {}
    data={**metadata,**(metadata.get('evaluation_data') or {})}
    try:
        record=validate_roles(data,roles)
    except (ValueError,TypeError) as exc:
        return {'status':'error','summary':str(exc)}
    similarity=dict(data.get('substrate_similarity') or {})
    similarity['accommodation']=apply_roles(data,record,similarity.get('accommodation'))
    data.update(reaction_roles=record,substrate_similarity=similarity)
    record={**record,'accession':accession,'recorded_at':datetime.now(timezone.utc).isoformat()}
    destination=root/'artifacts/reaction_roles'/accession
    destination.mkdir(parents=True,exist_ok=True)
    import uuid
    path=destination/(uuid.uuid4().hex+'.json')
    atomic_write_json(path,record)
    from enzyme_discovery_agent.models.reasoning_trace import capture_scientific_record
    capture_scientific_record(root,'participant_roles_recorded',path)
    update_candidate_registry(root,accession,None,{'reaction_roles':record,'evaluation_data':data,
                                                'substrate_similarity':similarity,'reaction_role_artifact':str(path.relative_to(root))})
    from enzyme_discovery_agent.models.event_log import append_event
    append_event(root,'reaction_roles_recorded',accession=accession,artifact=str(path.relative_to(root)),
                 role_transfer_status=(similarity.get('accommodation') or {}).get('role_transfer_status'))
    from enzyme_discovery_agent.output.csv_generator import refresh_ranked_candidates
    refresh_ranked_candidates(root)
    return {'status':'success','summary':'Source-grounded role interpretation recorded; unresolved transfers cannot be replaced by a legacy shape-fit score.',
            'data':{'role_record':record,'accommodation':similarity.get('accommodation'),'artifact':str(path.relative_to(root))}}
