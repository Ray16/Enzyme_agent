"""Opt-in atom-mapped step execution on contrasting synthetic development cases."""
import argparse
import json
from pathlib import Path

from enzyme_discovery_agent.bootstrap import register_builtin_tools
from enzyme_discovery_agent.context import compact_tool_definitions,build_model_context
from enzyme_discovery_agent.eval.adaptive_evals import response_record
from enzyme_discovery_agent.llm_client import LLMClient,TokenUsage
from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry,load_candidate_registry
from enzyme_discovery_agent.models.research_memory import recall_memory
from enzyme_discovery_agent.prompts import build_system_prompt
from enzyme_discovery_agent.state_store import atomic_write_json
from enzyme_discovery_agent.tool_registry import get_tool_info
from enzyme_discovery_agent.tools.reaction_roles import get_reaction_step_context,record_reaction_step


def run(output,model):
    output.mkdir(parents=True,exist_ok=False)
    register_builtin_tools()
    client=LLMClient(model=model)
    usage=TokenUsage(model=model)
    client.attach_usage(usage,output)
    client.budget_check=lambda: 'Step validation token allowance reached' if usage.total_tokens>=30000 else None
    tools=compact_tool_definitions([get_tool_info('record_reaction_step')['schema']])
    atomic_write_json(output/'model_tool_schemas.json',tools)
    rows=[]
    for kind in ('two_partners','one_cycle','oxygen_partners'):
        root=output/kind
        root.mkdir()
        context={'query':'Find a catalyst for direct intermolecular coupling of the supplied substrates, '
                        'then assess the remaining terminal groups and whether extension is supported.',
                 'substrate_smiles':'O=C(O)CCCC(=O)O.OCCCCO' if kind=='oxygen_partners' else 'O=C(O)CCCC(=O)O.NCCCCN'}
        atomic_write_json(root/'target_context.json',context)
        if kind in {'two_partners','oxygen_partners'}:
            reaction='2 synthetic substrate = synthetic coupled product + water'
            description=('Two copies of the substrate react at the same catalytic site: the terminal nitrogen of one copy '
                         'bonds to the terminal carbonyl carbon of the other copy, displacing the terminal hydroxyl as water. '
                         'The remaining terminal groups are retained. Activation intermediates are not resolved by this annotation.')
            if kind=='oxygen_partners':
                description=('Two copies of the substrate react at the same catalytic site: the terminal alcohol oxygen of '
                             'one copy bonds to the terminal carbonyl carbon of the other, displacing the acid hydroxyl '
                             'as water. The other terminal groups are retained. Activation intermediates are unresolved.')
        else:
            reaction='synthetic substrate = synthetic cyclic product + water'
            description=('One substrate molecule cyclizes at the catalytic site: its terminal nitrogen bonds to its own '
                         'terminal carbonyl carbon, displacing the terminal hydroxyl as water. The reaction is intramolecular. '
                         'No two-substrate turnover has been measured.')
        update_candidate_registry(root,'SYNTH_STEP','evaluated',{'accession':'SYNTH_STEP',
            'catalytic_activities':[reaction],'function_comments':[description],
            'native_substrates':[{'name':'synthetic substrate','smiles':'OCCC(=O)O' if kind=='oxygen_partners' else 'NCCCC(=O)O'}]})
        inspected=get_reaction_step_context(str(root),'SYNTH_STEP')
        messages=[{'role':'system','content':build_system_prompt()}, {'role':'user','content':
            'Use the inspected atom-mapped inputs below to record a native-versus-target reaction-step hypothesis. '
            'Construct native products consistent with the annotation and propose the target coupling product. '
            'Preserve every atom, including leaving water, and each molecular occurrence. '
            'Explain whether the reacting roles correspond and what is unresolved. Make one record_reaction_step call; '
            'recording a comparison does not require that its native and target steps be compatible.\n'+json.dumps(inspected)}]
        atomic_write_json(root/'model_input.json',messages)
        row={'case':kind,'passed':False}
        rows.append(row)
        response=client.call(messages=messages,tools=tools,max_tokens=5000,max_retries=1,usage_label=kind)
        if response is None:
            row['status']='model_unavailable'
            continue
        atomic_write_json(root/'model_response.json',response_record(response))
        calls,_,_=client.extract_tool_calls(response.choices[0].message)
        try:
            if len(calls)!=1:
                raise ValueError('Exactly one mapped-step action required')
            function=calls[0].function if hasattr(calls[0],'function') else calls[0]['function']
            name=function.name if hasattr(function,'name') else function['name']
            raw=function.arguments if hasattr(function,'arguments') else function['arguments']
            arguments=json.loads(raw) if isinstance(raw,str) else raw
            if name!='record_reaction_step' or arguments.get('accession')!='SYNTH_STEP':
                raise ValueError('Unexpected action or candidate')
            result=record_reaction_step(str(root),**arguments)
            atomic_write_json(root/'executed_result.json',result)
            if result.get('status')!='success':
                raise ValueError(result.get('summary'))
            # This fresh reader has neither the model response nor a tool transaction.
            recalled=recall_memory(Path(str(root)),'reasoning',query='SYNTH_STEP')
            long_history=[{'role':'system','content':'Use campaign evidence and preserve its uncertainty.'},
                          {'role':'user','content':context['query']}]
            long_history += [{'role':'assistant','content':'Obsolete narrative '*1000} for _ in range(40)]
            compacted=build_model_context(long_history,root)
            atomic_write_json(root/'compacted_restart_context.json',compacted)
            step=result['data']
            checks={'tool_execution':True,
                    'correct_occurrence_relationship':step['local_bond_changes_match']==(kind!='one_cycle'),
                    'recalled_from_disk':bool(recalled['records']),
                    'mapped_step_survives_compaction':'local_bond_changes_match' in json.dumps(compacted),
                    'continuation_uncertainty_preserved':'continuation_assessment' in json.dumps(compacted)}
            row.update(status='executed',checks=checks,passed=all(checks.values()),
                       local_bond_changes_match=step['local_bond_changes_match'],
                       activation_assessment=step['activation_assessment'],
                       continuation_assessment=step['continuation_assessment'])
        except (ValueError,TypeError,KeyError) as exc:
            row.update(status='failed',error=str(exc))
        atomic_write_json(output/'partial_results.json',rows)
    report={'passed':all(row['passed'] for row in rows),'cases':rows,'usage':usage.to_dict(),
            'scope':'Live structured step execution and disk-memory/compaction checks on synthetic development cases. '
                    'Source-to-structure scientific entailment and autonomous discovery efficacy are not established.'}
    atomic_write_json(output/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',required=True)
    args=parser.parse_args()
    report=run(args.output,args.model)
    print(json.dumps({'passed':report['passed'],'cases':report['cases'],'tokens':report['usage']['total_tokens']}))
    raise SystemExit(0 if report['passed'] else 1)
