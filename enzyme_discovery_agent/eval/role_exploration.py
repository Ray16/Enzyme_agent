"""Opt-in model test: annotation roles must change actual expansion selection.

Synthetic, paired development cases. No discovery recall or generalization claim.
Expected outcomes stay here and are never included in the model input.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from enzyme_discovery_agent.bootstrap import register_builtin_tools
from enzyme_discovery_agent.context import compact_tool_definitions
from enzyme_discovery_agent.eval.adaptive_evals import response_record
from enzyme_discovery_agent.llm_client import LLMClient, TokenUsage
from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry, update_candidate_registry
from enzyme_discovery_agent.models.expansion import select_expansion_seeds
from enzyme_discovery_agent.models.family_learning import refresh_family_learning
from enzyme_discovery_agent.models.reaction_roles import role_transfer_summary
from enzyme_discovery_agent.prompts import build_system_prompt
from enzyme_discovery_agent.state_store import atomic_write_json
from enzyme_discovery_agent.tool_registry import get_tool_info
from enzyme_discovery_agent.tools.reaction_roles import record_reaction_roles


INDIRECT = {
    'precursor': 'The synthetic participant is cleaved at a separate active site. Only a released atom '
                 'is conveyed through an internal tunnel to the product-forming site; its scaffold is discarded.',
    'cofactor': 'The synthetic participant transfers electrons and is regenerated in a separate step. '
                'Its molecular scaffold does not become part of the product.',
}
DIRECT = ('The synthetic participant reacts directly at the product-forming site. Its whole molecular '
          'scaffold is retained in the product at that site; no separate precursor-processing step is used.')


def fixture(root, indirect, label):
    root.mkdir(parents=True)
    atomic_write_json(root/'target_context.json', {
        'query':'Find an enzyme that directly transforms the supplied free substrate at its reacting site, '
                'retaining that substrate scaffold in the product.', 'substrate_smiles':'SYNTH_TARGET'})
    for candidate in ('A', 'B'):
        acc='SYNTH_'+candidate
        data={'accession':acc, 'protein_name':'Synthetic catalyst '+candidate, 'gene_names':[acc],
              'interpro_family_id':'SYNTH_SHARED_FOLD', 'claim_validation':{'promotion_eligible':True},
              'catalytic_activities':['Synthetic inputs are converted into the declared product.'],
              'function_comments':[INDIRECT[indirect] if candidate==label else DIRECT],
              'native_substrates':[{'name':'synthetic participant', 'smiles':'SYNTH_NATIVE'}],
              'required_cofactors':[], 'substrate_fit':.9,
              'substrate_similarity':{'accommodation':{'per_target':[
                  {'target_smiles':'SYNTH_TARGET','native_smiles':'SYNTH_NATIVE'}]},
                  'property_comparisons':[{'mw_similarity':.9}]}}
        update_candidate_registry(root, acc, 'evaluated', data)
    atomic_write_json(root/'pass1_ranking.json', [{'accession':'SYNTH_A','rank':1}, {'accession':'SYNTH_B','rank':2}])


def run(output, model, token_budget=40000):
    register_builtin_tools()
    output.mkdir(parents=True, exist_ok=False)
    client=LLMClient(model=model)
    usage=TokenUsage(model=model)
    client.attach_usage(usage, output)
    client.budget_check=lambda: 'Role validation token allowance reached' if usage.total_tokens>=token_budget else None
    tools=compact_tool_definitions([get_tool_info('record_reaction_roles')['schema']])
    rows=[]
    for kind in INDIRECT:
        for label in ('A', 'B'):
            root=output/(kind+'_'+label)
            fixture(root, kind, label)
            candidates=load_candidate_registry(root)['candidates']
            before=select_expansion_seeds(candidates, 1, root)
            messages=[{'role':'system','content':build_system_prompt()}, {'role':'user','content':
                'Review the participant roles of both evaluated analogues below for the next expansion. '
                'Record one source-grounded role interpretation for each candidate using the provided tool; '
                'multiple tool calls are permitted. Treat these annotations as the entire evidence available. '
                'Target activity remains untested.\n'+json.dumps({
                    'target':json.loads((root/'target_context.json').read_text()),
                    'candidates':[entry['metadata'] for entry in candidates.values()]})}]
            atomic_write_json(root/'model_input.json', messages)
            row={'case':kind+'_'+label, 'passed':False, 'before':before}
            rows.append(row)
            if client.budget_check():
                row['status']='budget_exhausted'
                continue
            response=client.call(messages=messages, tools=tools, max_retries=1,
                                 max_tokens=3500, usage_label=row['case'])
            if response is None:
                row['status']='model_unavailable'
                continue
            atomic_write_json(root/'model_response.json', response_record(response))
            calls, _, _=client.extract_tool_calls(response.choices[0].message)
            results=[]
            try:
                if len(calls)!=2:
                    raise ValueError('Both candidates require a recorded role assessment')
                assessed=set()
                for call in calls:
                    function=call.function if hasattr(call,'function') else call['function']
                    name=function.name if hasattr(function,'name') else function['name']
                    raw=function.arguments if hasattr(function,'arguments') else function['arguments']
                    arguments=json.loads(raw) if isinstance(raw,str) else raw
                    if name!='record_reaction_roles' or arguments.get('accession') not in candidates:
                        raise ValueError('Unexpected role action or candidate')
                    assessed.add(arguments['accession'])
                    results.append(record_reaction_roles(str(root), arguments['accession'], arguments['roles']))
                atomic_write_json(root/'executed_results.json', results)
                candidates=load_candidate_registry(root)['candidates']
                after=select_expansion_seeds(candidates, 1, root, method='foldseek')
                policy=refresh_family_learning(root)
                summaries={acc:role_transfer_summary(entry['metadata']['evaluation_data'])
                           for acc, entry in candidates.items() if entry['metadata'].get('evaluation_data')}
                expected='SYNTH_'+('B' if label=='A' else 'A')
                row.update(status='executed', after=after, role_summaries=summaries,
                    feedback=policy['candidates'],
                    passed=assessed==set(candidates) and all(r.get('status')=='success' for r in results)
                        and after==[expected] and summaries['SYNTH_'+label]['review_required']
                        and not summaries[expected]['review_required'])
            except (ValueError, TypeError, KeyError) as exc:
                row.update(status='failed', error=str(exc))
            atomic_write_json(output/'partial_results.json', rows)
    report={'recorded_at':datetime.now(timezone.utc).isoformat(), 'model':model,
            'passed':all(row['passed'] for row in rows), 'cases':rows, 'usage':usage.to_dict(),
            'scope':'Live participant classification and executed seed selection on synthetic development cases. '
                    'The model was asked to record roles. This does not measure autonomous retrieval, scientific entailment or discovery hit rate.'}
    atomic_write_json(output/'report.json', report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--token-budget', type=int, default=40000)
    args=parser.parse_args()
    report=run(args.output, args.model, args.token_budget)
    print(json.dumps({'passed':report['passed'], 'cases':[{k:r.get(k) for k in ('case','status','passed','after')} for r in report['cases']],
                      'tokens':report['usage']['total_tokens']}))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
