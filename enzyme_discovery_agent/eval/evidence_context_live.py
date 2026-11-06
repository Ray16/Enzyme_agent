"""Live component check of saved evidence -> context -> scoped model verdicts."""
import argparse
import json
from pathlib import Path
import subprocess
import types
from unittest.mock import patch

from enzyme_discovery_agent.context import build_model_context
from enzyme_discovery_agent.llm_client import LLMClient, TokenUsage
from enzyme_discovery_agent.models import research_decisions as current
from enzyme_discovery_agent.state_store import atomic_write_json


def run(output, model, revision):
    output.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(['git', 'show', revision+':enzyme_discovery_agent/models/research_decisions.py'], text=True)
    baseline = types.ModuleType('baseline_decisions')
    exec(compile(source, 'baseline_research_decisions.py', 'exec'), baseline.__dict__)
    root = output/'fixture'; root.mkdir()
    atomic_write_json(root/'target_context.json', {'query': 'Synthetic target: A1 plus B1 yields P11 with S1.'})
    ref = current.decision_context(root)['observations'][0]['id']
    plan = {'hypothesis': 'S1 may accept the two requested partners.',
        'alternative': 'Evidence may cover only one requested partner.', 'scope': 'retrieval',
        'prediction': 'Saved observations distinguish which partner has been tested.',
        'contrasting_prediction': 'No partner has experimentally supported acceptance.',
        'why_this_test': 'Retrieve the actual source observation before choosing a new experiment.',
        'revision_trigger': 'Revise each partner claim according to the observed substrate pairs.',
        'tool_name': 'synthetic_retrieval', 'arguments': {}, 'evidence_refs': [ref]}
    identifier = current.start_decision(root, plan, {'synthetic_retrieval'})
    current.finish_decision(root, identifier, {'status': 'success', 'data': [
        {'tool_name': 'get_candidate_evidence', 'arguments': {'accession': 'SYNTH_A'}, 'status': 'success',
         'data': {'accession': 'SYNTH_A', 'protein_name': 'Synthetic catalyst S1',
                  'catalytic_activities': ['A0 + B1 yields P01.'],
                  'function_comments': ['Purified S1 converts A0 plus B1 to identified P01 above inactive controls. A1 has never been tested.'],
                  'critical_warnings': ['Activity with A0 does not establish activity with A1.']}}]})
    client = LLMClient(model=model); usage = TokenUsage(model=model); client.attach_usage(usage, output)
    client.budget_check = lambda: usage.total_tokens >= 30000
    fields = {key: {'type': 'string', 'enum': ['supported','contradicted','unresolved']}
              for key in ('B1_acceptance_with_A0', 'complete_A1_B1_reaction')}
    fields['reason'] = {'type': 'string'}
    tool = {'type': 'function', 'function': {'name': 'record_scope_check', 'description': 'Record evidence support for the two scoped claims.',
        'parameters': {'type': 'object', 'properties': fields, 'required': list(fields)}}}
    rows = []
    for label, module in [('baseline', baseline), ('fixed', current)]:
        packet = module.decision_context(root); packet['review_due'] = True
        with patch('enzyme_discovery_agent.context.build_research_state', return_value={'adaptive_reasoning': packet}):
            messages = build_model_context([{'role': 'system', 'content': 'Use only supplied saved evidence; missing observations cannot establish activity.'},
                                           {'role': 'user', 'content': 'Assess S1 for A1 plus B1 yielding P11.'}], root)
        messages.append({'role': 'user', 'content': 'For this evaluation, call record_scope_check only. '
            'Assess whether S1 accepts B1 with A0 and whether it catalyzes the complete A1+B1 reaction. '
            'Use supported, contradicted or unresolved according to the visible observations, with a concise explanation.'})
        atomic_write_json(output/(label+'_request.json'), {'messages': messages, 'tools': [tool]})
        response = client.call(messages=messages, tools=[tool], usage_label=label, max_tokens=2500, max_retries=1)
        row = {'arm': label, 'status': 'provider_unavailable'}
        if response is not None:
            message = response.choices[0].message.model_dump()
            atomic_write_json(output/(label+'_response.json'), message)
            calls = message.get('tool_calls') or []
            if len(calls) == 1 and calls[0]['function']['name'] == 'record_scope_check':
                try:
                    raw = calls[0]['function']['arguments']
                    assessment = json.loads(raw) if isinstance(raw,str) else raw
                    if not isinstance(assessment, dict):
                        raise ValueError('Assessment must be an object')
                    row.update(status='recorded', assessment=assessment)
                except (ValueError, TypeError, KeyError) as exc:
                    row.update(status='invalid_output', error=str(exc))
            else:
                row['status'] = 'invalid_output'
        rows.append(row)
        atomic_write_json(output/'partial.json', rows)
    expected = {'baseline': {'B1_acceptance_with_A0': 'unresolved', 'complete_A1_B1_reaction': 'unresolved'},
                'fixed': {'B1_acceptance_with_A0': 'supported', 'complete_A1_B1_reaction': 'unresolved'}}
    for row in rows:
        row['passed'] = all((row.get('assessment') or {}).get(k) == v for k,v in expected[row['arm']].items())
    report = {'baseline_revision': revision, 'model': model, 'results': rows,
              'passed': all(row['passed'] for row in rows), 'usage': usage.to_dict(),
              'scope': 'Synthetic saved-observation/context/model component test. Not a full discovery run or a general chemical-reasoning benchmark.'}
    atomic_write_json(output/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--baseline-revision', required=True)
    args = parser.parse_args()
    result = run(args.output, args.model, args.baseline_revision)
    print(json.dumps({'passed': result['passed'], 'results': result['results'], 'tokens': result['usage']['total_tokens']}, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
