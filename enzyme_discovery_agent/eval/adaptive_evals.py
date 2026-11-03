"""Opt-in, serving-model counterfactual test of executed priority decisions.

All records are synthetic. The scorer is never sent to the model or imported by
production. This tests evidence-dependent actions, not enzyme discovery benefit.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from enzyme_discovery_agent.agent import EnzymeAgent
from enzyme_discovery_agent.bootstrap import register_builtin_tools
from enzyme_discovery_agent.context import compact_tool_definitions
from enzyme_discovery_agent.llm_client import LLMClient, TokenUsage
from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry_batch, load_candidate_registry
from enzyme_discovery_agent.models.research_decisions import decision_context
from enzyme_discovery_agent.models.rolling import fresh_evaluation_backlog
from enzyme_discovery_agent.state_store import atomic_write_json
from enzyme_discovery_agent.tool_registry import get_tool_info


def response_record(response):
    """Use the message interface shared by native and SDK-backed providers."""
    return {'model': getattr(response, '_enzyme_serving_model', getattr(response, 'model', None)),
            'message': response.choices[0].message.model_dump(),
            'usage': {key: getattr(response.usage, key, None) for key in
                      ('prompt_tokens', 'completion_tokens', 'total_tokens')}}


def fixture(root, mismatched):
    root.mkdir(parents=True)
    atomic_write_json(root/'target_context.json', {'query':
        'Compare two synthetic annotation groups for a declared transformation with two required substrate inputs.'})
    updates = []
    for group in ('alpha', 'beta'):
        metadata = {'screening_family_id': 'SYNTH_SHARED_FOLD', 'gene_names': [group],
            'ec_numbers': ['1.2.3.4'], 'protein_name': f'Synthetic {group} catalyst',
            'claim_validation': {'promotion_eligible': True},
            'substrate_similarity': {'accommodation': {
                'unaccommodated_targets': ['synthetic_input_role_2'] if group == mismatched else [],
                'all_target_substrates_accommodated': group != mismatched}},
            'catalytic_activities': ['Synthetic native reaction has the required direction; target activity has not been measured.']}
        for i in range(10):
            updates.append((f'SYNTH_{group}_{i}', 'evaluated', metadata))
        updates.append((f'SYNTH_PENDING_{group}', 'screened', metadata))
    update_candidate_registry_batch(root, updates)


def run(output, model, token_budget=60000):
    register_builtin_tools()
    output.mkdir(parents=True, exist_ok=False)
    client = LLMClient(model=model)
    usage = TokenUsage(model=model)
    client.attach_usage(usage, output)
    client.budget_check = lambda: 'Synthetic validation token allowance reached' if usage.total_tokens >= token_budget else None
    tools = [get_tool_info(name)['schema'] for name in ('plan_research_step', 'set_family_focus')]
    rows = []
    for index, mismatched in enumerate(('alpha', 'beta')):
        row = {'case': index+1, 'status': 'not_run'}
        rows.append(row)
        if client.budget_check():
            continue
        root = output/f'case_{index+1:02d}'
        fixture(root, mismatched)
        packet = decision_context(root)
        messages = [{'role':'system', 'content':
            'You are allocating the next evaluation tranche in a synthetic research study. '
            'Only the supplied observations are evidence. Make one call to plan_research_step, '
            'enclosing set_family_focus with appropriate expand/defer annotation scopes. '
            'Propose a testable compatibility hypothesis and a distinct competing explanation; '
            'state contrasting observations and how a later result would revise priorities. '
            'Keep non-native activity uncertain. Missing native accommodation is not proof of inactivity. '
            'Both groups have equal evidence-gathering costs and the same recorded reaction direction. '
            'Use exact recorded evidence IDs and executable tool arguments, not a narrative-only answer.'},
            {'role':'user', 'content':json.dumps(packet)}]
        atomic_write_json(root/'model_input.json', messages)
        response = client.call(messages=messages, tools=compact_tool_definitions(tools), usage_label=f'counterfactual_{index+1}')
        if response is None:
            row.update(status='model_unavailable', passed=False)
            continue
        atomic_write_json(root/'model_response.json', response_record(response))
        calls, _, _ = client.extract_tool_calls(response.choices[0].message)
        try:
            if len(calls) != 1:
                raise ValueError('Exactly one executable research proposal required')
            call = calls[0]
            name = call.function.name if hasattr(call, 'function') else call['function']['name']
            raw = call.function.arguments if hasattr(call, 'function') else call['function']['arguments']
            if name != 'plan_research_step':
                raise ValueError('Model bypassed the prediction-before-action contract')
            arguments = json.loads(raw) if isinstance(raw, str) else raw
            agent = EnzymeAgent.__new__(EnzymeAgent)
            agent.run_dir, agent.llm, agent.tools = root, client, tools
            agent._decision_action_tools, agent._require_research_decision = {'set_family_focus'}, True
            agent._search_history, agent._search_queries, agent._search_log = [], [], []
            agent.log = lambda *a, **k: None
            result = agent._execute_single_tool(name, arguments, index+1)
            atomic_write_json(root/'executed_result.json', result)
            surviving = fresh_evaluation_backlog(root)
            expected = [f'SYNTH_PENDING_{"beta" if mismatched == "alpha" else "alpha"}']
            row.update(status='executed', passed=result.get('status') == 'success' and surviving == expected,
                       surviving_work=surviving, proposal=arguments, decision=result.get('research_decision'))
            if result.get('research_decision') and not client.budget_check():
                # Controlled subsequent observation: the previously unresolved
                # native counterpart is established. Target activity stays unknown.
                updates = []
                for acc, entry in load_candidate_registry(root)['candidates'].items():
                    if (entry.get('metadata') or {}).get('gene_names') == [mismatched]:
                        updates.append((acc, None, {'substrate_similarity': {'accommodation': {
                            'unaccommodated_targets': [], 'all_target_substrates_accommodated': True}}}))
                update_candidate_registry_batch(root, updates)
                call_id = call.id if hasattr(call, 'id') else call['id']
                followup = messages + [response.choices[0].message.model_dump(),
                    {'role':'tool', 'tool_call_id':call_id, 'content':json.dumps(result)},
                    {'role':'user', 'content':'A new retrieved record is available. Compare it with the registered predictions and previous outcome, '
                     'then take one next research-decision action. Retain uncertainty about target activity.\n' + json.dumps(decision_context(root))}]
                atomic_write_json(root/'revision_input.json', followup)
                revised_response = client.call(messages=followup, tools=compact_tool_definitions(tools), usage_label=f'counterfactual_{index+1}_revision')
                if revised_response is None:
                    raise ValueError('Revision response unavailable')
                atomic_write_json(root/'revision_response.json', response_record(revised_response))
                revised_calls, _, _ = client.extract_tool_calls(revised_response.choices[0].message)
                if len(revised_calls) != 1:
                    raise ValueError('Exactly one executable revision required')
                revised_call = revised_calls[0]
                revised_name = revised_call.function.name if hasattr(revised_call, 'function') else revised_call['function']['name']
                revised_raw = revised_call.function.arguments if hasattr(revised_call, 'function') else revised_call['function']['arguments']
                if revised_name != 'plan_research_step':
                    raise ValueError('Revision bypassed the recorded-decision contract')
                revised_proposal = json.loads(revised_raw) if isinstance(revised_raw, str) else revised_raw
                revised_result = agent._execute_single_tool(revised_name, revised_proposal, index+1)
                atomic_write_json(root/'revision_executed_result.json', revised_result)
                surviving_after = fresh_evaluation_backlog(root)
                row.update(revision_proposal=revised_proposal, surviving_after_revision=surviving_after,
                    revision_passed=revised_result.get('status') == 'success' and set(surviving_after) == {'SYNTH_PENDING_alpha','SYNTH_PENDING_beta'},
                    passed=row['passed'] and revised_result.get('status') == 'success' and
                        set(surviving_after) == {'SYNTH_PENDING_alpha','SYNTH_PENDING_beta'})
            else:
                row.update(passed=False, revision_passed=False)
        except (ValueError, TypeError, KeyError) as exc:
            row.update(status='invalid_proposal', passed=False, error=str(exc))
        atomic_write_json(output/'partial_results.json', rows)
        print(json.dumps({'case': index+1, 'status': row['status'], 'passed': row.get('passed'),
                          'cumulative_tokens': usage.total_tokens}), flush=True)
    report = {'recorded_at': datetime.now(timezone.utc).isoformat(), 'model': model,
              'token_budget': token_budget, 'tokens': usage.total_tokens, 'accounting_complete': usage.unmetered_calls == 0 and not usage.legacy_accounting,
              'cases': rows, 'passed': all(row.get('passed', False) for row in rows),
              'scope': 'Live model; two counterfactual evidence-swap cases with subsequent evidence and executed priority revision in each.',
              'limits': ['This is a development sensitivity check, not a held-out estimate.',
                         'Correct priority changes do not establish scientific entailment of the generated explanations.',
                         'Enzyme hit-rate benefit needs budget-matched baseline/ablation and prospective evaluation.']}
    atomic_write_json(output/'results.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='claude-code:claude-sonnet-5-5')
    parser.add_argument('--token-budget', type=int, default=60000)
    args = parser.parse_args()
    report = run(args.output, args.model, args.token_budget)
    print(json.dumps({key:report[key] for key in ('passed', 'tokens', 'accounting_complete', 'scope')}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
