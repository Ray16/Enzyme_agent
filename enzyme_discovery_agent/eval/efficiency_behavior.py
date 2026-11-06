"""Bounded paired live comparison on pre-existing counterfactual action tests.

This isolates JSON whitespace normalization. It does not evaluate real-world
enzyme recovery and does not enable experimental evidence factoring.
"""
import argparse
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import patch

from enzyme_discovery_agent.eval.adaptive_evals import run
from enzyme_discovery_agent.state_store import atomic_write_json


def replay_guarded_revision(source, output):
    """Check actual controller rejection on a copy of an unexecuted revision.

    This does not change the original strict evaluation result or call a model.
    """
    from enzyme_discovery_agent.agent import EnzymeAgent
    from enzyme_discovery_agent.bootstrap import register_builtin_tools
    from enzyme_discovery_agent.models.rolling import fresh_evaluation_backlog
    from enzyme_discovery_agent.tool_registry import get_tool_info
    source = Path(source)
    if (source/'revision_executed_result.json').exists():
        raise ValueError('Revision already executed; cannot replay its pre-action state')
    message = json.loads((source/'revision_response.json').read_text())['message']
    register_builtin_tools()
    shutil.copytree(source, output)
    agent = EnzymeAgent.__new__(EnzymeAgent)
    agent.run_dir = Path(output)
    agent.llm = SimpleNamespace(active_label='offline_guard_replay')
    agent.tools = [get_tool_info(name)['schema'] for name in ('plan_research_step', 'set_family_focus')]
    agent._offered_direct_tools = {'plan_research_step'}
    agent._decision_action_tools = {'set_family_focus'}
    agent._require_research_decision = True
    agent._search_history, agent._search_queries, agent._search_log = [], [], []
    agent.log = lambda *args, **kwargs: None
    outcomes = []
    for call in message.get('tool_calls') or []:
        function = call['function']
        arguments = function['arguments']
        arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        result = agent._execute_model_tool(function['name'], arguments, 1)
        outcomes.append({'tool': function['name'], 'result': result})
    result = {'scope': 'Offline dispatch through the production guard on copied state; original failure retained.',
              'outcomes': outcomes, 'surviving_work': fresh_evaluation_backlog(output)}
    atomic_write_json(Path(output)/'guard_check.json', result)
    return result


def compare(output, model, token_budget=120000):
    if token_budget < 2:
        raise ValueError('Positive allowance required for both arms')
    output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for arm in ('baseline', 'optimized'):
        if arm == 'baseline':
            with patch('enzyme_discovery_agent.request_efficiency.compact_json_messages', lambda messages: messages):
                reports[arm] = run(output/arm, model, token_budget//2)
        else:
            reports[arm] = run(output/arm, model, token_budget//2)
        atomic_write_json(output/'partial_comparison.json', reports)
    def outcomes(report):
        return [{key: row.get(key) for key in ('case', 'passed', 'revision_passed',
                    'surviving_work', 'surviving_after_revision')} for row in report['cases']]
    result = {'model': model, 'token_allowance': token_budget,
              'both_arms_passed': all(report['passed'] for report in reports.values()),
              'matched_executed_outcomes': outcomes(reports['baseline']) == outcomes(reports['optimized']),
              'tokens': {arm: report['tokens'] for arm, report in reports.items()},
              'outcomes': {arm: outcomes(report) for arm, report in reports.items()},
              'scope': 'Pre-existing synthetic evidence-swap and executed revision cases. '
                       'Baseline disables only JSON whitespace normalization; schemas and scientific rules are identical.',
              'limitations': ['Single execution per arm; stochastic variation and cache-order effects remain.',
                              'These are development cases, not real enzyme discovery or a held-out hit-rate estimate.']}
    atomic_write_json(output/'comparison.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--token-budget', type=int, default=120000)
    args = parser.parse_args()
    result = compare(args.output, args.model, args.token_budget)
    print(json.dumps(result, indent=2))
    return 0 if result['both_arms_passed'] and result['matched_executed_outcomes'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
