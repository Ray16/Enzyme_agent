"""Run one ready modeling task with separate scientific and native execution stages."""
import json
from pathlib import Path
import time

from .agent import ModelingAgent
from .cofold_execution import execute
from enzyme_discovery_agent.state_store import atomic_write_json


TERMINAL = {'complete', 'inconclusive', 'inspection_stalled', 'setup_stalled',
            'failed', 'provider_unavailable', 'execution_failed', 'setup_blocked',
            'no_admitted_candidates'}


def run_candidate(*, task_path, run_dir, backend_path, execution_intent, model=None,
                  token_budget=100000, max_turns=20, runtime_budget_seconds=1800,
                  llm_client=None):
    """Budgets cover the whole saved candidate task; native work uses no model calls."""
    if execution_intent not in {'prepare_only', 'submit_after_checks_pass'}:
        raise ValueError('Declare whether validated agent-selected inputs may be submitted')
    if (type(max_turns) is not int or max_turns < 1
            or type(token_budget) is not int or token_budget < 1 or runtime_budget_seconds <= 0):
        raise ValueError('Require positive turn, token and runtime allowances')
    folder = Path(run_dir).resolve()
    started = time.monotonic()

    def load():
        resume = (folder / 'checkpoint.json').exists()
        return ModelingAgent(task_path=None if resume else task_path, run_dir=folder,
                             backend_path=backend_path, model=model, resume=resume,
                             llm_client=llm_client)

    agent = load()
    if agent.task.get('workflow_mode') != 'production_cofold' or len(agent.task['candidates']) != 1:
        raise ValueError('Production workers require one ready candidate in production_cofold mode')
    accession = agent.task['candidates'][0]['accession']
    if (agent.state['status']=='setup_stalled' and not agent.state.get('cached_review_workflow_enabled')
            and (agent.task['candidates'][0].get('evidence') or {}).get('existing_cofolds')):
        agent.state.update(status='running',cached_review_workflow_enabled=True,
            checkpoint_reason='Resume once through the direct cached-review workflow; previous tokens and turns are retained.')
        agent._save()
    while True:
        revisions = agent.state['candidates'][accession]['revisions']
        revision = revisions[-1] if revisions else {}
        measured = bool((revision.get('pose_validation') or {}).get('poses'))
        if agent.state['status'] in TERMINAL and (measured or not revision.get('job')):
            return agent._write_report()
        if revision.get('execution') and not measured and not agent.state.get('reuse_review_required'):
            proposal_path = folder / 'agent_selected_proposal.json'
            atomic_write_json(proposal_path, {'accession': accession, **{
                key: revision[key] for key in ('modeling_input', 'scientific_reasoning', 'rationale', 'alternatives')}})
            request = {'task_path': str(task_path), 'run_dir': str(folder),
                       'backend_path': str(backend_path), 'proposal_path': str(proposal_path),
                       'execution_intent': execution_intent,
                       'decision_origin': {'author': 'modeling_agent',
                                           'model': agent.state.get('model') or model,
                                           'mode': 'Saved agent-selected science; deterministic execution and measurement.'},
                       'wait_timeout_seconds': max(0, runtime_budget_seconds - (time.monotonic() - started))}
            atomic_write_json(folder / 'execution_request.json', request)
            result = execute(request)
            agent = load()
            if result['status'] == 'review_existing_predictions':
                agent.state.update(status='review_existing_predictions',reuse_review_required=True)
                agent._save()
            elif result['status'] != 'ensemble_ready_for_interpretation':
                return agent._write_report()
        usage = agent.state.get('usage') or {}
        remaining_tokens = token_budget - usage.get('total_tokens', 0)
        # Native render, status and validation events never consume these turns.
        calls_used = len(usage.get('calls') or []) + usage.get('unmetered_calls', 0)
        remaining_turns = max_turns - calls_used
        remaining_time = runtime_budget_seconds - (time.monotonic() - started)
        if remaining_tokens <= 0 or remaining_turns <= 0 or remaining_time <= 0:
            agent.state.update(status='budget_exhausted',
                               checkpoint_reason='Saved candidate task reached its production allowance; outputs and unresolved science remain resumable.')
            agent._save()
            return agent._write_report()
        before = (usage.get('total_tokens', 0), calls_used)
        report = agent.run(max_turns=remaining_turns, token_budget=remaining_tokens,
                           runtime_budget_seconds=remaining_time)
        if report['status'] in TERMINAL:
            return report
        after = report.get('usage') or {}
        progress = (after.get('total_tokens', 0),
                    len(after.get('calls') or []) + after.get('unmetered_calls', 0))
        if progress == before and report['status'] != 'ready_for_native_execution':
            # An unavailable/empty provider response cannot trigger a busy retry loop.
            return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--backend', required=True, type=Path)
    parser.add_argument('--execution-intent', required=True,
                        choices=['prepare_only', 'submit_after_checks_pass'])
    parser.add_argument('--model')
    parser.add_argument('--token-budget', type=int, default=100000)
    parser.add_argument('--max-turns', type=int, default=20)
    parser.add_argument('--runtime-budget-seconds', type=float, default=1800)
    args = parser.parse_args()
    result = run_candidate(task_path=args.task, run_dir=args.run_dir, backend_path=args.backend,
                           execution_intent=args.execution_intent, model=args.model,
                           token_budget=args.token_budget, max_turns=args.max_turns,
                           runtime_budget_seconds=args.runtime_budget_seconds)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] in {'complete', 'inconclusive'} else 2)
