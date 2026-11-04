"""Execute an agent-reviewed cofold plan without LLM polling or schema inspection."""
from pathlib import Path
import json
import time

from .agent import ModelingAgent,_controller_lock
from .scientific_reasoning import execution_reasoning_errors
from enzyme_discovery_agent.state_store import atomic_write_json


def action(agent,name,arguments):
    """Use the same scientific and resource gates as ordinary agent tool calls."""
    agent.state['turn']+=1
    result=agent.dispatch(name,arguments)
    artifact=agent._artifact(name,result)
    agent._event('workflow_tool',tool=name,artifact=artifact,
                 decision_origin=agent.state.get('decision_origin'),provider_call=False)
    agent._save()
    agent._write_report()
    return result


def prepare(agent,proposal,*,execution_intent):
    """Save choices, inspect consistency feedback, and render only a supported plan."""
    accession=proposal['accession']
    revisions=agent.state['candidates'][accession]['revisions']
    if not revisions:
        if agent.task.get('kind')=='discovery_handoff':
            action(agent,'retrieve_reference',{'kind':'uniprot','identifier':accession})
        revision=action(agent,'propose_model',proposal)
    else:revision=revisions[-1]
    checks=action(agent,'validate_setup',{'accession':accession})
    if checks['status']!='pass':
        return {'status':'blocked','reason':'Setup source/consistency checks did not pass','validation':checks}
    feedback=action(agent,'inspect_reasoning',{'accession':accession})
    assessment=feedback.get('assessment',feedback)
    contradictory=[p for p in assessment.get('predictions',[]) if p.get('status')=='contradicted']
    if contradictory or assessment.get('source_errors'):
        return {'status':'needs_scientific_revision','assessment':assessment}
    if assessment.get('requires_response') and not assessment.get('response_current'):
        if execution_intent!='submit_after_checks_pass':
            return {'status':'needs_scientific_decision','assessment':assessment}
        # This acknowledges input consistency under an already selected test.
        # It cannot erase a contradictory observation or select a different model.
        selected=revision['scientific_reasoning']['selected_hypothesis_id']
        action(agent,'respond_to_reasoning_feedback',{'accession':accession,
            'assessment_sha256':assessment['assessment_sha256'],'action':'retain',
            'rationale':'The agent selected this conditional test before execution. Source/identity/consistency checks passed and no contradictory observation was recorded. This is not confirmation of the biological hypothesis.',
            'addressed_hypotheses':[selected],
            'next_test':{'measurement':'Execute the already declared cofold ensemble and physical checks.',
                'expected_if_selected':'Some complete, physically consistent poses under the declared condition.',
                'expected_if_alternative':'Physical failures or unresolved results motivate a separate control or revised setup.',
                'controls':['The independently sourced reference setup retained in the scientific record.']}})
    gaps=execution_reasoning_errors(agent._current(accession))
    if gaps:return {'status':'blocked','reasoning_errors':gaps}
    if revision.get('execution'):
        return {'status':'rendered','input_yaml':revision['input_yaml'],'reused_render':True}
    return action(agent,'render_inputs',{'accession':accession})


def execute(request,*,wait=True):
    """No new provider calls: scientific choices are supplied as an explicit agent decision."""
    task=Path(request['task_path']);folder=Path(request['run_dir'])
    resume=(folder/'checkpoint.json').exists()
    agent=ModelingAgent(task_path=None if resume else task,run_dir=folder,
        backend_path=request['backend_path'],resume=resume)
    agent.state['decision_origin']=request['decision_origin']
    from enzyme_discovery_agent.provenance import snapshot_source_code
    snapshot=snapshot_source_code(folder)
    if not any(s['source_sha256']==snapshot['source_sha256'] for s in agent.state.get('code_snapshots',[])):
        agent.state.setdefault('code_snapshots',[]).append(snapshot)
    proposal=json.loads(Path(request['proposal_path']).read_text())
    accession=proposal['accession']
    with _controller_lock(folder):
        revisions=agent.state['candidates'][accession]['revisions']
        pending=(revisions[-1].get('job') or {}) if revisions else {}
        supplied=next(candidate for candidate in agent.task['candidates'] if candidate['accession']==accession)
        if not pending and (supplied.get('evidence') or {}).get('do_not_recofold'):
            return {'status':'review_existing_predictions','reason':'Existing cofold evidence must be reviewed before any new submission.',
                    'existing_predictions':(supplied.get('evidence') or {}).get('existing_cofolds',[])}
        receipt=None
        if pending and revisions[-1].get('execution'):
            # Reconciliation and measurement of saved outputs are read-only.
            # Pending scientific interpretation must not prevent either one.
            receipt=action(agent,'check_cofold',{'accession':accession})
        if receipt is None or receipt.get('status')=='not_found':
            result=prepare(agent,proposal,execution_intent=request.get('execution_intent'))
            if result.get('status')!='rendered':
                agent.state.update(status='setup_blocked',checkpoint_reason=result.get('reason','Source or setup review is required'))
                agent._save();agent._write_report();return result
            if request.get('execution_intent')!='submit_after_checks_pass':
                return {'status':'prepared','input_yaml':result['input_yaml']}
            receipt=action(agent,'submit_cofold',{'accession':accession})
        deadline=time.monotonic()+request.get('wait_timeout_seconds',21600)
        while receipt.get('status') in {'queued','running','unknown'} and wait:
            if time.monotonic()>=deadline:
                agent.state.update(status='waiting_for_backend');agent._save();agent._write_report()
                return {'status':'waiting_for_backend','job':receipt}
            time.sleep(request.get('poll_seconds',15))
            receipt=action(agent,'check_cofold',{'accession':accession})
        if receipt.get('status')!='complete':
            agent.state.update(status='waiting_for_backend' if receipt.get('status') in {'queued','running','unknown'} else 'execution_failed')
            agent._save();agent._write_report();return {'status':agent.state['status'],'job':receipt}
        report=action(agent,'validate_predictions',{'accession':accession,'directory':receipt['prediction_dir'],
                                                  'prediction_manifest':receipt['prediction_manifest']})
        agent.state.update(status='ensemble_ready_for_interpretation')
        agent._save();summary=agent._write_report()
        result={'status':'ensemble_ready_for_interpretation','pose_validation_status':report.get('status'),
                'n_predictions':report.get('n_predictions'),'prediction_dir':receipt['prediction_dir'],
                'modeling_report':str(folder/'report.json'),'provider_calls_for_execution':0}
        atomic_write_json(folder/'execution_status.json',result)
        return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True);parser.add_argument('--no-wait',action='store_true')
    args=parser.parse_args()
    print(json.dumps(execute(json.loads(Path(args.request).read_text()),wait=not args.no_wait),indent=2))
