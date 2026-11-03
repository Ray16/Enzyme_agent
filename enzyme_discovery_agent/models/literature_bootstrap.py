"""Bounded source reconnaissance before a cold annotation-driven expansion."""
from pathlib import Path
import json
from enzyme_discovery_agent.models.event_log import read_events
from enzyme_discovery_agent.run_layout import run_file

LITERATURE_TOOLS=frozenset({'search_pubmed','search_europepmc','search_openalex','search_semantic_scholar',
                           'search_crossref','deep_literature_search','read_primary_paper'})


def bootstrap_status(run_dir,available_tools,*,mechanisms_derived=False,max_attempts=2):
    """Source outages remain explicit and cannot lock discovery indefinitely."""
    configured=set(available_tools)&LITERATURE_TOOLS
    if not mechanisms_derived:
        # Explicit resume may have durable derivation output before gate state
        # has been reconstructed. A cold run has no such campaign-local file.
        try:
            saved=json.loads(run_file(Path(run_dir), 'mechanism_families.json').read_text())
            mechanisms_derived=isinstance(saved, list) and bool(saved)
        except (OSError, ValueError):
            pass
    if mechanisms_derived or not configured:
        return {'required':False,'status':'already_started' if mechanisms_derived else 'no_literature_tools',
                'attempts':0,'available_tools':sorted(configured)}
    attempts=[event for event in read_events(Path(run_dir))
              if event.get('type')=='tool_call' and event.get('tool') in LITERATURE_TOOLS]
    found=any(event.get('status') in {'success','ok'} and
              (event.get('tool')=='read_primary_paper' or isinstance(event.get('count'),(int,float)) and event['count']>0)
              for event in attempts)
    if not found:
        from enzyme_discovery_agent.models.research_decisions import _literature_observations
        found=bool(_literature_observations(Path(run_dir)))
    return {'required':not found and len(attempts)<max_attempts,
            'status':'retrieved_sources' if found else 'degraded_no_sources' if len(attempts)>=max_attempts else 'source_search_needed',
            'attempts':len(attempts),'available_tools':sorted(configured),
            'limits':'A failed or empty source lookup is not evidence that relevant enzymes do not exist.'}
