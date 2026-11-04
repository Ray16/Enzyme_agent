"""Explicit handoff from retrieved candidate evidence to a redesign library."""
from enzyme_discovery_agent.tool_registry import tool
from enzyme_design_agent.saturation.campaign import prepare
import json
from pathlib import Path


@tool(description='Prepare saturation libraries from an explicit saturation-1 JSON handoff. '
      'Requires selected candidates, source structures, sequences, bound components, '
      'assembly assumptions, folding inputs and separate metric definitions. '
      'Writes provenance and a variant manifest; does not submit computations.',
      category='structure', parallel_safe=False,
      params={'spec_path': {'type':'string', 'default':'', 'description':'Absolute path to saturation-1 JSON; alternatively supply spec_json.'},
              'spec_json': {'type':'string', 'default':'', 'description':'Explicit saturation-1 specification as JSON, including candidates and evidence.'},
              'output_dir': {'type':'string', 'description':'Campaign output directory.'}})
def prepare_saturation_campaign(output_dir: str, spec_path: str = '', spec_json: str = '') -> dict:
    try:
        if bool(spec_path) == bool(spec_json):
            raise ValueError('Supply exactly one of spec_path and spec_json')
        if spec_json:
            document = json.loads(spec_json)
            out = Path(output_dir).resolve()
            out.mkdir(parents=True, exist_ok=True)
            path = out / 'spec.json'
            if path.exists() and json.loads(path.read_text()) != document:
                raise ValueError('Existing campaign has another specification')
            if not path.exists():
                path.write_text(json.dumps(document, indent=2)+'\n')
            spec_path = str(path)
        result = prepare(spec_path, output_dir)
    except (ValueError, KeyError, OSError) as exc:
        return {'status':'error', 'summary':str(exc), 'data':None}
    return {'status':'success', 'summary':f"Prepared {result['n_variants']} variants",
            'data':{'manifest':str(result['output_dir'])+'/manifest.json',
                    'expected_models':result['expected_models'],
                    'candidates':[{k:c[k] for k in ('id','n_positions','n_variants')}
                                  for c in result['candidates']]}}
