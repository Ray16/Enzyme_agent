"""Read-only project-home status page for discovery campaigns."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time

from enzyme_discovery_agent.output.live_progress import snapshot, _atomic_text
from enzyme_discovery_agent.models.rolling import fresh_evaluation_backlog
from enzyme_discovery_agent.run_layout import run_file


def _run_signature(folder):
    files=[folder/'manifest.json',run_file(folder,'candidate_registry.json'),
           run_file(folder,'token_usage.json'),run_file(folder,'events.jsonl'),
           folder/'agent_state/invocation_resources.json',folder/'agent_state/family_decisions.json',
           run_file(folder,'pass1_ranking.json'),
           folder/'agent_state/discovery_owner.json']
    signature=tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in files if p.exists())
    # Include liveness on each pass even while a worker is waiting on I/O.
    owner=folder/'agent_state/discovery_owner.json'
    if owner.exists():
        identity=json.loads(owner.read_text());pid=identity.get('pid')
        proc=Path('/proc')/str(pid)/'stat'
        signature+=(('process_stat',proc.read_text() if pid and proc.exists() else None),)
    return signature


def collect_runs(project, *, limit=12, cache=None):
    """Reuse saved-state snapshots; never launch tools or alter campaign state."""
    project=Path(project).resolve()
    root=project/'enzyme_discovery_agent/runs'
    cache={} if cache is None else cache
    folders=sorted((p for p in root.iterdir() if p.is_dir() and not p.name.startswith('_')
                    and (p/'manifest.json').is_file()),
                   key=lambda p:(p/'manifest.json').stat().st_mtime_ns,reverse=True)[:limit]
    runs=[]
    for folder in folders:
        try:
            signature=_run_signature(folder)
            if cache.get(str(folder),{}).get('signature')!=signature:
                state=snapshot(folder)
                state['ready_count']=len(fresh_evaluation_backlog(folder))
                state['interpro_groups']=sum(row.get('canonical_assignment',False)
                                              for row in (state.get('frontier') or {}).get('families',[]))
                state['links']={}
                for label,relative in [('Ranked CSV','ranked_candidates.csv'),('Reasoning','reasoning/trace.html'),
                    ('Candidate registry','agent_state/candidate_registry.json'),('Report','discovery_report.md')]:
                    path=folder/relative
                    if path.is_file():state['links'][label]=str(path.relative_to(project))
                # The index needs no sequence payloads, plots or full evidence.
                state['transitions']=[]
                state['frontier']={'overview':(state.get('frontier') or {}).get('overview')}
                cache[str(folder)]={'signature':signature,'state':state}
            state=cache[str(folder)]['state']
            runs.append(state)
        except (OSError,ValueError,KeyError) as exc:
            runs.append({'run':folder.name,'status':'snapshot_unavailable','worker_alive':None,
                         'error':type(exc).__name__,'rows':[],'status_counts':{},'links':{}})
    runs.sort(key=lambda state:state.get('worker_alive') is not True)
    return {'generated_at':datetime.now(timezone.utc).isoformat(),'runs':runs,
            'scope':'Saved run records only. Evaluated native function does not establish target activity.'}


def publish_index(project, *, limit=12, cache=None):
    data=collect_runs(project,limit=limit,cache=cache)
    encoded=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
    output=Path(project)/'index.html'
    _atomic_text(output,HTML.replace('/*RUN_DATA*/',encoded))
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--interval',type=float,default=30)
    args=parser.parse_args()
    logs=args.project/'analysis/status_monitor';logs.mkdir(parents=True,exist_ok=True)
    with (logs/'index.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if hasattr(os,'sched_getaffinity'):
            os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
        cache={}
        while True:
            data=publish_index(args.project,cache=cache)
            _atomic_text(logs/'status.json',json.dumps({'pid':os.getpid(),'updated_at':data['generated_at'],
                'runs':len(data['runs']),'read_only':True},indent=2))
            print(json.dumps({'updated_at':data['generated_at'],'runs':len(data['runs'])}),flush=True)
            if not args.watch:return
            time.sleep(max(10,args.interval))


HTML=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="30">
<title>Enzyme discovery · Run status</title><style>
*{box-sizing:border-box}body{margin:0;color:#1d3342;background:#f4f7fa;font:14px/1.5 system-ui,sans-serif}main{max-width:1440px;margin:auto;padding:28px}header{display:flex;align-items:start;justify-content:space-between;gap:20px}h1{font-size:28px;margin:0}h2{font-size:18px;margin:0 0 12px}.muted{color:#526b7c}.badge{background:#e2ebf3;border-radius:14px;padding:4px 10px;display:inline-block}.active{background:#d9f0e3;color:#1b6743}.warn{background:#fff0ce;color:#7e550d}.panel{background:white;border:1px solid #d7e2ea;border-radius:10px;padding:20px;margin:18px 0}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center}select,input,button{font:inherit;border:1px solid #9fb4c3;border-radius:6px;padding:9px;background:white}#run-select{max-width:780px;flex:1}#search{min-width:280px;flex:1}.stats{display:grid;grid-template-columns:repeat(6,1fr);gap:12px}.stat{padding:14px;background:#f6f9fb;border-radius:7px}.stat strong{display:block;font-size:27px}.notice{border-left:4px solid #ae7e25;padding:10px 14px;background:#fff7e6}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.grid .panel{margin:0}.links{display:flex;gap:18px;flex-wrap:wrap;margin-top:14px}a{color:#1d62a0}.table-wrap{overflow:auto;max-height:650px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:10px;border-bottom:1px solid #e2e9ef;vertical-align:top}th{position:sticky;top:0;background:#eaf1f6;white-space:nowrap}.empty{padding:20px;color:#526b7c}.activity{max-height:220px;overflow:auto}.activity p{border-bottom:1px solid #e8eef2;padding-bottom:9px;overflow-wrap:anywhere}.mono{font:12px/1.5 ui-monospace,monospace;overflow-wrap:anywhere}#query{font-size:16px;margin:12px 0}button{cursor:pointer}@media(max-width:900px){.stats{grid-template-columns:repeat(3,1fr)}.grid{grid-template-columns:1fr}header{display:block}}
</style></head><body><main><header><div><h1>Enzyme discovery</h1><p class="muted">Run status, rolling evaluation and candidate rankings.</p></div><div><span id="updated" class="muted"></span><br><button onclick="location.reload()">Refresh saved status</button></div></header>
<div class="controls"><label for="run-select">Campaign</label><select id="run-select"></select><span id="status" class="badge"></span></div>
<p id="query"></p><p id="run-path" class="mono muted"></p><div id="error"></div>
<section class="panel"><div id="stats" class="stats"></div><div id="links" class="links"></div><p class="muted">Evaluated records have native-function evidence. They are not demonstrated hits for the requested reaction. InterPro groups are annotation groups, not verified functional independence.</p></section>
<div class="grid"><section class="panel"><h2>Current work</h2><div id="work"></div></section><section class="panel"><h2>Literature and reasoning</h2><div id="reasoning"></div></section></div>
<section class="panel"><h2>Candidate pool</h2><div class="controls"><input id="search" placeholder="Search enzyme, gene, accession or family…" aria-label="Search candidate pool"><select id="state-filter" aria-label="Candidate state"><option value="shortlisted">Suggested shortlist</option><option value="evaluated">All evaluated records</option><option value="">All records</option><option value="screened">Screened</option><option value="discovered">Discovered leads</option><option value="evidence_pending">Evidence pending</option><option value="rejected">Rejected</option></select><span id="row-count" class="muted"></span></div><div class="table-wrap"><table><thead><tr><th>Rank</th><th>Status</th><th>Gene / accession</th><th>Enzyme</th><th>Family</th><th>Evidence / unresolved question</th></tr></thead><tbody id="candidates"></tbody></table></div></section>
<section class="panel"><h2>Recent recorded activity</h2><div id="activity" class="activity"></div></section>
<p class="muted">Read-only saved-state monitor. Refreshes every 30 seconds; it cannot start, stop or steer the discovery worker.</p></main>
<script id="run-data" type="application/json">/*RUN_DATA*/</script><script>
'use strict';const data=JSON.parse(document.getElementById('run-data').textContent),runs=data.runs||[];let current=0;
let saved={};try{saved=JSON.parse(sessionStorage.getItem('enzyme-discovery-status')||'{}')}catch{};current=Math.max(0,runs.findIndex(run=>run.run===saved.run));
function remember(){try{sessionStorage.setItem('enzyme-discovery-status',JSON.stringify({run:runs[current]?.run,query:document.getElementById('search').value,state:document.getElementById('state-filter').value,scroll:window.scrollY}))}catch{}}
window.addEventListener('pagehide',remember);
const el=(tag,text,cls)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node};
const put=(id,text)=>document.getElementById(id).textContent=text, number=value=>typeof value==='number'?value.toLocaleString():'—';
document.getElementById('updated').textContent='Updated '+new Date(data.generated_at).toLocaleString();
for(const [i,run] of runs.entries()){const option=el('option',(run.worker_alive===true?'● ':'')+run.run);option.value=i;document.getElementById('run-select').append(option)}
function render(){const run=runs[current];if(!run){put('query','No saved discovery campaigns found.');return}const counts=run.status_counts||{},status=document.getElementById('status');
 const live=run.worker_alive===true,statusText=run.status==='running'&&run.worker_alive===false?'Stopped · saved state remains':run.status;
 status.textContent=(live?'Worker active · ':'')+(statusText||'Unknown');status.className='badge '+(live?'active':run.error?'warn':'');
 put('query',run.query||'');put('run-path',run.run);put('error',run.error?'Snapshot unavailable: '+run.error:'');
 const stats=document.getElementById('stats');stats.replaceChildren();for(const [title,value] of [['Evaluated',(counts.evaluated||0)+(counts.ranked||0)],['Ready to evaluate',run.ready_count],['Discovered leads',counts.discovered||0],['Evidence pending',counts.evidence_pending||0],['InterPro groups',run.interpro_groups],['Total records',run.candidate_count]]){const box=el('div',undefined,'stat');box.append(el('span',title),el('strong',number(value)));stats.append(box)}
 const links=document.getElementById('links');links.replaceChildren();for(const [label,href] of Object.entries(run.links||{})){const a=el('a',label);a.href=href;links.append(a)}
 const work=document.getElementById('work');work.replaceChildren();work.append(el('p','Serving model: '+(run.model||'unrecorded')),el('p','This allowance: '+number(run.invocation_tokens)+' / '+number(run.token_budget)+' tokens'),el('p','Accounted campaign total: '+number(run.tokens)+' tokens'),el('p','Last research event: '+(run.last_research_event?new Date(run.last_research_event).toLocaleString():'none recorded')));
 if(run.ready_count)work.append(el('p',run.ready_count+' screened records remain ready for rolling evaluation.','notice'));else work.append(el('p','No ready evaluation backlog in the saved snapshot.','muted'));
 const lit=run.literature||{},reason=run.reasoning||{},box=document.getElementById('reasoning');box.replaceChildren();for(const text of [number(lit.unique_papers)+' unique paper records · '+number(lit.full_texts)+' acquired full texts',number(reason.registered_predictions)+' registered predictions · '+number(reason.completed_actions)+' completed actions · '+number(reason.concluded_reviews)+' concluded reviews'])box.append(el('p',text));box.append(el('p','Retrieved papers and recorded reasoning are not proof of target activity.','muted'));
 const activity=document.getElementById('activity');activity.replaceChildren();for(const row of [...(run.decisions||[])].reverse().slice(0,8))activity.append(el('p',(row.time?new Date(row.time).toLocaleTimeString()+' · ':'')+row.type+' · '+(row.detail||row.tools.join(', '))));renderCandidates();}
function renderCandidates(){const run=runs[current]||{},query=document.getElementById('search').value.toLowerCase(),state=document.getElementById('state-filter').value,rows=(run.rows||[]).filter(row=>(!state||(state==='shortlisted'?Boolean(row.rank):state==='evaluated'?['evaluated','ranked'].includes(row.status):row.status===state))&&(!query||JSON.stringify(row).toLowerCase().includes(query)));put('row-count',rows.length+' shown');const body=document.getElementById('candidates');body.replaceChildren();for(const row of rows){const tr=el('tr');for(const value of [row.rank||'—',row.status,[row.gene,row.accession].filter(Boolean).join(' · '),row.database_name||row.enzyme,row.family||row.family_id||'Unresolved',row.reason])tr.append(el('td',value||'—'));body.append(tr)}if(!rows.length){const tr=el('tr'),td=el('td','No records match this filter.','empty');td.colSpan=6;tr.append(td);body.append(tr)}}
document.getElementById('run-select').value=String(current);document.getElementById('search').value=saved.query||'';if(saved.state!==undefined)document.getElementById('state-filter').value=saved.state;
document.getElementById('run-select').addEventListener('change',event=>{current=Number(event.target.value);render();remember()});document.getElementById('search').addEventListener('input',()=>{renderCandidates();remember()});document.getElementById('state-filter').addEventListener('change',()=>{renderCandidates();remember()});render();requestAnimationFrame(()=>window.scrollTo(0,saved.scroll||0));
</script></body></html>'''


if __name__=='__main__':
    main()
