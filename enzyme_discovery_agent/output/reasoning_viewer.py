"""Offline visual inspection of saved explicit reasoning and revision links."""
import json
from pathlib import Path


def write_reasoning_viewer(destination, trace, *, filename="trace.html", source_prefix="../", export_prefix=""):
    """Embed data safely; the viewer makes no network requests or model calls."""
    payload=json.dumps(trace,ensure_ascii=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    document=HTML.replace('/*TRACE_DATA*/',payload).replace('/*SOURCE_PREFIX*/',json.dumps(source_prefix)).replace('__EXPORT_PREFIX__',export_prefix)
    path=Path(destination)/filename
    path.write_text(document)
    return path


HTML=r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Saved scientific reasoning</title><style>
:root{color-scheme:light;--ink:#172b3a;--muted:#536775;--line:#d9e2e8;--blue:#2563aa;--paper:#f4f7fa}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,sans-serif}
header,main{max-width:1200px;margin:auto;padding:24px}header{padding-bottom:10px}h1{font-size:28px;margin:0 0 8px}h2{font-size:18px}p{margin:8px 0}
.muted{color:var(--muted)}.notice{background:#fff5dc;border-left:4px solid #b8841b;padding:12px 16px;margin:16px 0}
.stats,.controls,.links{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.stat{background:white;border:1px solid var(--line);border-radius:8px;padding:10px 18px}.stat strong{font-size:22px;margin-right:8px}
input,select{font:inherit;padding:9px;border:1px solid #aabcc8;border-radius:5px;background:white}input{flex:1;min-width:230px}a{color:var(--blue)}button{cursor:pointer}
.graph{max-height:440px;overflow:auto;background:white;border:1px solid var(--line);border-radius:9px;margin:16px 0}svg text{font:12px system-ui,sans-serif;fill:var(--ink)}svg .node{cursor:pointer}svg .node:focus{outline:3px solid #6ca4df}
.card{background:white;border:1px solid var(--line);border-left:5px solid var(--blue);border-radius:8px;padding:18px;margin:14px 0;scroll-margin-top:12px}.card h3{font-size:17px;margin:0}.card .meta{font:12px/1.5 ui-monospace,monospace;color:var(--muted);word-break:break-word}
.field{margin-top:12px}.field b{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}.field p{white-space:pre-wrap;overflow-wrap:anywhere}.tag{display:inline-block;border-radius:12px;background:#e5eef7;padding:2px 9px;font-size:12px;margin-right:8px}
pre{overflow:auto;max-height:460px;padding:14px;background:#f1f5f8;font:12px/1.5 ui-monospace,monospace}details{margin-top:14px}summary{cursor:pointer}.empty{padding:20px;color:var(--muted)}.pass{color:#176b42}.fail{color:#a12727}
</style></head><body><header><h1>Saved scientific reasoning</h1>
<p class="muted">Predictions, observations, revisions and participant interpretations from this campaign.</p>
<div class="notice">These are explicit recorded hypotheses and actions. Source integrity does not establish scientific correctness or target activity. Automated evidence processing is separate from a model decision.</div>
<div id="integrity"></div><p id="run" class="meta"></p><div class="stats" id="stats"></div><div id="audit-summary" class="notice"></div>
<p class="links"><a href="__EXPORT_PREFIX__trace.json">Complete JSON export</a><a href="__EXPORT_PREFIX__trace.md">Readable Markdown</a><a href="__EXPORT_PREFIX__trace.jsonl">Append-only journal</a></p>
</header><main><div class="controls"><input id="query" aria-label="Search reasoning" placeholder="Search candidate, hypothesis, source or uncertainty…"><select id="type" aria-label="Event type"><option value="">All event types</option></select><span id="count" class="muted"></span></div>
<h2>Decision and revision links</h2><p class="muted">Solid links connect a prediction to its result; dashed links show a recorded revision or review. Click a node to inspect its record.</p><div class="graph" id="graph"></div>
<h2>Recorded evidence and reasoning</h2><div id="records"></div></main>
<script id="trace-data" type="application/json">/*TRACE_DATA*/</script><script>
'use strict';
const sourcePrefix=/*SOURCE_PREFIX*/;
const trace=JSON.parse(document.getElementById('trace-data').textContent), events=trace.events||[], ns='http://www.w3.org/2000/svg';
const titles={prediction_registered:'Prediction before action',observation_recorded:'Observed result',review_concluded:'Provisional conclusion',participant_roles_recorded:'Participant roles',reaction_step_recorded:'Mapped reaction step',outcome_interpreted:'Outcome interpretation',mechanism_hypotheses_recorded:'Mechanism hypotheses',target_assessment_recorded:'Candidate assessment',target_review_batch_recorded:'Assessment attempt'};
const colors={prediction_registered:'#2563aa',observation_recorded:'#347a65',review_concluded:'#986225',participant_roles_recorded:'#7652a1',reaction_step_recorded:'#7652a1'};
const byId=new Map(events.map(e=>[e.id,e]));
const auditDetails=trace.audit_details||{}, auditSummary=trace.audit_summary||{};
function element(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e}
function field(card,label,value){if(value===undefined||value===null||value==='')return;const d=element('div',undefined,'field');d.append(element('b',label),element('p',typeof value==='string'?value:JSON.stringify(value,null,2)));card.append(d)}
function paths(value,found=new Set()){if(Array.isArray(value))value.forEach(v=>paths(v,found));else if(value&&typeof value==='object'){if(typeof value.path==='string'&&!value.path.startsWith('/')&&!value.path.split('/').includes('..'))found.add(value.path);Object.values(value).forEach(v=>paths(v,found))}return found}
const intact=trace.hash_chain_valid&&trace.source_files_intact;
document.getElementById('integrity').append(element('strong',intact?'Journal and referenced source checks passed':'Source verification needs attention',intact?'pass':'fail'));
document.getElementById('run').textContent=trace.run_dir||'';
document.getElementById('audit-summary').textContent='Recorded decisions: '+(auditSummary.registered_decisions??'unknown')+
 '; planned actions including batches: '+(auditSummary.planned_actions??'unknown')+
 '; exact request-linked tokens: '+(auditSummary.linked_request_tokens??'unavailable')+
 ' / '+(auditSummary.run_tokens??'unknown')+' total run tokens. '+(auditSummary.scope||'Usage attribution unavailable for this export.');
for(const [label,type] of [['Predictions','prediction_registered'],['Results','observation_recorded'],['Conclusions','review_concluded'],['Interpretations',null]]){const n=events.filter(e=>type?e.type===type:['participant_roles_recorded','reaction_step_recorded'].includes(e.type)).length;const d=element('div',undefined,'stat');d.append(element('strong',String(n)),element('span',label));document.getElementById('stats').append(d)}
for(const type of [...new Set(events.map(e=>e.type))]){const option=element('option',titles[type]||type);option.value=type;document.getElementById('type').append(option)}
function render(){const query=document.getElementById('query').value.toLowerCase(),type=document.getElementById('type').value;
 const selected=events.filter(e=>(!type||e.type===type)&&(!query||JSON.stringify(e).toLowerCase().includes(query)));
 document.getElementById('count').textContent=selected.length+' / '+events.length+' records';const records=document.getElementById('records');records.replaceChildren();
 if(!selected.length)records.append(element('p',events.length?'No records match the current filter.':'No explicit scientific records have been captured yet. This does not imply that no automated tool work occurred.','empty'));
 for(const event of selected){const r=event.record||{},card=element('article',undefined,'card');card.id='event-'+event.sequence;card.style.borderLeftColor=colors[event.type]||'#536775';
  card.append(element('h3',(titles[event.type]||event.type)+(r.accession?' · '+r.accession:'')),element('div','#'+event.sequence+' · '+event.occurred_at+' · '+event.capture_mode,'meta'),element('div',event.id,'meta'));
  if(event.capture_mode!=='live')card.append(element('p','Reconstructed from saved artifacts; this is not proof of prospective capture.','notice'));
  for(const [key,label] of [['scope','Scope'],['hypothesis','Hypothesis'],['alternative','Competing explanation'],['prediction','Prediction'],['contrasting_prediction','Contrasting prediction'],['why_this_test','Why this test'],['tool_name','Action'],['arguments','Action arguments'],['status','Execution status'],['summary','Observed summary'],['conclusion','Provisional conclusion'],['belief_update','Belief update'],['revision_trigger','Revisit when']])field(card,label,r[key]);
  if(r.outcome){field(card,'Observed outcome',r.outcome.summary);field(card,'Outcome status',r.outcome.status)}
  const audit=auditDetails[event.id]||{};
  if(event.type==='prediction_registered'){
   field(card,'Complete planned action batch',audit.actions);
   field(card,'Per-action observations and saved candidate facts',audit.outcomes);
   if((audit.candidate_revisits||[]).length)field(card,'Earlier decisions involving these candidates (revisits may be justified)',audit.candidate_revisits);
   field(card,'Exact request-linked token usage',audit.request_usage||'Not available; no cost inferred.');
   for(const issue of audit.issues||[])card.append(element('p',issue,'notice'));
   for(const assessment of audit.candidate_assessments||[]){const related=byId.get(assessment.event_id);if(related){
    const link=element('a','Saved assessment of '+(assessment.protein_name||'unnamed protein')+' ('+assessment.accession+'): '+assessment.decision+' · '+assessment.recorded_at);
    link.href='#event-'+related.sequence;const p=element('p');p.append(link);card.append(p)}}
   field(card,'Audit provenance',audit.provenance);
  }
  if(event.type==='target_assessment_recorded'){field(card,'Candidate',(audit.candidate_name||'Name not recorded')+' ('+r.accession+')');for(const [key,label] of [['decision','Decision'],['evidence_level','Recorded evidence level (not independently adjudicated)'],['reason','Rationale'],['mechanism_and_step','Mechanism and step'],['target_roles','Participant roles'],['strengths','Support'],['constraints','Constraints'],['unknowns','Unknowns'],['revisit_when','Reopen when']])field(card,label,r[key])}
  if(event.type==='target_review_batch_recorded'){field(card,'Assessment results',r.result);field(card,'Unresolved attempts and correction feedback',r.attempts)}
  if(event.type==='mechanism_hypotheses_recorded')field(card,'Provisional search hypotheses',r.families);
  if(typeof r.interpretation==='string')field(card,'Interpretation',r.interpretation);
  if(r.interpretation&&typeof r.interpretation==='object'){const step=r.interpretation;field(card,'Local bond changes correspond',step.local_bond_changes_match);field(card,'Interpretation',step.interpretation||step.participants);field(card,'Native mapped reaction',step.native_reaction);field(card,'Proposed target reaction',step.target_reaction);field(card,'Activation and site-access assessment',step.activation_assessment);field(card,'Remaining ends and continuation limits',step.continuation_assessment);field(card,'Verification scope',step.validation_scope)}
  if(r.revises){const ref=byId.get('prediction_registered:'+r.revises);const link=element('a','Revises decision '+r.revises.slice(0,8));if(ref)link.href='#event-'+ref.sequence;card.append(link)}
  const links=element('div',undefined,'links');for(const path of paths(r)){const a=element('a',path);a.href=sourcePrefix+path;links.append(a)}card.append(links);
  const details=element('details'),summary=element('summary','Complete saved event and checksums');details.append(summary);details.addEventListener('toggle',()=>{if(details.open&&!details.querySelector('pre'))details.append(element('pre',JSON.stringify(event,null,2)))});card.append(details);records.append(card)
 }
 const graph=document.getElementById('graph');graph.replaceChildren();const svg=document.createElementNS(ns,'svg');svg.setAttribute('width','1000');svg.setAttribute('height',String(Math.max(95,selected.length*72+24)));svg.setAttribute('role','img');svg.setAttribute('aria-label','Recorded prediction, observation and revision graph');
 const positions=new Map(selected.map((e,i)=>[e.id,{x:e.type==='prediction_registered'?70:e.type==='observation_recorded'?370:e.type==='review_concluded'?700:520,y:i*72+38}]));
 function edge(from,to,dashed){if(!positions.has(from)||!positions.has(to))return;const a=positions.get(from),b=positions.get(to),line=document.createElementNS(ns,'path');line.setAttribute('d',`M ${a.x} ${a.y} C ${a.x+100} ${a.y},${b.x-100} ${b.y},${b.x} ${b.y}`);line.setAttribute('fill','none');line.setAttribute('stroke',dashed?'#b07726':'#8a9ca8');line.setAttribute('stroke-width','2');if(dashed)line.setAttribute('stroke-dasharray','5 4');svg.append(line)}
 for(const e of selected){const r=e.record||{};if(e.type==='observation_recorded')edge('prediction_registered:'+r.decision_id,e.id,false);if(r.revises)edge(byId.has('observation_recorded:'+r.revises)?'observation_recorded:'+r.revises:'prediction_registered:'+r.revises,e.id,true)}
 for(const e of selected){const p=positions.get(e.id),group=document.createElementNS(ns,'g');group.classList.add('node');group.setAttribute('tabindex','0');group.setAttribute('role','button');group.setAttribute('aria-label',(titles[e.type]||e.type)+' '+e.sequence);const circle=document.createElementNS(ns,'circle');circle.setAttribute('cx',p.x);circle.setAttribute('cy',p.y);circle.setAttribute('r','8');circle.setAttribute('fill',colors[e.type]||'#536775');const text=document.createElementNS(ns,'text');text.setAttribute('x',p.x+16);text.setAttribute('y',p.y+4);text.textContent='#'+e.sequence+' '+(titles[e.type]||e.type);group.append(circle,text);const focus=()=>document.getElementById('event-'+e.sequence)?.scrollIntoView({behavior:'smooth',block:'start'});group.addEventListener('click',focus);group.addEventListener('keydown',key=>{if(key.key==='Enter'||key.key===' ')focus()});svg.append(group)}graph.append(svg)
}
document.getElementById('query').addEventListener('input',render);document.getElementById('type').addEventListener('change',render);render();
</script></body></html>'''
