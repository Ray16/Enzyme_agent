"""Analyze hash-verified local campaign structures using explicit reactive selectors."""
from pathlib import Path
import argparse,csv,hashlib,json,os,time
import numpy as np
from biotite.sequence import ProteinSequence
from geometry import evaluate,read_structure

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(path,text):
 tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(text);tmp.replace(path)
def main():
 p=argparse.ArgumentParser();p.add_argument('config',type=Path);a=p.parse_args();cfg=json.loads(a.config.read_text());root=Path(cfg['campaign_dir']);inv=json.loads((root/'download_inventory.json').read_text());rows=[];seqs={};errors=[]
 for remote in inv['rows']:
  path=root/'downloads'/Path(remote['path']).relative_to(cfg['remote_root']);h=remote.get('structure_sha256')
  if not path.exists() or not h or sha(path)!=h:continue
  row=dict(id=remote['id'],input_id=remote['input_id'],path=str(path),technical_pass=remote.get('technical_pass'),ame_gate='pending',geometry_scope='conditioned generation',error=remote.get('error',''))
  try:
   atoms=read_structure(path);row.update(evaluate(atoms,cfg['geometry']));protein=atoms[atoms.chain_id==cfg.get('protein_chain','A')];ca=protein[protein.atom_name=='CA']
   if len(set(ca.res_id))!=len(ca):raise ValueError('Duplicate protein residue numbering')
   seq=''.join(ProteinSequence.convert_letter_3to1(str(x)) for _,x in sorted(zip(ca.res_id,ca.res_name)));digest=hashlib.sha256(seq.encode()).hexdigest();row.update(protein_length=len(seq),sequence_sha256=digest)
   if remote.get('sequence_sha256') and digest!=remote['sequence_sha256']:raise ValueError('Sequence hash disagrees with remote analysis')
   seqs.setdefault(digest,{'sequence':seq,'ids':[]})['ids'].append(remote['id'])
   shard=remote['id'].split('/')[0];timing=inv.get('timings',{}).get(shard,{})
   row.update(generation_entrypoint_seconds=timing.get('generation_entrypoint_seconds'),average_seconds_per_sample=timing.get('entrypoint_seconds_per_completed_sample'))
  except Exception as e:row['local_analysis_error']=str(e);errors.append(dict(id=remote['id'],error=str(e)))
  rows.append(row)
 out=root/'local_analysis';out.mkdir(exist_ok=True)
 cols=['id','input_id','technical_pass','protein_length','sequence_sha256','attack_chain','attack_residue','attack_atom','distance_A','angle_deg','generation_entrypoint_seconds','average_seconds_per_sample','ame_gate','geometry_scope','path','error','local_analysis_error']
 tmp=out/f'metrics.{os.getpid()}.tmp'
 with tmp.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
 tmp.replace(out/'reactive_geometry.csv')
 groups={}
 for motif in sorted(set(r['input_id'] for r in rows)):
  valid=[r for r in rows if r['input_id']==motif and 'distance_A' in r and not r.get('local_analysis_error')]
  groups[motif]=dict(structures=len(valid),distance_A_range=[min(r['distance_A'] for r in valid),max(r['distance_A'] for r in valid)] if valid else None,angle_deg_range=[min(r['angle_deg'] for r in valid),max(r['angle_deg'] for r in valid)] if valid else None)
 summary=dict(downloaded_and_hashed=len(rows),unique_sequences=len(seqs),technical_pass=sum(r.get('technical_pass') is True for r in rows),local_analysis_errors=errors,by_motif=groups,duplicate_groups={h:s['ids'] for h,s in seqs.items() if len(s['ids'])>1},AME_status='pending independent refolding',geometry_scope='conditioned generation geometry; not independent cofold scores',updated=time.time())
 atomic(out/'summary.json',json.dumps(summary,indent=2)+'\n');atomic(out/'sequences.fasta',''.join('>'+s['ids'][0]+' sha256='+h+'\n'+s['sequence']+'\n' for h,s in sorted(seqs.items())))
 print(json.dumps({k:summary[k] for k in ('downloaded_and_hashed','unique_sequences','technical_pass','local_analysis_errors','by_motif')}))
if __name__=='__main__':main()
