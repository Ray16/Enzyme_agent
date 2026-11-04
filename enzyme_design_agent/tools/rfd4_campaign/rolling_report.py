"""Continuously report configurable attack geometry for completed batch records."""
from __future__ import annotations
import argparse,csv,glob,json,os,time,traceback
from pathlib import Path
from geometry import evaluate,read_structure

def write_atomic(path,writer):
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w',newline='') as handle:writer(handle)
    tmp.replace(path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',required=True,help='Glob for completed batch JSON records')
    p.add_argument('--geometry',required=True,type=Path,help='JSON atom-selector definition')
    p.add_argument('--output',required=True,type=Path,help='CSV report to update atomically')
    p.add_argument('--interval',type=float,default=30);p.add_argument('--once',action='store_true')
    a=p.parse_args()
    if a.interval<=0:raise ValueError('interval must be positive')
    spec=json.loads(a.geometry.read_text());cache={}
    columns=['record','structure','technical_qc','attack_chain','attack_residue','attack_atom','distance_A','angle_deg','batch_size','batch_seconds','amortized_seconds_per_structure','error']
    a.output.parent.mkdir(parents=True,exist_ok=True)
    while True:
        rows=[]
        for filename in sorted(glob.glob(a.records)):
            try: record=json.loads(Path(filename).read_text())
            except (OSError,ValueError):continue
            entries=record.get('rows',[]);seconds=record.get('generation_wall_seconds');count=len(entries)
            for entry in entries:
                row=dict(record=filename,structure=entry.get('path',''),technical_qc=entry.get('qc',{}).get('technical_pass'),batch_size=count,batch_seconds=seconds,amortized_seconds_per_structure=seconds/count if seconds is not None and count else None,error='')
                try:
                    f=Path(row['structure']);st=f.stat();key=(str(f),st.st_size,st.st_mtime_ns)
                    if key not in cache:cache[key]=evaluate(read_structure(f),spec)
                    row.update(cache[key])
                except Exception as e:row['error']=f'{type(e).__name__}: {e}'
                rows.append(row)
        def write(f):
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
        write_atomic(a.output,write)
        if a.once:return
        time.sleep(a.interval)
if __name__=='__main__':main()
