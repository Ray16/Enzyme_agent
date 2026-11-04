"""Apply the tutorial's three AME thresholds, retaining missing metrics as pending."""
from __future__ import annotations
import argparse,csv,math
from pathlib import Path

CRITERIA={
 'refolded_backbone_rmsd.rmsd':2.0,
 'refolded_motif_allatom_rmsd.rmsd':1.5,
 'clash_metrics.frac_atoms_clashing_inter_residue_polymer_ligand_1p5A':0.0,
}

def gate(row):
    missing=[];failed=[]
    for name,limit in CRITERIA.items():
        try:value=float(row[name])
        except (KeyError,ValueError,TypeError):missing.append(name);continue
        if not math.isfinite(value) or value<0:missing.append(name)
        elif value>limit:failed.append(f'{name}={value:g} > {limit:g}')
    status='fail' if failed else ('pending' if missing else 'pass')
    return status,'; '.join(failed+[f'missing/invalid: {k}' for k in missing])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('metrics',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with a.metrics.open(newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames or [];rows=list(reader)
    for row in rows:row['ame_gate'],row['ame_gate_reason']=gate(row)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=[k for k in fields if k not in ('ame_gate','ame_gate_reason')]+['ame_gate','ame_gate_reason']);w.writeheader();w.writerows(rows)
    print({k:sum(r['ame_gate']==k for r in rows) for k in ('pass','fail','pending')})
if __name__=='__main__':main()
