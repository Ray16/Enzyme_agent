"""Audit saved inputs and compile a candidacy methods backup; no model execution."""
from pathlib import Path
import collections,csv,hashlib,json,pickle,subprocess
import fitz,yaml
from rdkit import Chem

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PANEL=ROOT/'exp_data/amide_synthetase/boltz2'
MUT=ROOT/'enzyme_design_agent/campaigns/amide_AS9_AS17_AS23_step1_2Mg_saturation_5A_20261019'
SOURCES=[]

def source(p,role):
 p=Path(p)
 SOURCES.append({'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
                 'role':role,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})

panel=[]
for step in [1,2]:
 for p in sorted((PANEL/f'step{step}_1mg/inputs').glob('*.yaml')):
  cfg=yaml.safe_load(p.read_text()); ident=p.stem.removeprefix(f'step{step}_')
  prot=cfg['sequences'][0]['protein']
  with Path(prot['msa']).open() as h:
   reader=csv.DictReader(h);query=next(reader)['sequence']
  assert query==prot['sequence'],p
  ligands=[e['ligand'] for e in cfg['sequences'] if 'ligand' in e]
  assert sum(x.get('ccd')=='MG' for x in ligands)==1,p
  molpath=PANEL/f'step{step}_1mg/results/{ident}/boltz_results_step{step}_{ident}/processed/mols/{p.stem}.pkl'
  mols=pickle.loads(molpath.read_bytes())
  state=[{'key':k,'smiles':Chem.MolToSmiles(m),'formal_charge':Chem.GetFormalCharge(m)} for k,m in mols.items()]
  if step==1:
   assert len(state)==1 and state[0]['formal_charge']==0
   assert 'constraints' not in cfg
   assert cfg['properties'][0]['affinity']['binder']=='D'
  else:
   assert sorted(x['formal_charge'] for x in state)==[-2,1]
   pocket=cfg['constraints'][0]['pocket'];assert pocket['binder']=='C' and pocket['max_distance']==10 and pocket['force']
  pred=molpath.parents[2]/'predictions'/p.stem
  n=len(list(pred.glob(p.stem+'_model_*.pdb')))
  assert n==50,(p,n)
  panel.append({'id':ident,'step':step,'mg_count':1,'n_poses':n,'msa_query_matches':True,'msa_path':prot['msa'],
                'declared_ligands':ligands,'processed_smiles_ligands':state,
                'constraints':cfg.get('constraints',[]),'properties':cfg.get('properties',[])})
  source(p,'saved panel input');source(molpath,'saved predictor ligand graph')
assert collections.Counter(r['step'] for r in panel)=={1:49,2:49}
for ident in {r['id'] for r in panel}:
 assert len({r['msa_path'] for r in panel if r['id']==ident})==1

manifest=json.loads((MUT/'inputs/single_mutants/manifest.json').read_text())
mut=[]
for ident in ['AS9_WT','AS9_G211V','AS23_WT','AS23_P318R']:
 p=MUT/'inputs/single_mutants/inputs'/f'{ident}.yaml';cfg=yaml.safe_load(p.read_text())
 prot=cfg['sequences'][0]['protein']
 with Path(prot['msa']).open() as h:query=next(csv.DictReader(h))['sequence']
 assert query==prot['sequence']
 assert not cfg.get('constraints') and not cfg.get('properties')
 ligands=[e['ligand'] for e in cfg['sequences'] if 'ligand' in e]
 assert sum(x.get('ccd')=='MG' for x in ligands)==2
 substrate=next(x['smiles'] for x in ligands if 'smiles' in x)
 assert Chem.GetFormalCharge(Chem.MolFromSmiles(substrate))==-2
 mut.append({'id':ident,'mg_count':2,'adipate_input_charge':-2,'msa_query_matches':True,
             'constraints':[],'properties':[],'msa':prot['msa']})
 source(p,'mutation figure input');source(Path(prot['msa']),'sequence-specific mutation MSA')
for p,role in [
 (MUT/'chemistry_audit.json','cached CCD ATP formal charge audit'),
 (MUT/'scan_plan.json','sampling and mutation protocol'),
 (MUT/'aurora/runner/scan_worker.py','executed sampling command'),
 (ROOT/'structural_modeling_agent/CANDIDATE_RANKING_POLICY.md','mechanism-specific reaction coordinates'),
 (ROOT/'exp_data/amide_synthetase/results/structural_proxies/tables/model_setup_checks.json','discovery candidate coordinate audit'),
 (ROOT/'exp_data/amide_synthetase/results/analysis/current_two_step_metrics_20261023/summary.txt','paired panel scope and interpretation'),
 (PANEL/'step1_1mg/fleet/fold_loop.sh','panel sampling command'),
 (PANEL/'step2_1mg/fleet/fold_loop.sh','panel sampling command')]:source(p,role)
parser=Path('/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/boltz-2/lib/python3.12/site-packages/boltz/data/parse/schema.py')
source(parser,'local parser: standardization precedes ligand construction when affinity is requested')
audit={'panel':panel,'mutation_examples':mut,'mutation_seed':manifest['seed'],
       'mutation_samples':manifest['n_samples'],'sources':SOURCES,
       'scope':'Saved setup audit only; no new folding, no sensitivity experiment, no change to existing scores.',
       'charge_note':'Panel adipate input charge -2 becomes 0 in all 49 saved processed ligand graphs; local parser standardizes affinity ligands before construction. Cached CCD ATP charge 0 is documented by the mutant campaign chemistry audit; historical CCD cache identity is not independently pinned for every panel run.'}
(HERE/'sources.json').write_text(json.dumps(audit,indent=2)+'\n')

tex=r'''\documentclass{article}
\usepackage[paperwidth=16in,paperheight=9in,left=0.5in,right=0.5in,top=0.32in,bottom=0.32in]{geometry}
\usepackage[T1]{fontenc}\usepackage{lmodern,amsmath,booktabs,tabularx,array}
\usepackage[table]{xcolor}\usepackage{microtype}
\renewcommand{\familydefault}{\sfdefault}\pagestyle{empty}\setlength{\parindent}{0pt}
\definecolor{maroon}{HTML}{890B23}\definecolor{muted}{HTML}{59636C}
\definecolor{tint}{HTML}{F1F5F8}\definecolor{rose}{HTML}{F8EDEF}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\setlength{\tabcolsep}{8pt}\renewcommand{\arraystretch}{1.14}
\begin{document}
{\fontsize{14}{17}\selectfont\color{muted} BACKUP\quad Structural modeling methods}\par\vspace{5pt}
{\fontsize{27}{31}\selectfont\bfseries\color{maroon} Exact inputs behind the reported structural comparisons}\par
\vspace{7pt}
\fontsize{14.7}{17.2}\selectfont
\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{1.35in}YYY@{}}
\toprule
\textbf{Modeling choice} & \textbf{Step 1: activity panel} & \textbf{Step 2: activity panel} & \textbf{Step 1: mutant scan}\\
 & 48 homologs + SfaB reference & Same 49 enzymes & AS9 / AS17 / AS23; matched WT\\
\midrule
\textbf{Chemical state} & ATP (CCD) + adipate; pre-activation complex & Adipyl-AMP intermediate + HMDA; neutral attacking amine & ATP (CCD) + adipate; pre-activation complex\\
\rowcolor{tint}
\textbf{Formal charges} & Adipate: input $-2$; saved processed graph $0$\textsuperscript{*} & Adipyl-AMP $-2$; HMDA $+1$ with one NH$_2$ and one NH$_3^+$ & Adipate $-2$ input; cached CCD ATP $0$\textsuperscript{\ensuremath{\dagger}}\\
\textbf{Metal / protein} & 1 Mg$^{2+}$; one protein chain & 1 Mg$^{2+}$; one protein chain & 2 Mg$^{2+}$; one protein chain\\
\textbf{Spatial restraints} & None; adipate requested as affinity binder\textsuperscript{*} & Forced HMDA--pocket restraint: maximum distance 10\,\AA & None; no affinity property\\
\textbf{MSA} & Cached sequence-specific MSA & Same protein MSA as step 1 & Separate MSA for each mutant; query sequence verified\\
\textbf{Sampling} & Boltz-2; 50 diffusion samples; no explicit seed in launcher & Boltz-2; 50 diffusion samples; no explicit seed in launcher & 50 samples; seed 20261019; 3 recycles; 200 sampling steps\\
\textbf{Reported quantity} & Proximity-weighted O--P$_\alpha$--O3A angle; all 50 poses & Neutral amine-N to activated acyl-C distance & Change in weighted angle versus fresh two-Mg WT\\
\bottomrule
\end{tabularx}
\vspace{10pt}
{\fontsize{16}{19}\selectfont\bfseries Reaction products determine the activation coordinate}\par\vspace{5pt}
\begin{minipage}[t]{0.48\textwidth}
\colorbox{tint}{\parbox{0.95\linewidth}{\fontsize{14}{17}\selectfont
\textbf{AMP-forming: acyl-adenylate}\quad (e.g., DdaG)\\
Carboxyl O attacks P$_\alpha$; leaving bond P$_\alpha$--O3A\\
ATP $\rightarrow$ AMP + PP$_i$\quad |\quad $\theta=\angle$(O, P$_\alpha$, O3A)}}
\end{minipage}\hfill
\begin{minipage}[t]{0.48\textwidth}
\colorbox{tint}{\parbox{0.95\linewidth}{\fontsize{14}{17}\selectfont
\textbf{ADP-forming: acyl-phosphate}\quad (e.g., PuuA)\\
Carboxyl O attacks P$_\gamma$; leaving bond P$_\gamma$--O3B\\
ATP $\rightarrow$ ADP + P$_i$\quad |\quad $\theta=\angle$(O, P$_\gamma$, O3B)}}
\end{minipage}
\vspace{8pt}\par
{\fontsize{12}{14.2}\selectfont
\textbf{Discovery ranking (slide 17):} recorded inputs use 1 or 2 Mg ions by candidate; score the assigned P$_\alpha$ or P$_\gamma$ coordinate. Mg stoichiometry is an input assumption, independent of coordinate assignment.\\[3pt]
\textbf{Sensitivity remains to be tested:} compare within matched setups; the one-Mg panel does not validate transfer to the two-Mg mutant scan. Pose spread measures prediction variability.}
\vspace{6pt}\par
{\fontsize{10}{11.8}\selectfont\color{muted}
\textsuperscript{*}All 49 step-1 processed ligand graphs are neutral despite charged input SMILES; the installed parser standardizes affinity ligands before construction.
\textsuperscript{\ensuremath{\dagger}}The mutant campaign audit records CCD ATP charge 0; do not label it ATP$^{4-}$. Historical CCD identity is not pinned for every panel run.\\
Sources: saved panel YAMLs and ligand graphs; mutant campaign chemistry audit, inputs and launcher; discovery coordinate audit. Full identifiers, hashes and speaker notes accompany this slide.}
\end{document}
'''
(HERE/'structural_modeling_setup.tex').write_text(tex)
result=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','structural_modeling_setup.tex'],cwd=HERE,capture_output=True,text=True)
print(result.stdout[-1400:]);assert result.returncode==0
pdf=fitz.open(HERE/'structural_modeling_setup.pdf');assert len(pdf)==1,len(pdf)
page=pdf[0]
assert 'Overfull' not in (HERE/'structural_modeling_setup.log').read_text()
page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(HERE/'structural_modeling_setup.png')
low=max(b[3] for b in page.get_text('blocks'));assert low<page.rect.height-12,low
(HERE/'validation.json').write_text(json.dumps({'pages':1,'page_size_pt':list(page.rect),'lowest_text_y':low,
 'panel_inputs_verified':98,'panel_ligand_graphs_verified':98,'mutation_examples_verified':4,
 'historical_panel_poses_per_ensemble':50,'overfull_boxes':False},indent=2)+'\n')
original_path=HERE.parent/'Candidacy slides.pdf'
original_digest=hashlib.sha256(original_path.read_bytes()).hexdigest()
original=fitz.open(original_path)
review=fitz.open();review.insert_pdf(original)
added=review.new_page(width=original[0].rect.width,height=original[0].rect.height)
added.show_pdf_page(added.rect,pdf,0)
review.save(HERE/'Candidacy slides - modeling setup backup.pdf',garbage=3,deflate=True)
assert hashlib.sha256(original_path.read_bytes()).hexdigest()==original_digest
print('Ready: one-page backup and appended review copy; audited 98 panel inputs.')
