"""Render the verified historical descriptor inventory as a standalone LaTeX table."""
from pathlib import Path
import collections,csv,hashlib,json,subprocess
import fitz

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'exp_data/amide_synthetase/results/structural_proxies/attempts/proxy_inventory_20261011/historical_feature_inventory.csv'
GROUPS=[
 ('bat','Activation geometry and catalytic environment',
  'Attack angles and near-attack fractions; local metal geometry; reactive-group exposure; nearby charged residues.'),
 ('s2nac','Amine--acyl approach geometry (step 2)',
  r'Amine N--acyl C distance; B\"urgi--Dunitz angle; near-attack fraction; accepted-pose count.'),
 ('preorg','Reactive-group preorganization',
  'Carboxylate positional spread; attack-direction variability; mean angle; bidentate-contact fraction.'),
 ('hyp','Hypothesis-specific geometric descriptors',
  'Oxyanion contacts; substrate exposure and access; amine approach; adenine burial; accepted-pose count.'),
 ('sasa','Ligand exposure and pocket engagement',
  'Solvent-accessible surface areas of selected reactive atoms; intermediate--protein separation.'),
 ('tunnel','Substrate-access tunnel geometry',
  'Tunnel enclosure and length under alternative clearance thresholds.'),
 ('elec','Amine-site electrostatic environment',
  'Nearby acidic oxygens; distance to a basic residue; net charge within alternative probe radii.'),
 ('esp','Electrostatic potential and electric field',
  r'Potential and projected field at ATP P$_\alpha$ and the acyl carbonyl; ensemble means and variation.'),
 ('pka',r'Ionizable-residue and p$K_a$ descriptors',
  r'Histidine p$K_a$ proximity to 7; minimum lysine p$K_a$; ionizable-residue counts and ensemble variation.'),
 ('mg','Magnesium coordination geometry',
  'Mg--phosphorus and Mg--oxygen distances; coordination and bridging fractions across distance thresholds.'),
 ('psiptm','Protein--ligand interface confidence',
  'Directional pairwise ipTM; symmetric and extrema-based summaries across predicted poses.'),
 ('boltz','Minimum substrate--ATP distance',
  'Minimum donor oxygen--reactive phosphorus distance in the corresponding prediction ensemble.'),
]


def build():
    entries=list(csv.DictReader(SOURCE.open()))
    counts=collections.Counter(r['feature'].split(':')[0] for r in entries if r['status']=='retained')
    assert sum(counts.values())==88 and len(entries)==98
    assert set(counts)=={r[0] for r in GROUPS}
    lines=[r'''\documentclass{article}
\usepackage[paperwidth=16in,paperheight=10.5in,margin=0.42in]{geometry}
\usepackage[T1]{fontenc}\usepackage{lmodern,amsmath,booktabs,tabularx,array}
\usepackage[table]{xcolor}
\renewcommand{\familydefault}{\sfdefault}\pagestyle{empty}\setlength{\parindent}{0pt}
\definecolor{maroon}{HTML}{890B23}\definecolor{pale}{HTML}{F8EDEF}
\definecolor{stripe}{HTML}{F4F5F6}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\setlength{\tabcolsep}{9pt}\renewcommand{\arraystretch}{1.24}
\begin{document}
{\fontsize{30}{35}\selectfont\bfseries\color{maroon} Structural descriptors explored\par}
\vspace{10pt}
{\fontsize{16}{20}\selectfont Historical inventory grouped by the physical quantity being assessed.}\par
\vspace{15pt}
\fontsize{15.5}{20}\selectfont
\rowcolors{2}{stripe}{white}
\begin{tabularx}{\textwidth}{@{}>{\bfseries\raggedright\arraybackslash}p{4.2in}Y>{\raggedleft\arraybackslash}p{1.1in}@{}}
\toprule
\rowcolor{pale} Descriptor family & \textbf{Representative quantities} & \textbf{Distinct definitions}\\
\midrule''']
    for key,label,examples in GROUPS:
        lines.append(f'{label} & {examples} & {counts[key]} '+r'\\[3pt]')
    lines.extend([r'''\midrule
\rowcolor{pale}\textbf{Total} & Distinct, nonconstant definitions in the historical inventory & \textbf{88}\\
\bottomrule\end{tabularx}\par\vspace{14pt}
{\fontsize{11.5}{15}\selectfont
\textbf{Counting rule:} 98 exported fields = 88 distinct nonconstant definitions + 5 duplicate exports + 5 constant or undefined fields.
Counts include formula variants, threshold variants and sampling diagnostics; they are not counts of independent mechanistic hypotheses.
Related quantities can appear in different groups when their definitions or modeling setups differ. Later follow-up formulas overlap this inventory and are not added to these counts.\\[5pt]
SASA = solvent-accessible surface area; ipTM = interface predicted TM score. Pose variation describes generated predictions, not a molecular-dynamics trajectory.
This inventory does not establish that every descriptor predicts experimental activity.}
\end{document}'''])
    stem=HERE/'structural_proxy_search_inventory'
    stem.with_suffix('.tex').write_text('\n'.join(lines)+'\n')
    subprocess.run(['pdflatex','-interaction=batchmode','-halt-on-error',f'-output-directory={HERE}',str(stem.with_suffix('.tex'))],
                    check=True,capture_output=True,text=True)
    log=stem.with_suffix('.log').read_text()
    assert 'Overfull \\hbox' not in log and 'Overfull \\vbox' not in log
    pdf=fitz.open(stem.with_suffix('.pdf'));assert len(pdf)==1
    p=pdf[0]
    p.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(stem.with_suffix('.png'))
    svg=p.get_svg_image(text_as_path=False)
    stem.with_suffix('.svg').write_text('\n'.join(x.rstrip() for x in svg.splitlines())+'\n')
    report={'source':str(SOURCE.relative_to(ROOT)),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'total_distinct_definitions':88,'exported_fields':98,'latex_overflow':False,
            'groups':[{'source_prefix':k,'display_name':label,'examples':examples,'count':counts[k]} for k,label,examples in GROUPS]}
    (HERE/'descriptor_table_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(stem.with_suffix('.tex'))
    return report


if __name__=='__main__':build()
