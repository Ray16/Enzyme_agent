"""Build independent, source-linked family inventory figures; no deck assembly."""
from pathlib import Path
import json
import subprocess
import fitz

HERE=Path(__file__).resolve().parent
R={
 'NIS':'https://pmc.ncbi.nlm.nih.gov/articles/PMC7325057/',
 'ANL':'https://doi.org/10.1002/ange.201804592',
 'GH3':'https://pmc.ncbi.nlm.nih.gov/articles/PMC11253546/',
 'AsnA':'https://iubmb.qmul.ac.uk/enzyme/EC6/3/1/1.html',
 'AsnB':'https://pmc.ncbi.nlm.nih.gov/articles/PMC2610506/',
 'DdaG':'https://doi.org/10.1021/bi9013165',
 'NAD':'https://iubmb.qmul.ac.uk/enzyme/EC6/3/1/5.html',
 'aaRS':'https://doi.org/10.1186/1471-2164-13-621',
 'BirA':'https://iubmb.qmul.ac.uk/enzyme/EC6/3/4/15.html',
 'LplA':'https://iubmb.qmul.ac.uk/enzyme/EC6/3/1/20.html',
 'PylC':'https://doi.org/10.1016/j.jmb.2012.09.007',
 'GspS':'https://pmc.ncbi.nlm.nih.gov/articles/PMC1698887/',
 'TryS':'https://pmc.ncbi.nlm.nih.gov/articles/PMC2427367/',
 'PuuA':'https://pubmed.ncbi.nlm.nih.gov/18495664/',
 'GCL':'https://pmc.ncbi.nlm.nih.gov/articles/PMC523444/',
 'Mur':'https://www.rcsb.org/annotations/1P3D',
 'FolC':'https://journals.iucr.org/d/issues/2008/07/00/gx5127/gx5127.pdf',
 'CphA':'https://doi.org/10.1038/s41467-022-31542-7',
 'IaaL':'https://iubmb.qmul.ac.uk/enzyme/EC6/3/2/20.html',
 'PafA':'https://pmc.ncbi.nlm.nih.gov/articles/PMC3039397/',
 'GmaS':'https://iubmb.qmul.ac.uk/enzyme/EC6/3/4/12.html',
 'PurD':'https://www.rcsb.org/structure/1GSO',
}
# Labels deliberately mix family, subfamily and functional-group levels;
# overlapping entries are not counted as independent structural folds.
AMP=[
 ('NIS synthetases','IucA, IucC, DesD',
  'Carrier-independent siderophore assembly; substrates and cyclization behavior differ.','NIS'),
 ('ANL amide ligases','McbA',
  'Direct acid--amine coupling; other ANL enzymes perform different second half-reactions.','ANL'),
 ('GH3 acyl acid amido synthetases','GH3.5, GH3.11/JAR1, GH3.17',
  'Acyl-acid / phytohormone conjugation to amino acids; listed as a functional family.','GH3'),
 ('Asparagine synthetase A','AsnA',
  'Aspartate side-chain amidation using ammonia; specificity is not general amine ligation.','AsnA'),
 ('Asparagine synthetase B','AsnB',
  'Glutamine-derived ammonia supplied to the adenylating synthetase domain.','AsnB'),
 ('AsnB-related ring-forming synthetases',r'$\beta$-lactam synthetase; carbapenam synthetase (CarA)',
  'Intramolecular amide closure after adenylation; distinct from intermolecular diad formation.','AsnB'),
 ('Other characterized AMP-forming ligases','DdaG',
  'Fumarate--diaminopropanoate ligation; included without asserting an unverified fold assignment.','DdaG'),
 ('NAD synthetases','NadE / NAD synthetase',
  'Amidation of deamido-NAD; a specialized nucleotide substrate rather than a free diacid.','NAD'),
 ('NRPS adenylation domains (related)','PheA / DhbE-type A domains',
  'Carrier-dependent context: activation/loading precedes downstream peptide-bond formation.','ANL'),
 ('Aminoacyl-tRNA synthetases (related)','Class I and class II aaRS',
  'Acyl transfer to tRNA forms an ester; peptide-bond formation occurs downstream.','aaRS'),
 ('Biotin--protein ligases (related)','BirA / holocarboxylase synthetase',
  'Protein-bound lysine acceptor; covalent cofactor attachment rather than free-diamine ligation.','BirA'),
 ('Lipoate--protein ligases (related)','LplA / LplJ',
  'Lipoyl-AMP intermediate; transfer to a lysine in a protein lipoyl domain.','LplA'),
]
ADP=[
 ('ATP-grasp peptide ligases','DdaF; D-Ala--D-Ala ligases',
  'Peptide formation from activated carboxylates; family members differ in substrate requirements.','DdaG'),
 ('ATP-grasp isopeptide ligases','PylC',
  'Lysine side-chain acylation; structural evidence supports an acyl-phosphate intermediate.','PylC'),
 ('ATP-grasp glutathione / polyamine ligases','GshB, GspS, TryS',
  'Glutathione assembly or conjugation to spermidine; spermidine is a triamine.','GspS'),
 ('ATP-grasp purine-pathway ligases','PurD / glycinamide ribonucleotide synthetase',
  'Glycine ligation to phosphoribosylamine; specialized native substrates.','PurD'),
 ('GS-fold ligases','GS, PuuA, GmaS',
  'Different nitrogen donors: ammonia, putrescine or methylamine; a fold does not fix scope.','PuuA'),
 ('Glutamate--cysteine ligases',r'GCL / GshA / $\gamma$-glutamylcysteine synthetase',
  'Glutamate side-chain activation followed by cysteine addition; distinct from GshB.','GCL'),
 ('Mur peptide ligases','MurC, MurD, MurE, MurF; Mpl',
  'Peptidoglycan-precursor ligation; nucleotide-linked acceptors need separate accommodation tests.','Mur'),
 ('Folate glutamate ligases','FolC / DHFS / FPGS',
  'Folate synthesis or polyglutamylation via acyl-phosphate chemistry.','FolC'),
 ('Multidomain polymer ligases','Cyanophycin synthetase CphA1',
  'ATP-grasp G domain + Mur-like M domain; iterative peptide and isopeptide formation.','CphA'),
 ('Other characterized ADP-forming activity','IaaL / indoleacetate--lysine synthetase',
  'ADP production is established by enzyme nomenclature; no specific fold is assigned here.','IaaL'),
 ('Protein-conjugating ligases (related)','PafA / Pup ligase',
  'Protein-bound glutamate activation and transfer to substrate lysine; not free-substrate coupling.','PafA'),
 ('tRNA-dependent amidotransferases (related)','GatCAB system',
  'tRNA-bound acidic amino-acid side-chain amidation; different substrate context.','PafA'),
]

PREAMBLE=r'''\documentclass{article}
\usepackage[paperwidth=16in,paperheight=12in,margin=0.45in]{geometry}
\usepackage[T1]{fontenc}\usepackage{lmodern,amsmath,booktabs,tabularx,array,hyperref}
\usepackage[table]{xcolor}
\renewcommand{\familydefault}{\sfdefault}\pagestyle{empty}\setlength{\parindent}{0pt}
\definecolor{maroon}{HTML}{890B23}\definecolor{pale}{HTML}{F8EDEF}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\setlength{\tabcolsep}{8pt}\renewcommand{\arraystretch}{1.16}
\begin{document}
'''


def make_table(route, entries):
    stem=f'{route.lower()}_forming_family_inventory'
    title=f'{route}-forming ligases: extended family and substrate-context map'
    lines=[PREAMBLE,
           r'{\fontsize{27}{32}\selectfont\bfseries\color{maroon} '+title+r'\par}',
           r'\vspace{10pt}{\fontsize{16}{20}\selectfont Major relevant families and functional groups; overlapping entries are not independent folds.}\par\vspace{16pt}',
           r'\fontsize{16}{20}\selectfont',
           r'\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{3.9in}>{\raggedright\arraybackslash}p{3.25in}Y@{}}',
           r'\toprule\rowcolor{pale}\textbf{Family / functional group} & \textbf{Representative systems} & \textbf{Native context and limits of transfer}\\\midrule']
    for family, examples, context, source in entries:
        lines.append(r'\textbf{'+family+r'} & '+examples+' & '+context+
                     r' \href{'+R[source]+r'}{[source]} \\[5pt]')
    lines.extend([
        r'\bottomrule\end{tabularx}\par\vspace{13pt}',
        r'{\fontsize{13}{16}\selectfont A family label or nucleotide product does not establish activity on adipate and hexamethylenediamine. '
        r'For a new candidate, verify the native reacting partners, reaction direction, intermediate, molecular state and assembly. '
        r'Related carrier-, tRNA- and protein-dependent systems are included for mechanistic breadth, not as interchangeable direct amide ligases. '
        r'Carbamoyl-phosphate synthetase and biotin carboxylases are not placed in the core carboxyl-activating amide-ligase comparison because their activated substrates and reaction stoichiometries differ.}',
        r'\end{document}'])
    tex=HERE/f'{stem}.tex';tex.write_text('\n'.join(lines)+'\n')
    subprocess.run(['pdflatex','-interaction=batchmode','-halt-on-error',
                    f'-output-directory={HERE}',str(tex)],check=True,capture_output=True,text=True)
    d=fitz.open(HERE/f'{stem}.pdf')
    assert len(d)==1,(stem,len(d))
    log=(HERE/f'{stem}.log').read_text()
    assert 'Overfull \\hbox' not in log and 'Overfull \\vbox' not in log,stem
    # Crop unused paper around the table; preserve generous text margins.
    blocks=d[0].get_text('blocks');ink=fitz.Rect(blocks[0][:4])
    for b in blocks[1:]:ink|=fitz.Rect(b[:4])
    crop=ink+(-16,-16,16,16)
    out=fitz.open();p=out.new_page(width=crop.width,height=crop.height)
    p.show_pdf_page(p.rect,d,0,clip=crop)
    p.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(HERE/f'{stem}.png')
    svg=p.get_svg_image(text_as_path=False)
    (HERE/f'{stem}.svg').write_text('\n'.join(x.rstrip() for x in svg.splitlines())+'\n')
    return {'route':route,'entries':len(entries),'one_page':True,'latex_overflow':False}


if __name__=='__main__':
    HERE.mkdir(exist_ok=True)
    checks=[make_table('AMP',AMP),make_table('ADP',ADP)]
    (HERE/'family_inventory.json').write_text(json.dumps({'AMP':AMP,'ADP':ADP,'references':R,
        'scope':'Major relevant amide-ligase families and related activation systems; not an exhaustive count of independent folds.',
        'validation':checks},indent=2)+'\n')
    print(json.dumps(checks))
