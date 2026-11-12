"""Concrete chemistry examples, revised after the user's feedback.

Run with /homes/rzhu/miniforge3/envs/palm/bin/python build_revised_examples.py.
No new biological or structural computations are performed.
"""
import json
import xml.etree.ElementTree as ET
import fitz
from PIL import Image
import visual_helpers as v

O=v.OUT
v.begin('A shared fold can hide different substrate chemistry',1)
v.text(110,193,'ENZYME',23,v.MUTED,700)
v.text(490,193,'NATIVE NITROGEN DONOR',23,v.MUTED,700,'middle')
v.rect(85,550,850,145,'#EDF7F3')
rows=[('GS','N',280),('GmaS','CN',430),('PuuA','NCCCCN',580)]
for name,smi,y in rows:
    v.text(120,y+65,name,39,v.INK,700)
    v.mol(smi,330,y-5,470,125)
v.line(960,205,960,695,'#DCE5EB',2)
v.text(1265,235,'TARGET AMINE',23,v.BLUE,700,'middle')
v.text(1265,300,'Hexamethylenediamine',29,v.INK,600,'middle')
v.mol('NCCCCCCN',1030,345,470,160)
v.arrow(825,622,1050,540,v.GREEN,dash=True)
v.text(1265,600,'A diamine-compatible lead',28,v.GREEN,600,'middle')
v.text(800,770,'Search decision: investigate the PuuA-related lead',36,v.BLUE,700,'middle')
v.save('v2_01_native_chemistry_search','Native N-donor examples: IUBMB EC 6.3.1.2 / 6.3.4.12; Kurihara et al., 2008. Target-substrate activity remains unestablished.')

# MA/XA identity and direction are explicit; no score is used as a yield proxy.
v.begin('DesD: the same substrates give different products',2)
v.text(315,200,'DONOR',24,v.MUTED,700,'middle')
v.text(810,200,'ACCEPTOR',24,v.MUTED,700,'middle')
v.text(1320,200,'PRODUCT',24,v.MUTED,700,'middle')
ma='NCCCCCCNC(=O)CCCCC(=O)O'
xa='NCc1ccc(CNC(=O)CCCCC(=O)O)cc1'
def product(seq,y,label,color):
    for i,unit in enumerate(seq):
        c={'M':v.BLUE,'X':v.PURPLE,'A':'#667D8A'}[unit]
        v.rect(1160+i*78,y,68,62,c,r=7)
        v.text(1194+i*78,y+43,unit,35,'white',700,'middle')
    v.text(1315,y+112,label,31,color,600,'middle')

for y,donor,acceptor,ds,ac,seq,lab,col in [
    (250,'MA','XA',ma,xa,'MAXA','Higher yield',v.GREEN),
    (495,'XA','MA',xa,ma,'XAMA','Trace product',v.MUTED),
]:
    v.text(315,y+8,donor,31,v.INK,700,'middle')
    v.text(810,y+8,acceptor,31,v.INK,700,'middle')
    v.mol(ds,90,y+22,450,135,highlight='C(=O)[O;H1]',role='donor')
    v.text(565,y+112,'+',35,v.MUTED,600,'middle')
    v.mol(ac,585,y+22,450,135,highlight='[N;H2;X3]',role='acceptor')
    v.arrow(1050,y+95,1130,y+95)
    product(seq,y+52,lab,col)
v.line(90,465,1510,465,'#E5EBEF',2)
v.text(800,770,'Which step creates the preference: activation, aminolysis, or both?',33,v.GOLD,600,'middle')
v.save('v2_02_desd_scientific_puzzle','Literature product preference (regioselective oligoamide study, Fig. 7). Neutral identity drawings; molecular cause remains a hypothesis.')

# Amino-acid identities are exact; proposed contacts are not drawn as observations.
v.begin('AS9: two mutations, two mechanistic questions',3)
v.text(500,195,'WT SIDE CHAIN',23,v.MUTED,700,'middle')
v.text(855,195,'MUTANT SIDE CHAIN',23,v.MUTED,700,'middle')
v.text(1280,195,'HYPOTHESIS TO TEST',23,v.GOLD,700,'middle')
for y,case,mutation,wt,mut,query,q1,q2 in [
    (270,'PA66 diad','A215Q','*C','*CCC(=O)N','C(=O)N','Does the new amide','reposition the substrate?'),
    (520,'PA55 diad','F249K','*Cc1ccccc1','*CCCCN','N','Does the new amine','alter binding or packing?'),
]:
    v.text(110,y+20,case,28,v.MUTED,600)
    v.text(110,y+80,mutation,43,v.INK,700)
    v.mol(wt,345,y-30,280,165)
    v.arrow(635,y+55,690,y+55)
    v.mol(mut,705,y-30,320,165,highlight=query)
    v.text(1270,y+30,q1,30,v.GOLD,600,'middle')
    v.text(1270,y+78,q2,30,v.GOLD,600,'middle')
v.line(90,465,1510,465,'#E5EBEF',2)
v.text(800,770,'Next: matched structural comparisons + experimental controls',33,v.BLUE,700,'middle')
v.save('v2_03_as9_mutation_questions','User-reported yield improvements; assay values pending. * = backbone attachment. Identity drawings do not specify assay protonation states.')

names=['v2_01_native_chemistry_search','v2_02_desd_scientific_puzzle','v2_03_as9_mutation_questions']
pdf=fitz.open()
for name in names:
    ET.parse(O/f'{name}.svg')
    with fitz.open(O/f'{name}.pdf') as d:pdf.insert_pdf(d)
pdf.save(O/'revised_reasoning_examples.pdf')
preview=Image.new('RGB',(1000,1760),'#E3E9EE')
for i,name in enumerate(names):
    with Image.open(O/f'{name}.png') as im:
        im=im.convert('RGB');im.thumbnail((960,540));preview.paste(im,(20,20+i*580))
preview.save(O/'revised_preview.png')
checks=[]
for name in names:
    with fitz.open(O/f'{name}.pdf') as d:
        page=d[0]
        outside=[]
        for word in page.get_text('words'):
            if word[0]<0 or word[1]<0 or word[2]>1600 or word[3]>900:outside.append(word[4])
        assert not outside,(name,outside)
        checks.append({'name':name,'words':len(page.get_text().split()),'text_outside_slide':outside})
(O/'revised_validation.json').write_text(json.dumps({'checks':checks,'pages':len(pdf),'scope':'Scientific questions and chemical identities; no new assay data or causal validation.'},indent=2)+'\n')
print('Built and checked three revised chemistry figures.')
