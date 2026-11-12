"""Editable conceptual slide mockups; no synthetic assay or coordinate data.

Run with /homes/rzhu/miniforge3/envs/palm/bin/python build_visuals.py.
"""
from pathlib import Path
from html import escape
import json
import re
import fitz
from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

OUT = Path(__file__).resolve().parent
W, H = 1600, 900
INK, MUTED = '#172B3A', '#526774'
RED, BLUE, GREEN, GOLD = '#8D0000', '#0072B2', '#007E65', '#A86500'
PALE, WARM = '#EFF6FA', '#FFF6E8'
parts = []
checks = []

def text(x, y, label, size=30, fill=INK, weight=400, anchor='start'):
    parts.append(f'<text x="{x}" y="{y}" fill="{fill}" font-family="Helvetica" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{escape(label)}</text>')

def rect(x, y, w, h, fill, stroke='none', r=20):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')

def line(x1,y1,x2,y2,color=BLUE,width=4,dash=False):
    extra=' stroke-dasharray="10 8"' if dash else ''
    parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="{width}"{extra}/>')

def arrow(x1,y1,x2,y2,color=BLUE,width=5,dash=False):
    line(x1,y1,x2,y2,color,width,dash)
    import math
    a=math.atan2(y2-y1,x2-x1)
    p=[(x2,y2),(x2-18*math.cos(a-.5),y2-18*math.sin(a-.5)),(x2-18*math.cos(a+.5),y2-18*math.sin(a+.5))]
    parts.append('<polygon points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in p)+f'" fill="{color}"/>')

def mol(smiles,x,y,w,h,labels=None):
    m=Chem.MolFromSmiles(smiles)
    assert m is not None
    rdDepictor.Compute2DCoords(m)
    d=rdMolDraw2D.MolDraw2DSVG(w,h)
    o=d.drawOptions()
    o.clearBackground=False
    o.bondLineWidth=3
    o.fixedFontSize=33
    if labels:
        for atom in m.GetAtoms():
            if atom.GetAtomMapNum() in labels:
                o.atomLabels[atom.GetIdx()]=labels[atom.GetAtomMapNum()]
                atom.SetAtomMapNum(0)
    d.DrawMolecule(m)
    d.FinishDrawing()
    s=d.GetDrawingText()
    s=re.sub(r'<\?xml.*?\?>','',s,flags=re.S)
    s=s[s.index('<svg'):]
    s=s[s.index('>')+1:s.rindex('</svg>')]
    parts.append(f'<g transform="translate({x},{y})">{s}</g>')
    return m

def begin(title,number):
    global parts
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:rdkit="http://www.rdkit.org/xml" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><style>text{{font-family:Arial,Helvetica,sans-serif}}</style>']
    rect(0,0,W,H,'white',r=0)
    rect(65,65,9,58,RED,r=2)
    text(98,108,title,47,INK,700)
    line(70,820,1530,820,'#DCE5EB',2)
    text(1530,858,str(number),22,MUTED,anchor='end')

def save(name,footer):
    text(70,858,footer,20,MUTED)
    parts.append('</svg>')
    s='\n'.join(parts)
    target=OUT/f'{name}.svg'
    target.write_text(s)
    d=fitz.open(stream=s.encode(),filetype='svg')
    pdf=fitz.open('pdf',d.convert_to_pdf())
    pdf.save(OUT/f'{name}.pdf')
    pdf[0].get_pixmap(matrix=fitz.Matrix(1.2,1.2),alpha=False).save(OUT/f'{name}.png')
    checks.append({'name':name,'svg_xml':True,'pages':len(pdf),'size':[W,H]})

# 1: Chemical analogy is a reason to investigate, not a target-activity claim.
begin('Chemistry changes where we search',1)
rect(75,180,690,350,PALE)
rect(835,180,690,350,PALE)
text(110,230,'NATIVE EVIDENCE',23,GREEN,700)
text(870,230,'TARGET SUBSTRATE',23,BLUE,700)
text(420,292,'PuuA uses putrescine',34,INK,700,'middle')
text(1180,292,'Hexamethylenediamine',34,INK,700,'middle')
a=mol('NCCCCN',180,320,480,160)
b=mol('NCCCCCCN',940,320,480,160)
assert sum(at.GetSymbol()=='C' for at in a.GetAtoms())==4
assert sum(at.GetSymbol()=='C' for at in b.GetAtoms())==6
arrow(770,395,825,395,GOLD)
rect(350,565,900,80,WARM)
text(800,617,'Shared diamine chemistry; different chain length',32,GOLD,600,'middle')
arrow(800,655,800,700)
rect(350,710,900,75,BLUE)
text(800,758,'Investigate PuuA-related leads',35,'white',700,'middle')
save('01_search_decision','Native reaction: glutamate + putrescine (Kurihara et al., 2008) • Search analogy does not establish target activity.')

# 2: Generic acyl-state cartoons; no substrate-specific or geometrical data.
begin('DesD: the catalytic step determines the model',2)
rect(75,180,690,580,PALE)
rect(835,180,690,580,PALE)
text(110,230,'1  ACTIVATION',24,BLUE,700)
text(870,230,'2  AMINOLYSIS',24,BLUE,700)
text(420,290,'Which donor?',39,INK,700,'middle')
text(1180,290,'Which acceptor?',39,INK,700,'middle')
mol('[*:1]C(=O)O',105,355,260,150,{1:'D'})
text(395,445,'+ ATP',37,INK,600)
arrow(420,475,420,505)
text(420,550,'Acyl-AMP formation',31,MUTED,anchor='middle')
mol('[*:1]C(=O)O[*:2]',855,355,310,150,{1:'D',2:'AMP'})
text(1195,445,'+',37,INK,600)
mol('N[*:1]',1250,355,230,150,{1:'A'})
text(1180,550,'Amide-bond formation',31,MUTED,anchor='middle')
arrow(420,575,420,625)
arrow(1180,575,1180,625)
rect(125,645,590,75,BLUE)
rect(885,645,590,75,BLUE)
text(420,693,'Compare donor–ATP models',31,'white',600,'middle')
text(1180,693,'Compare acyl-AMP + acceptor models',29,'white',600,'middle')
save('02_reaction_state_decision','Generic reaction-state schematic • D = donor substituent; A = acceptor substituent • DesD: 2022 acyl-adenylate study.')

# 3: User-reported qualitative observation; alternatives and tests are proposed.
begin('AS9: turn a yield improvement into a test',3)
rect(70,185,440,575,'#EEF7F3')
rect(580,185,440,575,WARM)
rect(1090,185,440,575,PALE)
text(105,235,'OBSERVED',24,GREEN,700)
text(615,235,'HYPOTHESES',24,GOLD,700)
text(1125,235,'PROPOSED TESTS',24,BLUE,700)
text(290,380,'Mutant yield',44,INK,700,'middle')
text(290,450,'>',64,GREEN,700,'middle')
text(290,515,'WT yield',44,INK,700,'middle')
text(290,675,'Retrospective starting point',25,MUTED,anchor='middle')
arrow(520,460,570,460)
rect(615,315,370,130,'white')
text(800,365,'Catalytic positioning',29,INK,600,'middle')
text(800,410,'changes?',30,GOLD,600,'middle')
rect(615,505,370,130,'white')
text(800,555,'Expression / stability',29,INK,600,'middle')
text(800,600,'changes?',30,GOLD,600,'middle')
text(800,705,'Compare matched WT–mutant models',22,MUTED,anchor='middle')
arrow(1030,460,1080,460,dash=True)
for y,lab in [(330,'Equal purified enzyme'),(440,'Initial product-formation rates'),(550,'Abundance / stability controls')]:
    rect(1120,y,380,80,'white')
    text(1310,y+50,lab,25,INK,600,'middle')
text(1310,705,'Choose the next design from evidence',22,BLUE,600,'middle')
save('03_as9_retrospective_decision','Conceptual layout • No assay values, mutant identities or structural changes are invented • Hypotheses and tests are proposed.')

combined=fitz.open()
names=['01_search_decision','02_reaction_state_decision','03_as9_retrospective_decision']
for name in names:
    with fitz.open(OUT/f'{name}.pdf') as d:
        combined.insert_pdf(d)
combined.save(OUT/'reasoning_visual_mockups.pdf')

thumbs=[]
for name in names:
    im=Image.open(OUT/f'{name}.png').convert('RGB')
    im.thumbnail((960,540))
    thumbs.append(im)
preview=Image.new('RGB',(1000,3*580+20),'#E3E9EE')
for i,im in enumerate(thumbs):
    preview.paste(im,(20,20+i*580))
preview.save(OUT/'preview.png')

# Parse every vector file and check all rendered outputs.
import xml.etree.ElementTree as ET
for name in names:
    ET.parse(OUT/f'{name}.svg')
    with Image.open(OUT/f'{name}.png') as im:
        im.verify()
(OUT/'validation.json').write_text(json.dumps({'checks':checks,'combined_pages':len(combined),'molecule_carbon_counts':{'putrescine':4,'hexamethylenediamine':6},'scope':'Conceptual slide mockups, not new biological results.'},indent=2)+'\n')
print('Built 3 SVG/PNG/PDF mockups, combined PDF and preview.')
