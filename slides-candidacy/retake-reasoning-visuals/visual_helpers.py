"""Vector-first helpers for candidacy reasoning figures."""
from pathlib import Path
from html import escape
import math
import re
import fitz
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

OUT=Path(__file__).resolve().parent
INK='#172B3A'; MUTED='#526774'; BLUE='#0072B2'; GREEN='#007E65'
GOLD='#A86500'; RED='#8D0000'; PURPLE='#79509B'
parts=[]

def text(x,y,label,size=30,fill=INK,weight=400,anchor='start'):
    parts.append(f'<text x="{x}" y="{y}" fill="{fill}" font-family="Helvetica" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{escape(label)}</text>')

def rect(x,y,w,h,fill,stroke='none',r=12):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')

def line(x1,y1,x2,y2,color=BLUE,width=4,dash=False):
    extra=' stroke-dasharray="10 8"' if dash else ''
    parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="{width}"{extra}/>')

def arrow(x1,y1,x2,y2,color=BLUE,width=5,dash=False):
    line(x1,y1,x2,y2,color,width,dash)
    a=math.atan2(y2-y1,x2-x1)
    p=[(x2,y2),(x2-18*math.cos(a-.5),y2-18*math.sin(a-.5)),(x2-18*math.cos(a+.5),y2-18*math.sin(a+.5))]
    parts.append('<polygon points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in p)+f'" fill="{color}"/>')

def mol(smiles,x,y,w,h,highlight=None,role=None):
    m=Chem.MolFromSmiles(smiles)
    assert m is not None,smiles
    rdDepictor.Compute2DCoords(m)
    if role:
        acid=m.GetSubstructMatch(Chem.MolFromSmarts('C(=O)[O;H1]'))
        amine=m.GetSubstructMatch(Chem.MolFromSmarts('[N;H2;X3]'))
        assert acid and amine,(smiles,role)
        conf=m.GetConformer()
        acid_right=conf.GetAtomPosition(acid[0]).x>conf.GetAtomPosition(amine[0]).x
        if (role=='donor' and not acid_right) or (role=='acceptor' and acid_right):
            for i in range(m.GetNumAtoms()):
                p=conf.GetAtomPosition(i)
                conf.SetAtomPosition(i,(-p.x,p.y,p.z))
    d=rdMolDraw2D.MolDraw2DSVG(w,h)
    o=d.drawOptions(); o.clearBackground=False; o.bondLineWidth=3; o.fixedFontSize=32
    if highlight:
        ids=m.GetSubstructMatch(Chem.MolFromSmarts(highlight))
        d.DrawMolecule(m,highlightAtoms=list(ids),highlightAtomColors={i:(1,.88,.63) for i in ids})
    else:d.DrawMolecule(m)
    d.FinishDrawing()
    s=d.GetDrawingText()
    s=s[s.index('<svg'):]
    s=s[s.index('>')+1:s.rindex('</svg>')]
    parts.append(f'<g transform="translate({x},{y})">{s}</g>')
    return m

def begin(title,number):
    global parts
    parts=['<svg xmlns="http://www.w3.org/2000/svg" xmlns:rdkit="http://www.rdkit.org/xml" width="1600" height="900" viewBox="0 0 1600 900">']
    rect(0,0,1600,900,'white',r=0)
    rect(65,65,8,55,RED,r=2)
    text(98,106,title,45,INK,700)
    line(70,820,1530,820,'#DCE5EB',2)
    text(1530,860,str(number),20,MUTED,anchor='end')

def save(name,footer):
    text(70,858,footer,18,MUTED)
    parts.append('</svg>')
    s='\n'.join(parts)
    (OUT/f'{name}.svg').write_text(s)
    d=fitz.open(stream=s.encode(),filetype='svg')
    pdf=fitz.open('pdf',d.convert_to_pdf())
    pdf.save(OUT/f'{name}.pdf')
    pdf[0].get_pixmap(matrix=fitz.Matrix(1.2,1.2),alpha=False).save(OUT/f'{name}.png')
