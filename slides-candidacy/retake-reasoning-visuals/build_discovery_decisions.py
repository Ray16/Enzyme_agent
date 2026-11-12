"""Build two chemistry-led discovery slides; preserve earlier drafts.

Run with /homes/rzhu/miniforge3/envs/palm/bin/python build_discovery_decisions.py
Native reactions are literature evidence. Search actions are reconstructed from
the existing local audit; slide 2 is a proposed follow-up, not a recorded result.
"""
import json
import xml.etree.ElementTree as ET
import pymupdf as fitz
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import visual_helpers as v


def molecule(smiles, x, y, w, h, mark_n=False, mark_alpha=False):
    m = Chem.MolFromSmiles(smiles)
    assert m is not None
    rdDepictor.Compute2DCoords(m)
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    o = d.drawOptions()
    o.clearBackground = False
    o.bondLineWidth = 3
    o.fixedFontSize = 30
    o.fixedBondLength = 39
    atoms = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 7] if mark_n else []
    if mark_alpha:
        atoms = list(m.GetSubstructMatch(Chem.MolFromSmarts('[N]-[C;X4]')))
    d.DrawMolecule(m, highlightAtoms=atoms,
                   highlightAtomColors={i: (1.0, .86, .56) for i in atoms})
    d.FinishDrawing()
    s = d.GetDrawingText()
    s = s[s.index('<svg'):]
    s = s[s.index('>') + 1:s.rindex('</svg>')]
    v.parts.append(f'<g transform="translate({x},{y})">{s}</g>')


def label(x, y, s, size=28, color=None, bold=False, center=False):
    v.text(x, y, s, size, color or v.INK, 700 if bold else 400,
           'middle' if center else 'start')


v.begin('Native chemistry redirects the enzyme search', 1)
v.rect(70, 145, 1460, 150, '#EEF5FA')
label(100, 185, 'TARGET REACTION', 21, v.BLUE, True)
label(100, 238, 'Adipic acid + hexamethylenediamine (HMDA)', 30, bold=True)
molecule('NCCCCCCN', 920, 153, 540, 100, mark_n=True)
label(1190, 278, 'Six-carbon chain · two terminal amines', 23, v.BLUE, center=True)

label(105, 354, 'NATIVE NITROGEN DONOR', 22, v.MUTED, True)
label(700, 354, 'CHEMICAL RELEVANCE', 22, v.MUTED, True)
label(1190, 354, 'SEARCH DECISION', 22, v.MUTED, True)
v.rect(70, 380, 1460, 185, '#F3F5F6')
label(105, 420, 'GS', 30, bold=True)
label(105, 515, 'GmaS', 30, bold=True)
molecule('N', 260, 390, 260, 60)
molecule('CN', 260, 480, 260, 60)
label(530, 427, 'Ammonia', 23, v.MUTED)
label(530, 517, 'Methylamine', 23, v.MUTED)
label(700, 450, 'These native reactions', 27)
label(700, 490, 'do not establish', 27)
label(700, 530, 'diamine acceptance.', 27)
v.arrow(1060, 475, 1140, 475, v.MUTED)
label(1180, 463, 'Deprioritize', 30, v.MUTED, True)
label(1180, 505, 'these search scopes', 25, v.MUTED)

v.rect(70, 585, 1460, 170, '#E8F5EF')
label(105, 639, 'PuuA', 32, bold=True)
molecule('NCCCCN', 240, 601, 400, 95, mark_n=True)
label(440, 733, 'Putrescine · four-carbon diamine', 23, v.GREEN, center=True)
label(700, 638, 'A native linear diamine', 27)
label(700, 678, 'motivates an HMDA lead.', 27)
v.arrow(1060, 669, 1140, 669, v.GREEN)
label(1180, 657, 'Expand the search', 29, v.GREEN, True)
label(1180, 699, 'around the PuuA lead', 25, v.GREEN)
label(800, 797, 'Prioritize by native chemistry; target activity still needs testing.', 28,
      v.BLUE, True, True)
v.save('discovery_01_search_decision',
       'Native reactions: IUBMB EC 6.3.1.2 / 6.3.4.12; Kurihara et al., 2008. Actions: archived-run audit, events 51 and 122–125.')

v.begin('The chemical mismatch defines what to investigate next', 2)
label(75, 165, 'PuuA is a lead. Can it accept both target substrates?', 30, v.MUTED)
v.rect(70, 203, 700, 480, '#E8F5EF')
v.rect(800, 203, 730, 480, '#FFF5E6')
label(420, 252, 'NITROGEN DONOR', 23, v.GREEN, True, True)
label(1165, 252, 'CARBOXYLATE DONOR', 23, v.GOLD, True, True)
label(110, 305, 'Native', 22, v.MUTED)
label(840, 305, 'Native', 22, v.MUTED)
molecule('NCCCCN', 160, 308, 530, 110, mark_n=True)
molecule('N[C@@H](CCC(=O)O)C(=O)O', 895, 280, 550, 130, mark_alpha=True)
label(420, 446, 'Putrescine', 27, bold=True, center=True)
label(1165, 446, 'L-glutamic acid', 27, bold=True, center=True)
v.arrow(420, 465, 420, 500, v.GREEN)
v.arrow(1165, 465, 1165, 500, v.GOLD)
label(110, 530, 'Target', 22, v.MUTED)
label(840, 530, 'Target', 22, v.MUTED)
molecule('NCCCCCCN', 160, 510, 530, 105, mark_n=True)
molecule('O=C(O)CCCCC(=O)O', 900, 495, 530, 130)
label(420, 650, 'HMDA: longer by two carbons', 27, v.GREEN, True, True)
label(1165, 650, 'Adipic acid: different donor chemistry', 27, v.GOLD, True, True)
label(420, 725, 'Can the amine site accept the longer chain?', 27, center=True)
label(1165, 725, 'Can PuuA recognize and activate adipate?', 27, center=True)
v.rect(70, 755, 1460, 54, '#EEF5FA')
label(800, 792, 'PROPOSED NEXT STEP  ·  Evaluate the actual adipate–HMDA pair', 29,
      v.BLUE, True, True)
v.save('discovery_02_next_investigation',
       'PuuA native substrates: Kurihara et al., JBC 2008, PMID 18495664. Neutral identity drawings; target activity is unestablished.')

names = ['discovery_01_search_decision', 'discovery_02_next_investigation']
combined = fitz.open()
checks = []
for name in names:
    ET.parse(v.OUT / f'{name}.svg')
    with fitz.open(v.OUT / f'{name}.pdf') as d:
        page = d[0]
        outside = [w[4] for w in page.get_text('words')
                   if w[0] < 0 or w[1] < 0 or w[2] > page.rect.width or w[3] > page.rect.height]
        assert not outside, (name, outside)
        checks.append({'slide': name, 'text_outside_slide': outside})
        combined.insert_pdf(d)
combined.save(v.OUT / 'discovery_decision_slides.pdf')
preview = Image.new('RGB', (1280, 1460), '#DDE4E8')
for i, name in enumerate(names):
    with Image.open(v.OUT / f'{name}.png') as im:
        im.thumbnail((1240, 700))
        preview.paste(im, (20, 20 + i * 720))
preview.save(v.OUT / 'discovery_decision_preview.png')
(v.OUT / 'discovery_decision_sources.json').write_text(json.dumps({
    'sources': ['https://iubmb.qmul.ac.uk/enzyme/EC6/3/1/2.html',
                'https://iubmb.qmul.ac.uk/enzyme/EC6/3/4/12.html',
                'https://pubmed.ncbi.nlm.nih.gov/18495664/'],
    'local_action_sources': ['../reasoning-trace/audit/figure_provenance.json',
                             '../backup-finalization/speaker_notes.txt: Saved PuuA decision example'],
    'scope': 'Editorial reconstruction from existing audit. Original run path is no longer present. No new discovery run or performance comparison. Search-scope expansion does not assign PuuA specificity to all expanded members. Slide 2 is proposed.',
    'checks': checks,
}, indent=2) + '\n')
print('Created two slides, combined PDF, editable SVGs and preview.')
