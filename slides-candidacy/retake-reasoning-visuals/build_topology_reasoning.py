"""Discovery example that beats a substrate-similarity baseline.

The point this slide must carry: substrate-structure similarity ranked the WRONG
family highest; chemical reasoning about substrate *topology* (free small molecule
vs. carrier-protein / bulky-cofactor tethered) overrode that score and chose the
right family. This is the recorded decision the committee cannot dismiss as
"you could have just compared SMILES."

Provenance (all local, canonical run 20260917_135330):
  family_landscape.json   -> holistic substrate-fit scores (the similarity proxy)
  family_decisions.json   -> stop/expand decisions + reasons (timestamped)
  target_context.json     -> free_small_molecule topology rule derived up front
Representative members (candidate_registry.json):
  STOP   IPR041107 MptN  tetrahydromethanopterin:alpha-L-glutamate ligase (Q8Q0M5)
  EXPAND IPR022770 AsbA  spermidine--citrate ligase, B. anthracis (A0A0F7RJ52)

Run: /homes/rzhu/miniforge3/envs/palm/bin/python build_topology_reasoning.py
"""
import json
import xml.etree.ElementTree as ET
import pymupdf as fitz
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import visual_helpers as v


def molecule(smiles, x, y, w, h, highlight=None, hcolor=(1.0, .86, .56)):
    m = Chem.MolFromSmiles(smiles)
    assert m is not None, smiles
    rdDepictor.Compute2DCoords(m)
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    o = d.drawOptions()
    o.clearBackground = False
    o.bondLineWidth = 3
    o.fixedFontSize = 26
    o.fixedBondLength = 34
    ids = []
    if highlight:
        ids = list(m.GetSubstructMatch(Chem.MolFromSmarts(highlight)))
    d.DrawMolecule(m, highlightAtoms=ids,
                   highlightAtomColors={i: hcolor for i in ids})
    d.FinishDrawing()
    s = d.GetDrawingText()
    s = s[s.index('<svg'):]
    s = s[s.index('>') + 1:s.rindex('</svg>')]
    v.parts.append(f'<g transform="translate({x},{y})">{s}</g>')


def label(x, y, s, size=28, color=None, bold=False, center=False, end=False):
    anchor = 'middle' if center else ('end' if end else 'start')
    v.text(x, y, s, size, color or v.INK, 700 if bold else 400, anchor)


def fitbar(x, y, w, score, color):
    """Neutral similarity-score bar: same metric for both rows."""
    v.rect(x, y, w, 26, '#E3E9ED', r=6)
    v.rect(x, y, max(10, int(w * score)), 26, color, r=6)
    label(x + w + 16, y + 22, f'{score:.2f}', 26, v.INK, True)


def chip(cx, y, text, bg, fg):
    w = 170
    v.rect(cx - w / 2, y, w, 58, bg, r=12)
    label(cx, y + 39, text, 29, fg, True, center=True)


# ----------------------------------------------------------------------------
v.begin('Chemical reasoning overrides substrate similarity', 1)

# --- target banner ---------------------------------------------------------
v.rect(70, 140, 1460, 118, '#EEF5FA')
label(96, 180, 'TARGET REACTION', 20, v.BLUE, True)
label(96, 222, 'adipic acid  +  hexamethylenediamine', 29, bold=True)
molecule('O=C(O)CCCCC(=O)O', 640, 150, 320, 100)
label(972, 212, '+', 30, v.MUTED)
molecule('NCCCCCCN', 1000, 158, 300, 92, highlight='[NX3]')
v.rect(1322, 166, 188, 66, '#DCEFE6', r=10)
label(1416, 196, 'free diacid +', 21, v.GREEN, True, center=True)
label(1416, 222, 'free diamine', 21, v.GREEN, True, center=True)

# --- column headers --------------------------------------------------------
hy = 300
label(96, hy, 'CANDIDATE FAMILY (ATP-dependent amide ligase)', 19, v.MUTED, True)
label(720, hy, 'SUBSTRATE-SIMILARITY FIT', 19, v.MUTED, True, center=True)
label(930, hy, 'SUBSTRATE TOPOLOGY', 19, v.MUTED, True)
label(1445, hy, 'DECISION', 19, v.MUTED, True, center=True)

# --- Row A: the trap -- high similarity, wrong chemistry --------------------
ay = 320
v.rect(70, ay, 1460, 230, '#FBF0F0')
label(96, ay + 42, 'MptN  ·  tetrahydromethanopterin:', 26, bold=True)
label(96, ay + 75, 'L-glutamate ligase', 26, bold=True)
label(96, ay + 107, 'IucA/RimK ATP-grasp fold (IPR041107)', 19, v.MUTED)
# what similarity sees: a small glutamate
molecule('N[C@@H](CCC(=O)O)C(=O)O', 110, ay + 118, 280, 86,
         highlight='C(=O)[OH]')
label(250, ay + 220, 'glutamate: small diacid-like', 18, v.MUTED, center=True)
fitbar(600, ay + 74, 240, 0.83, v.BLUE)
label(720, ay + 150, 'ranked at the TOP by', 21, v.MUTED, center=True)
label(720, ay + 177, 'substrate structure + size', 21, v.MUTED, center=True)
# topology reality
label(930, ay + 66, 'Glutamate is ligated onto', 23)
label(930, ay + 97, 'tetrahydromethanopterin —', 23)
label(930, ay + 128, 'a bulky cofactor, not a', 23)
label(930, ay + 159, 'free acid.', 23, bold=True)
label(930, ay + 199, 'No free-substrate pose.', 20, v.RED, bold=True)
chip(1445, ay + 86, 'STOP', '#F3D6D6', v.RED)

# --- Row B: the right call -- lower similarity, right chemistry -------------
by = 568
v.rect(70, by, 1460, 200, '#ECF6F1')
label(96, by + 42, 'AsbA  ·  spermidine–citrate ligase', 26, bold=True)
label(96, by + 74, 'B. anthracis  (A0A0F7RJ52)', 19, v.MUTED)
label(96, by + 100, 'NIS synthetase, IucA/IucC (IPR022770)', 19, v.MUTED)
molecule('NCCCNCCCCN', 110, by + 108, 330, 78, highlight='[NX3]')
label(275, by + 192, 'free diamine (+ free citrate)', 18, v.MUTED, center=True)
fitbar(600, by + 70, 240, 0.40, v.BLUE)
label(720, by + 146, 'ranked LOWER by the', 21, v.MUTED, center=True)
label(720, by + 173, 'same similarity metric', 21, v.MUTED, center=True)
label(930, by + 74, 'Free diamine + free diacid,', 23)
label(930, by + 105, 'condensed via acyl-adenylate', 23)
label(930, by + 136, '— same topology as target.', 23, bold=True)
chip(1445, by + 72, 'EXPAND', '#CFE8DC', v.GREEN)

# --- takeaway --------------------------------------------------------------
v.rect(70, 786, 1460, 44, '#172B3A', r=10)
label(800, 815,
      'Similarity put the carrier-tethered family on top (0.83 vs 0.40) — '
      'reasoning about substrate presentation reversed the call.',
      21, 'white', True, center=True)

v.save('discovery_00_topology_over_similarity',
       'Recorded decision, canonical run 20260917_135330: fit = family_landscape.json; '
       'stop/expand = family_decisions.json. Prioritization, not a tested exclusion.')

# ---------------------------------------------------------------------------
name = 'discovery_00_topology_over_similarity'
ET.parse(v.OUT / f'{name}.svg')
with fitz.open(v.OUT / f'{name}.pdf') as d:
    page = d[0]
    outside = [w[4] for w in page.get_text('words')
               if w[0] < 0 or w[1] < 0 or w[2] > page.rect.width or w[3] > page.rect.height]
    assert not outside, outside
(v.OUT / 'discovery_topology_sources.json').write_text(json.dumps({
    'claim': 'Chemical reasoning about substrate topology overrode a substrate-similarity '
             'score: the highest-fit family (0.83) was stopped, a lower-fit family (0.40) expanded.',
    'run': '20260917_135330 (canonical amide-synthetase discovery run)',
    'evidence': {
        'fit_scores': 'family_landscape.json -> holistic_fit.median',
        'decisions': 'family_decisions.json -> decision + reason (timestamped)',
        'rule_derived_up_front': 'target_context.json -> target_reasoning: '
                                 'free_small_molecule topology, catalyst requiring '
                                 'different topology may not accept these inputs',
        'members': 'candidate_registry.json',
    },
    'stop_family': {'id': 'IPR041107', 'rep': 'MptN tetrahydromethanopterin:L-glutamate ligase (Q8Q0M5)',
                    'holistic_fit': 0.832, 'reason': 'carrier-protein-tethered / bulky-cofactor substrate'},
    'expand_family': {'id': 'IPR022770', 'rep': 'AsbA spermidine-citrate ligase (A0A0F7RJ52)',
                      'holistic_fit': 0.397, 'reason': 'free diamine + free diacid, acyl-adenylate'},
    'honest_scope': [
        'Recorded decision, not retrospective: the free-topology rule is in target_reasoning before family scores.',
        'Demonstrates chemical reasoning changes the search beyond substrate similarity.',
        'Does NOT alone prove an LLM agent beats a rule-based filter that also encodes topology -- that is the proposed controlled comparison.',
        'Stop/expand is prioritization; no member was experimentally tested against free adipate+HMDA.',
        'The fit score is the agent\'s own substrate/size feature, used here as a stand-in for a similarity ranker.',
    ],
}, indent=2) + '\n')
print('Built', name)
