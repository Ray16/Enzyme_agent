"""Build vector candidacy backup slides from reviewed sources, without model calls."""
from pathlib import Path
import hashlib
import json
import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
W, H = 1920, 1080
MAROON = '#890B23'
BLUE = '#0076BA'
INK = '#16191C'
MUTED = '#56616C'
PALE = '#EAF5FD'
ROSE = '#F8EDEF'
GRAY = '#F3F4F5'
DOC = fitz.open()
CHECKS = []
FONT_DIR = '/usr/share/fonts/truetype/dejavu'
ARCHIVE = fitz.Archive(FONT_DIR)
CSS = '''
@font-face {font-family: Deck; src: url(DejaVuSans.ttf);}
@font-face {font-family: Deck; src: url(DejaVuSans-Bold.ttf); font-weight: bold;}
@font-face {font-family: Deck; src: url(DejaVuSans-Oblique.ttf); font-style: italic;}
* {font-family: Deck; margin: 0; padding: 0;}
sub,sup {font-size: 70%;}
'''


def rgb(value):
    return tuple(int(value[i:i+2], 16)/255 for i in (1, 3, 5))


def rect(page, box, fill, stroke=None, width=1):
    page.draw_rect(fitz.Rect(box), color=rgb(stroke) if stroke else None,
                   fill=rgb(fill), width=width)


def txt(page, box, text, size=34, color=INK, bold=False, align='left'):
    requested_size = size
    while True:
        html = (f'<div style="font-size:{size}px;line-height:1.24;color:{color};'
                f'text-align:{align};font-weight:{"bold" if bold else "normal"}">{text}</div>')
        spare, scale = page.insert_htmlbox(fitz.Rect(box), html, css=CSS,
                                           archive=ARCHIVE, scale_low=1)
        if spare >= 0 or size <= requested_size*0.85:
            break
        size -= 1
    CHECKS.append({'page': page.number+1, 'text': text, 'font_size': size,
                   'requested_font_size': requested_size,
                   'box': list(box), 'spare_height': spare, 'scale': scale})
    if spare < 0 or scale != 1:
        raise ValueError(f'Text overflow on backup {page.number+1}: {text}')


def slide(title, subtitle, tag='FRAMEWORK', number=None):
    p = DOC.new_page(width=W, height=H)
    txt(p, (70, 42, 1850, 122), title, 51, MAROON, True, 'center')
    txt(p, (80, 133, 1840, 182), subtitle, 29, MUTED)
    rect(p, (0, 1030, W, H), MAROON)
    txt(p, (80, 1040, 1760, 1078), f'BACKUP · {tag}', 21, '#FFFFFF')
    txt(p, (1810, 1040, 1870, 1078), str(number or p.number+30), 23, '#FFFFFF', align='right')
    return p


def takeaway(p, text, y=931):
    rect(p, (80, y, 1840, y+66), PALE)
    txt(p, (101, y+11, 1820, y+62), text, 29, BLUE, True)


def table(p, x, y, widths, headings, rows, heights, size=31):
    xs = [x]
    for w in widths:
        xs.append(xs[-1]+w)
    h = heights[0]
    rect(p, (x, y, xs[-1], y+h), ROSE)
    for j, value in enumerate(headings):
        txt(p, (xs[j]+16, y+15, xs[j+1]-16, y+h-7), value, size, MAROON, True)
    y += h
    for i, row in enumerate(rows):
        h = heights[i+1]
        if i % 2 == 0:
            rect(p, (x, y, xs[-1], y+h), GRAY)
        for j, value in enumerate(row):
            txt(p, (xs[j]+16, y+12, xs[j+1]-16, y+h-6), value, size,
                INK if j else MUTED, j == 0)
        p.draw_line((x, y+h), (xs[-1], y+h), color=rgb('#DADDE0'), width=1)
        y += h
    return y


def chemistry():
    latex_table = HERE/'amp_adp_table.pdf'
    if latex_table.exists():
        p = DOC.new_page(width=W, height=H)
        with fitz.open(latex_table) as table_pdf:
            assert len(table_pdf) == 1
            p.show_pdf_page(p.rect, table_pdf, 0)
        rect(p, (0, 1030, W, H), MAROON)
        txt(p, (80, 1040, 1760, 1078), 'BACKUP · MECHANISTIC CHEMISTRY', 21, '#FFFFFF')
        txt(p, (1810, 1040, 1870, 1078), '30', 23, '#FFFFFF', align='right')
        return
    p = slide('AMP-forming versus ADP-forming synthetases',
              'Two routes to carboxyl activation during ATP-dependent amide ligation',
              'MECHANISTIC CHEMISTRY')
    table(p, 80, 213, [470, 645, 645],
          ['Feature', 'AMP-forming<br/>Adenylation', 'ADP-forming<br/>Phosphorylation'], [
          ['ATP attack site', 'α-phosphate (P<sub>α</sub>)', 'γ-phosphate (P<sub>γ</sub>)'],
          ['Activated intermediate', 'Acyl-adenylate (acyl-AMP)', 'Acyl-phosphate'],
          ['Activation coproduct', 'Pyrophosphate (PP<sub>i</sub>)', 'ADP'],
          ['Aminolysis leaving group', 'AMP', 'Inorganic phosphate (P<sub>i</sub>)'],
          ['Overall ATP products', 'AMP + PP<sub>i</sub>', 'ADP + P<sub>i</sub>'],
          ['Relevant families / examples', 'NIS synthetases (IucA);<br/>DdaG', 'GS-fold ligases (PuuA);<br/>ATP-grasp ligases (DdaF)'],
          ['Activation geometry', 'O<sub>carboxylate</sub>–P<sub>α</sub>–O3A', 'O<sub>carboxylate</sub>–P<sub>γ</sub>–O3B'],
          ], [100, 70, 70, 70, 70, 70, 103, 70], 30)
    takeaway(p, 'Assign the activation mechanism before choosing the intermediate and structural proxy.', 896)
    txt(p, (86, 970, 1840, 1022),
        'Scope: carboxyl-activating amide ligases. Family labels are examples, not exhaustive classifications. '
        'Metal stoichiometry requires separate evidence.<br/>'
        'Hollenhorst et al., Biochemistry (2009), doi:10.1021/bi9013165; '
        'Mydy et al., Biochemistry (2020), PMCID: PMC7325057; Kurihara et al., JBC (2008), PMID:18495664.',
        18, MUTED)


def save_table():
    chemistry()
    p = HERE/'AMP vs ADP synthetases.pdf'
    DOC.save(p, garbage=4, deflate=True)
    DOC[0].get_pixmap(matrix=fitz.Matrix(0.8, 0.8), alpha=False).save(HERE/'AMP vs ADP synthetases.png')
    print(p)


def card(p, x, y, w, h, heading, body, fill=PALE, size=30):
    rect(p, (x, y, x+w, y+h), fill)
    txt(p, (x+23, y+18, x+w-23, y+96), heading, 33, MAROON, True)
    txt(p, (x+23, y+103, x+w-23, y+h-16), body, size)


def arrow(p, start, end):
    p.draw_line(start, end, color=rgb(MUTED), width=3)
    x, y = end
    dx, dy = end[0]-start[0], end[1]-start[1]
    if abs(dx) > abs(dy):
        a = 1 if dx > 0 else -1
        points = [(x-13*a, y-7), end, (x-13*a, y+7)]
    else:
        a = 1 if dy > 0 else -1
        points = [(x-7, y-13*a), end, (x+7, y-13*a)]
    p.draw_polyline(points, color=rgb(MUTED), width=3)


def roles():
    p = slide('Agent roles, tools, and scientific responsibility',
              'Each component owns a scientific decision and produces reviewable evidence.')
    table(p, 80, 215, [350, 710, 700],
          ['Component', 'Scientific decisions', 'Tools and outputs'], [
          ['Enzyme discovery', 'Which reaction analogues and families?<br/>Which candidates merit investigation?',
           'Literature, reaction databases, UniProt,<br/>homolog and family search → shortlist'],
          ['Structural modeling', 'Which chemical state and assembly?<br/>Which test addresses the uncertainty?',
           'Cofolding, pose checks, geometry proxies;<br/>question-driven follow-up requests'],
          ['Enzyme design', 'Redesign an existing enzyme or generate<br/>a scaffold? Which motif or positions?',
           'Pocket scan, saturation libraries,<br/>motif/ligand-conditioned generation'],
          ['Experimental<br/>learning<br/>(proposed)', 'Which candidates distinguish hypotheses?<br/>How should outcomes revise the next round?',
           'Measured product, expression and controls<br/>→ revised hypotheses and test panel'],
          ], [90, 145, 145, 145, 145], 29)
    takeaway(p, 'The model chooses scientific actions; tools execute them and return inspectable results.')
    txt(p, (86, 1000, 1820, 1028),
        'Current component interfaces are implemented. The full prospective experimental learning loop is proposed work.', 19, MUTED)


def mechanism_comparison():
    p = slide('Activation geometry depends on the ATP cleavage route',
              'Saved predicted poses illustrate distinct reaction centers; they are not transition-state structures.',
              'STRUCTURAL COMPARISON')
    txt(p, (80, 201, 920, 257), 'AMP-forming · AS9', 34, MAROON, True, 'center')
    txt(p, (1000, 201, 1840, 257), 'ADP-forming · PuuA', 34, MAROON, True, 'center')
    p.insert_image(fitz.Rect(80, 262, 920, 912),
                   filename=str(HERE.parent/'attack-angle/AS9_attack_geometry.png'))
    p.insert_image(fitz.Rect(1000, 262, 1840, 912),
                   filename=str(HERE.parent/'adp-attack-geometry/PuuA_ADP_attack_geometry.png'))
    p.draw_line((960, 230), (960, 912), color=rgb('#DADDE0'), width=2)
    takeaway(p, 'AMP-forming: inspect Pα / O3A.     ADP-forming: inspect Pγ / O3B.')
    txt(p, (86, 1000, 1820, 1028),
        'Unchanged model-0 coordinates. AS9: one Mg; PuuA: two Mg. Metal count is model-specific, not determined by ATP cleavage class.', 19, MUTED)


def adaptive_loop():
    p = slide('How the agent selects and revises its next action',
              'Current decision protocol: hypothesis → predicted observation → tool result → revision')
    card(p, 80, 223, 530, 240, '1  Inspect the state',
         'Target constraints<br/>Sources and current hypotheses<br/>Unresolved or conflicting evidence')
    card(p, 695, 223, 530, 240, '2  Propose a test',
         'Choose a permitted tool<br/>State a discriminating prediction<br/>Record alternatives and rationale')
    card(p, 1310, 223, 530, 240, '3  Check the request',
         'Verify required evidence links<br/>Check tool and input contracts<br/>Retain unresolved assumptions')
    arrow(p, (620, 343), (685, 343)); arrow(p, (1235, 343), (1300, 343))
    card(p, 1310, 535, 530, 240, '4  Execute the tool',
         'Run the declared investigation<br/>Save raw outputs and status<br/>Record failures separately')
    card(p, 695, 535, 530, 240, '5  Compare with prediction',
         'Link the observed result<br/>Separate evidence from inference<br/>Retain or revise the hypothesis')
    card(p, 80, 535, 530, 240, '6  Update and continue',
         'Reprioritize the investigation<br/>Preserve previous observations<br/>Choose the next informative test')
    arrow(p, (1575, 475), (1575, 525))
    arrow(p, (1300, 655), (1235, 655)); arrow(p, (685, 655), (620, 655))
    rect(p, (80, 822, 1840, 902), GRAY)
    txt(p, (103, 837, 1817, 894),
        '<b>Stopping:</b> unresolved coverage or budget limit → checkpoint; new evidence can reopen a decision.', 28)
    takeaway(p, 'A completed tool call, a completed task list, and a supported scientific conclusion are distinct.')
    txt(p, (86, 1000, 1820, 1028),
        'The controller checks recorded contracts and provenance; it does not independently certify biochemical interpretations.', 19, MUTED)


def handoff():
    p = slide('Shared evidence and structured agent handoffs',
              'Communication carries explicit claims, source records, modeling proposals, and unresolved choices.')
    card(p, 80, 218, 530, 170, 'Discovery evidence', 'Candidate identity + native reaction', size=28)
    card(p, 695, 218, 530, 170, 'Modeling review', 'Independent setup + pose assessment', size=28)
    card(p, 1310, 218, 530, 170, 'Returned observations', 'Geometry + limitations + provenance', size=28)
    arrow(p, (620, 303), (685, 303)); arrow(p, (1235, 303), (1300, 303))
    table(p, 80, 430, [390, 1370], ['Handoff content', 'What must remain explicit'], [
          ['Target and identity', 'Bond change, substrate structures, protein name + accession, sequence provenance'],
          ['Scientific evidence', 'Native reaction, source passage, evidence status, contradictions and missing information'],
          ['Modeling proposal', 'Reaction step, reactive atoms, cofactors, charges, assembly and proposed restraints'],
          ['State and lineage', 'Admitted candidates separated from unresolved leads; source snapshots and revision identity'],
          ], [70, 98, 98, 98, 98], 29)
    takeaway(p, 'A discovery proposal is reviewed by the modeling agent before it becomes an executable setup.')
    txt(p, (86, 1000, 1820, 1028),
        'Current schema-3.0 interface. Structural evidence import is implemented; source integrity does not establish claim correctness.', 19, MUTED)


def worked_example():
    p = slide('Reaction evidence refines a broad family annotation',
              'Saved discovery case: distinct nitrogen donors lead to different search priorities.', 'SAVED CASE STUDY')
    card(p, 80, 230, 530, 430, 'Initial examples',
         '<b>Glutamine synthetase:</b><br/>glutamate + ammonia<br/><br/>'
         '<b>Glutamate–methylamine ligase:</b><br/>glutamate + methylamine<br/><br/>'
         'These examples were deprioritized<br/>for the target diamine.', GRAY, 29)
    card(p, 695, 230, 530, 430, 'Later retrieved evidence',
         '<b>PuuA</b><br/>Glutamate–putrescine ligase<br/>UniProt P78061<br/><br/>'
         'Native nitrogen donor:<br/>putrescine, a free C4 diamine<br/><br/>'
         'Target donor: a C6 diamine', PALE, 29)
    card(p, 1310, 230, 530, 430, 'Revised investigation',
         'Prioritize the PuuA lead<br/>and relevant family members.<br/><br/>'
         'Compare substrate scope<br/>and model target accommodation.<br/><br/>'
         '<b>Target activity remains unresolved.</b>', PALE, 29)
    arrow(p, (620, 445), (685, 445)); arrow(p, (1235, 445), (1300, 445))
    rect(p, (80, 715, 1840, 865), ROSE)
    txt(p, (105, 736, 1815, 856),
        '<b>Decision boundary:</b> a broad GS-fold annotation cannot determine nitrogen-donor specificity. '
        'Search deprioritization is reversible and does not demonstrate inactivity on the target substrates.', 32)
    takeaway(p, 'Evaluate the native reacting partners; avoid assigning one substrate scope to an entire fold.')
    txt(p, (86, 1000, 1820, 1028),
        'Archived September 17 run, events 51 and 122–125; reconstructed from saved decisions. Kurihara et al. (2008), PMID:18495664.', 19, MUTED)


def model_checks():
    p = slide('Model validation determines what can be interpreted',
              'Current structural controller: review the declared experiment before interpreting a favorable pose.')
    table(p, 80, 215, [370, 780, 610], ['Stage', 'Required assessment', 'Consequence'], [
          ['Identity and evidence', 'Resolve sequence identity; inspect sources,<br/>conflicts and intended reaction step.',
           'Unresolved identity or source conflict<br/>returns to investigation.'],
          ['Chemical setup', 'Declare ligand state, cofactors, assembly,<br/>restraints and reactive atom assignments.',
           'Failed setup checks require a<br/>new reviewed proposal.'],
          ['Prediction quality', 'Check output coverage, required atoms,<br/>declared physical criteria and file lineage.',
           'Incomplete or invalid outputs<br/>cannot support a structural conclusion.'],
          ['Mechanistic readout', 'Evaluate the reaction-specific geometry<br/>with its distance and sampling context.',
           'Geometric support remains distinct<br/>from measured catalytic activity.'],
          ['Follow-up choice', 'State what cofolding cannot answer and<br/>what observation would distinguish hypotheses.',
           'Dynamics, energetics or design<br/>requires its own scientific question.'],
          ], [80, 115, 115, 115, 115, 115], 28)
    takeaway(p, 'Input validity, prediction validity, mechanistic support, and experimental activity are separate.')
    txt(p, (86, 1000, 1820, 1028),
        'Generated cofold samples describe prediction variability; they are not a time trajectory or biological replicates.', 19, MUTED)


def evaluation():
    p = slide('How to test the value of the agentic framework',
              'Proposed evaluation: separate component accuracy from the benefit of adaptive coordination.', 'PROPOSED EVALUATION')
    table(p, 80, 215, [410, 730, 620], ['Scientific question', 'Comparison', 'Readout'], [
          ['Does chemistry help discovery?', 'Sequence / reaction retrieval baseline<br/>versus evidence-guided discovery',
           'Known-family recovery;<br/>precision among adjudicated candidates'],
          ['Does adaptation help?', 'Same model, tools and allowance;<br/>adaptive decisions versus a fixed workflow',
           'Useful evidence per unit cost;<br/>correction after conflicting evidence'],
          ['Does specialization help?', 'One agent with all tools versus specialized<br/>agents with the same total resource budget',
           'Decision consistency;<br/>handoff errors and final candidate quality'],
          ['Does feedback help design?', 'Initial ranking versus an updated ranking<br/>after measured outcomes',
           'Prospective hit rate;<br/>activity improvement and hypothesis resolution'],
          ], [80, 133, 133, 133, 133], 28)
    rect(p, (80, 837, 1840, 905), GRAY)
    txt(p, (103, 849, 1817, 898),
        '<b>Controls:</b> independent reactions, fixed data access, repeated runs, matched cost, and separate development/test sets.', 28)
    takeaway(p, 'A traceable explanation supports auditability; improved scientific performance needs a controlled comparison.')
    txt(p, (86, 1000, 1820, 1028),
        'Evaluation design, not completed comparative results. Unjudged candidates remain separate from confirmed negatives.', 19, MUTED)


def status():
    p = slide('Implementation and validation status',
              'Keep available software, preliminary scientific observations, and proposed validation distinct.', 'SCOPE AND STATUS')
    table(p, 80, 215, [340, 745, 675], ['Area', 'Available evidence / implementation', 'Next validation'], [
          ['Discovery', 'Saved target-reaction search and candidate<br/>shortlist; current source-linked decision records',
           'Independent reaction benchmark;<br/>adjudicated evidence and candidate quality'],
          ['Modeling and design', 'Saved cofold ensembles and proxy comparisons;<br/>matched WT–mutant structural predictions',
           'Setup sensitivity and prospective ranking;<br/>experimental mutant activity'],
          ['Coordination', 'Independent domain agents, structured<br/>handoffs, persistent state and feedback import',
           'Controlled benefit of specialization<br/>and evidence-driven reassessment'],
          ['Experimental learning', 'Measured homolog yields motivate hypotheses;<br/>candidate-panel and revision policies proposed',
           'Prospective repeated rounds with<br/>measured outcomes and matched controls'],
          ], [80, 140, 140, 140, 140], 28)
    takeaway(p, 'The thesis tests whether explicit chemical reasoning improves decisions and experimental outcomes.')
    txt(p, (86, 1000, 1820, 1028),
        'The September case study predates current controller upgrades; newer interfaces are not attributed retrospectively to that run.', 19, MUTED)


def modeling_setup():
    p = slide('Exact inputs behind the structural comparisons',
              'Saved setup audit: the activity panel and mutant campaign use distinct modeling conditions.', 'MODELING METHODS')
    table(p, 80, 215, [315, 482, 482, 481],
          ['Choice', 'Activity panel<br/>Activation step', 'Activity panel<br/>Amide-forming step', 'Mutant scan<br/>Activation step'], [
          ['Complex', 'ATP + adipate', 'Adipyl-AMP + HMDA', 'ATP + adipate'],
          ['Metal / protein', '1 Mg<sup>2+</sup>; one chain', '1 Mg<sup>2+</sup>; one chain', '2 Mg<sup>2+</sup>; one chain'],
          ['Spatial restraint', 'None', 'HMDA–pocket ≤10 Å;<br/>not an N–C attack restraint', 'None'],
          ['Sampling', 'Boltz-2; 50 samples<br/>per enzyme', 'Boltz-2; 50 samples<br/>per enzyme', '50 samples; fresh,<br/>matched WT controls'],
          ['Ligand-state audit', 'Adipate input −2;<br/>processed graph 0', 'Adipyl-AMP −2;<br/>HMDA +1, one NH<sub>2</sub>', 'Adipate −2 input;<br/>cached CCD ATP 0'],
          ], [110, 82, 82, 112, 108, 120], 27)
    rect(p, (80, 851, 1840, 913), ROSE)
    txt(p, (100, 860, 1820, 910),
        '<b>Open sensitivity tests:</b> metal count, processed ligand states, restraints, sampling and rank stability.', 27)
    takeaway(p, 'Interpret comparisons within matched setups; favorable geometry does not establish a catalytic barrier.')
    txt(p, (86, 1000, 1820, 1028),
        'Source: existing structural-modeling-setup audit and speaker notes. Historical panel ATP cache identity is not pinned for every run.', 19, MUTED)


def build_all():
    chemistry()
    mechanism_comparison()
    roles(); adaptive_loop(); handoff(); worked_example()
    model_checks(); evaluation(); status(); modeling_setup()
    output = HERE/'Candidacy backup slides.pdf'
    DOC.save(output, garbage=4, deflate=True)
    # Preserve the input PDF. Keep all main slides and move the existing full
    # proxy inventory to the end of the expanded backup section.
    source = HERE.parent/'Candidacy slides.pdf'
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    original = fitz.open(source)
    assert len(original) == 30
    combined = fitz.open()
    combined.insert_pdf(original, from_page=0, to_page=28)
    combined.insert_pdf(DOC)
    combined.insert_pdf(original, from_page=29, to_page=29)
    inventory_number = len(combined)
    last = combined[-1]
    # Renumber only the original inventory's footer in the new review copy.
    last.add_redact_annot(fitz.Rect(1874, 1035, 1915, 1074), fill=rgb(MAROON))
    last.apply_redactions(images=0, graphics=0)
    last.insert_text((1879.49, 1062.43), str(inventory_number), fontsize=24,
                     fontname='hebo', color=(1, 1, 1))
    for i in range(28):
        assert combined[i].get_text() == original[i].get_text()
        assert (combined[i].get_pixmap(matrix=fitz.Matrix(.12,.12)).digest ==
                original[i].get_pixmap(matrix=fitz.Matrix(.12,.12)).digest)
    titles = [page.get_text().splitlines()[0] for page in DOC]
    combined.set_toc([[1, 'Main presentation', 1], [1, 'Backup slides', 29]]+
                     [[2, title, 30+i] for i, title in enumerate(titles)]+
                     [[2, 'Full structural-proxy inventory', len(combined)]])
    combined_path = HERE/'Candidacy slides - with backups.pdf'
    combined.save(combined_path, garbage=4, deflate=True)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    previews = HERE/'previews'; previews.mkdir(exist_ok=True)
    individual = HERE/'individual'; individual.mkdir(exist_ok=True)
    for i, page in enumerate(DOC):
        page.get_pixmap(matrix=fitz.Matrix(0.6, 0.6), alpha=False).save(previews/f'backup-{i+1:02d}.png')
        (previews/f'backup-{i+1:02d}.svg').write_text(page.get_svg_image(text_as_path=False))
        one_slide = fitz.open(); one_slide.insert_pdf(DOC, from_page=i, to_page=i)
        one_slide.save(individual/f'backup-{i+1:02d}.pdf', garbage=4, deflate=True)
    # The table is also available independently for immediate use.
    one = fitz.open(); one.insert_pdf(DOC, from_page=0, to_page=0)
    one.save(HERE/'AMP vs ADP synthetases.pdf', garbage=4, deflate=True)
    DOC[0].get_pixmap(matrix=fitz.Matrix(0.8, 0.8), alpha=False).save(HERE/'AMP vs ADP synthetases.png')
    (HERE/'validation.json').write_text(json.dumps({
        'backup_pages': len(DOC), 'combined_pages': len(combined),
        'original_sha256': before, 'original_unchanged': True,
        'main_pages_preserved': 28, 'proxy_inventory_preserved': True,
        'main_page_text_and_render_checks': '28/28 identical',
        'latex_table_included': (HERE/'amp_adp_table.pdf').exists(),
        'text_boxes': len(CHECKS), 'all_text_fits_without_scaling': True,
        'checks': CHECKS,
    }, indent=2)+'\n')
    source_paths = [
        ROOT/'enzyme_discovery_agent/README.md',
        ROOT/'structural_modeling_agent/README.md',
        ROOT/'enzyme_design_agent/README.md',
        ROOT/'enzyme_discovery_agent/AGENTS.md',
        HERE.parent/'structural-modeling-setup/speaker_notes.txt',
        HERE.parent/'structural-modeling-setup/sources.json',
        HERE.parent/'reasoning-trace/build_trace.py',
        HERE.parent/'reasoning-trace/verification.json',
        HERE.parent/'attack-angle/geometry.json',
        HERE.parent/'adp-attack-geometry/geometry.json',
        HERE.parent/'adp-attack-geometry/validation.json',
    ]
    sources = [{'path':str(path.relative_to(ROOT)),
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
               for path in source_paths]
    (HERE/'sources.json').write_text(json.dumps({
        'scope':'Presentation preparation only. No model inference, fleet submission, or scientific experiments.',
        'local_sources':sources,
        'primary_references':[
            'https://doi.org/10.1021/bi9013165',
            'https://pmc.ncbi.nlm.nih.gov/articles/PMC7325057/',
            'https://pubmed.ncbi.nlm.nih.gov/18495664/',
        ]},indent=2)+'\n')
    print(output); print(combined_path)


if __name__ == '__main__':
    import sys
    if '--table-only' in sys.argv:
        save_table()
    else:
        build_all()
