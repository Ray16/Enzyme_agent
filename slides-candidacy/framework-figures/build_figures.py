"""Export standalone framework figures from reviewed presentation artwork.

This reuses reviewed drawing functions in memory. It does not modify or assemble
a slide deck. Each output is an independent figure with its own caption.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import fitz

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'backup-finalization/build_backups.py'
OUTPUT=HERE.parent/'backup-evidence-figures'
ACTIVE={'adaptive_decision_loop','shared_evidence_handoff','proposed_framework_evaluation','modeling_setup_audit'}
FIGURES = [
    (0, 'agent_roles_tools', (80,215,1840,895),
     'Current component roles and tools. Prospective experimental-learning integration is proposed.'),
    (1, 'adaptive_decision_loop', (75,215,1845,910),
     'Current adaptive protocol: inspect state, propose a test, check the request, execute, compare and revise. '
     'Controller checks establish contract and source integrity, not biochemical correctness.'),
    (2, 'shared_evidence_handoff', (75,210,1845,920),
     'Current handoff content and source lineage. The modeling agent independently reviews discovery proposals. '
     'Fingerprints establish byte integrity, not evidence entailment.'),
    (3, 'evidence_driven_search_example', (75,220,1845,875),
     'Reconstruction of archived September 17 decisions. Native nitrogen donors distinguish GS-fold leads. '
     'The example does not establish target activity or prospectively captured reasoning.'),
    (4, 'structural_validation', (80,215,1840,875),
     'Current structural validation stages. Setup validity, physical acceptance, mechanism-specific geometry '
     'and experimental activity are different assessments.'),
    (5, 'proposed_framework_evaluation', (80,215,1840,910),
     'Proposed comparisons and readouts. These are evaluation plans, not measured performance results. '
     'Use independent tasks, matched resources, repeated runs and adjudicated outcomes.'),
    (6, 'implementation_validation_status', (80,215,1840,890),
     'Available software and artifacts are distinguished from proposed scientific validation. '
     'Current controller capabilities are not attributed retrospectively to the September run.'),
    (7, 'modeling_setup_audit', (80,215,1840,917),
     'Saved-input audit of the activity panel and mutant campaign: different metal counts, ligand processing '
     'and restraints. This is not a completed sensitivity analysis. Read the existing modeling-setup speaker notes.'),
]


def main():
    HERE.mkdir(exist_ok=True)
    original_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    spec=importlib.util.spec_from_file_location('reviewed_framework_artwork',SOURCE)
    artwork=importlib.util.module_from_spec(spec);spec.loader.exec_module(artwork)
    for name in ['roles','adaptive_loop','handoff','worked_example','model_checks',
                 'evaluation','status','modeling_setup']:
        getattr(artwork,name)()
    source=artwork.DOC
    assert len(source)==8
    manifest=[]
    for page_number, name, box, caption in FIGURES:
        if name not in ACTIVE:continue
        crop=fitz.Rect(box)
        figure=fitz.open()
        page=figure.new_page(width=crop.width, height=crop.height)
        page.show_pdf_page(page.rect, source, page_number, clip=crop)
        figure.save(OUTPUT/f'{name}.pdf',garbage=4,deflate=True)
        page.get_pixmap(matrix=fitz.Matrix(1.4,1.4),alpha=False).save(OUTPUT/f'{name}.png')
        svg=page.get_svg_image(text_as_path=False)
        (OUTPUT/f'{name}.svg').write_text('\n'.join(line.rstrip() for line in svg.splitlines())+'\n')
        manifest.append({'name':name,'drawing_index':page_number,'crop':box,'caption':caption,
                         'outputs':[f'{name}.{ext}' for ext in ['png','pdf','svg']]})
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==original_hash
    (HERE/'figures.json').write_text(json.dumps({'source':str(SOURCE),
        'source_sha256':original_hash,'source_unchanged':True,'figures':manifest},indent=2)+'\n')
    print(f'Exported {len(manifest)} independent PNG / SVG / PDF figure sets to {OUTPUT}')


if __name__=='__main__':
    main()
