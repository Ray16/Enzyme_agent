"""Archive unused slide assets using the reviewed 29-page PDF image audit.

Default: write a dry-run plan. --apply: rename unused files into the archive,
then verify every source hash and the unchanged presentation PDF.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT/'slides-candidacy'
AUDIT=Path('/tmp/candidacy-cleanup-audit/image_audit.json')
DEST=ROOT/'archive/unused_20261101'

def sha(p):
    if p.is_symlink():
        return 'symlink:'+os.readlink(p)
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()

def plan():
    audit=json.loads(AUDIT.read_text())
    assert sha(ROOT/'Candidacy slides.pdf')==audit['pdf_sha256'],'PDF changed: rerun the audit'
    keep={}
    def add(path,reason):
        assert (ROOT/path).exists(),path
        keep.setdefault(path,[]).append(reason)
    # Choose one canonical asset for pixel-identical duplicates.
    for page in audit['pages']:
        for im in page['images']:
            if im['exact']:
                add(im['exact'][0],f'Exact decoded-pixel match to PDF page {page["page"]}, image xref {im["xref"]}')
    bundles={
      'Title structure, page 1; coordinate source for several retained renderers':[
        'title-as9/render_as9.py','title-as9/AS9_model_0.pdb','title-as9/render_metadata.json'],
      'Nylon chemistry, page 5; RDKit generator and imported chemistry validator':[
        'nylon-synthesis/render_nylon_synthesis_rdkit.py','nylon-synthesis/render_nylon_synthesis.py',
        'nylon-synthesis/rdkit_render_manifest.json'],
      'Offline discovery reconstruction, page 12':[
        'reasoning-trace/build_trace.py','reasoning-trace/verification.json','reasoning-trace/audit/figure_provenance.json'],
      'Family coverage, page 13':[
        'candidate-panel-summary/build_figure.py','candidate-panel-summary/sources.json'],
      'Mechanism animation source of the movie still on page 14; retained native 4K master and its rebuild inputs':[
        'as9-mechanism/AS9_mechanism_4K.mp4','as9-mechanism/keyframes/01_adipate.png',
        'as9-mechanism/build_states.py','as9-mechanism/make_animation.py','as9-mechanism/export_4k.py',
        'as9-mechanism/render_context.py','as9-mechanism/render_context_4k.py',
        'as9-mechanism/AS9_animation_context.pse','as9-mechanism/context_camera.json',
        'as9-mechanism/chemical_states.json','as9-mechanism/projection_fiducials.png',
        'as9-mechanism/protein_context.png','as9-mechanism/protein_context_4k.png',
        'as9-mechanism/4K_export_manifest.json'],
      'AI-generated source prompts for the specific variants actually used on page 15':[
        'structural-proxy-exploration/individual-panels/clean-visuals/reaction_geometry.prompt.txt',
        'structural-proxy-exploration/individual-panels/clean-visuals/substrate_access.prompt.txt',
        'structural-proxy-exploration/individual-panels/clean-visuals/substrate_access.boundary.prompt.txt',
        'structural-proxy-exploration/individual-panels/clean-visuals/pocket_accommodation.prompt.txt'],
      'AS9 and AS13 protein renderings on pages 15 and 24; shared display metadata':[
        'AS9-AS13-structure-comparison/render_comparison.py','AS9-AS13-structure-comparison/sources.json',
        'enzyme_visualization/style.py','enzyme_visualization/style.json','molecular_style.json'],
      'Attack-geometry illustration, page 16; raw render and projection calibration are composition inputs':[
        'attack-angle/render_site.py','attack-angle/compose_figure.py','attack-angle/geometry.json',
        'attack-angle/AS9_site_render.png','attack-angle/projection_fiducials.png'],
      'Yield versus geometry plot, page 16; original image is read for provenance by its generator':[
        'proximity-yield-explained/build_plots.py','proximity-yield-explained/sources.json',
        'proximity-yield-explained/plot_data.csv','main-talk/synthetase_simple_geometry.png'],
      'Discovery ranking, page 17; original image is read for provenance by its generator':[
        'discovery-ranking-simplified/build_plot.py','discovery-ranking-simplified/sources.json',
        'discovery-ranking-simplified/ranking_and_modeling_details.csv','main-talk/synthetase_simple_discovery.png'],
      'AI-generated sequence and pocket illustrations, page 19':[
        'in-silico-mutagenesis/saturation_mutagenesis_sequence.prompt.txt',
        'in-silico-mutagenesis/saturation_mutagenesis_pocket_illustration_v2.prompt.txt',
        'in-silico-mutagenesis/saturation_mutagenesis_pocket_illustration_v2_README.txt'],
      'Mutation figures, pages 20 and 21; shared composites remain necessary for their slide crops':[
        'saturation-mutagenesis-design/results-20261027/compose_structures.py',
        'saturation-mutagenesis-design/results-20261027/render_structures.py',
        'saturation-mutagenesis-design/results-20261027/make_candidate_figure.py',
        'saturation-mutagenesis-design/results-20261027/make_data_figure.py',
        'saturation-mutagenesis-design/results-20261027/make_scatter_figure.py',
        'saturation-mutagenesis-design/results-20261027/figure_data.json',
        'saturation-mutagenesis-design/results-20261027/representative_poses.json',
        'saturation-mutagenesis-design/results-20261027/residue_contact_map.json',
        'saturation-mutagenesis-design/results-20261027/as17_n344r_poses.json'],
      'Conformational comparison, page 24; overlay and ligand layer are exact composition inputs':[
        'conformational_sampling/render_cofold.py','conformational_sampling/render_ligand_composite.py',
        'conformational_sampling/compose_ligand_overlay.py','conformational_sampling/sources.json',
        'conformational_sampling/AS9_bioemu2_cofold_active_site.pse',
        'conformational_sampling/AS9_bioemu2_cofold_overlay.png','conformational_sampling/ligand_layer.png'],
      'Exploitation plot, page 26; narrow generator imports definitions from the shared builder':[
        'aim3_experimental_learning/selection_policy_figures/build_exploitation_narrow.py',
        'aim3_experimental_learning/selection_policy_figures/build_figures.py'],
      'Exploration comparison, page 26':[
        'aim3_experimental_learning/exploration_pose_consistency/build_final_slide.py',
        'aim3_experimental_learning/exploration_pose_consistency/final_sources.json'],
      'AS9 versus AS4 yield-angle plot, page 27':[
        'aim3_experimental_learning/unexpected_outcomes_slide/build_figure.py',
        'aim3_experimental_learning/unexpected_outcomes_slide/sources.json'],
    }
    for reason,paths in bundles.items():
        for p in paths:add(p,reason)
    mut='saturation-mutagenesis-design/results-20261027'
    rep=json.loads((ROOT/mut/'representative_poses.json').read_text())
    for e in ('AS9','AS23'):
        for suffix in ('geom.json','fiducials.png','wt_active_site.png','mut_active_site.png','wt_full.png','mut_full.png'):
            add(f'{mut}/{e}_{suffix}','Direct composition input for the used WT/mutant active-site composite')
        for role in ('wt','mut'):
            r=rep[e][role]
            add(f'{mut}/pulled/{r["ident"]}/{r["ident"]}_model_{r["pose"]}.pdb',
                'Selected structural coordinate input for the used mutation views')
    for p in (ROOT/'conformational_sampling/cofold_frames').glob('*.pdb'):
        add(str(p.relative_to(ROOT)),'Four propagated endpoints used on page 24')
    # The bundled encoder is a required import/runtime for the retained movie exporter.
    for p in (ROOT/'as9-mechanism/.runtime').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.name!='.lock':
            add(str(p.relative_to(ROOT)),'Bundled imageio-ffmpeg runtime used by the retained 4K movie generator')
    add('Candidacy slides.pdf','Authoritative updated 29-page deck; unchanged by cleanup')
    files=[p for p in ROOT.rglob('*') if (p.is_file() or p.is_symlink()) and p.relative_to(ROOT).parts[0]!='archive']
    rows=[]
    for p in sorted(files):
        relative=str(p.relative_to(ROOT))
        rows.append({'path':relative,'bytes':p.lstat().st_size,'sha256':sha(p),
                     'action':'retain' if relative in keep else 'archive',
                     'reasons':keep.get(relative,['Not used in the audited PDF and not required by a retained generator']),
                     'archive_path':str((DEST/relative).relative_to(ROOT)) if relative not in keep else None})
    return {'pdf_sha256':audit['pdf_sha256'],'pages':29,'archive':str(DEST),
            'image_matches':audit['pages'],'files':rows,
            'new_active_site_package':'New deliverable is not yet embedded in the PDF, so it is archived intact under the same rule as other unused assets.',
            'external_sources_unchanged':[
                'exp_data/amide_synthetase/results/analysis/as9_as4_evidence_20261031',
                'exp_data/amide_synthetase/scripts/as9_as4_evidence_audit_20261031.py',
                'exp_data/amide_synthetase/scripts/render_as9_as4_evidence_20261031.py',
                'enzyme_design_agent/campaigns/amide_AS9_AS17_AS23_step1_2Mg_saturation_5A_20261019',
                '0_prev_agent_archive/enzyme_discovery_agent_2026_0916/runs/20260917_135330_find_amide_synthetases_that_ligates_adipic_acid_and_hexamethylenediamine_for_nyl']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');args=ap.parse_args()
    report=plan()
    rows=report['files']
    counts={a:sum(r['action']==a for r in rows) for a in ('retain','archive')}
    sizes={a:sum(r['bytes'] for r in rows if r['action']==a) for a in counts}
    report['counts']=counts;report['bytes']=sizes
    Path('/tmp/candidacy-cleanup-audit/cleanup_plan.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'counts':counts,'bytes':sizes,'pdf_sha256':report['pdf_sha256']},indent=2),flush=True)
    if not args.apply:return
    assert not DEST.exists(),'Archive destination already exists; do not overwrite'
    assert sha(ROOT/'Candidacy slides.pdf')==report['pdf_sha256']
    # Validate the complete source snapshot and every destination before mutations.
    for r in rows:
        assert sha(ROOT/r['path'])==r['sha256'],r['path']
        if r['action']=='archive':assert not (ROOT/r['archive_path']).exists()
    DEST.mkdir(parents=True)
    (DEST/'MOVE_MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n')
    for r in rows:
        if r['action']=='archive':
            target=ROOT/r['archive_path'];target.parent.mkdir(parents=True,exist_ok=True)
            (ROOT/r['path']).rename(target)
    for p in sorted(ROOT.rglob('*'),key=lambda p:len(p.parts),reverse=True):
        if p.is_dir() and p.relative_to(ROOT).parts[0]!='archive':
            try:p.rmdir()
            except OSError:pass
    for r in rows:
        p=ROOT/(r['archive_path'] if r['action']=='archive' else r['path'])
        assert sha(p)==r['sha256'],str(p)
    assert sha(ROOT/'Candidacy slides.pdf')==report['pdf_sha256']
    report['verification']={'all_files_accounted_for':True,'all_file_hashes_match':True,'pdf_unchanged':True}
    (DEST/'MOVE_MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n')
    active={'pdf_sha256':report['pdf_sha256'],'pages':29,'files':[r for r in rows if r['action']=='retain'],
            'archive_manifest':str((DEST/'MOVE_MANIFEST.json').relative_to(ROOT)),
            'new_active_site_package':str((DEST/'aim3_experimental_learning/AS9_AS4_active_site').relative_to(ROOT)),
            'verification':report['verification']}
    (ROOT/'ASSET_MANIFEST.json').write_text(json.dumps(active,indent=2)+'\n')
    print('APPLIED AND VERIFIED',flush=True)

if __name__=='__main__':main()
