"""Move supplementary generated figures to an archive and centralize keepers."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
HERE=Path(__file__).resolve().parent
SLIDES=HERE.parent
KEEP={'assay_key_candidates_and_controls','current_metric_shortlist_recovery',
      'all_explored_proxy_comparison','activation_vs_aminolysis_readouts',
      'complete_mutant_geometry_tradeoffs','structural_proxy_search_inventory',
      'proposed_multistate_motif_constraints','adaptive_decision_loop',
      'shared_evidence_handoff','proposed_framework_evaluation','modeling_setup_audit',
      'AMP versus ADP mechanism comparison','ADP-forming PuuA geometry','AMP-forming AS9 geometry'}


def main():
    receipt=HERE/'cleanup_manifest.json'
    if receipt.exists():
        print(json.loads(receipt.read_text())['archive']);return
    archive=SLIDES/'archive'/('backup_figures_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    archive.mkdir(parents=True)
    old=json.loads((HERE/'figure_catalog.json').read_text())
    for name in ['figure_catalog.json','FIGURE_INDEX.txt','quantitative_figures.json',
                 'method_figures.json','asset_validation.json','README.txt']:
        shutil.copy2(HERE/name,archive/name)
    moves=[];kept=[]

    def move(p,destination):
        if not p.exists():return
        destination.parent.mkdir(parents=True,exist_ok=True)
        moves.append({'original':str(p),'archived':str(destination),
                      'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        shutil.move(str(p),str(destination))

    for item in old['figures']:
        name=item['name'];selected=name in KEEP
        if selected:kept.append(name)
        for f in item['absolute_files']:
            p=Path(f)
            if not p.exists():continue
            if not selected:
                move(p,archive/p.relative_to(SLIDES));continue
            if p.parent==HERE:continue
            if name=='AMP versus ADP mechanism comparison':
                suffix=p.suffix;destination=HERE/f'amp_adp_comparison{suffix}'
                if suffix=='.tex':shutil.copy2(p,destination)
                else:destination.parent.mkdir(exist_ok=True);shutil.move(str(p),str(destination))
            elif name=='AMP-forming AS9 geometry':
                # This source is also used in the main talk. Keep it in place.
                shutil.copy2(p,HERE/p.name)
            else:
                shutil.move(str(p),str(HERE/p.name))
    assert len(kept)==14,kept
    # Earlier review-deck exports and duplicate previews are no longer active
    # deliverables. Keep them intact in the same reversible archive.
    legacy=SLIDES/'backup-finalization'
    for pattern in ['*.png','*.svg','*.pdf']:
        for p in legacy.glob(pattern):move(p,archive/p.relative_to(SLIDES))
    for folder in [legacy/'previews',legacy/'individual',HERE/'previews']:
        if folder.exists():
            dest=archive/folder.relative_to(SLIDES);dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(folder),str(dest))
    result={'active_figures':14,'previous_figures':len(old['figures']),
            'archive':str(archive),'active_folder':str(HERE),'kept':kept,
            'archived_file_moves':moves,'main_AS9_source_preserved':True}
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    (archive/'RESTORE.txt').write_text('Supplementary figures were moved here, not deleted.\n'
        'The original catalog and source-folder structure are retained.\n'
        f'Cleanup receipt: {receipt}\n')
    print(json.dumps({'active_figures':14,'archived_figures':len(old['figures'])-14,
                      'archive':str(archive)}))


if __name__=='__main__':main()
