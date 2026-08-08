#!/usr/bin/env python3
"""Generate co-folding pipeline diagram — essential content, slide-friendly."""
import graphviz, os

def main():
    dot = graphviz.Digraph('CofoldPipeline')
    font = "Helvetica Neue,Helvetica,Arial,Liberation Sans,sans-serif"
    dot.attr(rankdir='TB', dpi='300', nodesep='0.55', ranksep='0.6',
             compound='true', newrank='true', size='14,28')
    dot.attr('node', shape='box', style='filled,rounded', fontname=font,
             fontsize='13', penwidth='1.4', margin='0.22,0.13')
    dot.attr('edge', fontname=font, fontsize='10', color='#555555',
             penwidth='1.2', arrowhead='vee')

    C_IN='#F7FAFC'; B_IN='#4A5568'
    C_P1='#EBF8FF'; B_P1='#2B6CB0'
    C_P2='#FAF5FF'; B_P2='#6B46C1'
    C_P3='#FFFBEB'; B_P3='#C05621'
    C_P4='#E6FFFA'; B_P4='#276749'
    C_OUT='#F0FFF4'; B_OUT='#276749'
    C_EXT='#FFF5F5'; B_EXT='#C53030'
    C_NOTE='#FFFFF0'; B_NOTE='#B7791F'

    dot.node('input', 'Candidate Enzymes\n(AlphaFold structures + sequences)',
             fillcolor=C_IN, color=B_IN, fontsize='15', penwidth='2')

    # ── Phase 1 ────────────────────────────────────────────────────────────
    with dot.subgraph(name='cluster_p1') as p1:
        p1.attr(label='Phase 1: Map Catalytic Residues', style='bold,rounded',
                color=B_P1, fontcolor=B_P1, fontname=font, fontsize='15',
                bgcolor='#F7FBFF')
        p1.node('p1a1', 'Own UniProt annotations\nCore: Active site\nShell: Binding + Metal binding',
                fillcolor=C_P1, color=B_P1)
        p1.node('p1fam', 'Annotation complete\nfor this family?',
                fillcolor='#DBEAFE', color='#1a6fa8', shape='diamond', style='filled',
                fontsize='12')
        p1.node('p1a2', 'Foldseek homolog transfer (TM ≥ 0.7)\nCore: LLM validates mechanism match\nShell: transferred only if no UniProt\nshell annotation · consensus ≥50%',
                fillcolor='#EBF8FF', color='#1a6fa8', style='filled,rounded,dashed')
        p1.node('p1_out', 'Core + Shell Residues · ≤18 Å spread',
                fillcolor='#BEE3F8', color=B_P1, penwidth='2')
        p1.edge('p1a1', 'p1fam')
        p1.edge('p1fam', 'p1_out', label='yes')
        p1.edge('p1fam', 'p1a2', label='no', style='dashed', color='#1a6fa8')
        p1.edge('p1a2', 'p1_out')

    dot.node('ext1', 'UniProt API', shape='ellipse', fillcolor=C_EXT, color=B_EXT, fontsize='11')
    dot.node('ext2', 'PDB / AlphaFold', shape='ellipse', fillcolor=C_EXT, color=B_EXT, fontsize='11')
    dot.edge('input', 'p1a1')
    dot.edge('p1a1', 'ext1', style='dashed', dir='back', color=B_EXT, constraint='false')
    dot.edge('p1b', 'ext2', style='dashed', dir='back', color=B_EXT, constraint='false')

    # ── Phase 2 ────────────────────────────────────────────────────────────
    with dot.subgraph(name='cluster_p2') as p2:
        p2.attr(label='Phase 2: Detect Binding Pockets', style='bold,rounded',
                color=B_P2, fontcolor=B_P2, fontname=font, fontsize='15',
                bgcolor='#FAF5FF20')
        p2.node('alphafill', 'AlphaFill (primary)\nCofactor contacts from PDB homologs',
                fillcolor=C_P2, color=B_P2)
        p2.node('uniprot_ann', 'UniProt annotation (2nd)',
                fillcolor=C_P2, color=B_P2)
        p2.node('foldseek_p2', 'Foldseek homolog TM ≥ 0.5 (3rd)',
                fillcolor=C_P2, color=B_P2)
        p2.node('kvfinder', 'pyKVFinder (always runs)\nCavity geometry + volume',
                fillcolor='#E9D8FD', color=B_P2)
        p2.node('psel', 'Pocket Selection\nOverlap with Phase 1 residues',
                fillcolor='#D6BCFA', color=B_P2)
        p2.node('p2_out', 'Binding Pocket + Cofactor Contacts + Metal Sites',
                fillcolor='#D6BCFA', color=B_P2, penwidth='2')
        with p2.subgraph() as s:
            s.attr(rank='same')
            s.node('alphafill'); s.node('uniprot_ann'); s.node('foldseek_p2'); s.node('kvfinder')
        p2.edge('alphafill', 'uniprot_ann', label='fallback', style='dashed', color=B_P2)
        p2.edge('uniprot_ann', 'foldseek_p2', label='fallback', style='dashed', color=B_P2)
        p2.edge('alphafill', 'psel'); p2.edge('uniprot_ann', 'psel', style='dashed')
        p2.edge('foldseek_p2', 'psel', style='dashed'); p2.edge('kvfinder', 'psel')
        p2.edge('psel', 'p2_out')

    dot.edge('p1_out', 'alphafill', lhead='cluster_p2')

    # ── Phase 3 ────────────────────────────────────────────────────────────
    with dot.subgraph(name='cluster_p3') as p3:
        p3.attr(label='Phase 3: Co-folding (Boltz-2)', style='bold,rounded',
                color=B_P3, fontcolor=B_P3, fontname=font, fontsize='15',
                bgcolor='#FFFBEB20')
        p3.node('template', 'AlphaFold template · cofactor analogs normalised (ANP→ATP)',
                fillcolor=C_NOTE, color=B_NOTE)
        p3.node('cof', 'Cofactor-Guided\nAlphaFill CCD available',
                fillcolor='#FEFCBF', color=B_P3)
        p3.node('poc', 'Pocket-Constrained\nPocket evidence, no cofactor',
                fillcolor='#FED7AA', color=B_P3)
        p3.node('blind', 'Blind\nNo evidence',
                fillcolor='#FED7D7', color=B_P3)
        p3.node('metals', 'Metals (Mg²⁺/Zn²⁺/Mn²⁺) · Coherence ≤15 Å',
                fillcolor=C_P3, color=B_P3)
        p3.node('best', 'Boltz-2: 5 seeds · 3 recycling + 200 sampling steps\nBest model selected',
                fillcolor='#FEEBC8', color=B_P3, penwidth='2')
        with p3.subgraph() as s:
            s.attr(rank='same'); s.node('cof'); s.node('poc'); s.node('blind')
        p3.edge('template','cof'); p3.edge('template','poc'); p3.edge('template','blind')
        p3.edge('cof','metals'); p3.edge('poc','metals'); p3.edge('blind','metals')
        p3.edge('metals','best')

    dot.edge('p2_out', 'template', lhead='cluster_p3')

    # ── Phase 4 ────────────────────────────────────────────────────────────
    with dot.subgraph(name='cluster_p4') as p4:
        p4.attr(label='Phase 4: Score & Validate', style='bold,rounded',
                color=B_P4, fontcolor=B_P4, fontname=font, fontsize='15',
                bgcolor='#F0FFF420')
        p4.node('conf', 'Confidence\niPTM · pLDDT · iPLDDT', fillcolor=C_P4, color=B_P4)
        p4.node('geom', 'Binding Geometry\nSASA burial · min distance', fillcolor=C_P4, color=B_P4)
        p4.node('chem', 'Pocket Chemistry\nHydrophobic · charged · polar', fillcolor=C_P4, color=B_P4)
        p4.node('ccs', 'Catalytic Center Score\n|pocket ∩ core| / |core|\nApo vs holo comparison',
                fillcolor='#C6F6D5', color=B_P4)
        with p4.subgraph() as s:
            s.attr(rank='same'); s.node('conf'); s.node('geom'); s.node('chem')
        p4.edge('conf','ccs'); p4.edge('geom','ccs'); p4.edge('chem','ccs')

    dot.edge('best', 'conf', lhead='cluster_p4')
    dot.edge('best', 'geom'); dot.edge('best', 'chem')

    dot.node('ranked', 'Ranked Candidate List\nfor Experimental Testing',
             fillcolor=C_OUT, color=B_OUT, fontsize='15', penwidth='2.5', shape='box3d')
    dot.edge('ccs', 'ranked')

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', 'cofold_pipeline')
    dot.render(out, format='png', cleanup=True)
    dot.render(out, format='pdf', cleanup=True)
    dot.render(out, format='svg', cleanup=True)
    print(f"Saved to {out}.{{png,pdf,svg}}")

if __name__ == '__main__':
    main()
