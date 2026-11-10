"""Render audited contact cells with PyMOL, then compose overview and print pages.

The render command is one shard: launch disjoint shards with explicit CPU/thread
limits. Compose refuses missing cells. All picture bonds are separate from the
contact classification; ligand bonds are restored from the input chemical graph.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import pickle

import gemmi


ONE = dict(zip('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(), 'ARNDCQEGHILKMFPSTWYV'))


def ligand_bonds(cmd, row, chain):
    from rdkit import Chem
    atoms = {a.name: a.symbol for a in cmd.get_model(f'm and chain {chain}').atom}
    graphs = pickle.loads(Path(row['chemistry']['source']).read_bytes())
    bonds = None
    for mol in graphs.values():
        heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
        if all(a.HasProp('name') for a in heavy) and {a.GetProp('name'): a.GetSymbol() for a in heavy} == atoms:
            bonds = [(b.GetBeginAtom().GetProp('name'), b.GetEndAtom().GetProp('name'),
                      4 if b.GetIsAromatic() else int(b.GetBondTypeAsDouble())) for b in mol.GetBonds()
                     if b.GetBeginAtom().GetAtomicNum() != 1 and b.GetEndAtom().GetAtomicNum() != 1]
            break
    if bonds is None and chain == 'B' and row['campaign'] == 'M4':
        from enzyme_design_agent.tools.audit_as9_contacts import CAMPS
        path = CAMPS['M4'].parent / 'ccd/A/ATP/ATP.cif'
        block = gemmi.cif.read(str(path)).sole_block()
        bonds = [(gemmi.cif.as_string(r[0]), gemmi.cif.as_string(r[1]), {'SING': 1, 'DOUB': 2, 'TRIP': 3, 'AROM': 4}[r[2]])
                 for r in block.find(['_chem_comp_bond.atom_id_1', '_chem_comp_bond.atom_id_2', '_chem_comp_bond.value_order'])
                 if gemmi.cif.as_string(r[0]) in atoms and gemmi.cif.as_string(r[1]) in atoms]
        expected = {gemmi.cif.as_string(r[0]): r[1] for r in block.find(['_chem_comp_atom.atom_id', '_chem_comp_atom.type_symbol']) if r[1] != 'H'}
        if expected != atoms:
            raise ValueError('CCD ATP atom identities differ from pose')
    if not bonds:
        raise ValueError(f'No exact chemical graph for chain {chain}')
    # Delete inferred bonds, including any spurious ligand/protein or Mg bond.
    cmd.unbond(f'm and chain {chain}', 'm')
    for a, b, order in bonds:
        sa, sb = f'm and chain {chain} and name "{a}"', f'm and chain {chain} and name "{b}"'
        if cmd.count_atoms(sa) != 1 or cmd.count_atoms(sb) != 1:
            raise ValueError(f'Ambiguous chemical graph atom selection: {a}, {b}')
        cmd.bond(sa, sb, order=order)
    if len(cmd.get_model(f'm and chain {chain}').bond) != len(bonds):
        raise ValueError('Rendered ligand bond count differs from source graph')
    return len(bonds)


def render(args):
    from pymol import cmd, util
    from pymol.cgo import BEGIN, END, LINES, COLOR, VERTEX
    import numpy as np
    from enzyme_design_agent.tools.audit_as9_contacts import read_atoms
    rows = [json.loads(Path(p).read_text()) for p in json.loads((args.audit / 'result_index.json').read_text())]
    rows = [r for r in rows if r['panel_cell'] and r['campaign'] == args.campaign]
    specs = json.loads((args.audit / 'inputs' / f'{args.campaign.lower()}_panel_full.json').read_text())
    by_id = {r['panel_cell']['id']: r for r in rows}
    rows = [by_id[c['id']] for c in specs]
    if args.only:
        rows = [r for r in rows if args.only in r['panel_cell']['id']]
    ref = by_id['WT_AS9']['panel_cell']
    cmd.reinitialize()
    cmd.set('max_threads', args.threads)
    cmd.bg_color('white')
    for key, value in {'ray_opaque_background': 1, 'antialias': 2, 'ray_shadows': 0,
                       'ambient': .5, 'stick_radius': .15, 'sphere_scale': .25,
                       'label_size': 20, 'label_color': 'black', 'label_outline_color': 'white',
                       'label_font_id': 7, 'label_bg_color': 'white', 'label_bg_transparency': .1,
                       'float_labels': 1,
                       'valence': 1, 'valence_size': .13}.items():
        cmd.set(key, value)
    cmd.set_color('contact_ionic', [0, .447, .698])
    cmd.set_color('contact_polar', [.835, .369, 0])
    cmd.set_color('substrate_c', [.9, .62, 0])
    cmd.load(ref['pdb'], 'ref')
    ref_atoms, _ = read_atoms(ref['pdb'])
    lig = [a for a in ref_atoms if a.chain in {'B', 'D'}]
    site = sorted({a.resi for a in ref_atoms if a.chain == 'A' and a.name == 'CA'
                   and min(math.dist(a.xyz, b.xyz) for b in lig) < 10}, key=int)
    if len(site) < 10:
        raise ValueError('Too few shared active-site anchors')
    cmd.orient('ref and (chain D or (chain B and elem P))')
    cmd.turn('y', 20)
    cmd.turn('x', -12)
    cmd.zoom('ref and (chain D or chain B)', buffer=5.8, complete=1)
    cmd.clip('slab', 35)
    view = cmd.get_view()
    cmd.disable('ref')
    dest = args.audit / 'cells' / args.campaign
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    for idx, row in enumerate(rows):
        if idx % args.shards != args.shard:
            continue
        cell = row['panel_cell']
        cmd.delete('m')
        cmd.delete('ct_*')
        cmd.delete('reaction_proximity')
        cmd.delete('lab_*')
        cmd.delete('label_leaders')
        cmd.load(cell['pdb'], 'm')
        fit = []
        for ri in site:
            a, b = f'm and chain A and resi {ri} and name CA', f'ref and chain A and resi {ri} and name CA'
            if cmd.count_atoms(a) != 1 or cmd.count_atoms(b) != 1:
                raise ValueError(f'Unresolved active-site residue mapping: {ri}')
            fit.extend([a, b])
        fit_rmsd = cmd.pair_fit(*fit, quiet=1)
        graph_counts = {ch: ligand_bonds(cmd, row, ch) for ch in ('B', 'D')}
        cmd.hide('everything', 'm')
        cmd.show('cartoon', 'm and chain A and byres (m and chain A within 10 of (m and (chain B or chain D)))')
        cmd.color('gray85', 'm and chain A')
        cmd.set('cartoon_transparency', .91, 'm and chain A')
        pocket = 'm and chain A and byres (m and chain A within 5.0 of (m and chain D))'
        cmd.show('sticks', f'({pocket}) and not name N+C+O+OXT')
        cmd.color('gray78', pocket)
        cmd.set('stick_radius', .08, pocket)
        cmd.set('stick_transparency', .55, pocket)
        for ch, color in [('D', 'substrate_c'), ('B', 'teal')]:
            cmd.show('sticks', f'm and chain {ch}')
            cmd.color(color, f'm and chain {ch} and elem C')
            util.cnc(f'm and chain {ch}')
        cmd.show('spheres', 'm and elem Mg')
        cmd.color('forest', 'm and elem Mg')
        # No inferred covalent bond to magnesium in the drawing.
        cmd.unbond('m and elem Mg', 'm')
        accepted = [c for c in row['group_contacts'] if not c['conditional']]
        labeled = {}
        com = cmd.centerofmass('m and chain D')
        for j, c in enumerate(accepted):
            p = c['pairs'][0]
            ri = c['ri']
            residue = f'm and chain A and resi {ri}'
            cmd.show('sticks', f'{residue} and not name N+C+O+OXT')
            cmd.set('stick_radius', .15, residue)
            cmd.set('stick_transparency', 0, residue)
            cmd.color('gray45', f'{residue} and elem C')
            util.cnc(residue)
            name = f'ct_{j}'
            cmd.distance(name, f'{residue} and name {p["atom"]}', f'm and chain D and name {p["O"]}')
            cmd.hide('labels', name)
            cmd.set('dash_color', 'contact_ionic' if c['kind'] == 'potential_ionic' else 'contact_polar', name)
            frequent = c['consistency'] >= .5
            cmd.set('dash_width', 4 if frequent else 2.2, name)
            cmd.set('dash_radius', .035 if frequent else .023, name)
            cmd.set('dash_length', .18 if frequent else .05, name)
            cmd.set('dash_gap', .20 if frequent else .25, name)
            labeled.setdefault((c['res'], ri), set()).add('D' if c['where'] == 'distal' else 'A')
        anchors = []
        # PyMOL stores this matrix in OpenGL column-major order.
        rotation = np.asarray(view[:9]).reshape(3, 3).T
        origin = np.asarray(view[12:15])
        for (res, ri), roles in labeled.items():
            sel = f'm and chain A and resi {ri} and name CB'
            if not cmd.count_atoms(sel):
                sel = f'm and chain A and resi {ri} and name CA'
            label = f'{ONE.get(res, res)}{ri} [{"/".join(sorted(roles))}]'
            cb = np.asarray(cmd.get_atom_coords(sel))
            anchors.append(dict(label=label, cb=cb, camera=rotation @ (cb-origin)))
        # Place labels in two external columns, ordered by their anchor height.
        # Thin gray leaders identify residues without masking donor atoms or ATP.
        height = 2*abs(view[11])*math.tan(math.radians(float(cmd.get('field_of_view')))/2)
        width = height*1500/1150
        center_x = (rotation @ (np.asarray(com)-origin))[0]
        sides = [[a for a in anchors if a['camera'][0] < center_x],
                 [a for a in anchors if a['camera'][0] >= center_x]]
        leaders, label_manifest = [BEGIN, LINES, COLOR, .6, .6, .6], []
        for sign, side in zip([-1, 1], sides):
            side.sort(key=lambda a: a['camera'][1])
            ys = [float(np.clip(a['camera'][1], -.25*height, .25*height)) for a in side]
            for k in range(1, len(ys)):
                ys[k] = max(ys[k], ys[k-1]+.075*height)
            if ys and ys[-1] > .30*height:
                ys = [y-(ys[-1]-.30*height) for y in ys]
            for a, y in zip(side, ys):
                position = origin + rotation.T @ np.array([sign*.35*width, y, a['camera'][2]])
                leader_end = origin + rotation.T @ np.array([sign*.26*width, y, a['camera'][2]])
                obj = f'lab_{len(label_manifest)}'
                cmd.pseudoatom(obj, pos=position.tolist())
                cmd.hide('everything', obj)
                cmd.label(obj, repr(a['label']))
                cmd.set('label_position', [0, 0, 0], obj)
                leaders += [VERTEX, *a['cb'].tolist(), VERTEX, *leader_end.tolist()]
                label_manifest.append(dict(text=a['label'], world_position=position.tolist()))
        if anchors:
            cmd.load_cgo(leaders+[END], 'label_leaders')
            cmd.set('cgo_line_width', .7, 'label_leaders')
            cmd.set('cgo_line_radius', .009, 'label_leaders')
        cmd.distance('reaction_proximity', f'm and chain D and name {cell["attackO"]}',
                     f'm and chain B and name {cell["paname"]}')
        cmd.hide('labels', 'reaction_proximity')
        cmd.set('dash_color', 'black', 'reaction_proximity')
        cmd.set('dash_width', 2.2, 'reaction_proximity')
        cmd.set('dash_radius', .025, 'reaction_proximity')
        cmd.set('dash_gap', .14, 'reaction_proximity')
        cmd.set('dash_length', .08, 'reaction_proximity')
        cmd.set_view(view)
        cmd.set('ray_trace_mode', 0)
        path = dest / f'{row["ident"]}.png'
        if args.preview:
            cmd.set('antialias', 1)
        cmd.ray(750 if args.preview else 1500, 575 if args.preview else 1150)
        if args.preview:
            path = path.with_stem(path.stem+'_preview')
        cmd.png(str(path), dpi=400)
        # Individual files retain enough resolution for a 3.5-inch print panel.
        manifest.append(dict(ident=row['ident'], png=str(path.resolve()),
                             active_site_fit_rmsd_A=fit_rmsd, ligand_bonds=graph_counts,
                             contact_lines=len(accepted), reference_residues=site,
                             labels=label_manifest,
                             pid=os.getpid(), affinity=sorted(os.sched_getaffinity(0))))
        print(f'{args.campaign} {row["ident"]}: {len(accepted)} contact lines', flush=True)
    (dest / f'manifest_{args.shard}.json').write_text(json.dumps(manifest, indent=2))


def compose(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.backends.backend_pdf import PdfPages
    rows = [json.loads(Path(p).read_text()) for p in json.loads((args.audit / 'result_index.json').read_text())]
    rows = [r for r in rows if r['panel_cell'] and r['campaign'] == args.campaign]
    specs = json.loads((args.audit / 'inputs' / f'{args.campaign.lower()}_panel_full.json').read_text())
    lookup = {r['panel_cell']['id']: r for r in rows}
    rows = [lookup[c['id']] for c in specs]
    # Recover WT ensemble labels from the exact controls, not historical constants.
    from enzyme_design_agent.tools.audit_as9_contacts import BASE, CAMPS
    import numpy as np
    wt_metrics = (BASE/'as9_motif_m0_m1_20261106/wt_controls_charged_20261107/jobs/WT_AS9_charged/geometry.json'
                  if args.campaign == 'M3' else CAMPS['M4'].parent/'wt_charged_control_20261109/wt_ccdATP_glutarate_geom.json')
    control = json.loads(wt_metrics.read_text())
    geom = [p['geometry'] for p in control['poses'] if p.get('geometry')]
    if len(geom) != 50:
        raise ValueError('Incomplete WT geometry ensemble')
    dd = np.asarray([g['distance_A'] for g in geom])
    aa = np.asarray([g['angle_deg'] for g in geom])
    rows[0]['geometry'] = dict(A=float(np.sum(aa/dd)/np.sum(1/dd)), dh=float(len(dd)/np.sum(1/dd)),
                               nac=float(np.mean((dd < 4)&(aa > 150))))
    for row in rows:
        if not (args.audit / 'cells' / args.campaign / f'{row["ident"]}.png').is_file():
            raise ValueError(f'Missing rendered cell: {row["ident"]}')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    ligand = 'adipate (C6); ATP from SMILES, charge −4' if args.campaign == 'M3' else 'glutarate (C5); CCD ATP'
    handles = [Line2D([], [], color='#0072B2', ls='--', lw=2, label='Potential ionic contact'),
               Line2D([], [], color='#D55E00', ls='--', lw=2, label='Donor–acceptor polar contact'),
               Line2D([], [], color='black', ls=':', lw=1.3, label='Substrate O ··· ATP Pα proximity')]

    def page(subset, ncols, width, title_size, print_page=False):
        nrows = math.ceil(len(subset) / ncols)
        height = nrows * (3.35 if print_page else 2.65) + 2.65
        fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False)
        fig.subplots_adjust(left=.025, right=.975, bottom=.035, top=1-2.35/height,
                            hspace=.30, wspace=.10)
        for ax in axes.ravel():
            ax.axis('off')
        for ax, row in zip(axes.ravel(), subset):
            ax.imshow(plt.imread(args.audit / 'cells' / args.campaign / f'{row["ident"]}.png'))
            ax.axis('on')
            ax.set_xticks([])
            ax.set_yticks([])
            g = row['geometry']
            name = row['panel_cell']['id'].replace('AS9D ', '')
            metric = f'A {g["A"]:.1f}° · dₕ {g["dh"]:.2f} Å · NAC {100*g["nac"]:.0f}%' if g else 'WT reference'
            ax.set_title(f'{name}   |   {metric}\n{row["n_distal"]} distal + {row["n_attack"]} attack; {row["n_conditional"]} flagged omitted',
                         fontsize=8 if print_page else 8.5, loc='left', linespacing=1.35, pad=5)
            for s in ax.spines.values():
                s.set_color('#999999' if row['ident'] != 'WT_AS9' else '#333333')
                s.set_linewidth(.7)
        top = lambda inches: 1 - inches/height
        fig.text(.025, top(.16), f'AS9 {args.campaign} | chemically compatible substrate contacts',
                 fontsize=title_size, fontweight='bold', va='top')
        fig.text(.025, top(.43), f'{ligand} | WT + {len(rows)-1} selected designs', fontsize=8.5 if print_page else 10, va='top')
        fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.015, top(.62)),
                   ncol=1 if print_page else 3, frameon=False, fontsize=7 if print_page else 9)
        text = ('One line per residue–carboxylate pair. [D]/[A]: distal/attack. Gray sticks: pocket; fine gray lines: label leaders.\n'
                'Bold dashes: ≥50% prediction consistency; fine dots: <50%. Neither is bond occupancy.\n'
                'Polar 2.4–3.5 Å; potential ionic 2.4–4.0 Å. His/Cys and anomalously short groups omitted; no added H.\n'
                'Heavy-atom contacts ≠ proven bonds. Geometry labels use 50 models; NAC: d <4 Å and angle >150°.\n'
                'Membership and representative poses preserved. One residue may contact both carboxylates; counts refer to groups.')
        fig.text(.025, top(1.30 if print_page else 1.0), text, fontsize=6.5 if print_page else 8.5,
                 va='top', linespacing=1.4)
        return fig
    out = args.audit / 'figures'
    out.mkdir(exist_ok=True)
    fig = page(rows, 6, 19.2, 15)
    fig.savefig(out / f'{args.campaign}_essential_contacts.png', dpi=300)
    plt.close(fig)
    with PdfPages(out / f'{args.campaign}_contact_atlas.pdf') as pdf:
        for i in range(0, len(rows), 4):
            fig = page(rows[i:i+4], 2, 7.2, 10, True)
            pdf.savefig(fig, dpi=400)
            if i == 0 or any('df32bc' in r['ident'] or 'b7bd8a' in r['ident'] for r in rows[i:i+4]):
                fig.savefig(out / f'{args.campaign}_printproof_{i//4+1}.png', dpi=180)
            plt.close(fig)
    print(f'Composed {len(rows)} {args.campaign} cells', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['render', 'compose'])
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--campaign', choices=['M3', 'M4'], required=True)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--threads', type=int, default=3)
    p.add_argument('--only')
    p.add_argument('--preview', action='store_true')
    args = p.parse_args()
    if args.command == 'render':
        if not 0 <= args.shard < args.shards:
            p.error('Invalid shard')
        render(args)
    else:
        compose(args)


if __name__ == '__main__':
    main()
