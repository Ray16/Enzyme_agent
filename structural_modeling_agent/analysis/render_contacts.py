"""Pair a STRUCTURAL pose render with a CONTACT-FREQUENCY bar chart (the agent's visualization of the
substrate_contacts detector + statistics). Left = one representative pose (substrate orange, ATP teal,
metals green spheres, anchoring residues as sticks colored by interaction category); right = per-residue
contact frequency across the Boltz-pose ensemble (or MD), colored by the same categories.

Enzyme-agnostic: residues + substrate are taken from the frequency JSON and detected in pymol by
composition (chain containing P = cofactor; metal elements = metals; the rest = substrate). The bar
chart is the robust signal; the pose is one illustrative snapshot.

Run (in an env with matplotlib+PIL; shells out to the pymol binary):
  python render_contacts.py --pose model.pdb --freq-json ens.json --title "GlnA3 (Q9KZC7)" --out fig.png
"""
import json, argparse, subprocess, os, tempfile

CAT_COLOR = {
    "salt_bridge": "#1b7837",      # green
    "h_bond": "#2166ac",           # blue
    "hydrophobic": "#999999",      # gray
}
PYMOL = "/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/pymol/bin/pymol"


def _top_residues(freq, min_freq=0.5, max_res=6):
    """From the ensemble/MD frequency JSON, collect residues to highlight: dominant category +
    max frequency per residue, keeping anchoring interactions (salt bridge / metal bridge / H-bond)."""
    best = {}
    for gk, g in freq.get("groups", {}).items():
        role = gk.split(":")[0]
        for c in g["contacts"]:
            # key anchors = side-chain-specific interactions only; drop non-specific contacts
            # (hydrophobic packing, backbone-amide H-bonds e.g. Ala/Gly) from the figure.
            if c["category"] in ("hydrophobic", "backbone_hbond"):
                continue
            key = c["res"]
            if key not in best or c["frequency"] > best[key]["frequency"]:
                best[key] = {"res": c["res"], "category": c["category"],
                             "frequency": c["frequency"], "role": role}
    rows = [r for r in best.values() if r["frequency"] >= min_freq]
    rows.sort(key=lambda r: -r["frequency"])
    return rows[:max_res]


def _resid(res):
    return "".join(ch for ch in res if ch.isdigit())


def _resname(res):
    return "".join(ch for ch in res if ch.isalpha()).upper()


# A reactive carboxylate poised to attack the electrophilic P sits ~3-4 Å from it; beyond this the
# substrate's reactive end is not at the reaction center and any anchors are holding a non-productive
# register. The panel's 2026-11 data separate cleanly: the 4 docked hits are all <=4.6 Å, the rest >=6.1.
DOCK_CUT_A = 5.0


def _dockedness(freq):
    """From the frequency JSON, read the REACTIVE carboxylate's mean distance to the reaction center and
    its metal-coordination frequency. Returns (react_d_A | None, docked: bool, metal_coord_freq | None)."""
    react = next((g for k, g in freq.get("groups", {}).items() if k.startswith("reactive")), None)
    if react is None:
        return None, False, None
    d = react.get("d_to_center_A_mean")
    mg = react.get("metal_coordination_freq")
    return d, (d is not None and d <= DOCK_CUT_A), mg


_AA3 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
        "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "HIE", "HID", "HIP", "CYX", "ASH", "GLH", "LYN"}   # incl. Amber protonation variants
_METAL = {"MG", "MN", "ZN", "CA", "FE", "NI", "CO", "CU"}
_SKIP = {"HOH", "WAT", "SOL", "TIP3", "T3P", "NA", "CL", "K", "NA+", "CL-", "K+", "SOD", "CLA"}


def _parse_atoms(pose):
    """Parse ATOM+HETATM -> list of (serial, name, chain, resi, resn, x, y, z). Record-type agnostic
    (works for cofold poses AND MD/prep PDBs, where ligands may be ATOM records)."""
    out = []
    for ln in open(pose):
        if ln[:6] not in ("ATOM  ", "HETATM"):
            continue
        out.append((int(ln[6:11]), ln[12:16].strip(), ln[21], ln[22:26].strip(), ln[17:20].strip(),
                    float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    return out


def _ligand_groups(atoms):
    """Group non-protein, non-solvent atoms by (chain, resname) -> classify each group as cofactor
    (has P), metal, or substrate. Keying on (chain,resname) splits both cofold poses (chains B/C/E,
    all resn LIG) and MD structures (one chain, resn ATP/SUB/MG)."""
    by = {}
    for a in atoms:
        rn = a[4].upper()
        if rn in _AA3 or rn in _SKIP:
            continue
        by.setdefault((a[2], rn), []).append(a)
    cofactor, metals, substrate = [], [], []
    for (ch, rn), al in by.items():
        names = {a[1] for a in al}
        elems = {"".join(c for c in a[1] if c.isalpha()).upper() for a in al}
        if names & {"PA", "PB", "PG"} or (any(n.startswith("P") for n in names) and len(al) > 6):
            cofactor += al
        elif elems <= _METAL or (len(al) == 1 and next(iter(elems)) in _METAL):
            metals += al
        else:
            substrate += al
    return cofactor, metals, substrate


def _cb_coords(pose, resids):
    """CB coord (CA fallback) for each protein resid."""
    want = set(resids); got = {}
    for a in _parse_atoms(pose):
        if a[4].upper() not in _AA3 or a[3] not in want:
            continue
        if a[1] == "CB" or (a[1] == "CA" and a[3] not in got):
            got[a[3]] = (a[5], a[6], a[7])
    return got


def _contact_pairs(pose, resids):
    """One clean dash per residue: its single closest atom pair to the substrate or a metal."""
    atoms = _parse_atoms(pose)
    cof, metals, substrate = _ligand_groups(atoms)
    partners = substrate + metals
    sidechain = {}
    for a in atoms:
        if a[4].upper() in _AA3 and a[3] in set(resids) and a[1] not in ("N", "CA", "C", "O"):
            sidechain.setdefault(a[3], []).append(a)
    out = {}
    for rid, sc in sidechain.items():
        best = None
        for ra in sc:
            for pa in partners:
                d = (ra[5]-pa[5])**2 + (ra[6]-pa[6])**2 + (ra[7]-pa[7])**2
                if best is None or d < best[0]:
                    best = (d, ra[1], pa[0])
        if best and best[0] <= 4.2**2:
            out[rid] = (best[1], best[2])
    return out


def _attack_pair(pose):
    """The reaction coordinate, GENERICALLY: the closest (substrate nucleophile heteroatom O/N/S) <->
    (cofactor electrophile) pair. Electrophile = cofactor phosphorus if present (phosphoryl transfer:
    adenylation/acyl-phosphate), else cofactor carbon (carbonyl, e.g. thioester). No hardcoded atom
    names, so it adapts to the chemistry. Returns (nuc_serial, elec_serial, elec_name, distance) or None."""
    atoms = _parse_atoms(pose)
    cof, metals, substrate = _ligand_groups(atoms)
    elec = [a for a in cof if a[1][:1] == "P"] or [a for a in cof if a[1][:1] == "C"]
    nuc = [a for a in substrate if a[1][:1] in ("O", "N", "S")]
    best = None
    for o in nuc:
        for p in elec:
            d = ((o[5]-p[5])**2 + (o[6]-p[6])**2 + (o[7]-p[7])**2) ** 0.5
            if best is None or d < best[-1]:
                best = (o[0], p[0], p[1], d)
    return best


def render_structure(pose, residues, out_png):
    """Headless pymol render of the pose; highlight residues colored by interaction category."""
    pml = [f"load {pose}, m", "hide everything", "bg_color white",
           # strip water + neutralizing counter-ions outright (solvated MD structures bring ~10^4
           # waters + Na/Cl that otherwise swamp the view).
           "remove solvent", "remove (resn WAT+HOH+SOL+TIP3+T3P+NA+CL+K)", "remove (elem Na+Cl+K)",
           "set ray_opaque_background, 0", "set cartoon_transparency, 0.9",
           "set stick_radius, 0.16", "set valence, 0",
           # resname-agnostic ligand selection (cofold resn LIG OR MD resn ATP/SUB/MG).
           # byres (NOT bychain): in single-chain MD structures bychain would grab the whole protein.
           "select atp_ch, byres (not polymer and elem P)",
           "select metals, (not polymer and elem Mg+Mn+Zn+Fe+Ca+Ni+Co+Cu)",
           "select substrate, (not polymer and not atp_ch and not metals)",
           "show cartoon, polymer", "color gray90, polymer",
           "show sticks, substrate", "util.cbao substrate", "set stick_radius, 0.28, substrate",
           "show sticks, atp_ch", "util.cbac atp_ch",
           "show spheres, metals", "color forest, metals", "set sphere_scale, 0.45, metals"]
    # interactions as THIN DASHED GRAY lines (residue <-> substrate/metal); residue names labelled
    # OUTWARD (radially from the active-site centre) so text clears the sticks.
    pml += ["set label_size, 16", "set label_color, black", "set label_outline_color, white",
            "set label_font_id, 7",
            "set dash_color, gray40", "set dash_radius, 0.045", "set dash_gap, 0.35",
            "set dash_length, 0.25"]
    cbs = _cb_coords(pose, [_resid(r["res"]) for r in residues])
    pairs = _contact_pairs(pose, [_resid(r["res"]) for r in residues])
    rids = []
    if cbs:
        cx = sum(v[0] for v in cbs.values()) / len(cbs)
        cy = sum(v[1] for v in cbs.values()) / len(cbs)
        cz = sum(v[2] for v in cbs.values()) / len(cbs)
    for r in residues:
        rid = _resid(r["res"])
        if not rid:
            continue
        rids.append(rid)
        col = CAT_COLOR.get(r["category"], "#999999")
        pml += [f"select r{rid}, (polymer and resi {rid} and not name N+C+O)", f"show sticks, r{rid}",
                f"set_color c{rid}, [{int(col[1:3],16)/255:.3f},{int(col[3:5],16)/255:.3f},{int(col[5:7],16)/255:.3f}]",
                f"color c{rid}, r{rid} and elem C"]
        if rid in pairs:  # ONE thin dashed line = the residue's single closest contact
            a1, serial = pairs[rid]
            pml += [f"distance dd{rid}, (polymer and resi {rid} and name {a1}), (id {serial})",
                    f"hide labels, dd{rid}"]
        if rid in cbs:
            v = cbs[rid]; dx, dy, dz = v[0] - cx, v[1] - cy, v[2] - cz
            n = (dx * dx + dy * dy + dz * dz) ** 0.5 or 1.0
            s = 2.3 / n   # modest outward offset: clears the stick yet stays inside the tight frame
            pml.append(f"set label_position, ({dx*s:.2f}, {dy*s:.2f}, {dz*s:.2f}), r{rid}")
        pml.append(f"label (polymer and resi {rid} and name CB), '{r['res']}'")
    # THE reaction coordinate: reactive carboxylate O -> electrophilic P, as a BLUE dashed line
    # (same thickness as the gray contact dashes), with its Angstrom value shown.
    ap = _attack_pair(pose)
    if ap:
        o_s, p_s, pname, dist = ap
        pml += ["set_color attackblue, [0.12, 0.31, 0.85]",
                f"distance attack, (id {o_s}), (id {p_s})",
                "set dash_color, attackblue, attack", "set dash_radius, 0.045, attack",
                "set label_color, attackblue, attack"]
    # frame the whole active site (substrate + metals + anchor residues) for a balanced composition
    site = "substrate or metals" + "".join(f" or r{rid}" for rid in rids)
    pml += ["set cartoon_side_chain_helper, 1",
            # per-candidate camera: orient looks down the site's thinnest axis (least occlusion),
            # auto-adapting to each enzyme's geometry -- no fixed turns.
            f"select site, ({site})", "orient site", "zoom site, 1.0",
            f"ray 1200, 1050", f"png {out_png}, dpi=200", "quit"]
    f = tempfile.NamedTemporaryFile("w", suffix=".pml", delete=False)
    f.write("\n".join(pml)); f.close()
    r = subprocess.run([PYMOL, "-cq", f.name], capture_output=True, text=True, timeout=300)
    return os.path.exists(out_png), r.stderr[-500:] if not os.path.exists(out_png) else ""


def _md_lookup(md_freq):
    """resid number -> max contact frequency (any category) across MD groups; for Boltz<->MD matching."""
    best = {}
    if not md_freq:
        return best
    for g in md_freq.get("groups", {}).values():
        for c in g["contacts"]:
            num = "".join(ch for ch in c["res"] if ch.isdigit())
            best[num] = max(best.get(num, 0.0), c["frequency"])
    return best


def build_figure(pose, freq, title, out_png, md_freq=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np
    residues = _top_residues(freq)
    md = _md_lookup(md_freq)
    for i, r in enumerate(residues, 1):     # number keys structure labels <-> bar labels
        r["idx"] = i
    # ---- dockedness gate: is the reactive carboxylate actually AT the reaction center? ----
    react_d, docked, react_mg = _dockedness(freq)
    if react_d is None:
        dock_txt, dock_col = "reactive-end distance to electrophile: unknown", "#777777"
    elif docked:
        dock_txt = f"reactive COO⁻ → electrophile: {react_d:.1f} Å   ✓ docked"
        if react_mg is not None:
            dock_txt += f"   ·   reactive end Mg-coordinated in {react_mg*100:.0f}% of poses"
        dock_col = "#1b7837"
    else:
        dock_txt = (f"reactive COO⁻ → electrophile: {react_d:.1f} Å   ✗ NOT docked (> {DOCK_CUT_A:.0f} Å)"
                    f"  —  anchors below hold a non-productive register")
        dock_col = "#b23024"
    # ---- structural panel ----
    struct_png = out_png.replace(".png", "_struct.png")
    ok, err = render_structure(pose, residues, struct_png)

    order = {"reactive": 0, "secondary": 1}
    if docked:   # only claim "reactive / carboxylate activated" when the substrate is actually poised
        SECTION = {"reactive": "REACTIVE end  (carboxylate activated)",
                   "secondary": "OTHER / distal end  (anchor)"}
    else:        # not docked: name the ends geometrically, make no catalytic claim
        SECTION = {"reactive": "end nearest electrophile  (not docked)",
                   "secondary": "other end"}
    groups = {}
    for r in residues:
        groups.setdefault(r["role"], []).append(r)
    roles = sorted(groups, key=lambda k: order.get(k, 9))

    fig = plt.figure(figsize=(15, 6.6))
    outer = fig.add_gridspec(1, 2, width_ratios=[0.82, 1.18], wspace=0.22)
    axL = fig.add_subplot(outer[0])
    if ok:
        axL.imshow(plt.imread(struct_png)); axL.axis("off")
        axL.set_title("Representative active-site pose\n(substrate orange, ATP teal, metal green;\ngray dashes = contacts)",
                      fontsize=11, pad=4)
    else:
        axL.text(0.5, 0.5, "structural render unavailable", ha="center"); axL.axis("off")

    if not roles:   # no key side-chain anchor (>=0.5): itself the signal of a non-specific / drifter pose
        axR = fig.add_subplot(outer[1]); axR.axis("off")
        axR.text(0.5, 0.5, "No persistent side-chain anchor\n(no salt bridge / H-bond at ≥ 0.50 of poses)\n\n"
                 "→ substrate not specifically held",
                 ha="center", va="center", fontsize=14, color="#b23024", fontweight="bold")
        fig.suptitle(title, fontsize=15, fontweight="bold", y=1.03)
        fig.text(0.5, 0.965, dock_txt, ha="center", va="top", fontsize=11,
                 color=dock_col, fontweight="bold")
        fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"wrote {out_png}  (no key anchors)" + ("" if ok else f"  (pymol failed: {err})"))
        return

    # right side: one VERTICAL-bar subplot per substrate end, side by side -> "both ends gripped"
    rgs = outer[1].subgridspec(1, len(roles), width_ratios=[max(1, len(groups[r])) for r in roles],
                               wspace=0.18)
    axes = []
    for i, role in enumerate(roles):
        ax = fig.add_subplot(rgs[i], sharey=axes[0] if axes else None); axes.append(ax)
        rows = sorted(groups[role], key=lambda r: -r["frequency"])
        xp = np.arange(len(rows))
        cols = [CAT_COLOR.get(r["category"], "#999999") for r in rows]
        if md:   # grouped: Boltz poses (solid) vs MD (hatched), same category color
            w = 0.4
            ax.bar(xp - w/2, [r["frequency"] for r in rows], width=w, color=cols,
                   edgecolor="black", linewidth=0.6)
            mdv = [md.get(_resid(r["res"]), 0.0) for r in rows]
            ax.bar(xp + w/2, mdv, width=w, color=cols, edgecolor="black", linewidth=0.6, hatch="////")
            for x, r in zip(xp, rows):
                ax.text(x - w/2, r["frequency"] + 0.02, f"{r['frequency']:.2f}", ha="center", fontsize=8.5)
            for x, v in zip(xp, mdv):
                ax.text(x + w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5)
        else:
            ax.bar(xp, [r["frequency"] for r in rows], color=cols, edgecolor="black",
                   linewidth=0.6, width=0.7)
            for x, r in zip(xp, rows):
                ax.text(x, r["frequency"] + 0.02, f"{r['frequency']:.2f}", ha="center",
                        fontsize=10, fontweight="bold")
        # His salt bridges are protonation-dependent (His must be protonated to be cationic) -> star them
        xlabs = [r["res"] + ("*" if (r["category"] == "salt_bridge" and _resname(r["res"]) == "HIS") else "")
                 for r in rows]
        ax.set_xticks(xp); ax.set_xticklabels(xlabs, fontsize=10.5)
        ax.set_ylim(0, 1.18); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(SECTION.get(role, role), fontsize=12, fontweight="bold", color="#222")
        if i == 0:
            ax.set_ylabel("contact frequency", fontsize=12)
        else:
            ax.tick_params(labelleft=False)
    cats = sorted({r["category"] for r in residues})
    handles = [Patch(facecolor=CAT_COLOR[c], edgecolor="black", label=c.replace("_", " ")) for c in cats]
    if md:
        handles += [Patch(facecolor="#cccccc", edgecolor="black", label=f"Boltz poses (n={freq.get('n_structures','?')})"),
                    Patch(facecolor="#cccccc", edgecolor="black", hatch="////",
                          label=f"MD (n={md_freq.get('n_frames','?')} frames)")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9.5,
               bbox_to_anchor=(0.72, -0.03), framealpha=0.95)
    # protonation caveat for any His salt bridge shown
    if any(r["category"] == "salt_bridge" and _resname(r["res"]) == "HIS" for r in residues):
        fig.text(0.72, -0.075, "* His salt bridge — protonation-dependent (requires protonated His)",
                 ha="center", va="top", fontsize=8.5, color="#555555", style="italic")
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.05)
    fig.text(0.5, 0.985, dock_txt, ha="center", va="top", fontsize=11.5,
             color=dock_col, fontweight="bold")
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_png}" + (" [NOT docked]" if not docked else "") +
          ("" if ok else f"  (pymol render failed: {err})"))


def _cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pose", required=True)
    ap.add_argument("--freq-json", required=True, help="output of substrate_contacts ensemble/traj")
    ap.add_argument("--title", default="")
    ap.add_argument("--md-freq-json", default=None, help="optional MD traj frequency to overlay")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    freq = json.load(open(a.freq_json))
    md_freq = json.load(open(a.md_freq_json)) if a.md_freq_json else None
    build_figure(a.pose, freq, a.title, a.out, md_freq)


if __name__ == "__main__":
    _cli()
