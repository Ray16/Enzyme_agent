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


def _attack_geometry(pose):
    """The reaction coordinate for THIS pose, GENERICALLY: forming distance = closest (substrate
    nucleophile O/N/S) <-> (cofactor electrophile) pair, and attack angle = nucleophile-P-leaving, where
    the leaving atom is the P-bonded atom most ANTI to the nucleophile (the in-line displacement axis; an
    ideal in-line attack -> ~180 deg). Electrophile = cofactor P if present (phosphoryl transfer), else C
    (carbonyl). Returns dict(nuc, p, lg, dist, angle_deg, mid=(x,y,z)) or None."""
    import math
    atoms = _parse_atoms(pose)
    cof, metals, substrate = _ligand_groups(atoms)
    elec = [a for a in cof if a[1][:1] == "P"] or [a for a in cof if a[1][:1] == "C"]
    nuc = [a for a in substrate if a[1][:1] in ("O", "N", "S")]
    if not elec or not nuc:
        return None

    def _d(a, b):
        return ((a[5]-b[5])**2 + (a[6]-b[6])**2 + (a[7]-b[7])**2) ** 0.5
    best = None
    for o in nuc:
        for p in elec:
            d = _d(o, p)
            if best is None or d < best[0]:
                best = (d, o, p)
    d, o, p = best

    def _ang(a, b, c):   # angle a-b-c in degrees
        v1 = (a[5]-b[5], a[6]-b[6], a[7]-b[7]); v2 = (c[5]-b[5], c[6]-b[6], c[7]-b[7])
        dot = sum(i*j for i, j in zip(v1, v2))
        n1 = sum(i*i for i in v1) ** 0.5; n2 = sum(i*i for i in v2) ** 0.5
        return math.degrees(math.acos(max(-1.0, min(1.0, dot / (n1 * n2 + 1e-9)))))
    bonded = [a for a in atoms if a[0] not in (p[0], o[0]) and _d(a, p) < 2.1]
    lg, angle = None, None
    for x in bonded:                    # leaving = P-substituent most anti to the nucleophile
        an = _ang(o, p, x)
        if angle is None or an > angle:
            angle, lg = an, x
    return {"nuc": o[0], "p": p[0], "lg": lg[0] if lg else None, "dist": d, "angle_deg": angle,
            "mid": ((o[5]+p[5]) / 2, (o[6]+p[6]) / 2, (o[7]+p[7]) / 2)}


# distinct saturated RGB tags for the locator pass -> one per anchor residue, recovered by colour
# matching from a flat-lit render so each residue's 2D screen position is known exactly.
_LOC_RGB = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
            (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60)]
_ATTACK_RGB = (0, 255, 255)        # reserved locator tag for the reaction-coordinate midpoint
_ATTACK_KEY = "__ATTACK__"
_IMG_W, _IMG_H = 1200, 1050


def render_structure(pose, residues, out_png):
    """Headless pymol render of the pose (NO baked-in text), colouring anchor residues by interaction
    category. Residue names are NOT drawn here -- instead a second 'locator' pass renders each anchor as
    a uniquely-coloured flat sphere so its 2D screen position can be recovered by colour matching; the
    caller places the names in matplotlib, in clear space, with no leader lines.
    Returns (ok, err, label_xy) where label_xy maps the residue label -> (px, py) in the CLEAN image."""
    cbs = _cb_coords(pose, [_resid(r["res"]) for r in residues])
    pairs = _contact_pairs(pose, [_resid(r["res"]) for r in residues])
    locator_png = out_png.replace(".png", "_locator.png")

    pml = [f"load {pose}, m", "hide everything", "bg_color white",
           "remove solvent", "remove (resn WAT+HOH+SOL+TIP3+T3P+NA+CL+K)", "remove (elem Na+Cl+K)",
           "set ray_opaque_background, 0", "set cartoon_transparency, 0.9",
           "set stick_radius, 0.16", "set valence, 0",
           "select atp_ch, byres (not polymer and elem P)",
           "select metals, (not polymer and elem Mg+Mn+Zn+Fe+Ca+Ni+Co+Cu)",
           "select substrate, (not polymer and not atp_ch and not metals)",
           "show cartoon, polymer", "color gray90, polymer",
           "show sticks, substrate", "util.cbao substrate", "set stick_radius, 0.28, substrate",
           "show sticks, atp_ch", "util.cbac atp_ch",
           "show spheres, metals", "color forest, metals", "set sphere_scale, 0.45, metals",
           "set dash_color, gray40", "set dash_radius, 0.058", "set dash_gap, 0.35",
           "set dash_length, 0.25"]
    rids = []
    loc_lines = []          # pseudoatom + colour commands for the locator pass (created, then hidden)
    res_tag = {}            # residue label -> locator RGB
    for i, r in enumerate(residues):
        rid = _resid(r["res"])
        if not rid or rid not in cbs:
            continue
        rids.append(rid)
        col = CAT_COLOR.get(r["category"], "#999999")
        pml += [f"select r{rid}, (polymer and resi {rid} and not name N+C+O)", f"show sticks, r{rid}",
                f"set_color c{rid}, [{int(col[1:3],16)/255:.3f},{int(col[3:5],16)/255:.3f},{int(col[5:7],16)/255:.3f}]",
                f"color c{rid}, r{rid} and elem C"]
        if rid in pairs:    # ONE thin dashed line = the residue's single closest contact
            a1, serial = pairs[rid]
            pml += [f"distance dd{rid}, (polymer and resi {rid} and name {a1}), (id {serial})",
                    f"hide labels, dd{rid}"]
        rgb = _LOC_RGB[i % len(_LOC_RGB)]
        res_tag[r["res"]] = rgb
        x, y, z = cbs[rid]
        loc_lines += [f"pseudoatom loc{rid}, pos=[{x:.3f}, {y:.3f}, {z:.3f}]",
                      f"set_color lc{rid}, [{rgb[0]/255:.3f}, {rgb[1]/255:.3f}, {rgb[2]/255:.3f}]",
                      f"color lc{rid}, loc{rid}"]
    # reaction coordinate: nucleophile -> electrophilic P as a BLUE dash. The pose's forming distance +
    # attack angle are placed later in matplotlib (via a locator tag at the dash midpoint) so they never
    # collide with the sticks. pymol's own distance label is hidden.
    attack = None
    ag = _attack_geometry(pose)
    if ag:
        attack = (ag["dist"], ag["angle_deg"])
        o_s, p_s = ag["nuc"], ag["p"]
        pml += ["set_color attackblue, [0.12, 0.31, 0.85]",
                f"distance attack, (id {o_s}), (id {p_s})",
                "set dash_color, attackblue, attack", "set dash_radius, 0.058, attack",
                "hide labels, attack"]
        mx, my, mz = ag["mid"]
        res_tag[_ATTACK_KEY] = _ATTACK_RGB
        loc_lines += [f"pseudoatom locatk, pos=[{mx:.3f}, {my:.3f}, {mz:.3f}]",
                      f"set_color lcatk, [{_ATTACK_RGB[0]/255:.3f}, {_ATTACK_RGB[1]/255:.3f}, {_ATTACK_RGB[2]/255:.3f}]",
                      "color lcatk, locatk"]
    site = "substrate or metals" + "".join(f" or r{rid}" for rid in rids)
    pml += ["set cartoon_side_chain_helper, 1",
            f"select site, ({site})", "orient site", "zoom site, 1.0"]
    pml += loc_lines + ["hide everything, loc*"]     # create locator atoms AFTER orient; keep hidden
    # two renders at one camera: CLEAN (final) then LOCATOR (flat-lit colour tags only)
    pml += ["python",
            "from pymol import cmd",
            "v = cmd.get_view()",
            f"cmd.png('{out_png}', width={_IMG_W}, height={_IMG_H}, dpi=200, ray=1)",
            "cmd.hide('everything')", "cmd.bg_color('white')",
            "cmd.show('spheres', 'loc*')", "cmd.set('sphere_scale', 0.9, 'loc*')",
            "cmd.set('ambient', 1.0)", "cmd.set('direct', 0.0)", "cmd.set('spec_reflect', 0.0)",
            "cmd.set('ray_shadows', 0)", "cmd.set('antialias', 0)",
            "cmd.set_view(v)",
            f"cmd.png('{locator_png}', width={_IMG_W}, height={_IMG_H}, dpi=200, ray=1)",
            "python end", "quit"]
    f = tempfile.NamedTemporaryFile("w", suffix=".pml", delete=False)
    f.write("\n".join(pml)); f.close()
    r = subprocess.run([PYMOL, "-cq", f.name], capture_output=True, text=True, timeout=300)
    if not os.path.exists(out_png):
        return False, r.stderr[-500:], {}, None
    label_xy = _detect_label_positions(locator_png, res_tag)
    try:
        os.remove(locator_png)
    except OSError:
        pass
    return True, "", label_xy, attack


def _detect_label_positions(locator_png, res_tag, tol=60):
    """Recover each anchor residue's 2D screen position (pixel centroid of its colour tag)."""
    if not os.path.exists(locator_png):
        return {}
    import numpy as np
    from PIL import Image
    arr = np.asarray(Image.open(locator_png).convert("RGB")).astype(int)
    out = {}
    for lab, (R, G, B) in res_tag.items():
        mask = ((abs(arr[..., 0] - R) < tol) & (abs(arr[..., 1] - G) < tol) &
                (abs(arr[..., 2] - B) < tol))
        if mask.sum() < 4:
            continue
        ys, xs = np.nonzero(mask)
        out[lab] = (float(xs.mean()), float(ys.mean()))
    return out


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


def _place_residue_labels(axL, img, label_xy, residues, attack=None):
    """Draw anchor residue names (+ the pose's forming distance / attack angle) on the structure panel at
    OPTIMISED positions (no leader lines): seed each name just OUTSIDE its residue along the outward radial
    direction (away from the active-site centre, where the sticks are), then let adjustText separate any
    that still collide while a firm pull keeps each label hugging its residue. His salt bridges get a
    protonation star; the reaction coordinate is labelled in blue at the dash midpoint."""
    import numpy as np
    import matplotlib.patheffects as pe
    if not label_xy:
        return
    halo = [pe.withStroke(linewidth=3.0, foreground="white")]
    cat = {r["res"]: r["category"] for r in residues}
    H, W = img.shape[0], img.shape[1]
    rgb = img[..., :3] if img.ndim == 3 else np.dstack([img] * 3)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    ink = rgb.min(axis=2) < 0.82          # sticks/spheres/dashes = darker than the near-white bg/cartoon
    ys, xs = np.nonzero(ink)
    if len(xs) > 600:                     # subsample the obstacle cloud for speed
        k = np.random.default_rng(0).choice(len(xs), 600, replace=False)
        xs, ys = xs[k], ys[k]

    attack_xy = label_xy.pop(_ATTACK_KEY, None)
    # active-site centre = mean of all tagged positions; seed each label pushed radially outward from it
    pts = list(label_xy.values()) + ([attack_xy] if attack_xy else [])
    cx = sum(p[0] for p in pts) / len(pts); cy = sum(p[1] for p in pts) / len(pts)
    off = 0.11 * (W + H) / 2              # modest outward nudge: clears the stick, stays beside the residue

    texts = []
    for lab, (px, py) in label_xy.items():
        name = lab + ("*" if (cat.get(lab) == "salt_bridge" and _resname(lab) == "HIS") else "")
        col = CAT_COLOR.get(cat.get(lab), "#333333")
        dx, dy = px - cx, py - cy
        n = (dx*dx + dy*dy) ** 0.5 or 1.0
        sx = min(max(px + dx/n * off, 0.05*W), 0.95*W)
        sy = min(max(py + dy/n * off, 0.05*H), 0.95*H)
        texts.append(axL.text(sx, sy, name, fontsize=11, fontweight="bold", color=col,
                              ha="center", va="center", zorder=10, path_effects=halo))
    if attack and attack[0] is not None:   # reserve the bottom-right corner for the attack read-out box
        gx = np.linspace(0.60 * W, 0.99 * W, 9)
        gy = np.linspace(0.82 * H, 0.99 * H, 6)
        mesh = np.array([(x, y) for x in gx for y in gy])
        xs = np.concatenate([xs, mesh[:, 0]]); ys = np.concatenate([ys, mesh[:, 1]])
    try:
        from adjustText import adjust_text
        # firm pull keeps each label beside its residue; text-text force separates co-located anchors;
        # light obstacle force nudges off the sticks. Seeds already sit outside the cluster.
        adjust_text(texts, x=xs, y=ys, ax=axL, avoid_self=True,
                    force_text=(0.5, 0.7), force_static=(0.12, 0.2), force_pull=(0.5, 0.5),
                    expand=(1.25, 1.4), ensure_inside_axes=True, only_move="xy")
    except Exception:
        pass                               # fall back to seed positions if adjustText unavailable
    axL.set_xlim(0, W); axL.set_ylim(H, 0)


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
    ok, err, label_xy, attack = render_structure(pose, residues, struct_png)

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
        img = plt.imread(struct_png)
        axL.imshow(img); axL.axis("off")
        _place_residue_labels(axL, img, label_xy, residues, attack)
        if attack and attack[0] is not None:   # pose's reaction coordinate, as a corner read-out
            d, a = attack
            txt = f"attack distance: {d:.1f} Å"
            if a is not None:
                txt += f"\nattack angle: {a:.0f}°"
            axL.text(0.985, 0.015, txt, transform=axL.transAxes, ha="right", va="bottom", ma="left",
                     fontsize=10.5, fontweight="bold", color="#1a1a1a", linespacing=1.4,
                     bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                               edgecolor="#555555", linewidth=1.0, alpha=0.9))
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
