#!/usr/bin/env python3
"""Slide-ready PNG gallery of the NATIVE reactions of a run's ranked candidates.

For each ranked candidate it draws the 2D structures of its native reaction's
organic participants (UniProt/Rhea-annotated SMILES straight from
``ranked_candidates.csv`` — no inference, no fabricated products) under the
authoritative catalytic-activity equation.  One panel per candidate, laid out in
a grid and ordered by rank, accent-coloured by tier.

Reusable across runs so galleries can be compared as the agent improves.

Usage
-----
  python analysis/native_reaction_grid.py RUN_DIR [--out PNG] [--top N] [--cols 4]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import textwrap
from difflib import SequenceMatcher
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D
RDLogger.DisableLog("rdApp.*")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# Phosphoryl-transfer cofactors and inorganic ions carry no discriminating
# chemistry.  Dropped ONLY when --drop-cofactors is passed; by default every
# annotated native-reaction participant is drawn (the point of the gallery).
# NOTE: ammonium is deliberately NOT here — for glutamine synthetases / CPS /
# asparagine synthetase it is the actual amine SUBSTRATE, not a spectator.
# Matched as a standalone name, never as a substring, so "1,4-butanediammonium"
# and "gamma-L-glutamylputrescinium" are always kept.
_COFACTORS = {
    "atp", "adp", "amp", "gtp", "gdp", "ctp", "utp", "itp", "idp",
    "phosphate", "hydrogenphosphate", "diphosphate", "pyrophosphate",
    "h2o", "water", "h(+)", "h+", "hydron", "o2", "co2",
    "nad(+)", "nadh", "nadp(+)", "nadph", "coa", "coenzyme a",
    "mg(2+)", "mn(2+)", "k(+)", "na(+)",
}

_TIER_COLOR = {"1": "#0f8a4f", "2": "#b06a00", "3": "#7a4fd0"}


# Names in ``native_substrates`` are protonation-state forms ("...ium(1+)",
# "... zwitterion") while the equation uses neutral names, so a plain string
# compare fails.  Normalisation strips charge tags + state words and maps the few
# genuine synonyms the equations use; matching then survives the naming drift.
_SYN = {"ammonium": "nh4", "azanium": "nh4"}


def _norm(s: str) -> str:
    s = re.sub(r"\(\d*[+-]\)", "", s.lower())
    s = re.sub(r"\b(zwitterion|cation|anion|dication|trication|tetraanion|ion|residue)\b", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return _SYN.get(s, s)


def _rxn_sides(cat: str) -> tuple[set[str], set[str]]:
    """Normalised reactant / product name tokens, unioned over ALL reactions of the
    entry (a multi-reaction enzyme's substrate of reaction 2 must not read as a
    product of reaction 1)."""
    lhs: set[str] = set()
    rhs: set[str] = set()
    for rxn in (cat or "").split(" | "):
        if " = " not in rxn:
            continue
        left, right = rxn.split(" = ", 1)
        lhs |= {_norm(x) for x in left.split(" + ")}
        rhs |= {_norm(x) for x in right.split(" + ")}
    return lhs, rhs


def _ratio(n: str, toks: set[str]) -> float:
    # also try dropping a trailing "ium" so tyraminium ~ tyramine
    variants = (n, n[:-3]) if n.endswith("ium") else (n,)
    return max((SequenceMatcher(None, v, t).ratio() for v in variants for t in toks),
               default=0.0)


def _is_product(name: str, lhs: set[str], rhs: set[str]) -> bool:
    """A participant is a product when it matches the equation's RHS more closely
    than its LHS.  Species matching neither side (a differently-named substrate,
    e.g. 2,3-diaminopropanoate vs '3-amino-L-alanine') default to reactant, so no
    substrate is lost."""
    n = _norm(name)
    return _ratio(n, rhs) > 0.85 and _ratio(n, rhs) > _ratio(n, lhs)


def _participants(names: list[str], smis: list[str], drop_cof: bool,
                  cat: str, reactants_only: bool) -> list[tuple[str, str]]:
    """Annotated native-reaction participants, in order.  ``drop_cof`` removes the
    phosphoryl-transfer cofactors / inorganic ions in _COFACTORS; ``reactants_only``
    additionally drops species identified as reaction products."""
    lhs, rhs = _rxn_sides(cat) if reactants_only else (set(), set())
    out = []
    for n, s in zip(names, smis):
        if drop_cof:
            base = n.lower().strip()
            if base in _COFACTORS or base.split("(")[0].strip() in _COFACTORS:
                continue
        if reactants_only and _is_product(n, lhs, rhs):
            continue
        out.append((n, s))
    return out


def _parse(smiles: str) -> tuple["Chem.Mol | None", bool]:
    """Return (mol, strict_ok).  A strict parse failure (e.g. an unkekulizable
    aromatic ring from a malformed extracted SMILES like bare 'n' for indole) is
    retried leniently so the structure can still be drawn — but strict_ok=False
    marks it as an extraction defect for the correctness report."""
    m = Chem.MolFromSmiles(smiles)
    if m is not None:
        return m, True
    m = Chem.MolFromSmiles(smiles, sanitize=False)
    if m is None:
        return None, False
    try:
        m.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
            m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    except Exception:  # noqa: BLE001
        return None, False
    return m, False


def _mol_png(mol: "Chem.Mol", kekulize: bool, w: int = 240, h: int = 185) -> Image.Image | None:
    d = rdMolDraw2D.MolDraw2DCairo(w, h)
    o = d.drawOptions()
    o.bondLineWidth = 2
    o.padding = 0.10
    o.clearBackground = True
    try:
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol, kekulize=kekulize)
    except Exception:  # noqa: BLE001
        return None
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText())).convert("RGBA")


def _placeholder(name: str, w: int = 240, h: int = 185) -> Image.Image:
    """A dashed-border box naming a species whose SMILES would not parse — so a
    broken extraction is shown explicitly, never silently dropped."""
    im = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    d = ImageDraw.Draw(im)
    for x in range(0, w, 10):
        d.line([(x, 2), (x + 5, 2)], fill=(200, 60, 60), width=2)
        d.line([(x, h - 3), (x + 5, h - 3)], fill=(200, 60, 60), width=2)
    for y in range(0, h, 10):
        d.line([(2, y), (2, y + 5)], fill=(200, 60, 60), width=2)
        d.line([(w - 3, y), (w - 3, y + 5)], fill=(200, 60, 60), width=2)
    lines = textwrap.wrap(name, 22)[:4]
    d.text((w // 2, h // 2 - 18), "SMILES unavailable", fill=(180, 40, 40), anchor="mm")
    for i, ln in enumerate(lines):
        d.text((w // 2, h // 2 + 2 + i * 13), ln, fill=(90, 90, 90), anchor="mm")
    return im


def _plus_strip(imgs: list[Image.Image], gap: int = 26) -> Image.Image:
    """Composite substrate images side by side with '+' gaps into one strip."""
    if not imgs:
        return Image.new("RGBA", (240, 185), (255, 255, 255, 0))
    h = max(im.height for im in imgs)
    w = sum(im.width for im in imgs) + gap * (len(imgs) - 1)
    strip = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    x = 0
    for i, im in enumerate(imgs):
        strip.paste(im, (x, (h - im.height) // 2), im)
        x += im.width + gap
    return strip


def _clean_rxn(cat: str) -> str:
    first = (cat or "").split(" | ")[0].strip()
    return first.replace(" = ", "  →  ") if first else "(no annotated reaction)"


def load_rows(run_dir: Path, top: int | None) -> list[dict]:
    rows = list(csv.DictReader((run_dir / "ranked_candidates.csv").open()))
    rows.sort(key=lambda r: int(r.get("rank") or 10**9))
    return rows[:top] if top else rows


def build(run_dir: Path, out: Path, cols: int, top: int | None, cap: int | None,
          drop_cof: bool, reactants_only: bool) -> Path:
    rows = load_rows(run_dir, top)
    n = len(rows)
    rowsN = math.ceil(n / cols)

    query = ""
    tc = run_dir / "target_context.json"
    if tc.exists():
        try:
            query = json.loads(tc.read_text()).get("query", "")
        except (OSError, ValueError):
            pass

    bad_smiles: list[tuple] = []  # (rank, gene, name, smiles) — extraction errors
    fig, axes = plt.subplots(
        rowsN, cols, figsize=(cols * 4.7, rowsN * 3.0), dpi=150,
        squeeze=False, facecolor="white")

    for idx in range(rowsN * cols):
        ax = axes[idx // cols][idx % cols]
        ax.axis("off")
        if idx >= n:
            continue
        r = rows[idx]
        names = [x.strip() for x in (r.get("native_substrates") or "").split(";") if x.strip()]
        smis = [x.strip() for x in (r.get("native_substrate_smiles") or "").replace(".", ";").split(";") if x.strip()]
        org = _participants(names, smis, drop_cof, r.get("catalytic_activities", ""),
                            reactants_only)
        shown = org if cap is None else org[:cap]
        imgs = []
        for nm, s in shown:
            mol, strict_ok = _parse(s)
            im = _mol_png(mol, kekulize=strict_ok) if mol is not None else None
            if im is None:
                bad_smiles.append((r.get("rank"), r.get("gene"), nm, s, "unparseable"))
                im = _placeholder(nm)
            elif not strict_ok:
                # drawn via lenient parse, but the extracted SMILES is defective
                bad_smiles.append((r.get("rank"), r.get("gene"), nm, s, "malformed-aromatic (drawn leniently)"))
            imgs.append(im)
        strip = _plus_strip(imgs)
        ax.imshow(strip)
        extra = f"  (+{len(org) - cap} more)" if (cap and len(org) > cap) else ""

        tier = str(r.get("tier") or "").strip()
        color = _TIER_COLOR.get(tier, "#444")
        gene = (r.get("gene") or r.get("protein_name") or "?").strip()[:24]
        pname = (r.get("protein_name") or "").strip()
        title = f"#{int(r.get('rank') or 0)}  {gene}"
        ax.set_title(title, fontsize=12.5, fontweight="bold", color=color,
                     loc="left", pad=6)
        # protein-name subtitle + accession
        ax.text(0, -0.02, f"{pname[:40]}  ·  {r.get('accession','')}  ·  T{tier or '?'}",
                transform=ax.transAxes, fontsize=7.3, color="#555",
                va="top", ha="left")
        # equation caption
        eqn = _clean_rxn(r.get("catalytic_activities", "")) + extra
        wrapped = "\n".join(textwrap.wrap(eqn, 52)[:3])
        ax.text(0.5, -0.13, wrapped, transform=ax.transAxes, fontsize=7.6,
                color="#1a1a1a", va="top", ha="center", family="monospace")
        # tier accent bar
        ax.plot([0, 1], [1.02, 1.02], transform=ax.transAxes, color=color,
                lw=3, solid_capstyle="butt", clip_on=False)

    sup = "Native reactions of ranked candidates"
    if query:
        sup += f"\n{query}"
    fig.suptitle(sup, fontsize=15, fontweight="bold", y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=3.4, w_pad=1.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Correctness check — always emitted.  Any drawn reactant whose SMILES will
    # not parse is an EXTRACTION error in the run's data; surface it loudly and
    # record it beside the figure rather than letting the structure vanish.
    report = out.with_suffix(".validation.txt")
    if bad_smiles:
        lines = [f"#{rk} {gn}: [{st}] reactant '{nm}': {sm}"
                 for rk, gn, nm, sm, st in bad_smiles]
        report.write_text("NATIVE-SUBSTRATE SMILES DEFECTS (extraction errors in the run)\n"
                          + "\n".join(lines) + "\n")
        print(f"WARNING: {len(bad_smiles)} reactant(s) have defective SMILES; see {report}")
        for ln in lines:
            print("  " + ln)
    else:
        report.write_text("All drawn reactant SMILES parsed cleanly.\n")
        print("validation: all drawn reactant SMILES parsed cleanly")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=None, help="limit to top-N ranked")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--cap", type=int, default=None,
                    help="max structures drawn per panel (default: no cap — draw all)")
    ap.add_argument("--drop-cofactors", action="store_true",
                    help="omit phosphoryl-transfer cofactors/ions (ammonium is kept — it is a substrate)")
    ap.add_argument("--include-products", action="store_true",
                    help="also draw reaction products (default: reactants only)")
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    out = Path(a.out) if a.out else run_dir / "native_reaction_grid.png"
    p = build(run_dir, out, a.cols, a.top, a.cap, a.drop_cofactors,
              reactants_only=not a.include_products)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
