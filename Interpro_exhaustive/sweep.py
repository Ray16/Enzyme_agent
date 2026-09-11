"""Substrate-first sweep of the reviewed EC 6.3.- ligase space.

Front-end alternative to LLM fold-proposal: enumerate ALL reviewed amide/C-N
ligases (reviewed_63.tsv), then a DETERMINISTIC chemistry gate keeps only
reactions that, in the formation direction, condense a FREE carboxylic acid with a
FREE amine into an amide (the nylon-6,6 target chemistry) -- structurally dropping
glutamine amidotransferases (amide from Gln), ammonia ligases (glutamine
synthetase), and nucleotide/cofactor ligases (purine, argininosuccinate, ...)
without any EC blacklist.

Reports: how many entries pass, how many DISTINCT enzymes after ortholog collapse,
which InterPro folds they span, and whether ground-truth DdaG (E2JA32) survives.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
CHEBI = json.loads((HERE / "chebi_cache.json").read_text())

# --- functional-group SMARTS -------------------------------------------------
_ACID = Chem.MolFromSmarts("[CX3](=[OX1])[OX2H1,OX1-]")          # -COOH / -COO-
_AMIDE = Chem.MolFromSmarts("[NX3][CX3](=[OX1])")                # -C(=O)N<
# free primary/secondary amine (incl. protonated), NOT an amide N, NOT aromatic,
# NOT nitrile/imine.  Ammonia is excluded separately by identity.
_AMINE = Chem.MolFromSmarts("[NX3;H1,H2;!$(N C=O);!$(N=*);!$([n])]".replace(" ", ""))
_AMMONIUM = Chem.MolFromSmarts("[NX4;H2,H3,H4;+;!$(N C=O)]".replace(" ", ""))

# cofactors / leaving groups that are never the "substrate of interest"
_COFACTOR_NAMES = re.compile(
    r"\b(ATP|ADP|AMP|GTP|GDP|GMP|CTP|UTP|dATP|phosphate|diphosphate|triphosphate|"
    r"H2O|water|H\(\+\)|hydron|Mg\(2\+\)|Mn\(2\+\)|K\(\+\)|Na\(\+\)|CoA|"
    r"coenzyme A|NAD|NADP|NADH|NADPH|CO2|carbon dioxide|hydrogencarbonate|"
    r"hydrogen carbonate|O2|dioxygen|pyrophosphate)\b", re.I)
# ammonia as an "amine" source is glutamine-synthetase chemistry, not a free amine
_AMMONIA = re.compile(r"\bammoni(a|um)\b", re.I)
_GLUTAMINE = re.compile(r"\bL-glutamine\b", re.I)
_GLUTAMATE = re.compile(r"\bL-glutamate\b", re.I)


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\(\d*[+-]\)", "", s)     # drop charge notation (1+),(2-)
    s = s.replace("zwitterion", "")
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def _mol(chebi_id: str):
    sm = (CHEBI.get(chebi_id) or {}).get("smiles")
    return Chem.MolFromSmiles(sm) if sm else None


def _has(mol, patt) -> bool:
    return bool(mol) and mol.HasSubstructMatch(patt)


def _reactions(cat_activity: str):
    """Yield (lhs_text, rhs_text, [chebi_ids]) per reaction block."""
    for block in cat_activity.split("CATALYTIC ACTIVITY:"):
        block = block.strip()
        if not block or "Reaction=" not in block:
            continue
        m = re.search(r"Reaction=(.*?);\s*Xref=(.*?)(?:;\s*EC=|$)", block, re.S)
        if not m:
            continue
        rxn, xref = m.group(1), m.group(2)
        if "=" not in rxn:
            continue
        lhs, rhs = rxn.split("=", 1)
        chebis = re.findall(r"CHEBI:\d+", xref)
        yield lhs.strip(), rhs.strip(), chebis


def _participants(side_text: str) -> list[str]:
    """Normalized participant tokens of one reaction side (split on ' + ',
    stripping stoichiometry prefixes like '2 ' or 'n ')."""
    parts = []
    for p in side_text.split(" + "):
        p = re.sub(r"^\s*(\d+|n)\s+", "", p.strip())   # drop '2 '/'n ' coefficients
        parts.append(_norm(p))
    return [p for p in parts if p]


def _assign_side(chebis, lhs_text, rhs_text):
    """Map each ChEBI to substrate (LHS) / product (RHS) by matching its name to a
    single PARTICIPANT token on one side.  Token-level (not whole-side) matching so
    an acylation product that embeds a substrate's name does not make that
    substrate look present on both sides.  Exact normalized match wins; a unique
    containment match is the fallback."""
    lhs_p, rhs_p = _participants(lhs_text), _participants(rhs_text)
    subs, prods = [], []
    for c in chebis:
        nm = _norm((CHEBI.get(c) or {}).get("name") or "")
        if not nm:
            continue
        exact_l = nm in lhs_p
        exact_r = nm in rhs_p
        if exact_l and not exact_r:
            subs.append(c); continue
        if exact_r and not exact_l:
            prods.append(c); continue
        if exact_l and exact_r:
            continue  # genuinely on both sides (rare) -> ignore
        # fallback: unique containment within a single participant token
        in_l = sum(nm in p or p in nm for p in lhs_p)
        in_r = sum(nm in p or p in nm for p in rhs_p)
        if in_l and not in_r:
            subs.append(c)
        elif in_r and not in_l:
            prods.append(c)
    return subs, prods


def gate(cat_activity: str) -> dict:
    """Return {'pass': bool, 'reason': str} for the free-acid + free-amine amide
    formation test, evaluated over all of the entry's reactions (any pass wins).

    Side-AGNOSTIC over the reaction participants: ChEBI canonical labels are often
    a synonym of the name used in the reaction text, so mapping a species to the
    substrate/product side by name is unreliable (it silently drops the very
    substrate we test for).  Every EC 6.3.- reaction is an ATP-dependent LIGATION
    (formation), so instead of assigning sides we ask whether the reaction's
    non-cofactor participants together carry a free carboxylic acid, a free
    (non-ammonia) amine, and an amide product -- the amide-ligation signature."""
    best = {"pass": False, "reason": "no parseable reaction"}
    for lhs, rhs, chebis in _reactions(cat_activity):
        noncof = []
        for c in chebis:
            nm = (CHEBI.get(c) or {}).get("name") or ""
            if _COFACTOR_NAMES.search(nm):
                continue
            noncof.append((c, nm, _mol(c)))
        acid = any(_has(m, _ACID) for _, _, m in noncof)
        amine = any(
            (not _AMMONIA.search(nm)) and (_has(m, _AMINE) or _has(m, _AMMONIUM))
            for _, nm, m in noncof
        )
        amide = any(_has(m, _AMIDE) for _, _, m in noncof)
        # glutamine amidotransferase: amide N donated by L-glutamine (-> glutamate)
        amidotransferase = bool(_GLUTAMINE.search(lhs) and _GLUTAMATE.search(rhs))
        if acid and amine and amide and not amidotransferase:
            return {"pass": True, "reason": "free acid + free amine -> amide"}
        if amidotransferase:
            best["reason"] = "glutamine amidotransferase (amide from Gln)"
        elif not amide:
            best["reason"] = "no amide among products"
        elif not amine:
            best["reason"] = "no free amine substrate (ammonia/cofactor only)"
        elif not acid:
            best["reason"] = "no free acid substrate"
    return best


def _gene_key(gene: str, protein: str, acc: str) -> str:
    if gene:
        return f"gene:{gene.lower()}"
    n = re.sub(r"\b\d+\b", "", protein.lower())
    n = re.sub(r"\s+", " ", n).strip()
    return f"name:{n}" if n else f"acc:{acc}"


def main() -> None:
    rows = list(csv.DictReader((HERE / "reviewed_63.tsv").open(), delimiter="\t"))
    passed, reasons = [], Counter()
    for r in rows:
        g = gate(r.get("Catalytic activity", "") or "")
        reasons[g["reason"] if not g["pass"] else "PASS"] += 1
        if g["pass"]:
            passed.append(r)

    # ortholog collapse
    groups = defaultdict(list)
    for r in passed:
        groups[_gene_key(r.get("Gene Names (primary)", ""),
                         r.get("Protein names", ""), r["Entry"])].append(r)
    # folds
    folds = Counter()
    for r in passed:
        for ipr in re.findall(r"IPR\d+", r.get("InterPro", "") or ""):
            folds[ipr] += 1

    print(f"reviewed EC 6.3.- entries scanned : {len(rows)}")
    print(f"PASS free-acid+free-amine amide  : {len(passed)}")
    print(f"distinct enzymes (ortholog-collapsed): {len(groups)}")
    print(f"distinct InterPro folds spanned  : {len(folds)}")
    print("\n-- gate outcome breakdown --")
    for k, v in reasons.most_common():
        print(f"  {v:6d}  {k}")
    print("\n-- top folds among passers --")
    for ipr, n in folds.most_common(15):
        print(f"  {ipr}  {n}")

    # ground-truth check
    gt = {"E2JA32": "DdaG"}
    print("\n-- ground-truth check --")
    pass_accs = {r["Entry"] for r in passed}
    for acc, name in gt.items():
        print(f"  {name} ({acc}): {'RECOVERED' if acc in pass_accs else 'missed'}")

    # persist
    with (HERE / "passed_candidates.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
        w.writeheader()
        # one representative (longest) per ortholog group
        for grp in groups.values():
            grp.sort(key=lambda r: int(r.get("Length") or 0), reverse=True)
            w.writerow(grp[0])
    print(f"\nwrote {len(groups)} representatives -> passed_candidates.tsv")


if __name__ == "__main__":
    sys.exit(main())
