#!/usr/bin/env python3
"""Stage 2 (standalone): collapse Rhea-matched enzymes to InterPro folds and
compare Rhea-anchored coverage vs EC-anchored (reviewed_63) vs the live run.

Efficient: cross-references the local reviewed_63.tsv (reviewed EC 6.3.- with
InterPro) for free; only queries UniProt for the OUT-of-EC-6.3 amide-formers,
which is precisely the completeness Rhea buys over EC-scoping.
"""
from __future__ import annotations
import csv, json, time, collections, urllib.parse, urllib.request
from pathlib import Path
import rhea_coverage as R

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
ROOT = HERE.parent
REV63 = ROOT / "Interpro_exhaustive" / "reviewed_63.tsv"
RUN = ROOT / ("enzyme_discovery_agent_improved/runs/"
              "20260914_220431_find_amide_synthetases_that_ligate_adipic_acid_and_hexamethylenediamine_for_nylo")


def uniprot_fields(accs, chunk=200):
    """Batch-fetch accession -> (protein_name, ec, interpro_ids) from UniProt REST."""
    out = {}
    accs = list(accs)
    for i in range(0, len(accs), chunk):
        part = accs[i:i + chunk]
        q = " OR ".join(f"accession:{a}" for a in part)
        url = ("https://rest.uniprot.org/uniprotkb/search?query=" + urllib.parse.quote(q) +
               "&fields=accession,protein_name,ec,xref_interpro&format=tsv&size=500")
        for attempt in range(4):
            try:
                raw = urllib.request.urlopen(url, timeout=60).read().decode()
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        else:
            continue
        rd = csv.DictReader(raw.splitlines(), delimiter="\t")
        for row in rd:
            acc = row.get("Entry", "")
            ipr = [x for x in (row.get("InterPro") or "").split(";") if x.strip()]
            out[acc] = (row.get("Protein names", ""), row.get("EC number", ""), ipr)
        time.sleep(0.3)
    return out


def main():
    hit_rhea, accs = R.matched_accessions()
    print(f"Rhea-matched reviewed accessions: {len(accs)}")

    # local reviewed EC 6.3.- with InterPro
    rev63 = {r["Entry"]: r for r in csv.DictReader(open(REV63), delimiter="\t")}
    in63 = accs & set(rev63)
    out63 = accs - set(rev63)
    print(f"  in reviewed EC 6.3.- (local): {len(in63)}")
    print(f"  OUT of EC 6.3.- (Rhea completeness win, will query): {len(out63)}")

    # fold key = first InterPro id; gather (fold -> exemplar record)
    fold = {}                      # ipr -> (acc, name, ec)
    def add(acc, name, ec, iprs):
        if not iprs:
            return
        k = iprs[0]
        if k not in fold:
            fold[k] = (acc, name, ec)
    for acc in in63:
        r = rev63[acc]
        add(acc, r.get("Protein names", ""), r.get("EC number", ""),
            [x for x in (r.get("InterPro") or "").split(";") if x.strip()])
    # query only the out-of-6.3 set
    extra = uniprot_fields(out63) if out63 else {}
    for acc, (name, ec, iprs) in extra.items():
        add(acc, name, ec, iprs)

    print(f"\n=== Rhea-anchored fold coverage: {len(fold)} InterPro folds ===")

    # recovery
    def has_name(subs):
        return any(any(s in (n or "").lower() for s in subs) for _, n, _ in fold.values())
    for lab, subs in [("panC", ["pantothenate", "pantoate"]),
                      ("gss", ["glutathionylspermidine"]),
                      ("NIS/aerobactin", ["iuca", "iucc", "aerobactin", "siderophore synthase", "putrebactin"]),
                      ("DesD/desferrioxamine", ["desferrioxamine"]),
                      ("cyanophycin", ["cyanophycin"])]:
        print(f"  {'RECOVERED' if has_name(subs) else 'missing  '}: {lab}")

    # out-of-EC-6.3 folds (completeness win)
    out_folds = {k: v for k, v in fold.items()
                 if not (v[2] or "").strip().startswith("6.3")}
    print(f"\n  folds whose exemplar is NOT EC 6.3.- (missed by EC-6.3 scoping): {len(out_folds)}")
    for k, (acc, name, ec) in list(out_folds.items())[:12]:
        print(f"    {k} {acc} EC={ec or '-'}  {name[:44]}")

    # precision: NH3-amidation / off-target folds present
    offt = {k: v for k, v in fold.items()
            if any(t in (v[1] or "").lower() for t in
                   ("carboxylase", "carbamoyl", "glutamine synthetase", "asparagine synthase",
                    "amidotransferase", "gmp synthase", "nad synthase"))}
    print(f"\n  likely off-target / NH3-amidation folds (precision cost): {len(offt)}")
    for k, (acc, name, ec) in list(offt.items())[:10]:
        print(f"    {k} {acc}  {name[:46]}")

    # compare to the live run's surveyed folds
    summ = json.load(open(RUN / "exploration_summary.json"))["families"]
    names = list(summ.keys()) if isinstance(summ, dict) else [f.get("name", "") for f in summ]
    import re
    run_iprs = set(re.findall(r"IPR\d+", " ".join(names)))
    netnew = set(fold) - run_iprs
    print(f"\n=== coverage vs live run ===")
    print(f"  run surveyed folds: {len(run_iprs)}")
    print(f"  Rhea-anchored folds: {len(fold)}")
    print(f"  net-new folds Rhea adds: {len(netnew)}")

    # persist
    with open(CACHE / "rhea_folds.tsv", "w") as fh:
        fh.write("interpro\taccession\tec\tname\n")
        for k, (acc, name, ec) in sorted(fold.items()):
            fh.write(f"{k}\t{acc}\t{ec}\t{name}\n")
    print(f"\nwrote cache/rhea_folds.tsv ({len(fold)} folds)")


if __name__ == "__main__":
    main()
