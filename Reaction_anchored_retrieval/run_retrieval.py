"""Standalone validation harness: reaction-anchored enzyme retrieval.

Pipeline-independent. Tests whether anchoring discovery on the TARGET REACTION
(ATP-dependent amide ligation, detected by chemistry from Rhea reaction SMILES)
recovers on-target amide ligases — including annotated ones the EC 6.3.- /
InterPro sweep misses — from an answer-agnostic query.

Stages (each cached to ./cache so reruns are cheap):
  1. download Rhea bulk TSVs
  2. detect ATP-dependent amide-ligation reactions (detect.py)
  3. map matching Rhea IDs -> UniProt (reviewed + unreviewed)
  4. report scale + overlap with a prior run's candidate set
  5. annotate the reviewed members (native substrate / family / EC) for eyeballing

Nothing here is wired into the agent. If the verdict is negative, the whole
directory is archived with the verdict recorded — no dead code in the tree.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect import classify_reaction  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

RHEA_BASE = "https://ftp.expasy.org/databases/rhea/tsv"
FILES = {
    "smiles": "rhea-reaction-smiles.tsv",
    "sprot": "rhea2uniprot_sprot.tsv",
    "trembl": "rhea2uniprot_trembl.tsv.gz",
}
UA = {"User-Agent": "reaction-anchored-validation/1.0 (research)"}


def _download(name: str) -> Path:
    dest = CACHE / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = f"{RHEA_BASE}/{name}"
    print(f"  downloading {name} ...", flush=True)
    r = requests.get(url, headers=UA, timeout=300)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("rt")


def stage_detect() -> dict:
    """Return {rhea_id: flags} for ATP-dependent amide-ligation reactions."""
    cached = CACHE / "matched_reactions.json"
    if cached.exists():
        return json.loads(cached.read_text())
    smiles_path = _download(FILES["smiles"])
    matched: dict[str, dict] = {}
    n = 0
    with smiles_path.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rhea_id, rxn = parts[0], parts[1]
            n += 1
            hit, flags = classify_reaction(rxn)
            if hit:
                matched[rhea_id] = {"reaction_smiles": rxn, **flags}
    print(f"  scanned {n} reactions -> {len(matched)} ATP-amide-ligation reactions")
    cached.write_text(json.dumps(matched, indent=1))
    return matched


def stage_map(matched: dict) -> dict:
    """Map matched Rhea IDs -> {accession: {'reviewed': bool, 'rhea_ids': [...]}}."""
    cached = CACHE / "members.json"
    if cached.exists():
        return json.loads(cached.read_text())
    matched_ids = set(matched)
    members: dict[str, dict] = {}

    def ingest(path: Path, reviewed: bool):
        with _open_text(path) as fh:
            header = fh.readline()  # RHEA_ID DIRECTION MASTER_ID ID
            for line in fh:
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 4:
                    continue
                rhea_id, acc = cols[0], cols[3]
                # MASTER_ID collapses the 4 directional children; match on any id.
                if rhea_id not in matched_ids and cols[2] not in matched_ids:
                    continue
                rec = members.setdefault(acc, {"reviewed": reviewed, "rhea_ids": []})
                rec["reviewed"] = rec["reviewed"] or reviewed
                anchor = rhea_id if rhea_id in matched_ids else cols[2]
                if anchor not in rec["rhea_ids"]:
                    rec["rhea_ids"].append(anchor)

    ingest(_download(FILES["sprot"]), reviewed=True)
    ingest(_download(FILES["trembl"]), reviewed=False)
    n_rev = sum(1 for v in members.values() if v["reviewed"])
    print(f"  mapped members: {len(members)} total | {n_rev} reviewed | "
          f"{len(members) - n_rev} unreviewed")
    cached.write_text(json.dumps(members, indent=1))
    return members


def stage_report(members: dict, prior_accessions_path: str | None) -> None:
    prior = set()
    if prior_accessions_path and Path(prior_accessions_path).exists():
        prior = set(Path(prior_accessions_path).read_text().split())
    reviewed = {a for a, v in members.items() if v["reviewed"]}
    print("\n  === OVERLAP WITH PRIOR RUN ===")
    print(f"  prior-run candidate accessions: {len(prior)}")
    print(f"  reaction-anchored reviewed members: {len(reviewed)}")
    if prior:
        new_rev = reviewed - prior
        print(f"  reviewed members NOT in prior run (new, annotated): {len(new_rev)}")
        (CACHE / "new_reviewed.json").write_text(json.dumps(sorted(new_rev), indent=1))


def _uniprot_batch(accs: list[str]) -> dict:
    """Fetch annotation for a batch of accessions via UniProt REST."""
    fields = ("accession,id,gene_names,protein_name,ec,organism_name,"
              "xref_interpro,cc_catalytic_activity,cc_function,reviewed")
    q = " OR ".join(f"accession:{a}" for a in accs)
    url = "https://rest.uniprot.org/uniprotkb/search"
    r = requests.get(url, params={"query": q, "fields": fields,
                                  "format": "json", "size": len(accs)},
                     headers=UA, timeout=120)
    r.raise_for_status()
    out = {}
    for e in r.json().get("results", []):
        acc = e.get("primaryAccession")
        desc = (e.get("proteinDescription", {}).get("recommendedName", {})
                .get("fullName", {}).get("value", ""))
        genes = ";".join(g.get("geneName", {}).get("value", "")
                         for g in e.get("genes", []) if g.get("geneName"))
        ecs = ";".join(
            ec.get("value", "") for ec in
            e.get("proteinDescription", {}).get("recommendedName", {}).get("ecNumbers", [])
        )
        interpro = ";".join(
            x.get("id", "") for x in e.get("uniProtKBCrossReferences", [])
            if x.get("database") == "InterPro"
        )
        cat = []
        for c in e.get("comments", []):
            if c.get("commentType") == "CATALYTIC ACTIVITY":
                rx = c.get("reaction", {})
                if rx.get("name"):
                    cat.append(rx["name"])
        out[acc] = {
            "gene": genes, "protein_name": desc, "ec": ecs,
            "organism": e.get("organism", {}).get("scientificName", ""),
            "interpro": interpro, "catalytic_activity": " | ".join(cat),
            "reviewed": e.get("entryType", "").startswith("UniProtKB reviewed"),
        }
    return out


def stage_annotate(members: dict, only_new: bool) -> None:
    reviewed = sorted(a for a, v in members.items() if v["reviewed"])
    if only_new and (CACHE / "new_reviewed.json").exists():
        new = set(json.loads((CACHE / "new_reviewed.json").read_text()))
        targets = [a for a in reviewed if a in new]
        tag = "new_reviewed"
    else:
        targets = reviewed
        tag = "all_reviewed"
    print(f"\n  annotating {len(targets)} {tag} members ...")
    rows = []
    for i in range(0, len(targets), 25):
        batch = targets[i:i + 25]
        try:
            ann = _uniprot_batch(batch)
        except requests.RequestException as exc:
            print(f"    batch {i} failed: {exc}")
            ann = {}
        for acc in batch:
            a = ann.get(acc, {})
            rows.append({
                "accession": acc,
                "reviewed": a.get("reviewed", True),
                "gene": a.get("gene", ""),
                "protein_name": a.get("protein_name", ""),
                "ec": a.get("ec", ""),
                "interpro": a.get("interpro", ""),
                "organism": a.get("organism", ""),
                "catalytic_activity": a.get("catalytic_activity", ""),
                "rhea_ids": ";".join(members[acc]["rhea_ids"]),
            })
        time.sleep(0.4)  # courtesy to the REST endpoint
    out = HERE / f"candidates_{tag}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["accession"])
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default="/tmp/current_run_accessions.txt",
                    help="file of prior-run candidate accessions, one per line")
    ap.add_argument("--annotate", choices=["none", "new", "all"], default="new")
    args = ap.parse_args()

    print("STAGE 1-2: detect ATP-dependent amide-ligation reactions")
    matched = stage_detect()
    print("STAGE 3: map to UniProt members")
    members = stage_map(matched)
    print("STAGE 4: report scale + overlap")
    stage_report(members, args.prior)
    if args.annotate != "none":
        stage_annotate(members, only_new=(args.annotate == "new"))


if __name__ == "__main__":
    main()
