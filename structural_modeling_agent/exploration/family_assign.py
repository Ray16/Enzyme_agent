#!/usr/bin/env python3
"""Family assignment for ESM Atlas hits -- what makes dark-matter candidates actionable.

An Atlas neighbor (from ``esm_atlas_expand.py``) is an unannotated metagenomic
sequence. It only becomes an *actionable* candidate once we can assign it a family,
because the reaction coordinate (mechanism), the developability priors and the proxy
calibration are all FAMILY-level properties that TRANSFER to a homolog of a known
family (see ``modeling_reasoning/knowledge/reaction_coordinates.json``, whose
coordinates are keyed by InterPro family). This module turns a hit into a set of
InterPro family accessions -- the same ``IPRxxxxxx`` keys the coordinates use -- with
provenance, via a pluggable set of assigners:

  * ``SequenceInterProAssigner`` -- EBI InterProScan5 REST on the hit sequence
    (always returned by the Foldseek server as ``tseq``). Returns InterPro
    accessions directly.
  * ``StructureFoldseekAssigner`` -- local Foldseek search of a hit STRUCTURE
    against the annotated AFDB/SwissProt DB, then UniProt -> InterPro on the best
    hit. Reaches into the twilight zone where sequence identity is too low for
    InterProScan to fire, so it is the fallback when a structure is available.

Network/compute I/O is kept separate from the pure result parsers so the parsers can
be unit-tested against fixtures without touching the network. EBI is a shared public
service and the local Foldseek DB sits on shared NFS: keep volume modest, poll
politely, and cap threads.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict

# --- EBI InterProScan5 REST -------------------------------------------------------
IPRSCAN_BASE = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"
# EBI requires an email to identify the job's owner; it is not used for anything else.
DEFAULT_EMAIL = os.environ.get("IPRSCAN_EMAIL", "rzhu16@uchicago.edu")

# --- local annotated Foldseek DB (structure route) --------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))  # .../Enzyme_agent
DEFAULT_FOLDSEEK_DB = os.path.join(_REPO, "foldseek_db", "swissprot", "afdb_swissprot")
# foldseek is env-local, not on PATH; these envs are known to carry it.
_FOLDSEEK_CANDIDATES = [
    os.path.expanduser(f"~/miniforge3/envs/{env}/bin/foldseek")
    for env in ("boltz-2", "datasail", "palm")
]
UNIPROT_ENTRY = "https://rest.uniprot.org/uniprotkb/{acc}.json"

# AFDB/SwissProt Foldseek targets look like "AF-<UniProt>-F1-model_v4".
_AFDB_ACC = re.compile(r"AF-([A-Z0-9]+)-F\d+")


@dataclass
class FamilyAssignment:
    """A family assignment for one hit, carrying its provenance."""
    target: str
    family_ids: tuple[str, ...] = ()
    method: str | None = None
    evidence: list[dict] = field(default_factory=list)

    @property
    def assigned(self) -> bool:
        return bool(self.family_ids)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["family_ids"] = list(self.family_ids)
        return d


# =================================================================================
# Pure parsers (no I/O -- unit-testable against fixtures)
# =================================================================================
def parse_interproscan_json(payload: dict) -> list[dict]:
    """Extract integrated InterPro entries from an InterProScan5 JSON result.

    Returns one record per distinct InterPro accession found across all matches,
    keeping the member signatures that support it as evidence. Matches whose
    signature has no integrated InterPro entry (``entry == null``) are skipped --
    only InterPro accessions key the reaction coordinates.
    """
    by_ipr: dict[str, dict] = {}
    for result in payload.get("results") or []:
        for match in result.get("matches") or []:
            sig = match.get("signature") or {}
            entry = sig.get("entry")
            if not entry or not entry.get("accession"):
                continue
            ipr = entry["accession"]
            rec = by_ipr.setdefault(ipr, {
                "accession": ipr,
                "name": entry.get("name"),
                "type": entry.get("type"),
                "signatures": [],
            })
            lib = (sig.get("signatureLibraryRelease") or {}).get("library")
            rec["signatures"].append({
                "accession": sig.get("accession"),
                "name": sig.get("name"),
                "library": lib,
            })
    return list(by_ipr.values())


def parse_uniprot_interpro(entry: dict) -> list[dict]:
    """Extract InterPro cross-references from a UniProtKB JSON entry."""
    out: list[dict] = []
    for xref in entry.get("uniProtKBCrossReferences") or []:
        if xref.get("database") != "InterPro":
            continue
        name = None
        for prop in xref.get("properties") or []:
            if prop.get("key") in ("EntryName", "entry name"):
                name = prop.get("value")
        out.append({"accession": xref.get("id"), "name": name})
    return out


def parse_foldseek_m8(text: str) -> list[dict]:
    """Parse a Foldseek tab-separated hit table (query,target,evalue,bits...)."""
    hits: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        cols = line.split("\t")
        if len(cols) < 4:
            continue
        try:
            evalue = float(cols[2])
            bits = float(cols[3])
        except ValueError:
            continue
        hits.append({"query": cols[0], "target": cols[1], "evalue": evalue, "bits": bits})
    return hits


def uniprot_from_foldseek_target(target: str) -> str | None:
    """Recover the UniProt accession from an AFDB/SwissProt Foldseek target id."""
    m = _AFDB_ACC.search(target or "")
    return m.group(1) if m else None


# =================================================================================
# I/O helpers
# =================================================================================
def _http_get(url: str, timeout: int = 60, accept: str | None = None) -> bytes:
    headers = {"Accept": accept} if accept else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _http_post_form(url: str, fields: dict, timeout: int = 60) -> bytes:
    data = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _resolve_foldseek(explicit: str | None = None) -> str | None:
    if explicit and os.path.exists(explicit):
        return explicit
    for cand in _FOLDSEEK_CANDIDATES:
        if os.path.exists(cand):
            return cand
    return None


# =================================================================================
# Assigners
# =================================================================================
class SequenceInterProAssigner:
    """Assign InterPro families from a hit's amino-acid sequence via EBI InterProScan5."""

    def __init__(self, email: str = DEFAULT_EMAIL, *, poll_s: float = 10.0,
                 timeout_s: float = 900.0, applications: tuple[str, ...] = ()):
        self.email = email
        self.poll_s = poll_s
        self.timeout_s = timeout_s
        self.applications = applications  # e.g. ("PfamA",) to restrict; () = InterProScan default

    def submit(self, sequence: str) -> str:
        fields = {"email": self.email, "sequence": sequence, "stype": "p"}
        if self.applications:
            fields["appl"] = ",".join(self.applications)
        job_id = _http_post_form(f"{IPRSCAN_BASE}/run", fields).decode().strip()
        if not job_id or " " in job_id:
            raise RuntimeError(f"unexpected InterProScan job id: {job_id!r}")
        return job_id

    def wait(self, job_id: str) -> str:
        start = time.time()
        while time.time() - start < self.timeout_s:
            status = _http_get(f"{IPRSCAN_BASE}/status/{job_id}").decode().strip()
            if status in ("FINISHED", "ERROR", "FAILURE", "NOT_FOUND"):
                return status
            time.sleep(self.poll_s)
        return "TIMEOUT"

    def result(self, job_id: str) -> dict:
        return json.loads(_http_get(f"{IPRSCAN_BASE}/result/{job_id}/json", accept="application/json"))

    def assign(self, hit: dict) -> FamilyAssignment | None:
        target = hit.get("target")
        seq = hit.get("tseq")
        if not seq:
            return None
        job_id = self.submit(seq)
        status = self.wait(job_id)
        if status != "FINISHED":
            return FamilyAssignment(target=target, method="interproscan",
                                    evidence=[{"job_id": job_id, "status": status}])
        entries = parse_interproscan_json(self.result(job_id))
        fams = [e["accession"] for e in entries if (e.get("type") in (None, "FAMILY", "DOMAIN"))]
        return FamilyAssignment(
            target=target,
            family_ids=tuple(dict.fromkeys(fams)),
            method="interproscan",
            evidence=[{"job_id": job_id, "entries": entries}],
        )


class StructureFoldseekAssigner:
    """Assign families from a hit STRUCTURE by Foldseek search vs annotated AFDB/SwissProt."""

    def __init__(self, db: str = DEFAULT_FOLDSEEK_DB, *, foldseek: str | None = None,
                 threads: int = 4, max_targets: int = 5, evalue: float = 1e-3):
        self.db = db
        self.foldseek = _resolve_foldseek(foldseek)
        self.threads = threads  # capped: shared node courtesy
        self.max_targets = max_targets
        self.evalue = evalue

    def search(self, structure_path: str, tmp_dir: str) -> list[dict]:
        if not self.foldseek:
            raise RuntimeError("foldseek binary not found (looked in boltz-2/datasail/palm envs)")
        os.makedirs(tmp_dir, exist_ok=True)
        out_m8 = os.path.join(tmp_dir, "hits.m8")
        cmd = [
            self.foldseek, "easy-search", structure_path, self.db, out_m8,
            os.path.join(tmp_dir, "fs_tmp"),
            "--format-output", "query,target,evalue,bits",
            "-e", str(self.evalue), "--max-seqs", str(self.max_targets),
            "--threads", str(self.threads),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        with open(out_m8) as fh:
            return parse_foldseek_m8(fh.read())

    def families_for_uniprot(self, acc: str) -> list[dict]:
        entry = json.loads(_http_get(UNIPROT_ENTRY.format(acc=acc), accept="application/json"))
        return parse_uniprot_interpro(entry)

    def assign(self, hit: dict, *, tmp_dir: str) -> FamilyAssignment | None:
        target = hit.get("target")
        structure_path = hit.get("structure_path")
        if not structure_path or not os.path.exists(structure_path):
            return None
        fs_hits = self.search(structure_path, tmp_dir)
        for fs in sorted(fs_hits, key=lambda h: -h["bits"]):
            acc = uniprot_from_foldseek_target(fs["target"])
            if not acc:
                continue
            entries = self.families_for_uniprot(acc)
            fams = [e["accession"] for e in entries if e.get("accession")]
            if fams:
                return FamilyAssignment(
                    target=target,
                    family_ids=tuple(dict.fromkeys(fams)),
                    method="foldseek_afdb_swissprot",
                    evidence=[{"foldseek_hit": fs, "uniprot": acc, "entries": entries}],
                )
        return FamilyAssignment(target=target, method="foldseek_afdb_swissprot",
                                evidence=[{"foldseek_hits": fs_hits}])


# =================================================================================
# Bridge into the reaction-coordinate transfer
# =================================================================================
def coordinate_for(assignment: FamilyAssignment, coordinates_path: str | None = None) -> dict:
    """Look up the reaction coordinate an assignment's families inherit.

    Uses the same family-first selection the modeling_reasoning compiler uses, so an
    Atlas hit assigned to a known family inherits that family's mechanism coordinate.
    Reads the coordinate JSON directly (no cross-package import) to stay decoupled.
    """
    path = coordinates_path or os.path.join(
        _REPO, "structural_modeling_agent", "modeling_reasoning",
        "knowledge", "reaction_coordinates.json")
    with open(path) as fh:
        mechanisms = (json.load(fh).get("mechanisms")) or {}
    fam = set(assignment.family_ids or ())
    for key, mech in mechanisms.items():
        coord_fams = set(((mech.get("keys") or {}).get("families")) or [])
        if fam & coord_fams:
            return {
                "coordinate_key": key,
                "basis": "family",
                "matched_families": sorted(fam & coord_fams),
                "n_steps": len(mech.get("steps") or []),
            }
    return {"coordinate_key": None, "basis": None, "matched_families": []}


def annotate_hits(hits: list[dict], assigner: SequenceInterProAssigner,
                  coordinates_path: str | None = None) -> list[dict]:
    """Assign a family + reaction coordinate to each Atlas hit; return enriched records."""
    out: list[dict] = []
    for hit in hits:
        assignment = assigner.assign(hit)
        record = dict(hit)
        if assignment is not None:
            record["family_assignment"] = assignment.to_dict()
            record["reaction_coordinate"] = coordinate_for(assignment, coordinates_path)
        out.append(record)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--neighbors", required=True, help="atlas_neighbors.json from esm_atlas_expand")
    ap.add_argument("--out", help="write family-annotated neighbors JSON here")
    ap.add_argument("--email", default=DEFAULT_EMAIL, help="EBI job-owner email")
    ap.add_argument("--limit", type=int, default=0, help="assign only the first N hits (0 = all)")
    ap.add_argument("--coordinates", help="path to reaction_coordinates.json")
    args = ap.parse_args(argv)

    hits = json.load(open(args.neighbors))
    if args.limit:
        hits = hits[: args.limit]
    assigner = SequenceInterProAssigner(email=args.email)
    annotated = annotate_hits(hits, assigner, args.coordinates)

    n_fam = sum(1 for h in annotated if h.get("family_assignment", {}).get("family_ids"))
    n_coord = sum(1 for h in annotated if h.get("reaction_coordinate", {}).get("coordinate_key"))
    print(f"assigned families to {n_fam}/{len(annotated)} hits; "
          f"{n_coord} inherited a reaction coordinate")
    for h in annotated:
        fa = h.get("family_assignment", {})
        rc = h.get("reaction_coordinate", {})
        fams = ",".join(fa.get("family_ids", [])) or "-"
        print(f"  {h.get('target'):28} families={fams:40} coord={rc.get('coordinate_key')}")
    if args.out:
        json.dump(annotated, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
