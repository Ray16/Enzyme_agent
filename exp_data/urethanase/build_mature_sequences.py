#!/usr/bin/env python3
"""Strip expression tags from the panel's published sequences, with validation.

Replaces the undocumented trim that produced the original `panel_mature.fasta`,
which left TEV-site remnants on 10 sequences (`ENLYF`) and over-trimmed 2 native
residues from 3 others. Both defects were invisible because no script recorded
what it had removed or checked the result against anything.

Rules, applied only where the motif is unambiguous:

  C-terminus
    - TEV site `ENLYFQ` found in the last 40 residues -> native ends before it.
      Uses the FIRST occurrence in that window: BLPUR4 carries the site twice
      (`QIENLYFQG|ENLYFQGLEHHHHHH`) and the native protein ends before both.
    - otherwise a trailing His run (>=6 H, optionally after an `LE` linker)
      -> native ends before the linker.

  N-terminus
    - TEV site `ENLYFQ` in the first 60 residues -> cleavage leaves everything
      after Q. Any residual cloning linker is NOT guessed away; it is resolved
      against a crystal if one exists, otherwise kept and flagged.

Every emitted sequence is checked against the highest-identity PDB entry for it.
A sequence that disagrees with its own crystal is reported as FAIL and is not
written, because a cofold that ran on the wrong sequence is indistinguishable
downstream from one that ran on the right sequence.

Usage:
  build_mature_sequences.py --published panel/panel_as_published.fasta \
      --out panel/panel_mature.fasta --report panel/tag_stripping_report.md
  # add --crystals panel/crystal_refs.json to enable validation
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

TEV = "ENLYFQ"
HIS_TAIL = re.compile(r"(?:LE)?H{6,}$")


def read_fasta(path: Path) -> dict[str, str]:
    seqs, name, order = {}, None, []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            name = line[1:]
            seqs[name] = ""
            order.append(name)
        elif name:
            seqs[name] += line
    return {k: seqs[k] for k in order}


def strip_cterm(seq: str) -> tuple[str, str]:
    """-> (sequence, what_was_removed_and_why)"""
    window = seq[-40:]
    i = window.find(TEV)
    if i >= 0:
        cut = len(seq) - 40 + i
        n = window.count(TEV)
        why = f"C-term TEV site{' (x%d, cut before the first)' % n if n > 1 else ''}"
        return seq[:cut], f"{why}: -{len(seq) - cut} `{seq[cut:]}`"
    m = HIS_TAIL.search(seq)
    if m:
        return seq[: m.start()], f"C-term His tag: -{len(seq) - m.start()} `{m.group()}`"
    return seq, ""


def strip_nterm(seq: str) -> tuple[str, str]:
    i = seq[:60].find(TEV)
    if i >= 0:
        cut = i + len(TEV)
        return seq[cut:], f"N-term TEV site: -{cut} `{seq[:cut]}`"
    return seq, ""


def pdb_seq(pdb_id: str) -> str | None:
    url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
    try:
        d = json.load(urllib.request.urlopen(url, timeout=30))
    except Exception:
        return None
    return d["entity_poly"]["pdbx_seq_one_letter_code_can"].replace("\n", "")


def validate(name: str, seq: str, ref: dict) -> tuple[str, str, str]:
    """-> (verdict, detail, corrected_sequence).

    A reference is only authoritative at identity 1.0. Below that, point
    substitutions are expected and a mismatch says nothing about tags, so the
    result is UNVALIDATED rather than FAIL — calling it FAIL would reject a
    correct sequence for having a merely-close crystal.

    At identity 1.0 the crystal IS the evidence for where the native protein
    starts and ends, so leftover cloning linker that no motif rule can resolve
    (e.g. CCPUR1's `GAMG`) is trimmed here and recorded.
    """
    pdb_id, ident = ref["pdb_id"], ref["identity"]
    cs = pdb_seq(pdb_id)
    if cs is None:
        return "SKIP", f"{pdb_id}: could not fetch", seq
    if seq in cs:
        off = cs.find(seq)
        extra = len(cs) - len(seq) - off
        return "PASS", (
            f"{pdb_id}: exact substring, crystal has {off} tag residue(s) before "
            f"and {extra} after"
        ), seq
    if ident < 1.0:
        return "UNVALIDATED", (
            f"{pdb_id}: closest crystal is only {ident:.1%} identical, so a "
            f"mismatch is point substitutions, not tags — cannot validate tags here"
        ), seq
    for trim in range(1, 13):
        if seq[trim:] in cs:
            return "CORRECTED", (
                f"{pdb_id}: crystal starts {trim} residue(s) later; trimmed "
                f"leftover linker `{seq[:trim]}` on crystal evidence"
            ), seq[trim:]
        if seq[:-trim] in cs:
            return "CORRECTED", (
                f"{pdb_id}: crystal ends {trim} residue(s) earlier; trimmed "
                f"`{seq[-trim:]}` on crystal evidence"
            ), seq[:-trim]
    if cs.find(seq[20:60]) < 0:
        return "FAIL", f"{pdb_id}: no shared 40-mer; wrong reference or wrong protein", seq
    return "FAIL", f"{pdb_id}: diverges internally at 100% claimed identity", seq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--published", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--crystals", help="JSON {name: pdb_id} for validation")
    ap.add_argument("--allow-fail", action="store_true",
                    help="write sequences that fail crystal validation (NOT default)")
    args = ap.parse_args()

    pub = read_fasta(Path(args.published))
    crystals = json.loads(Path(args.crystals).read_text()) if args.crystals else {}

    rows, out_seqs, failures = [], {}, []
    for header, seq in pub.items():
        name = header.split("|")[0]
        s, c_why = strip_cterm(seq)
        s, n_why = strip_nterm(s)
        removed = "; ".join(x for x in (n_why, c_why) if x) or "nothing removed"

        verdict, detail = "NO REF", "no crystal reference supplied"
        if name in crystals:
            verdict, detail, s2 = validate(name, s, crystals[name])
            if s2 != s:
                removed += f"; {detail.split(': ', 1)[1]}"
                s = s2

        rows.append([name, len(seq), len(s), removed, verdict, detail])
        if verdict == "FAIL" and not args.allow_fail:
            failures.append(name)
        else:
            out_seqs[header] = s

    # A member with no crystal may still have its N-terminal start settled by a
    # close panel homolog whose crystal DID settle it (CCPUR2 <- CCPUR1 at 89%).
    # This is an inference, not a measurement, so it is labelled as one.
    anchored = {
        r[0]: out_seqs[h]
        for h, r in zip(list(out_seqs), rows)
        if r[4] in ("PASS", "CORRECTED")
    }
    for header in list(out_seqs):
        name = header.split("|")[0]
        row = next(r for r in rows if r[0] == name)
        if row[4] != "NO REF":
            continue
        s = out_seqs[header]
        for aname, aseq in anchored.items():
            # same start motif, our sequence carries extra residues before it
            k = s.find(aseq[:12])
            if 0 < k <= 8:
                ident = sum(a == b for a, b in zip(s[k:], aseq)) / min(len(s) - k, len(aseq))
                if ident >= 0.80:
                    out_seqs[header] = s[k:]
                    row[2] = len(s) - k
                    row[3] += (f"; N-term start inferred from {aname} "
                               f"({ident:.0%} identical, crystal-anchored): "
                               f"trimmed `{s[:k]}`")
                    row[4] = "INFERRED"
                    row[5] = (f"N-term taken from homolog {aname}, whose start is "
                              f"crystal-confirmed. NOT independently validated.")
                    break

    Path(args.out).write_text(
        "".join(f">{h}\n{s}\n" for h, s in out_seqs.items())
    )

    lines = [
        "# Tag stripping report",
        "",
        f"Source: `{args.published}`  ->  `{args.out}`",
        "",
        "| name | published | mature | removed | crystal check |",
        "|---|---|---|---|---|",
    ]
    for name, lp, lm, removed, verdict, detail in rows:
        lines.append(f"| {name} | {lp} | {lm} | {removed} | **{verdict}** — {detail} |")
    if failures:
        lines += [
            "",
            "## Not written (failed crystal validation)",
            "",
            *(f"- `{f}`" for f in failures),
            "",
            "Re-run with `--allow-fail` only after deciding the crystal is the "
            "wrong reference, and record why.",
        ]
    Path(args.report).write_text("\n".join(lines) + "\n")

    print(f"wrote {len(out_seqs)} sequence(s) to {args.out}")
    print(f"report: {args.report}")
    for name, _, _, removed, verdict, detail in rows:
        if verdict in ("FAIL", "SKIP"):
            print(f"  {verdict} {name}: {detail}")
    if failures:
        print(f"{len(failures)} sequence(s) NOT written; see the report.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
