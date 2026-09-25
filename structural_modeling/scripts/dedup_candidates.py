#!/usr/bin/env python3
"""Dedup a discovery-agent candidate list to one representative per ortholog cluster.

Pipeline: fetch UniProt sequences -> mmseqs cluster at a chosen identity/coverage ->
pick best-ranked member per cluster -> write representatives + an auditable cluster report.

Usage:
  python dedup_candidates.py --candidates ranked_candidates.csv --out <dir> \
      [--min-seq-id 0.9] [--cov 0.8] [--acc-col accession] [--rank-col rank]

The identity threshold is the ONE knob for "what counts as the same enzyme":
  0.90  (default) collapse only near-identical orthologs; keep distinct paralogs.
  0.70  looser: also fuses divergent orthologs/close paralogs (fewer cofolds, more risk of merging distinct enzymes).
  0.95+ stricter: collapse only near-duplicates.
Always eyeball the printed cluster report (families/EC should be uniform per cluster).
"""
import argparse, csv, collections, hashlib, os, re, subprocess, sys, urllib.request

UNIPROT = "https://rest.uniprot.org/uniprotkb/accessions?accessions={acc}&format=fasta"


def fetch_sequences(accs, out_fa, chunk=40):
    """Fetch FASTA for accessions from UniProt in chunks; reheader to bare accession."""
    raw = []
    for i in range(0, len(accs), chunk):
        url = UNIPROT.format(acc=",".join(accs[i:i + chunk]))
        raw.append(urllib.request.urlopen(url).read().decode())
    recs = {}; acc = None; buf = []
    for line in "".join(raw).splitlines():
        if line.startswith(">"):
            if acc: recs[acc] = "".join(buf)
            m = re.match(r">(?:sp|tr)\|([^|]+)\|", line) or re.match(r">(\S+)", line)
            acc = m.group(1); buf = []
        else:
            buf.append(line.strip())
    if acc: recs[acc] = "".join(buf)
    with open(out_fa, "w") as f:
        for a, s in recs.items():
            f.write(f">{a}\n{s}\n")
    return recs


def cluster(fasta, workdir, min_seq_id, cov):
    os.makedirs(workdir, exist_ok=True)
    pref = os.path.join(workdir, "clu")
    tmp = os.path.join(workdir, "tmp")
    subprocess.run(["mmseqs", "easy-cluster", fasta, pref, tmp,
                    "--min-seq-id", str(min_seq_id), "-c", str(cov), "--cov-mode", "0"],
                   check=True, stdout=open(os.path.join(workdir, "mmseqs.log"), "w"),
                   stderr=subprocess.STDOUT)
    clu = collections.defaultdict(list)
    for line in open(pref + "_cluster.tsv"):
        rep, mem = line.split()
        clu[rep].append(mem)
    return clu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="CSV with accession + rank columns")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--min-seq-id", type=float, default=0.90, help="ortholog identity threshold (default 0.90)")
    ap.add_argument("--cov", type=float, default=0.80, help="min alignment coverage (default 0.80)")
    ap.add_argument("--acc-col", default="accession")
    ap.add_argument("--rank-col", default="rank")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rows = list(csv.DictReader(open(a.candidates)))
    meta = {r[a.acc_col]: r for r in rows}
    accs = [r[a.acc_col] for r in rows]
    print(f"{len(accs)} candidates, {len(set(accs))} unique accessions")

    fa = os.path.join(a.out, "candidates.fasta")
    seq = fetch_sequences(accs, fa)
    print(f"fetched {len(seq)} sequences (len {min(map(len,seq.values()))}-{max(map(len,seq.values()))})")

    clu = cluster(fa, os.path.join(a.out, "mmseqs_work"), a.min_seq_id, a.cov)
    print(f"clustered {len(seq)} -> {len(clu)} representatives at id>={a.min_seq_id} cov>={a.cov}")

    def rank(m):
        try: return int(meta[m][a.rank_col])
        except Exception: return 10**9
    reps = []
    for rep, mems in clu.items():
        best = min(mems, key=lambda m: (rank(m), len(seq[m])))   # best (lowest) rank; tie -> shorter
        reps.append((best, [m for m in mems if m != best]))
    reps.sort(key=lambda x: rank(x[0]))

    # write representatives.fasta + representatives.csv (carry all metadata + collapsed list)
    with open(os.path.join(a.out, "representatives.fasta"), "w") as f:
        for r, _ in reps:
            f.write(f">{r}\n{seq[r]}\n")
    cols = list(rows[0].keys()) + ["collapsed_orthologs"]
    with open(os.path.join(a.out, "representatives.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r, others in reps:
            row = dict(meta[r]); row["collapsed_orthologs"] = ";".join(others); w.writerow(row)

    # auditable cluster report: flag any cluster mixing families/EC (a possible false merge)
    print("\n=== multi-member clusters (verify these are true orthologs) ===")
    warn = 0
    for r, others in reps:
        if not others: continue
        mems = [r] + others
        fams = {meta[m].get("interpro_family", "") for m in mems}
        ecs = {meta[m].get("ec_numbers", "") for m in mems}
        flag = "  !! MIXED FAMILY/EC" if (len(fams) > 1 or len(ecs) > 1) else ""
        if flag: warn += 1
        print(f"  {r} ({meta[r].get('gene','?')}) n={len(mems)} fams={len(fams)} ec={len(ecs)}{flag}")
        for m in others:
            print(f"      -> {m} ({meta[m].get('gene','?')}, {meta[m].get('interpro_family','')[:30]})")
    print(f"\n{len(reps)} representatives written to {a.out}/representatives.{{fasta,csv}}")
    if warn:
        print(f"WARNING: {warn} cluster(s) mixed family/EC — lower --min-seq-id may be merging distinct enzymes.", file=sys.stderr)


if __name__ == "__main__":
    main()
