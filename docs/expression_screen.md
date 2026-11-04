# E. coli expression sequence screen

`enzyme_design_agent.expression_screen` is a reusable CPU-only screen for arbitrary
canonical protein sequences. It retrieves native coding DNA through UniProt EMBL
cross-references, NCBI protein `coded_by` coordinates and bounded nucleotide ranges.
Alternatively, supply your own CDS. Every CDS must translate to the exact input
protein before any codon metrics are calculated. DNA is never inferred by reverse
translating a protein.

The default host is **E. coli BL21(DE3), CP001509.3**. The reference for codon
adaptation index (CAI) consists of its annotated 30S/50S structural ribosomal-protein
CDSs. The reference genes, codon weights, source record hash and software version
are recorded with each run. Use `--reference-embl` to supply another host genome.

```bash
python -m enzyme_design_agent.expression_screen \
  --uniprot A0A2S5ZW91 A0A0U1DKV8 \
  --cache /path/to/expression-cache \
  --out /path/to/expression-results
```

For protein FASTA, use `--protein-fasta proteins.fasta`. Add
`--cds-fasta native_or_designed_cds.fasta` when DNA is available locally; identifiers
must match. Swiss-Prot/TrEMBL FASTA headers such as `sp|P12345|...` are recognized.
Arbitrary FASTA identifiers are accepted, but generally need supplied CDS to obtain
DNA-level metrics. A missing DNA link does not prevent protein descriptors.

For batches with metadata, use `--input-json records.json`:

```json
[
  {"id": "example", "sequence": "MA", "cds": "ATGGCCTAA", "arms": ["example_group"]}
]
```

Optional fields are `organism`, `arms`, `cds` and `translation_table` (default 11).
`--input-json`, `--protein-fasta` and `--uniprot` are mutually exclusive. An optional
`--uniprot-json` accepts a cached UniProt stream response with a `results` list;
`--protein-genbank` accepts cached GenPept records. `--offline` disallows HTTP and
uses cached records or supplied data. Network retrieval is cached and NCBI calls
are globally limited to two per second; `--workers` is capped at four.

Outputs include:

- `candidate_expression_features.csv` and `records.json`: per-protein metrics.
- `native_CDS.fasta` and `CDS_provenance.json`: verified DNA, linked protein,
  native coordinates, URLs, hashes and translation checks.
- `summary.json`: group summaries and the full host reference definition.
- `report.html`: readable results with interpretation limits.

DNA metrics are CAI, coding GC, GC3, codon counts and positions with relative
adaptiveness below 0.1 in the selected reference, the count in the first 30 codons,
and the maximum count in a sliding 10-codon window. This definition is a reference
codon-usage measure, not a direct measurement of tRNA availability. There is no
universal CAI pass/fail threshold. Missing or mismatched DNA has null metrics and
an explicit error, never an implied pass.

Protein descriptors include length, molecular weight, pI, GRAVY, cysteine count and
maximum 19-residue Kyte–Doolittle hydropathy. Hydropathy windows are contextual
descriptors; they are not membrane or solubility predictions.

An installed **NetSolP-1.0** predictor can add protein-level E. coli solubility and
usability scores in one batch:

```bash
python -m enzyme_design_agent.expression_screen \
  --protein-fasta proteins.fasta --cds-fasta coding_sequences.fasta \
  --cache /path/to/cache --out /path/to/results \
  --netsolp-dir /path/to/NetSolP-1.0/PredictionServer \
  --netsolp-python /path/to/python-with-torch-and-onnxruntime
```

This runs the authors' quantized distilled ONNX models on CPU with two threads.
It does not reserve or use a GPU. Predictions require exact input/output sequence
and ID matches. Inputs above 1022 residues are declined to prevent the upstream
tool's silent truncation. Missing models or failed predictions remain unassessed.
Model and script hashes are saved in `netsolp_provenance.json`.

Generate reusable comparison plots after a run (requires Matplotlib):

```bash
python -m enzyme_design_agent.expression_plots \
  --results-dir /path/to/results \
  --reference-id AS9_control --highlight A0A2S5ZW91 A0A0U1DKV8
```

This produces `codon_comparison` and `codon_preferences` in PNG, SVG and PDF.
The first compares groups defined by each record's `arms`; the second shows the
host reference and selected sequences. Reference and highlight arguments are
optional. Plain FASTA inputs without group metadata form one candidate group.

CAI and protein predictor scores are separate endpoints. Neither is a guarantee
of expression, soluble yield or catalytic activity. Native DNA results do not
describe a future codon-optimized construct. Vector, promoter, RBS, tags, 5′ UTR,
induction conditions and mRNA initiation accessibility are not specified here.

References: [BL21(DE3) genome](https://www.ncbi.nlm.nih.gov/nuccore/CP001509.3),
[Biopython CAI implementation](https://biopython.org/docs/latest/api/Bio.SeqUtils.html#Bio.SeqUtils.CodonAdaptationIndex),
[NetSolP](https://doi.org/10.1093/bioinformatics/btab801).
