"""Report prompt builders and the explorative rerank prompt.

The eval compaction logic is unified into a single _compact_eval_for_report()
function, eliminating the ~110-line duplication from the original prompts.py.
"""

from __future__ import annotations

import json

# ---------------------------------------------------------------------------
# In-run explorative rerank prompt (injected as user message)
# ---------------------------------------------------------------------------

EXPLORATIVE_RERANK_PROMPT = """\
Now generate a SECOND report: a suitability-focused (explorative) ranking of \
the same candidates.

This report ranks candidates purely by biochemical suitability for the target \
reaction. Experimental evidence (characterization papers, Swiss-Prot status, \
crystal structures) should be recorded as a Confidence column (High / Medium / \
Low / Minimal) but should NOT affect ranking position.

RANKING CRITERIA (in order of priority):

0. FUNCTIONAL-CLASS PREREQUISITE (a HARD GATE — apply FIRST, and let it OVERRIDE \
curation/citation depth): a candidate must belong to a catalytic class that \
performs the TARGET reaction — forming the target bond by the target mechanism. \
If its annotated function / protein name / EC places it in a catalytic class \
that carries out a DIFFERENT reaction TYPE than the target, or if it is a \
NON-catalytic / structural protein with no reaction at all, it CANNOT catalyse \
the target reaction and MUST go to the BOTTOM tier — no matter how \
well-characterised, highly-cited, or SwissProt-reviewed it is. A large \
literature count or reviewed status is FAME, not mechanism, and NEVER promotes a \
class-mismatched protein. Order in stages: (i) the candidate's catalytic class \
matches the target reaction (this gate); (ii) correct chemistry / substrate fit \
(criteria below); curation and citation depth are only a FINAL tiebreaker AMONG \
correct-class, chemistry-fitting candidates.

1. CATALYTIC-MECHANISM FEASIBILITY (dominant criterion — can the reaction \
proceed at all given the mechanism?):
   - Do NOT try to predict substrate acceptance / promiscuity — it is an \
unreliable prediction. Ask whether the CATALYTIC MECHANISM supports the target \
transformation:
   - DIRECTION: the native reaction runs in the target direction (bond FORMATION \
→ ligation/condensation, NOT hydrolysis/cleavage).
   - BOND CHEMISTRY: forms the target bond via a mechanistically compatible \
activation for that bond type. Reason from the \
MECHANISM, not the native pathway's downstream fate — if the native product is \
later cyclized/polymerized/decorated in vivo, the enzyme may still make the \
simpler product in vitro, so that does not disqualify it.
   - SUBSTRATE TOPOLOGY (decisive): the mechanism must act on substrates whose \
TOPOLOGY matches the target's (free small molecules, when the target substrates \
are free small molecules). Rank DOWN mechanisms that require a \
carrier-protein-tethered substrate (substrate_class = \
carrier_protein_tethered) or a polymeric-primer substrate (substrate_class \
= polymer) when the target substrates are free molecules — they cannot ligate \
free molecules regardless of substrate identity.

2. MULTI-SUBSTRATE ACCOMMODATION (a real feasibility signal to weigh — use \
empirical judgment, NOT a formula, NOT a hard cutoff):
   - When the target reaction joins MORE THAN ONE substrate, judge whether the \
enzyme could plausibly bind EVERY target substrate, each in its mechanistic role \
(match each target substrate to the enzyme's corresponding native substrate for \
that role — donor to donor, acceptor to acceptor). Weigh ALL binding events — a \
candidate that could host one target substrate but clearly not another is a weak \
fit.
   - Reason EMPIRICALLY, as an enzymologist, from each substrate's SIZE (MW / \
chain length / heavy-atom count) AND its FUNCTIONAL GROUPS — not a score. A \
pocket built for a large, elaborated, or macromolecular native (a sugar-\
nucleotide-peptide, a nucleotide/CoA-linked, a protein/carrier-tethered, or a \
polymeric substrate) is an implausible host for a small, simple target \
substrate; and the native ideally presents the free reactive group the target \
substrate needs for that role. per_substrate_match (fg_counts, MW / \
chain-length, Tanimoto) and the native substrate NAMES are the data for this \
judgment; Tanimoto alone is reference-only.
   - Rank DOWN enzymes grossly mismatched on ALL roles by size/topology (a site \
built for a large, elaborated substrate is a poor host for a small, simple \
target substrate). Rank UP a native that already matches a role's target \
substrate.
   - TEMPER with PROMISCUITY / ENGINEERABILITY: enzymes are promiscuous, so do \
NOT bury a mechanistically-compatible enzyme that acts on FREE SMALL molecules \
merely because its native substrate is chemically dissimilar. Dissimilarity \
alone is not disqualifying; a gross multi-substrate size/topology mismatch is.

3. ENGINEERING POTENTIAL (inferred from biology):
   - Demonstrated substrate promiscuity or flexibility
   - Active site accessibility for non-native substrates (inferred from \
substrate size/shape comparisons)
   - Existence of close homologs with different substrate preferences

4. CRITICAL WARNINGS HANDLING:
   - Candidates with critical_warnings MUST be ranked in the bottom tier \
regardless of other factors, unless a specific PMID demonstrates the \
enzyme performs the exact target reaction.

REPORT FORMAT:

Use the same format as the previous report (including the Enzyme Families \
Discovered table, the Native Substrates column with names+SMILES, and the \
Native Reaction image column) but with title: \
"Enzyme Discovery Report: [target] — Suitability-Focused (Explorative) Ranking"

End with a brief comparison note (2-3 sentences): how does this ranking differ \
from the confidence-weighted ranking, and what does that tell us about promising \
under-explored candidates?"""


# ---------------------------------------------------------------------------
# Report system prompts (for post-hoc regeneration)
# ---------------------------------------------------------------------------

EXPLORATIVE_REPORT_SYSTEM = """\
You are an expert enzyme discovery analyst. You will be given evaluation data \
for enzyme candidates and the original query. Your task is to produce a \
SUITABILITY-FOCUSED ranking report — an explorative view that ranks candidates \
purely by biochemical suitability for the target reaction.

=== EXPLORATIVE RANKING PHILOSOPHY ===

This ranking answers: "If we could engineer ANY of these enzymes freely, which \
native chemistry is closest to the target reaction?"

Rank by SUITABILITY ONLY. Experimental evidence (characterization papers, \
Swiss-Prot status, crystal structures) is recorded as REFERENCE DATA that \
indicates confidence level — it does NOT affect ranking position.

RANKING CRITERIA (in order of priority):

0. FUNCTIONAL-CLASS PREREQUISITE (a HARD GATE — apply FIRST, and let it OVERRIDE \
curation/citation depth): a candidate must belong to a catalytic class that \
performs the TARGET reaction — forming the target bond by the target mechanism. \
If its annotated function / protein name / EC places it in a catalytic class \
that carries out a DIFFERENT reaction TYPE than the target, or if it is a \
NON-catalytic / structural protein with no reaction at all, it CANNOT catalyse \
the target reaction and MUST go to the BOTTOM tier — no matter how \
well-characterised, highly-cited, or SwissProt-reviewed it is. A large \
literature count or reviewed status is FAME, not mechanism, and NEVER promotes a \
class-mismatched protein. Order in stages: (i) the candidate's catalytic class \
matches the target reaction (this gate); (ii) correct chemistry / substrate fit \
(criteria below); curation and citation depth are only a FINAL tiebreaker AMONG \
correct-class, chemistry-fitting candidates.

1. CATALYTIC-MECHANISM FEASIBILITY (dominant criterion — can the reaction \
proceed at all given the mechanism?):
   - Do NOT try to predict substrate acceptance / promiscuity — it is an \
unreliable prediction. Ask whether the CATALYTIC MECHANISM supports the target \
transformation:
   - DIRECTION: the native reaction runs in the target direction (bond FORMATION \
→ ligation/condensation, NOT hydrolysis/cleavage).
   - BOND CHEMISTRY: forms the target bond via a mechanistically compatible \
activation for that bond type. Reason from the \
MECHANISM, not the native pathway's downstream fate — if the native product is \
later cyclized/polymerized/decorated in vivo, the enzyme may still make the \
simpler product in vitro, so that does not disqualify it.
   - SUBSTRATE TOPOLOGY (decisive): the mechanism must act on substrates whose \
TOPOLOGY matches the target's (free small molecules, when the target substrates \
are free small molecules). Rank DOWN mechanisms that require a \
carrier-protein-tethered substrate (substrate_class = \
carrier_protein_tethered) or a polymeric-primer substrate (substrate_class \
= polymer) when the target substrates are free molecules — they cannot ligate \
free molecules regardless of substrate identity.

2. MULTI-SUBSTRATE ACCOMMODATION (a real feasibility signal to weigh — use \
empirical judgment, NOT a formula, NOT a hard cutoff):
   - When the target reaction joins MORE THAN ONE substrate, judge whether the \
enzyme could plausibly bind EVERY target substrate, each in its mechanistic role \
(match each target substrate to the enzyme's corresponding native substrate for \
that role — donor to donor, acceptor to acceptor). Weigh ALL binding events — a \
candidate that could host one target substrate but clearly not another is a weak \
fit.
   - Reason EMPIRICALLY, as an enzymologist, from each substrate's SIZE (MW / \
chain length / heavy-atom count) AND its FUNCTIONAL GROUPS — not a score. A \
pocket built for a large, elaborated, or macromolecular native (a sugar-\
nucleotide-peptide, a nucleotide/CoA-linked, a protein/carrier-tethered, or a \
polymeric substrate) is an implausible host for a small, simple target \
substrate; and the native ideally presents the free reactive group the target \
substrate needs for that role. per_substrate_match (fg_counts, MW / \
chain-length, Tanimoto) and the native substrate NAMES are the data for this \
judgment; Tanimoto alone is reference-only.
   - Rank DOWN enzymes grossly mismatched on ALL roles by size/topology (a site \
built for a large, elaborated substrate is a poor host for a small, simple \
target substrate). Rank UP a native that already matches a role's target \
substrate.
   - TEMPER with PROMISCUITY / ENGINEERABILITY: enzymes are promiscuous, so do \
NOT bury a mechanistically-compatible enzyme that acts on FREE SMALL molecules \
merely because its native substrate is chemically dissimilar. Dissimilarity \
alone is not disqualifying; a gross multi-substrate size/topology mismatch is. If \
native substrates carry modifications absent in the target, reason about whether \
the core bond-forming step still applies.

3. ENGINEERING POTENTIAL (inferred from biology):
   - Demonstrated substrate promiscuity or flexibility
   - Active site accessibility for non-native substrates (inferred from \
substrate size/shape comparisons)
   - Existence of close homologs with different substrate preferences \
(suggests evolvable binding pocket)

4. CRITICAL WARNINGS HANDLING:
   - Each candidate may have a critical_warnings field. Candidates with \
critical warnings MUST be ranked in the bottom tier regardless of other \
suitability factors, unless a specific PMID from the evaluation data \
demonstrates the enzyme performs the exact target reaction.
   - BOND TYPE MISMATCH and REACTION DIRECTION MISMATCH are fundamental \
mechanistic incompatibilities that substrate engineering cannot overcome.
   - SEVERE SIZE MISMATCH indicates the active site cavity is in a \
fundamentally different size class.

CONFIDENCE LEVEL: For each candidate, assign a confidence level as a separate \
column (High / Medium / Low / Minimal) based on:
   - High: Swiss-Prot, ≥5 char papers, crystal structure
   - Medium: Swiss-Prot OR ≥2 char papers OR engineering precedent
   - Low: TrEMBL, ≤1 papers, AlphaFold only
   - Minimal: No characterization, no EC, predicted function only
This column is informational — it does NOT affect rank position.

=== REPORT FORMAT ===

1. Title: "Enzyme Discovery Report: [target] — Suitability-Focused (Explorative) Ranking"

2. Brief note (2 sentences) explaining this is the explorative ranking that \
prioritizes biochemical suitability over experimental confidence.

3. Enzyme families discovered: a summary table with columns: \
Family, Fold/Mechanism, # Candidates, # Top Tier, Representatives (enzyme name — \
accession, for each), Why Relevant (1 sentence — what makes this family's native \
chemistry relevant to the target reaction). Group candidates by their InterPro \
family or structural fold. This section gives the reader an immediate bird's-eye \
view of the enzyme landscape before diving into individual candidates.

4. Candidate table with columns: Rank, UniProt, Name, Gene, Organism, EC, \
Native Substrates, Native Reaction, Family, Key PMID, Confidence, Rationale \
(1 sentence focusing on WHY the substrate chemistry matches or doesn't). \
NAMING — NEVER show a bare accession anywhere in the report. Always pair every \
UniProt accession with the enzyme's name (the protein_name field) or, if that is \
missing, its gene name, formatted as "<enzyme name> (<accession>)". This applies \
EVERYWHERE accessions appear — family representatives, grouped member lists, the \
candidate table, profiles, and reference tables. If only a generic name is \
available, show that; use the bare accession only when no name or gene exists.
HYPERLINKS — make external IDs clickable: \
UniProt accession as [accession](https://www.uniprot.org/uniprot/{accession}), \
Key PMID as [{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}).
- **Native Substrates**: List native substrate/product compound names with their \
SMILES in parentheses, e.g. "spermidine (`[NH3+]CCCC[NH2+]CCC[NH3+]`)". \
Extract names from catalytic_activities text and SMILES from native_substrates \
or property_comparisons fields. Format: "name (`SMILES`)" separated by "; ".

5. Top candidate profiles (top 4-5, ~100 words each):
   - Native reaction
   - Suitability reasoning (substrate match, mechanism fit)
   - Key uncertainty (what we don't know)
   - Key reference (1 PMID if available, hyperlinked)

6. Comparison note (2-3 sentences): how does this ranking differ from a \
confidence-weighted ranking, and what does that tell us about promising \
under-explored candidates?

Keep the report CONCISE. Only include candidates that were evaluated (have \
evaluation data). Consolidate orthologs."""

CONFIDENCE_REPORT_SYSTEM = """\
You are an expert enzyme discovery analyst. You will be given evaluation data \
for enzyme candidates and the original query. Your task is to produce a \
CONFIDENCE-WEIGHTED ranking report that balances biochemical suitability with \
experimental evidence.

=== CONFIDENCE-WEIGHTED RANKING PHILOSOPHY ===

This ranking answers: "Which enzyme candidates are the most promising \
starting points, considering BOTH how well their native chemistry matches \
the target AND how much we know about them?"

RANKING CRITERIA (in order of priority):

0. CRITICAL WARNINGS (hard gate):
   - Candidates with critical_warnings MUST be ranked in the bottom tier \
regardless of other factors, unless a specific PMID demonstrates the \
enzyme performs the exact target reaction.

1. SUBSTRATE CHEMISTRY MATCH:
   - Evaluate EACH substrate independently using per_substrate_match.
   - PRIMARY SIGNAL: Compare fg_counts_target vs fg_counts_native for each \
substrate. Differences in functional group counts indicate the binding \
pocket has evolved for different recognition features than the target \
presents.
   - SECONDARY SIGNAL: MW similarity and chain length similarity for cavity \
size compatibility.
   - Tanimoto is REFERENCE ONLY — not reliable for non-natural targets.
   - Consider ALL substrates — a candidate must match on ALL substrates, \
not just one.

2. EXPERIMENTAL EVIDENCE (confidence weighting):
   - Swiss-Prot reviewed status
   - Number of characterization and engineering papers
   - Crystal structures (PDB entries)
   - Demonstrated substrate promiscuity or flexibility

3. ENGINEERING TRACTABILITY:
   - Availability of structural data for rational design
   - Existing mutagenesis data
   - Expression system compatibility

=== REPORT FORMAT ===

Write a well-structured, informative report. Use numbered section headers. \
No filler text, but DO include enough detail to be scientifically useful.

## 1. Executive Summary
3-5 sentences covering: (a) the target reaction and substrates with chemical \
names; (b) the mechanistic families searched (by name); (c) total candidates \
found; (d) the most promising family and top 1-2 candidates with names/IDs; \
(e) WHY the top candidate is the best match (one-sentence mechanistic rationale).

## 2. Enzyme Families Discovered
Summary table: Family, Fold/Mechanism, # Candidates, # Top Tier, \
Representatives (enzyme name — accession, for each), Why Relevant (1 sentence).

## 3. Candidate Table
CRITICAL REQUIREMENT: Every single candidate in the evaluation data MUST appear \
as its own row in this table. Do NOT consolidate, omit, or summarise any \
candidate — even mechanistically incompatible ones. The only exception is true \
sequence-identical orthologs (same gene, same organism, different accession), \
which may be merged into one row with a note in Rationale. \
The number of rows must equal the number of candidates in the evaluation data.

Use these columns:

| Rank | UniProt | Name | Gene | Organism | EC | Native Substrates | Native Reaction | Family/Fold | Swiss-Prot | PDB | AF | \
Char. | Eng. | Tanimoto | Key PMID | Tier | Rationale |

- **Native Substrates**: List native substrate/product compound names with their \
SMILES in parentheses. Extract from native_substrates field (preferred) or \
from catalytic_activities text + property_comparisons SMILES. Format each as \
"compound_name (`SMILES`)" separated by "; ".

NAMING — NEVER show a bare accession anywhere in the report. Always pair every \
UniProt accession with the enzyme's name (protein_name) or, if missing, its gene \
name, formatted as "<enzyme name> (<accession>)". This applies EVERYWHERE \
accessions appear — the Representative column, grouped member lists, the \
candidate table, Top Candidate Profiles, and the Candidate(s) column of the \
references table. If only a generic name exists, show that; use a bare accession \
only when no name or gene exists at all.
HYPERLINKS — make external IDs clickable: \
UniProt as [accession](https://www.uniprot.org/uniprot/{accession}), \
PDB as [{pdb_id}](https://www.rcsb.org/structure/{pdb_id}), \
PMID as [{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}).

## 4. Top Candidate Profiles (Tier 1)
One subsection per Tier 1 candidate:
### Rank N — UniProt · Name · *Organism* · EC
**Native reaction:** one-line equation
**Why it ranks here:** 2-4 sentences
**Key limitation:** 1-2 sentences
**Key reference:** PMID (Author et al., Journal Year)

## 5. Key Literature References
Table: PMID (hyperlinked), Authors, Title, Candidate(s), Relevance. 8-12 papers. \
The Candidate(s) column lists each supported candidate as "<enzyme name> \
(<accession>)", so readers can trace evidence to specific named enzymes.

## 6. Mechanistic Summary
Table: Family/Fold, Mechanism, EC Class, Candidates, Best Representative.
End with 1-2 sentences on overall strategy.

Do NOT include: experimental strategy, risk assessment, substrate compatibility \
matrices, mechanism diagrams, cofactor SMILES, or sections that restate \
information already in the candidate table."""


# ---------------------------------------------------------------------------
# Shared eval compaction (replaces 2x ~110-line duplicated blocks)
# ---------------------------------------------------------------------------

_PROPERTY_KEYS = frozenset({
    "target_smiles", "native_smiles",
    "mw_target", "mw_native", "mw_similarity",
    "chain_length_target", "chain_length_native", "chain_length_similarity",
    "functional_group_similarity",
    "functional_group_counts_target", "functional_group_counts_native",
    "charge_ph8_target", "charge_ph8_native",
})


def _compact_eval_for_report(
    evaluation_data: list[dict],
    include_pdb_ids: bool = False,
) -> list[dict]:
    """Compact evaluation data to essentials for report LLM calls.

    This is the single implementation replacing the two identical blocks
    that were in build_explorative_report_prompt and build_confidence_report_prompt.
    """
    compact_evals = []
    for ev in evaluation_data:
        d = ev.get("data", ev)
        compact = {
            "accession": d.get("accession"),
            "protein_name": d.get("protein_name"),
            "gene_names": d.get("gene_names"),
            "organism": d.get("organism"),
            "ec_numbers": d.get("ec_numbers"),
            "is_swissprot": d.get("is_swissprot"),
            "catalytic_activities": d.get("catalytic_activities"),
            "function_comments": d.get("function_comments"),
            "interpro_family": d.get("interpro_family"),
            "strengths": d.get("strengths"),
            "weaknesses": d.get("weaknesses"),
            "requires_polymer_substrate": d.get("requires_polymer_substrate"),
            "critical_warnings": d.get("critical_warnings", []),
            "substrate_class": d.get("substrate_class", []),
            "per_substrate_match": d.get("per_substrate_match"),
        }

        sub_sim = d.get("substrate_similarity", {})
        if sub_sim:
            compact["tanimoto_score"] = sub_sim.get("tanimoto_score")
            per_sub = sub_sim.get("per_target_best")
            if per_sub and len(per_sub) > 1:
                compact["per_substrate_tanimoto"] = per_sub
            props = sub_sim.get("property_comparisons", [])
            if props:
                compact["property_comparisons"] = [
                    {k: v for k, v in p.items() if k in _PROPERTY_KEYS}
                    for p in props
                ]

        native_subs = d.get("native_substrates", [])
        if native_subs:
            compact["native_substrates"] = [
                {"name": ns.get("name"), "smiles": ns.get("smiles")}
                for ns in native_subs if ns.get("smiles")
            ]

        lit = d.get("literature", {})
        compact["characterization_papers"] = lit.get("characterization_papers", 0)
        compact["engineering_papers"] = lit.get("engineering_papers", 0)

        struct = d.get("structural_data", {})
        compact["has_pdb"] = bool(struct.get("pdb_ids"))
        compact["pdb_count"] = len(struct.get("pdb_ids", []))
        compact["has_alphafold"] = struct.get("has_alphafold", False)
        if include_pdb_ids:
            compact["pdb_ids"] = struct.get("pdb_ids", [])

        mut = d.get("mutagenesis", {})
        compact["n_mutations"] = mut.get("n_mutations", 0)

        compact_evals.append(compact)

    return compact_evals


def _pass1_compact(ev: dict) -> dict:
    """Ultra-compact representation for Pass 1 ranking.

    Includes all signals needed for P0-P5 ranking but omits SMILES,
    reaction images, and catalytic activity prose to minimise tokens.
    Target: ~200 tokens per candidate.
    """
    d = ev.get("data", ev)

    # Substrate signals: FG counts, MW similarity, Tanimoto per substrate
    sub_sim = d.get("substrate_similarity", {})
    props = sub_sim.get("property_comparisons", [])
    substrate_signals = []
    for p in props:
        substrate_signals.append({
            "fg_counts_target": p.get("functional_group_counts_target"),
            "fg_counts_native": p.get("functional_group_counts_native"),
            "fg_similarity": p.get("functional_group_similarity"),
            "mw_similarity": p.get("mw_similarity"),
            "chain_length_similarity": p.get("chain_length_similarity"),
        })

    # Expression
    expr = d.get("expression_data", {})

    # Native substrate names only (no SMILES)
    native_names = [
        ns.get("name", "") for ns in d.get("native_substrates", [])
        if ns.get("name")
    ]

    lit = d.get("literature", {})
    struct = d.get("structural_data", {})
    mut = d.get("mutagenesis", {})

    return {
        "accession": d.get("accession"),
        "gene": (d.get("gene_names") or [""])[0],
        "organism": d.get("organism", ""),
        "ec": (d.get("ec_numbers") or [""])[0],
        "family": d.get("interpro_family", ""),
        "is_swissprot": d.get("is_swissprot", False),
        "native_substrate_names": native_names,
        "tanimoto": sub_sim.get("tanimoto_score"),
        "substrate_signals": substrate_signals,
        "critical_warnings": d.get("critical_warnings", []),
        "strengths": d.get("strengths", []),
        "weaknesses": d.get("weaknesses", []),
        "requires_polymer": d.get("requires_polymer_substrate", False),
        "char_papers": lit.get("characterization_papers", 0),
        "eng_papers": lit.get("engineering_papers", 0),
        "n_mutations": mut.get("n_mutations", 0),
        "pdb_count": len(struct.get("pdb_ids", [])),
        "has_alphafold": struct.get("has_alphafold", False),
        "ecoli_expression": expr.get("ecoli_expression", False),
        "bacterial_expression": expr.get("bacterial_expression", False),
        "is_eukaryote": expr.get("is_eukaryote", False),
    }


PASS1_RANKING_SYSTEM = """\
You are an expert enzyme discovery analyst. You will be given compact \
evaluation data for ALL enzyme candidates from a discovery run. Your task \
is to produce a COMPLETE RANKED LIST of every candidate — no omissions.

RANKING FRAMEWORK (apply in strict priority order):

P0. CRITICAL WARNINGS (hard gate): any candidate with a critical_warning \
(bond type mismatch, direction mismatch, severe size mismatch) goes to the \
bottom of the list regardless of other signals.

P1. SUBSTRATE COMPATIBILITY (primary signal):
- Compare fg_counts_target vs fg_counts_native for EACH substrate.
  Differences in functional group counts mean the binding pocket has evolved \
  for different recognition features. A candidate where both substrates have \
  matching FG counts ranks higher.
- Secondary: mw_similarity and chain_length_similarity (cavity size).
- Tanimoto is REFERENCE ONLY — do not use as primary criterion.

P2. CHARACTERISATION DEPTH: Swiss-Prot reviewed + more char_papers ranks higher.

P3. EXPRESSION EVIDENCE: ecoli_expression=true > bacterial_expression=true \
> unknown > is_eukaryote=true (eukaryotic origin requires codon optimisation).

P4. STRUCTURAL DATA: pdb_count > 0 > has_alphafold > neither.

P5. ENGINEERING PRECEDENT: eng_papers and n_mutations (variants).

OUTPUT FORMAT — return ONLY a JSON array, no prose, no markdown, NO rationale \
field (the ranking is ordering-only; prose is written later):
[
  {"rank": 1, "accession": "...", "gene": "...", "tier": 1},
  {"rank": 2, "accession": "...", "gene": "...", "tier": 2},
  ...
]

Emit ONE compact object per line so the array stays parseable even if long. \
Tier assignment: Tier 1 = best 20%, Tier 2 = next 50%, Tier 3 = bottom 30%. \
Every candidate must appear exactly once. Total entries must equal the number \
of candidates given.
"""


def _pass1_table_row(c: dict) -> str:
    """Render one compact candidate as a pipe-delimited row for Pass 1."""
    # FG signals: show target vs native counts per substrate
    sigs = c.get("substrate_signals", [])
    fg_parts = []
    for s in sigs:
        tgt = {k: v for k, v in (s.get("fg_counts_target") or {}).items() if v}
        nat = {k: v for k, v in (s.get("fg_counts_native") or {}).items() if v}
        fg_sim = round(s.get("fg_similarity") or 0, 2)
        mw_sim = round(s.get("mw_similarity") or 0, 2)
        fg_parts.append(
            f"fg_sim={fg_sim} mw_sim={mw_sim} "
            f"[tgt:{tgt} nat:{nat}]"
        )
    fg_str = "; ".join(fg_parts) if fg_parts else "no_data"

    crit = " | ".join(c.get("critical_warnings", [])) or "none"
    strengths = "; ".join(c.get("strengths", [])[:3]) or "none"
    weaknesses = "; ".join(c.get("weaknesses", [])[:2]) or "none"
    native = ", ".join(c.get("native_substrate_names", [])[:3]) or "unknown"

    ecoli = "yes" if c.get("ecoli_expression") else (
        "bact" if c.get("bacterial_expression") else (
            "euk" if c.get("is_eukaryote") else "unk"))
    struct = f"PDB:{c.get('pdb_count',0)} AF:{int(c.get('has_alphafold',False))}"

    return (
        f"{c['accession']} | {c['gene']} | {c.get('organism','')[:30]} | "
        f"{c.get('ec','')} | {c.get('family','')[:35]} | "
        f"SP:{int(c.get('is_swissprot',False))} | "
        f"char:{c.get('char_papers',0)} eng:{c.get('eng_papers',0)} "
        f"mut:{c.get('n_mutations',0)} | {struct} | ecoli:{ecoli} | "
        f"tan:{round(c.get('tanimoto') or 0,3)} | "
        f"substrate_signals:[{fg_str}] | "
        f"native:[{native}] | "
        f"CRITICAL:[{crit}] | "
        f"strengths:[{strengths}] | "
        f"weaknesses:[{weaknesses}]"
    )


def build_pass1_ranking_prompt(
    query: str,
    all_evals: list[dict],
    notebook_context: str = "",
) -> list[dict]:
    """Build Pass 1 messages: rank ALL candidates from compact tabular data."""
    compact = [_pass1_compact(ev) for ev in all_evals]

    header = (
        "accession | gene | organism | ec | family | swissprot | "
        "char eng mut | struct | ecoli | tanimoto | "
        "substrate_signals | native_substrates | CRITICAL_WARNINGS | "
        "strengths | weaknesses"
    )
    rows = "\n".join(_pass1_table_row(c) for c in compact)
    table = f"{header}\n{'-'*80}\n{rows}"

    parts = [f"Target query: {query}"]
    if notebook_context.strip():
        lines = notebook_context.split("\n")
        rxn_lines = []
        in_rxn = False
        for line in lines:
            if "reaction_analysis" in line.lower() or "## target" in line.lower():
                in_rxn = True
            if in_rxn:
                rxn_lines.append(line)
                if len(rxn_lines) > 30:
                    break
        if rxn_lines:
            parts.append("## Reaction Analysis\n" + "\n".join(rxn_lines))

    parts.append(f"## Candidates ({len(compact)} total)\n\n{table}")
    parts.append(
        "Rank ALL candidates. Return only the JSON array. "
        "Every candidate must appear exactly once."
    )

    return [
        {"role": "system", "content": PASS1_RANKING_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


PASS1_SCORING_SYSTEM = """\
You are an expert enzyme discovery analyst. You will be given compact evaluation \
data for a batch of enzyme candidates and a fixed set of calibration anchors. \
Score each candidate 1-10 based on suitability for the target reaction using \
the framework below (applied in strict priority order).

P0. CRITICAL WARNINGS (hard gate): any candidate with a critical_warning \
(bond type mismatch, direction mismatch, severe size mismatch) must receive \
score 1-2 regardless of other signals.

P1. SUBSTRATE COMPATIBILITY (primary signal, ~40% of score):
- Compare fg_counts_target vs fg_counts_native for EACH substrate pair.
  Matching functional group counts indicate the binding pocket evolved for \
  similar recognition features. A candidate where ALL substrates have \
  matching FG counts ranks higher.
- Secondary: mw_similarity and chain_length_similarity (cavity size).
- Tanimoto is REFERENCE ONLY — do not use as primary criterion.

P2. CHARACTERISATION DEPTH (~20%): Swiss-Prot reviewed + more char_papers = higher.

P3. EXPRESSION EVIDENCE (~15%): ecoli=yes > bact > unknown > euk.

P4. STRUCTURAL DATA (~15%): pdb_count > 0 > has_alphafold > neither.

P5. ENGINEERING PRECEDENT (~10%): eng_papers and n_mutations.

SCORING SCALE:
  9-10: Excellent substrate match, well-characterised, high confidence
  7-8:  Good match with moderate evidence or good match with limited evidence
  5-6:  Moderate substrate match or mixed signals
  3-4:  Poor substrate match or minimal characterisation
  1-2:  Critical warnings OR fundamentally incompatible mechanism

CALIBRATION: The prompt includes a set of CALIBRATION ANCHORS — candidates \
that appear in every batch. Use them to maintain consistent scores across batches. \
Score anchor entries just like regular candidates; they are included solely for \
cross-batch consistency.

OUTPUT FORMAT — return ONLY a JSON array, no prose, no markdown fences, \
no explanatory text before or after:
[
  {"accession": "...", "gene": "...", "score": 7.5, "tier": 1,
   "rationale": "one sentence biochemical reasoning"},
  ...
]

Tier assignment: 1 = score >= 7, 2 = score 4-6.9, 3 = score < 4.
Every candidate AND every calibration anchor in the input must appear \
exactly once in the output. The total number of entries must equal the \
number of candidates given (anchors included).
"""


MECHANISTIC_RANKING_SYSTEM = """\
You are an expert enzyme discovery analyst. You will be given mechanistic \
evaluation data for ALL enzyme candidates from a discovery run. Your task is \
to rank every candidate by suitability for the target reaction.

RANKING FRAMEWORK (apply in this order):

1. BOND TYPE AND MECHANISM (primary):
   - Read catalytic_activity to understand what bond the enzyme actually forms.
   - Does it match the target bond type? Is the reaction direction correct?
   - Focus on whether the core bond-forming chemistry matches the target, not on \
the specific identity of native substrates or downstream pathway products.

2. SUBSTRATE POCKET COMPATIBILITY:
   - Compare tgt_fg vs nat_fg for EACH substrate independently.
   - A pocket evolved for substrates with the same functional group profile as \
the target substrates is more compatible than one evolved for substrates with \
additional or different decorations (extra functional groups or substituents not \
present in the target).
   - mw_sim and chain_sim indicate cavity size fit — large mismatches suggest \
the binding pocket has different spatial requirements.
   - Reason from the actual functional group names, not derived similarity scores.

3. SUBSTRATE PROMISCUITY:
   - Engineering papers and mutagenesis variants suggest the enzyme tolerates \
substrate variation. Weight these positively for borderline candidates.

4. PRACTICAL FEASIBILITY:
   - expr:ecoli > expr:bact > expr:unk > expr:euk
   - struct:PDB > struct:AF > struct:none
   - Higher char count = better understood, lower risk.

5. HARD BOTTOM (flags):
   - flags:POLYMER — requires polymer primer, cannot ligate two free molecules. \
Rank these last regardless of other signals.
   - flags containing BOND TYPE MISMATCH or REACTION DIRECTION MISMATCH — \
fundamental incompatibilities, rank last.

OUTPUT FORMAT — return ONLY a JSON array, no prose, no markdown fences:
[
  {"rank": 1, "accession": "...", "gene": "...", "tier": 1,
   "rationale": "one sentence mechanistic reasoning"},
  ...
]

Tier: 1 = top 20% (strong mechanistic match), 2 = middle 50%, 3 = bottom 30%.
Every candidate must appear exactly once. Total entries must equal the number \
of candidates given.
"""


def _mechanistic_compact_row(ev: dict) -> str:
    """Render one candidate as a compact mechanistically rich row."""
    d = ev.get("data", ev)

    acc = d.get("accession", "")
    gene = (d.get("gene_names") or [""])[0]
    ec = (d.get("ec_numbers") or [""])[0]

    # Bond type from reaction chemistry
    rxn_chem = d.get("reaction_chemistry", {})
    bond_types = rxn_chem.get("bond_types", [])
    rxn_dir = rxn_chem.get("reaction_direction", "")
    bond_str = "+".join(bond_types) if bond_types else "unknown"
    if rxn_dir:
        bond_str += f"/{rxn_dir}"

    # Catalytic activity text (truncated)
    cats = d.get("catalytic_activities", [])
    cat_text = cats[0][:150] if cats else "unknown"

    # Build SMILES -> name map from native_substrates
    smiles_to_name = {}
    for ns in (d.get("native_substrates") or []):
        smi = ns.get("smiles", "")
        name = ns.get("name", "")
        if smi and name:
            smiles_to_name[smi] = name

    # Per-substrate FG comparison with native substrate names
    sub_sim = d.get("substrate_similarity", {})
    prop_comps = sub_sim.get("property_comparisons", [])
    sub_parts = []
    for pc in prop_comps[:2]:
        fg_tgt = pc.get("functional_groups_target", [])
        fg_nat = pc.get("functional_groups_native", [])
        mw_sim = round(pc.get("mw_similarity", 0), 2)
        chain_sim = round(pc.get("chain_length_similarity", 0), 2)
        native_smi = pc.get("native_smiles", "")
        native_name = smiles_to_name.get(native_smi, "")
        name_str = f"{native_name}:" if native_name else ""
        sub_parts.append(
            f"[{name_str}tgt:{fg_tgt} nat:{fg_nat} mw:{mw_sim} chain:{chain_sim}]"
        )
    sub_str = " ".join(sub_parts) if sub_parts else "no_substrate_data"

    # Expression
    expr = d.get("expression_data", {})
    if expr.get("ecoli_expression"):
        expr_str = "ecoli"
    elif expr.get("bacterial_expression"):
        expr_str = "bact"
    elif expr.get("is_eukaryote"):
        expr_str = "euk"
    else:
        expr_str = "unk"

    # Structure
    struct = d.get("structural_data", {})
    pdb_n = len(struct.get("pdb_ids", []))
    af = struct.get("has_alphafold", False)
    struct_str = f"PDB:{pdb_n}" if pdb_n else ("AF" if af else "none")

    # Literature
    lit = d.get("literature", {})
    char_n = lit.get("characterization_papers", 0)
    eng_n = lit.get("engineering_papers", 0)

    # Flags
    crit = d.get("critical_warnings", [])
    polymer = d.get("requires_polymer_substrate", False)
    flags = []
    if polymer:
        flags.append("POLYMER")
    flags += [w[:60] for w in crit[:1]]
    flag_str = "; ".join(flags) if flags else "none"

    return (
        f"{acc} | {gene} | {ec} | bond:{bond_str} | "
        f"rxn:{cat_text} | "
        f"substrates:{sub_str} | "
        f"expr:{expr_str} struct:{struct_str} char:{char_n} eng:{eng_n} | "
        f"flags:{flag_str}"
    )


def build_mechanistic_ranking_prompt(
    query: str,
    all_evals: list[dict],
    notebook_context: str = "",
) -> list[dict]:
    """Build ranking messages with mechanistically rich compact format.

    Each row includes catalytic activity text, bond type, and per-substrate
    functional group comparison — giving the LLM real mechanistic signals
    rather than derived similarity scores.
    """
    rows = "\n".join(_mechanistic_compact_row(ev) for ev in all_evals)

    parts = [f"Target query: {query}"]

    if notebook_context.strip():
        lines = notebook_context.split("\n")
        rxn_lines = []
        in_rxn = False
        for line in lines:
            if "reaction_analysis" in line.lower() or "## target" in line.lower():
                in_rxn = True
            if in_rxn:
                rxn_lines.append(line)
                if len(rxn_lines) > 30:
                    break
        if rxn_lines:
            parts.append("## Reaction Analysis\n" + "\n".join(rxn_lines))

    header = (
        "accession | gene | ec | bond_type | catalytic_activity | "
        "substrate_FG_comparison | expression/struct/char/eng | flags"
    )
    parts.append(
        f"## All Candidates ({len(all_evals)} total)\n\n"
        f"{header}\n{'-'*80}\n{rows}"
    )
    parts.append(
        f"Rank ALL {len(all_evals)} candidates. Return only the JSON array. "
        "Every candidate must appear exactly once."
    )

    return [
        {"role": "system", "content": MECHANISTIC_RANKING_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def build_scoring_prompt(
    query: str,
    batch_candidates: list[dict],
    calibration_anchors: list[dict],
    notebook_context: str = "",
) -> list[dict]:
    """Build Pass 1 scoring messages for a single batch.

    Includes calibration_anchors in every batch so the LLM scores
    consistently across batches.

    Args:
        query: The target reaction query string.
        batch_candidates: Compact dicts from _pass1_compact() for this batch.
        calibration_anchors: Compact dicts for the fixed anchor candidates
            (same set in every batch call).
        notebook_context: Optional agent notebook text for reaction context.

    Returns:
        OpenAI-compatible messages list.
    """
    header = (
        "accession | gene | organism | ec | family | swissprot | "
        "char eng mut | struct | ecoli | tanimoto | "
        "substrate_signals | native_substrates | CRITICAL_WARNINGS | "
        "strengths | weaknesses"
    )

    def render_section(candidates: list[dict], label: str) -> str:
        rows = "\n".join(_pass1_table_row(c) for c in candidates)
        return f"## {label} ({len(candidates)} entries)\n{header}\n{'-'*80}\n{rows}"

    parts = [f"Target query: {query}"]

    if notebook_context.strip():
        lines = notebook_context.split("\n")
        rxn_lines = []
        in_rxn = False
        for line in lines:
            if "reaction_analysis" in line.lower() or "## target" in line.lower():
                in_rxn = True
            if in_rxn:
                rxn_lines.append(line)
                if len(rxn_lines) > 30:
                    break
        if rxn_lines:
            parts.append("## Reaction Analysis\n" + "\n".join(rxn_lines))

    if calibration_anchors:
        parts.append(render_section(calibration_anchors, "Calibration Anchors (appear in every batch)"))

    parts.append(render_section(batch_candidates, "Candidates to Score"))
    parts.append(
        f"Score every candidate above (including all {len(calibration_anchors)} calibration anchors "
        f"and all {len(batch_candidates)} batch candidates). "
        f"Total entries in output must be {len(calibration_anchors) + len(batch_candidates)}. "
        "Return only the JSON array."
    )

    return [
        {"role": "system", "content": PASS1_SCORING_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def build_pass2_profile_prompt(
    query: str,
    top_evals: list[dict],
    pass1_ranking: list[dict],
    notebook_context: str = "",
    system_prompt: str = CONFIDENCE_REPORT_SYSTEM,
    report_label: str = "confidence-weighted",
    authoritative_ranking: bool = True,
) -> list[dict]:
    """Build Pass 2 messages: write narrative report for top-tier candidates.

    Uses full evaluation data for top candidates only, with Pass 1 ranking
    as authoritative context for the complete picture.

    ``system_prompt``/``report_label`` let the same robust top-N render drive
    either report: the confidence-weighted view (default) or the suitability-
    only explorative view (``EXPLORATIVE_REPORT_SYSTEM``).

    ``authoritative_ranking`` — when True the supplied ranking fixes the report
    order (confidence view).  When False (explorative view) the ranking is only
    the candidate POOL and the model RE-RANKS it by biochemical suitability,
    ignoring curation/citation depth for ordering — the whole point of the
    explorative view.  The deterministic spine cannot produce that order itself
    when the substrate-accommodation signal is sparse, so the model does it here
    by reasoning over the full per-substrate data provided above.
    """
    compact_top = _compact_eval_for_report(top_evals, include_pdb_ids=True)

    # Build ranked summary for all candidates to give the LLM full context.
    # Same-enzyme orthologs are folded into their representative and listed
    # inline, so one enzyme occupies one ranked entry (not one row per organism).
    def _summary_line(r: dict) -> str:
        line = (f"  #{r['rank']} (Tier {r['tier']}) {r['gene']} ({r['accession']}): "
                f"{r.get('rationale','')}")
        orths = r.get("orthologs") or []
        if orths:
            shown = ", ".join(
                f"{o['accession']} ({o.get('organism','')})" for o in orths[:8]
            )
            more = "" if len(orths) <= 8 else f", +{len(orths) - 8} more"
            line += (f"\n      └ same enzyme in {len(orths)} other organism(s): "
                     f"{shown}{more}")
        return line

    all_ranked_summary = "\n".join(_summary_line(r) for r in pass1_ranking)

    parts = [f"Target query: {query}"]
    if notebook_context.strip():
        parts.append(f"## Agent Research Notebook\n{notebook_context}")
    parts.append(
        f"## Complete Ranking (all {len(pass1_ranking)} candidates, "
        f"from Pass 1 analysis)\n{all_ranked_summary}"
    )
    parts.append(
        f"## Full Evaluation Data for Top {len(top_evals)} Candidates\n\n"
        + json.dumps(compact_top, indent=2, default=str)
    )
    parts.append(
        f"Write the {report_label} report. The ranking above is authoritative "
        "— use it for section 3 (candidate table). Write detailed Tier 1 profiles "
        "only for the top candidates whose full data is provided above. "
        "For candidates not in the full data, use the ranking summary above.\n\n"
        "IMPORTANT — orthologs: where an entry lists 'same enzyme in N other "
        "organisms', it is ONE enzyme. Give it a SINGLE table row / profile using "
        "the representative accession, and note the organism spread (e.g. "
        "'conserved across N organisms incl. …'). Do NOT create a separate row per "
        "organism, and do NOT let one enzyme's orthologs fill multiple ranks."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _build_report_messages(
    system_prompt: str,
    query: str,
    evaluation_data: list[dict],
    notebook_context: str,
    report_type_label: str,
    include_pdb_ids: bool = False,
) -> list[dict]:
    """Build messages for a report generation LLM call."""
    compact_evals = _compact_eval_for_report(evaluation_data, include_pdb_ids=include_pdb_ids)
    eval_json = json.dumps(compact_evals, indent=2, default=str)

    user_parts = [f"Original query: {query}"]

    if notebook_context.strip():
        user_parts.append(
            "## Agent's Research Notebook\n"
            "The following is the agent's synthesized reasoning from the "
            "discovery run — reaction analysis, enzyme landscape, substrate "
            "chemistry insights, and evaluation notes. Use this context to "
            "inform your ranking.\n"
            f"{notebook_context}"
        )

    user_parts.append(
        f"## Evaluation Data ({len(compact_evals)} candidates)\n\n{eval_json}"
    )
    user_parts.append(f"Generate the {report_type_label} ranking report.")

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def build_explorative_report_prompt(
    query: str,
    evaluation_data: list[dict],
    notebook_context: str = "",
) -> list[dict]:
    """Build messages for the explorative (suitability-focused) report."""
    return _build_report_messages(
        system_prompt=EXPLORATIVE_REPORT_SYSTEM,
        query=query,
        evaluation_data=evaluation_data,
        notebook_context=notebook_context,
        report_type_label="suitability-focused (explorative)",
    )


def build_confidence_report_prompt(
    query: str,
    evaluation_data: list[dict],
    notebook_context: str = "",
) -> list[dict]:
    """Build messages for the confidence-weighted report (fallback generation)."""
    return _build_report_messages(
        system_prompt=CONFIDENCE_REPORT_SYSTEM,
        query=query,
        evaluation_data=evaluation_data,
        notebook_context=notebook_context,
        report_type_label="confidence-weighted",
        include_pdb_ids=True,
    )
