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

    The single projection used by the two-pass report prompts.
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
    if authoritative_ranking:
        order_instruction = (
            f"Write the {report_label} report. The ranking above is authoritative "
            "— use it for section 3 (candidate table). Write detailed Tier 1 profiles "
            "only for the top candidates whose full data is provided above. "
            "For candidates not in the full data, use the ranking summary above.\n\n"
        )
    else:
        order_instruction = (
            f"Write the {report_label} report. Treat the list above as the CANDIDATE "
            "POOL, not a fixed order: RE-RANK it by biochemical suitability for the "
            "target reaction alone — how closely each enzyme's native chemistry "
            "(bond formed, reaction direction, substrate size/topology/functional "
            "groups) matches the target — and do NOT let curation status, citation "
            "depth, or structure availability change a rank position (record those "
            "only in the Confidence column). Reason over the full per-substrate data "
            "provided above for the top candidates. Write detailed profiles for your "
            "top candidates; for the rest, place them by suitability using the pool "
            "summary above.\n\n"
        )
    parts.append(
        order_instruction +
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


