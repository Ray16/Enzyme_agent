# Implementation Plan: Aim 1 & 2 Features

## Context

This plan implements features from the PhD candidacy proposal's Aim 1 (enzyme discovery agent) and Aim 2 (structure-aware evaluation). The work targets two codebases:
- **enzyme_agent_refactor** (`/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/enzyme_agent_refactor/`) -- new discovery tools
- **experimental_feedback_refactor** (`/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/experimental_feedback_refactor/`) -- structural analysis features

All new code is additive (new files or new functions appended to existing files). No existing functions are modified.

---

## Part A: Aim 1 -- Enzyme Discovery Agent (enzyme_agent_refactor)

### A1. ModelSEED/KBase Reaction Database Tool

**File:** `enzyme_agent_refactor/tools/kbase.py` (new file)

**What it does:** Queries the ModelSEED Solr API (the biochemistry database underlying KBase) to systematically discover enzyme families by EC class and reaction chemistry. This replaces the LLM's incomplete brainstorming of natural substrates with deterministic database traversal.

**API confirmed working:**
- Endpoint: `https://modelseed.org/solr/reactions/select`
- Supports: EC wildcard queries, text search on reaction definitions, compound ID lookups
- Response: JSON with `id`, `name`, `ec_numbers`, `definition`, `compound_ids`, `aliases`, `pathways`
- Compound details: `https://modelseed.org/solr/compounds/select` returns `name`, `formula`, `smiles`, `aliases`

**Tool 1: `search_modelseed_reactions`**
- Category: `database`, parallel_safe: `True`
- Parameters:
  - `ec_class` (str): EC prefix to search (e.g., "6.3" for all ligases forming C-N bonds)
  - `keyword` (str, optional): Text filter on reaction definition (e.g., "amide", "amine")
  - `max_results` (int, default 50): Result cap
- Logic:
  1. Query ModelSEED Solr: `q=ec_numbers:{ec_class}.*` (wildcard)
  2. If keyword provided, add `AND definition:*{keyword}*`
  3. Request fields: `id,name,ec_numbers,definition,compound_ids,pathways,aliases`
  4. For each reaction, extract unique EC numbers and compound IDs
  5. Return reactions grouped by EC subclass
- Returns:
  ```python
  {
      "status": "success",
      "summary": "Found 370 reactions in EC 6.3.* (42 unique EC numbers)",
      "data": [
          {
              "reaction_id": "rxn00187",
              "name": "L-Glutamate:ammonia ligase (ADP-forming)",
              "ec_numbers": ["6.3.1.2"],
              "definition": "ATP + NH3 + L-Glutamate => ADP + Pi + L-Glutamine",
              "compound_ids": ["cpd00002", "cpd00013", ...],
              "pathways": [...]
          }, ...
      ],
      "count": 370,
      "ec_summary": {"6.3.1": 45, "6.3.2": 120, "6.3.4": 85, ...},
      "suggestions": ["Use compound IDs to fetch substrate structures via get_modelseed_compound"]
  }
  ```

**Tool 2: `get_modelseed_compound`**
- Category: `database`, parallel_safe: `True`
- Parameters:
  - `compound_id` (str): ModelSEED compound ID (e.g., "cpd00053")
- Logic:
  1. Query: `https://modelseed.org/solr/compounds/select?q=id:{compound_id}`
  2. Extract name, formula, SMILES, aliases (KEGG, ChEBI cross-refs)
- Returns: `{status, summary, data: {id, name, formula, smiles, aliases}}`
- This lets the agent resolve ModelSEED compound IDs to SMILES for fingerprint similarity comparison

**Prompt integration:** Add a new section to `prompts/sections.py`:
```
_MODELSEED_GUIDANCE = """
When to use ModelSEED: After decomposing the target reaction, use search_modelseed_reactions
to systematically enumerate all EC subclasses that catalyze the target bond type. This 
provides deterministic coverage that keyword searches may miss. Use get_modelseed_compound
to resolve substrate structures for fingerprint comparison.
"""
```

**Authentication:** None required. ModelSEED Solr API is fully public. KBase Workspace API (which would add genome-level gene-protein-reaction mappings) requires a KBase Developer Account with Argonne Lab registration -- deferred to future work.

**Dependencies:** Only `urllib`, `json` -- uses `tools/_http.py:request_with_retry()` for resilience.

---

### A2. Sequence-Based Literature Search Tool (PaperBLAST Alternative)

**File:** `enzyme_agent_refactor/tools/literature.py` (append to existing)

**Why not PaperBLAST directly:** PaperBLAST CGI is behind Cloudflare JS challenge, blocking programmatic access. Instead, we implement the same concept (sequence -> homologs -> literature) using a hybrid approach: UniProt's REST API for curated references (high precision) + PubMed gene-name search (high recall).

**Rate limits:** UniProt REST API has no hard rate limit but recommends <1 req/sec for batch operations. Our `request_with_retry()` handles 429 throttling. For ~20 homologs, total time is ~30 seconds.

**Effectiveness vs PaperBLAST:** PaperBLAST text-mines 1.27M articles via automated linking. UniProt curated references are more conservative (~3-10 per protein). To bridge this gap, the tool combines two strategies:
1. Fetch UniProt's curated references for each homolog (high precision)
2. Search PubMed by each homolog's gene name using existing `search_pubmed` (high recall)
3. Semantic re-rank the combined results

**Tool: `sequence_literature_search`**
- Category: `literature`, parallel_safe: `True`
- Parameters:
  - `accession` (str): UniProt accession of a representative family member
  - `max_homologs` (int, default 20): Max homologs to process
  - `rerank_query` (str, optional): Target reaction description for semantic re-ranking
- Logic:
  1. **Find homologs via InterPro family:** Fetch the protein's InterPro family ID from `GET https://rest.uniprot.org/uniprotkb/{accession}.json` -> extract `crossReferences` for InterPro entries of type "Family". Then search UniProt for reviewed members of that family: `GET https://rest.uniprot.org/uniprotkb/search?query=(database:interpro-{family_id}) AND reviewed:true&size={max_homologs}&fields=accession,gene_names,organism_name,protein_name,lit_pubmed_id`
  2. **Collect publications (two sources per homolog):**
     - *UniProt references:* `GET https://rest.uniprot.org/uniprotkb/{hit}.json` -> `references[].citation` -> extract title, journal, year, PubMed ID
     - *PubMed gene-name search:* For each homolog's gene name(s), call existing `_search_pubmed_raw()` with query `"{gene_name}" AND enzyme` (capped at 5 results per gene). This catches papers UniProt curators haven't linked yet.
  3. **Deduplicate** by PubMed ID (exact) + title similarity (token Jaccard, 80% threshold for papers without PMIDs)
  4. **Semantic re-rank** using existing `_semantic_rerank()` with rerank_query
  5. **Extract entities** (gene names, EC numbers) using existing `_extract_entities_from_papers()`
  6. Throttle: 0.5s delay between UniProt JSON fetches to stay well under rate limits
- Returns:
  ```python
  {
      "status": "success",
      "summary": "Found 47 publications across 15 homologs of Q81RQ8 (InterPro family IPR010089)",
      "data": {
          "family_id": "IPR010089",
          "family_name": "IucA/IucC family",
          "homologs": [{"accession": "...", "gene": "...", "organism": "...", "n_papers": 5}],
          "publications": [{"title": "...", "year": 2023, "pmid": "...", "source_accessions": ["..."], "semantic_score": 0.82}],
          "extracted_entities": {"gene_names": [...], "ec_numbers": [...]}
      },
      "count": 47,
      "suggestions": ["Search for newly discovered gene names: ..."]
  }
  ```

**Key advantage:** Surfaces characterization papers about homologous enzymes that keyword searches miss because they use organism-specific gene names unrelated to the target reaction. The InterPro family lookup ensures homologs share the same structural fold, not just sequence similarity.

**Prompt integration:** Add guidance to `_LITERATURE_GUIDANCE` in `prompts/sections.py`:
```
When you have identified a promising candidate family but suspect there may be 
characterized homologs with relevant substrate data, use sequence_literature_search 
with a representative accession to find publications about related proteins in the 
same InterPro family. This is especially useful for enzyme families where different 
organisms use unrelated gene names for the same protein.
```

---

### A3. Structural Evidence Integration Tool

**File:** `enzyme_agent_refactor/tools/structural_evidence.py` (new file)

**What it does:** Reads co-folding pipeline results (CSV/JSON) and returns a structured structural report for each candidate, allowing the agent to incorporate structural data into its evaluation and ranking.

**Tool: `get_structural_evaluation`**
- Category: `reasoning`, parallel_safe: `True`
- Parameters:
  - `accession` (str): UniProt accession
  - `cofold_results_dir` (str): Path to experimental_feedback_refactor analysis_results directory
- Logic:
  1. Read `comprehensive_enzyme_table.csv` (or individual CSVs: `cofolding_metrics.csv`, `holo_pocket_chemistry.csv`, `apo_pocket_chemistry.csv`, `pocket_volume_analysis.csv`, `holo_vs_apo_comparison.csv`)
  2. Filter to matching UID (accession)
  3. Compile structural report:
     - **Binding quality:** substrate iPTM, SASA burial %, cofactor iPTM
     - **Pocket chemistry:** holo pocket charged/hydrophobic/polar fractions
     - **Apo-holo comparison:** residue Jaccard index, volume change %
     - **Pocket geometry:** volume, shape classification, SA/V ratio
     - **Clash data:** from `clash_analysis.json` if available
     - **Sub-region descriptors:** from `substrate_pocket_subregions.csv` if available (from Part B2)
  4. Generate interpretive assessment (e.g., "High hydrophobic fraction (43%) compatible with uncharged aliphatic substrate")
- Returns:
  ```python
  {
      "status": "success",
      "summary": "Structural evaluation for Q81RQ8: iPTM=0.983, burial=98.7%, pocket=20.7% hydrophobic",
      "data": {
          "binding_quality": {"substrate_iptm": 0.983, "sasa_burial_pct": 98.7, ...},
          "pocket_chemistry": {"frac_charged": 0.362, "frac_hydrophobic": 0.207, "frac_polar": 0.431},
          "apo_holo_comparison": {"residue_jaccard": 0.85, "volume_change_pct": -12.3},
          "pocket_geometry": {"volume_A3": 1250, "shape": "enclosed", "sa_v_ratio": 0.65},
          "sub_regions": [...],  # if available
          "assessment": "..."
      }
  }
  ```

**Prompt integration:** Add to `_CANDIDATE_EVALUATION_GUIDANCE`:
```
When structural evaluation data is available (co-folding results), use 
get_structural_evaluation to incorporate binding quality, pocket chemistry, and 
apo-holo stability into your candidate assessment. Weight structural evidence 
alongside chemistry and literature evidence.
```

---

## Part B: Aim 2 -- Structural Evaluation (experimental_feedback_refactor)

### B1. Substrate-Aware Sub-Region Pocket Descriptors

**File:** `experimental_feedback_refactor/substrate_pocket_descriptors.py` (new file)

**What it does:** Decomposes the binding pocket into sub-regions defined by 5A shells around each substrate functional group, providing spatially resolved chemistry rather than whole-pocket averages.

**Inputs:**
- Co-folded PDB structures (`cofold_output/{uid}/*_best_model.pdb`)
- Substrate SMILES from `config.yaml`
- Holo pocket residues from `analysis_results/holo_pocket_residues.json`

**Algorithm:**
1. **Identify substrate functional groups** from SMILES using RDKit SMARTS patterns (reuse the 19 patterns from `enzyme_agent_refactor/tools/substrate.py` or `plot_substrate_diversity.py`):
   - Carboxyl (`[CX3](=O)[OX1H0-,OX2H1]`), primary amine (`[NX3;H2;!$(NC=O)]`), amide (`[NX3][CX3](=[OX1])`)
   - Each match maps to atom indices in the 3D conformer
2. **Map SMILES atom indices to PDB coordinates:** Parse the ligand HETATM records from the co-folded PDB to get 3D coordinates of each substrate atom
3. **Define sub-regions:** For each functional group, compute centroid of matched atoms -> define a 5A sphere -> collect all protein residues with any heavy atom within 5A
4. **Characterize each sub-region:**
   - Residue composition: charged/hydrophobic/polar fractions (reuse `classify_residues()` from `utils.py`)
   - Hydrogen-bond capacity: count of H-bond donors (Ser, Thr, Tyr, Asn, Gln, Arg, Lys, His) and acceptors (Asp, Glu, Asn, Gln, Ser, Thr, His) in the sub-region
   - Local electrostatic character: net charge of sub-region residues (Arg+Lys = +1, Asp+Glu = -1, His = +0.5 at pH 7)
   - Kyte-Doolittle mean hydropathy (reuse `compute_kd_hydropathy()` from `utils.py`)
5. **Identify mismatches:** For each sub-region, flag if the local chemistry is inconsistent with the substrate functional group:
   - Carboxylate sub-region should have cationic residues (Arg, Lys) for coordination
   - Amine sub-region should have anionic/polar residues, NOT excess cationic (electrostatic repulsion)
   - Aliphatic chain sub-region should be hydrophobic

**Output:** `analysis_results/substrate_pocket_subregions.csv`
- Columns: `project, uid, functional_group, fg_type, fg_atom_indices, centroid_x, centroid_y, centroid_z, n_residues, residue_list, frac_charged, frac_hydrophobic, frac_polar, n_hbond_donors, n_hbond_acceptors, net_charge, mean_kd_hydropathy, mismatch_flag, mismatch_reason`

**Also outputs:** `analysis_results/substrate_pocket_subregions.json` (structured, per-UID)

**Dependencies:** RDKit (SMARTS matching, conformer), prody or BioPython (PDB parsing), numpy. All already available.

**Integration with existing pipeline:** Add as a new Phase 6b step after `holo_pocket_analysis.py`, called from `run_structural_modeling.py`.

---

### B2. LigandMPNN Active-Site Design Integration

**File:** `experimental_feedback_refactor/design_active_site.py` (new file)

**What it does:** Uses sub-region mismatch data to identify pocket positions to redesign, runs LigandMPNN to generate variant sequences, then verifies designs by re-cofolding with Boltz-2.

**Inputs:**
- Co-folded PDB: `cofold_output/{uid}/*_best_model.pdb`
- Sub-region descriptors: `analysis_results/substrate_pocket_subregions.csv` (from B1)
- Catalytic site data: `catalytic_site_validation.json` (residues to keep fixed)
- Config: `config.yaml` (substrates, cofactors)

**Algorithm:**
1. **Identify redesign targets:** From sub-region descriptors, select residues in sub-regions with `mismatch_flag=True` that are NOT in the catalytic residue set (core or shell)
2. **Prepare LigandMPNN input:**
   - `--pdb_path`: the best_model.pdb (holo structure with substrate + cofactor)
   - `--model_type ligand_mpnn`: use ligand-context-aware model
   - `--fixed_residues`: catalytic residues + cofactor-coordinating residues (from `catalytic_site_validation.json`)
   - `--redesigned_residues`: the mismatch residues identified above
   - `--ligand_mpnn_use_atom_context 1`: include substrate/cofactor atoms as design context
   - `--number_of_batches 10`: generate 10 designs per candidate
   - `--temperature 0.1`: low temperature for conservative designs
   - `--pack_side_chains 1`: generate side chain conformers
   - Amino acid bias: optionally bias toward residue types matching the target sub-region chemistry (e.g., bias toward Arg/Lys in carboxylate sub-region)
3. **Run LigandMPNN:** subprocess call to `python /path/to/LigandMPNN/run.py <args>`
4. **Parse outputs:** Read FASTA sequences from `out_folder/seqs/`, extract confidence scores from headers
5. **Prepare re-cofolding:** For top N designs (by overall_confidence), generate Boltz-2 YAML inputs with the redesigned sequence
6. **Verify by re-cofolding:** Run Boltz-2 on designs, compute iPTM and SASA burial, compare to wild-type

**Output:**
- `design_output/{uid}/designs.fasta` - designed sequences
- `design_output/{uid}/design_summary.csv` - per-design: sequence, overall_confidence, ligand_confidence, seq_rec, redesigned_positions, mutations
- `design_output/{uid}/verification/` - re-cofolding results (if run)

**CLI interface (both modes via --verify flag):**
```bash
# Prepare-only mode (default): runs LigandMPNN, outputs designs + Boltz-2 YAMLs
python design_active_site.py amide_synthetase \
    --uids A0A0F7RJ52 Q81RQ8 \
    --n-designs 10 \
    --ligandmpnn-path /path/to/LigandMPNN

# Auto-verify mode: runs LigandMPNN + Boltz-2 re-cofolding end-to-end
python design_active_site.py amide_synthetase \
    --uids A0A0F7RJ52 Q81RQ8 \
    --n-designs 10 \
    --ligandmpnn-path /path/to/LigandMPNN \
    --verify --verify-top-n 3  # re-cofold top 3 designs per candidate
```

**Dependencies:** LigandMPNN (subprocess), Boltz-2 (subprocess, for --verify mode only), RDKit, same conda env as pipeline.

---

## Implementation Order

1. **A1** (ModelSEED tools) -- self-contained, no dependencies on other new code
2. **A2** (sequence literature search) -- self-contained, extends existing literature.py
3. **B1** (sub-region pocket descriptors) -- self-contained structural analysis
4. **A3** (structural evidence integration) -- depends on B1 output format being defined
5. **B2** (LigandMPNN design) -- depends on B1 for mismatch identification

---

## Verification Plan

### A1: ModelSEED Tools
```bash
cd /nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/enzyme_agent_refactor
python -c "from tools.kbase import search_modelseed_reactions, get_modelseed_compound; \
  r = search_modelseed_reactions(ec_class='6.3.1'); print(r['summary']); \
  c = get_modelseed_compound(compound_id='cpd00053'); print(c['data']['name'], c['data']['smiles'])"
```
Expected: Returns 45+ reactions in EC 6.3.1.*, compound resolves to L-Glutamine with SMILES.

### A2: Sequence Literature Search
```bash
python -c "from tools.literature import sequence_literature_search; \
  r = sequence_literature_search(accession='Q81RQ8'); \
  print(r['summary']); print(len(r['data']['publications']), 'publications')"
```
Expected: Returns publications about AsbB homologs across organisms.

### A3: Structural Evidence Integration
```bash
python -c "from tools.structural_evidence import get_structural_evaluation; \
  r = get_structural_evaluation(accession='A0A0F7RJ52', \
    cofold_results_dir='/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/experimental_feedback/analysis_results'); \
  print(r['summary'])"
```
Expected: Returns binding quality, pocket chemistry, and geometry for AsbA.

### B1: Sub-Region Descriptors
```bash
cd /nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/experimental_feedback_refactor
python substrate_pocket_descriptors.py amide_synthetase --dry-run
```
Expected: Prints identified functional groups and sub-region residue counts without writing files.

### B2: LigandMPNN Design
```bash
python design_active_site.py amide_synthetase --uids A0A0F7RJ52 --n-designs 2 --dry-run \
  --ligandmpnn-path /nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/LigandMPNN
```
Expected: Prints LigandMPNN command that would be run, shows redesign targets and fixed residues.

---

## Files Modified/Created Summary

| File | Action | Codebase |
|------|--------|----------|
| `tools/kbase.py` | NEW | enzyme_agent_refactor |
| `tools/literature.py` | APPEND (1 new function) | enzyme_agent_refactor |
| `tools/structural_evidence.py` | NEW | enzyme_agent_refactor |
| `prompts/sections.py` | APPEND (guidance text) | enzyme_agent_refactor |
| `substrate_pocket_descriptors.py` | NEW | experimental_feedback_refactor |
| `design_active_site.py` | NEW | experimental_feedback_refactor |

No existing functions are modified. All changes are additive.
