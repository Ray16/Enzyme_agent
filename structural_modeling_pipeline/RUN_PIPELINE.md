# Structural Modeling Pipeline

## Quick Start (single command)

```bash
python run_structural_modeling.py \
    --project my_enzyme_family \
    --uniprots P31663 P0A6W9 P39641 P0AES0 E2JA31 \
    --substrates "hexamethylenediamine=[NH3+]CCCCCC[NH3+]" \
                 "adipic_acid=[O-]C(=O)CCCCC(=O)[O-]" \
    --reaction-name "Amide bond synthesis"
```

This single command runs: fetch structures, auto-discover cofactors, detect binding pockets, map catalytic residues (both phases), run Boltz-2 cofolding, score results, and run MD.

To resume after interruption: `python run_structural_modeling.py --project my_enzyme_family --resume`

**What the single command does and does not cover.** It runs Phases 0, 0b, 1, 2, 3a/3b, 4a, 4b, 4c, and 6 (MD). It does **not** run the reporting and visualization phases (4d through 4i); run those manually afterwards, or use the manual pipeline below.

**MD runs by default.** Phase 6 prepares AmberTools systems and runs OpenMM production for every candidate, which is the most expensive part of the pipeline. Pass `--skip-md` to stop after scoring, `--md-candidates UID1 UID2` to restrict it, or `--md-time-ns` to change the production length (default 1 ns).

**Catalytic-site phases.** Phase 3 invokes `identify_catalytic_sites.py`, which runs Phase A (InterPro/UniProt) and Phase B (FoldSeek transfer with LLM validation). Phase B is what raises catalytic-residue coverage from roughly 20% to roughly 70%. Use `--skip-catalytic-enhance` to run Phase A only, or `--skip-catalytic-llm` to keep all FoldSeek-mapped residues without LLM filtering.

## Prerequisites

- Python 3.10+ with: prody, numpy, biopython, rdkit, pyKVFinder, pyyaml
- Foldseek binary (default: `~/tools/foldseek/bin/foldseek`, override with `FOLDSEEK_BIN` env var) with PDB database (default: `~/tools/foldseek_db/pdb`, override with `FOLDSEEK_DB` env var)
- Boltz-2 in conda environment (default: `boltz-2`, override with `BOLTZ_CONDA_ENV` env var)
- NVIDIA GPU(s) with >1 GB free memory for cofolding
- Internet access for UniProt, AlphaFill, and AlphaFold APIs

## Configuration

**`pipeline_config.yaml`** (top-level) lists all projects to analyze:
```yaml
projects:
  - path: amide_synthetase
    name: synthetase
  - path: amide_hydrolase
    name: hydrolase
```

Each project has its own **`config.yaml`** defining substrates, cofactors, and metals. Optional fields for new enzyme families:
```yaml
target_reaction:
  additional_cofactor_ccds: []    # extend built-in cofactor CCD set
  cofactor_mappings: {}           # extend analog→physiological mapping (e.g., ANP: ATP)
  metal_mappings: {}              # extend metal mapping (e.g., MN: MG)
```

## Pipeline Scripts (run in order)

Each script takes one or more project directories as arguments (e.g., `amide_synthetase`, `amide_hydrolase`).

### Phase 0 -- Generate Config (optional, for new projects)

```bash
python generate_config.py <project_dir> \
    --substrates "substrate_a=SMILES1" "substrate_b=SMILES2" \
    --reaction-name "Reaction description"
```

- **Input:** `structure_sequence_summary.csv` or `all_sequences.fasta` (candidate UniProt IDs)
- **Output:** `config.yaml`
- **What it does:** Queries AlphaFill for a sample of candidates to auto-discover cofactors (e.g., ATP) and catalytic metals (e.g., Mg2+). Writes `config.yaml` with `relevant_ligands`, `catalytic_metals`, and substrate SMILES.
- **Skip if:** `config.yaml` already exists and is correct.

### Phase 0b -- Fetch Structures

```bash
python fetch_structures.py <project_dir>
```

- **Input:** `structure_sequence_summary.csv` (UniProt IDs)
- **Output:** `alphafold_structures/` (AlphaFold PDB/CIF files), updated `structure_sequence_summary.csv` (with sequences)
- **What it does:** Downloads AlphaFold predicted structures and sequences for each UniProt ID. Checks AlphaFold DB first, falls back to UniProt sequence-only if no structure available.

### Phase 1 -- Detect Binding Pockets

```bash
python pocket_analysis.py amide_synthetase amide_hydrolase
```

- **Input:** AlphaFold/PDB structures in `alphafold_structures/`, `pdb_structures/`, `ref_structure/`; `config.yaml`
- **Output:** `pocket_analysis.json`
- **Dependencies:** Foldseek database, AlphaFill API, UniProt API
- **Active-site transfer:** For FoldSeek homologs, active-site contacts are transferred via structural superposition (spatial projection of ligand/metal positions), not residue-number mapping. Cascade: (1) project non-polymer entities from homolog PDB, (2) project UniProt active-site annotations if no ligands, (3) motif search fallback (find catalytic clusters in query structure). This handles diverse catalytic mechanisms (Zn metallohydrolases, cysteine/serine protease triads, substrate channels) without hardcoded motif definitions.

### Phase 2 -- Identify Catalytic Residues

```bash
python identify_catalytic_sites.py amide_synthetase amide_hydrolase --workers 4
```

- **Input:** `pocket_analysis.json` (from Phase 1), `foldseek_results.tsv` (from Phase 1)
- **Output:** `catalytic_site_validation.json`
- **Dependencies:** Argo LLM proxy (for residue-type validation in Phase B), RCSB PDB API, UniProt API
- **What it does:** Two-phase catalytic residue identification:
  - **Phase A** (`validate_catalytic_sites.py`): InterPro/UniProt sequence-based identification. Fetches UniProt "Active site" annotations directly and via InterPro domain homologs. High confidence but low coverage (~20% of enzymes).
  - **Phase B** (`enhance_catalytic_sites.py`): FoldSeek structural homolog transfer. For remaining `NO-CAT-DATA` entries, finds structural homologs (TM >= 0.7), maps catalytic positions via ProDy alignment, then validates each transfer with an LLM to filter biochemically implausible mappings (e.g., rejects Ile at a nucleophile position).
- **Options:** `--skip-enhance` (skip Phase B), `--skip-llm` (skip LLM validation in Phase B)
- **Note:** FoldSeek-transferred catalytic residues are excluded from cofolding constraints as a safety measure, but are used for pocket quality scoring and visualization.

### Phase 3a -- Prepare Cofolding Inputs

```bash
python prepare_cofold_inputs.py amide_synthetase
python prepare_cofold_inputs.py amide_hydrolase
```

- **Input:** `pocket_analysis.json`, `catalytic_site_validation.json`, `p2rank_pockets.json` (optional), `config.yaml`, `all_sequences.fasta`
- **Output:** `cofold_input/*.yaml` (one per candidate)
- **Constraint strategy:**
  - **Cofactor-guided (synthetases):** Substrate candidate pool excludes pocket lining residues (geometry-only noise). Only cofactor contacts, catalytic residues, and active-site residues are used. Candidates filtered to within 15 A of cofactor contact centroid.
  - **Pocket-constrained (hydrolases):** Pocket lining included (no cofactor to anchor). `select_tight_constraints()` picks the 10 tightest residues from the candidate pool.
- **Options:** `--affinity` (request binding affinity prediction), `--max-distance 6.0` (pocket constraint distance), `--no-pocket-constraint` (blind cofolding)

### Phase 3b -- Run Cofolding

```bash
python run_cofold.py amide_synthetase
python run_cofold.py amide_hydrolase
```

- **Input:** `cofold_input/*.yaml`
- **Output:** `cofold_output/{uid}/` (PDB structures + confidence JSON per candidate)
- **Requirements:** GPU, Boltz-2 conda environment
- **Pre-cofolding validation:** `run_cofold.py` automatically runs `validate_cofold_inputs.py` before launching Boltz-2. This catches constraint errors (e.g., metal anchored far from cofactor) that would produce unphysical structures. The run aborts if critical errors are found.
- **Options:** `--skip-gpu 0` (reserve GPU 0), `--recycling-steps 3`, `--sampling-steps 200`

### Phase 4a -- Score Cofolding Results

```bash
python analyze_cofolding.py
```

- **Input:** `cofold_output/` (Boltz-2 predictions), `pocket_analysis.json`, `catalytic_site_validation.json`, `cofold_input/*.yaml` (for all projects in `pipeline_config.yaml`)
- **Output:** `analysis_results/cofolding_metrics.csv`, `analysis_results/cofolding_summary.txt`

### Phase 4b -- Check Steric Clashes

```bash
python check_clashes.py
```

- **Input:** `cofold_output/` (PDB files, for all projects in `pipeline_config.yaml`)
- **Output:** `analysis_results/clash_report.json`, `analysis_results/clash_summary.txt`

### Phase 4c -- Analyze Pocket Geometry

```bash
python pocket_volume_analysis.py
```

- **Input:** `pocket_analysis.json`, `analysis_results/cofolding_metrics.csv`, `config.yaml` (for all projects in `pipeline_config.yaml`)
- **Output:** `analysis_results/pocket_volume_analysis.csv`, `analysis_results/pocket_volume_summary.txt`

### Phase 4d -- Select Best Model

```bash
python select_best_model.py amide_synthetase
python select_best_model.py amide_hydrolase
```

- **Input:** `cofold_output/` (multi-seed Boltz-2 predictions)
- **Output:** `*_best_model.pdb` symlink in each prediction directory
- **What it does:** When multiple diffusion samples are generated, selects the best model by substrate-cofactor proximity and creates a `*_best_model.pdb` file.

### Phase 4e -- Holo Pocket Analysis and Chemistry

```bash
python holo_pocket_analysis.py
```

- **Input:** `cofold_output/` (cofolded PDBs with `*_best_model.pdb`), AlphaFold structures (for all projects)
- **Output:** `analysis_results/holo_pocket_analysis.csv`, `analysis_results/holo_vs_apo_comparison.csv`
- **What it does:** Extracts protein from cofolded structures, runs pyKVFinder pocket detection on the holo form, compares holo vs apo (AlphaFold) pocket geometry.

```bash
python apo_pocket_analysis.py
```

- **Input:** `pocket_analysis.json`, `cofolding_metrics.csv`
- **Output:** `analysis_results/apo_pocket_chemistry.csv`
- **What it does:** Computes apo pocket chemistry (hydrophobic/charged/polar residue fractions). Holo pocket chemistry is now computed by `holo_pocket_analysis.py` above.

### Phase 4f -- Classify Enzyme Families

```bash
python classify_enzyme_families.py
```

- **Input:** `structure_sequence_summary.csv` (UniProt IDs for all projects in `pipeline_config.yaml`)
- **Output:** `analysis_results/enzyme_family_classification.csv`, `analysis_results/enzyme_family_summary.md`
- **Dependencies:** UniProt API (for InterPro/Pfam domain lookups)
- **What it does:** Classifies each enzyme candidate into protein families using InterPro/Pfam domain annotations from UniProt. Groups enzymes by shared domains and identifies the catalytic mechanism for each family.

### Phase 4g -- Generate Comprehensive Table

```bash
python generate_comprehensive_table.py
```

- **Input:** All CSV/JSON outputs from previous phases
- **Output:** `analysis_results/comprehensive_enzyme_table.csv`
- **What it does:** Merges pocket analysis, catalytic validation, cofolding metrics, clash reports, volume analysis, and holo pocket data into a single table.

### Phase 4h -- Collect Best Structures

```bash
python collect_all_structures.py
```

- **Input:** `alphafold_structures/` (apo PDBs), `cofold_output/` (cofolded PDBs), `structure_sequence_summary.csv`, `analysis_results/cofolding_metrics.csv` (for enzyme names)
- **Output:** `all_structures/amide_synthetases_candidates/{alphafold_apo_structure,cofolded_structure}/`, `all_structures/amide_hydrolase_candidates/{alphafold_apo_structure,cofolded_structure}/`
- **What it does:** Copies AlphaFold apo and best cofolded PDBs for all candidates into a unified directory with human-readable filenames (`{enzyme}_{uid}.pdb`).

### Phase 4i -- Generate Visualizations

```bash
python generate_pocket_html.py
python generate_pymol_scripts.py
```

- **Input:** All analysis data, cofolded PDB structures, AlphaFold structures, `holo_pocket_residues.json`, `catalytic_site_validation.json`, Boltz-2 confidence JSON files
- **Output:**
  - `all_structures/pocket_visualization_synthetases.html`, `all_structures/pocket_visualization_hydrolases.html` — interactive 3D viewers (3Dmol.js) with per-substrate iPTM, pocket chemistry, and binding site overlap metrics
  - `all_structures/*/alphafold_apo_structure/*_pocket_analysis.pml` — PyMOL scripts for apo structures
  - `all_structures/*/cofolded_structure/*_pocket_analysis.pml` — PyMOL scripts for cofolded structures with YRB surface coloring, ligand sticks, polar contacts, and catalytic residue annotations

Optional: `python render_pocket_images.py` to batch-render .pml files to PNG images.

### Phase 5 -- Molecular Dynamics (optional, expensive)

```bash
python prepare_md_systems.py <project_dir> [--candidates UID1 UID2]
python run_md.py <project_dir> --sim-time-ns 1 [--candidates UID1 UID2]
python analyze_md.py <project_dir> [--candidates UID1 UID2]
```

- **Input:** `cofold_output/` best models, `config.yaml`
- **Output:** `md_prep/` (AmberTools parameterized systems), `md_output/` (OpenMM trajectories), `md_output/md_stability_metrics.csv`
- **Dependencies:** AmberTools (parameterization), OpenMM (production), GPU
- **What it does:** Solvates and parameterizes each cofolded complex, runs production MD, then reports per-candidate stability metrics (ligand RMSD, contact persistence).
- **Note:** `run_structural_modeling.py` runs all three of these by default. Pass `--skip-md` to omit them.

## Manual Pipeline (individual scripts)

```bash
# Phase 0 (setup)
# python generate_config.py amide_synthetase/ --substrates "hmd=[NH3+]CCCCCC[NH3+]" "adipic=[O-]C(=O)CCCCC(=O)[O-]"
# python fetch_structures.py amide_synthetase
# python fetch_structures.py amide_hydrolase

# Phase 1 (pocket detection + spatial active-site transfer)
python pocket_analysis.py amide_synthetase amide_hydrolase

# Phase 2 (catalytic residue identification)
python identify_catalytic_sites.py amide_synthetase amide_hydrolase --workers 4

# Phase 3 (cofolding)
python prepare_cofold_inputs.py amide_synthetase
python prepare_cofold_inputs.py amide_hydrolase
python run_cofold.py amide_synthetase
python run_cofold.py amide_hydrolase

# Phase 4 (scoring + visualization)
python analyze_cofolding.py
python check_clashes.py
python pocket_volume_analysis.py
python select_best_model.py amide_synthetase
python select_best_model.py amide_hydrolase
python apo_pocket_analysis.py
python holo_pocket_analysis.py
python classify_enzyme_families.py
python generate_comprehensive_table.py
python collect_all_structures.py
python generate_pocket_html.py
python generate_pymol_scripts.py

# Phase 5 (MD, optional)
python prepare_md_systems.py amide_synthetase
python run_md.py amide_synthetase --sim-time-ns 1
python analyze_md.py amide_synthetase
```
