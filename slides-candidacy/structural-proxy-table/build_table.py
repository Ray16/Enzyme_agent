"""Build a single-page, slide-proportioned LaTeX inventory from saved study records."""
from pathlib import Path
import csv, hashlib, json, subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'exp_data/amide_synthetase/results/structural_proxies/attempts'
INV = BASE / 'proxy_inventory_20261011'
records = list(csv.DictReader((INV / 'historical_feature_inventory.csv').open()))
inventory = json.loads((INV / 'inventory.json').read_text())

# Each historical export prefix occurs once below. Every original field is retained
# in sources.json and in LaTeX comments, including duplicate and constant fields.
rows = [
('bat', 'Activation geometry / local environment',
 r'Mean distance-gated in-line angle; in-line fraction among gated poses; near-attack fraction; reactive-group SASA; Mg--P and Mg--O distances; Mg coordination; basic and acidic residue counts within 6\,\AA; gated-pose count.'),
('boltz', 'Geometry summaries',
 r'Boltz-2 minimum approach distance, gated mean angle, near-attack fraction, and total pose count.'),
('s2nac', 'Amine attack (step 2)',
 r'Minimum amine-N--acyl-C distance; near-attack fraction; gated B\"urgi--Dunitz angle; gated-pose count.'),
('hyp', 'Mechanistic hypothesis battery',
 r'Reactive-carboxylate polar contacts; distal-carboxylate SASA; acceptor-channel clearance; distal-minus-reactive SASA; minimum amine approach; amine approach fraction within 6\,\AA; adenine burial; acyl-carbonyl polar contacts; step-1 gated and step-2 bound pose counts.'),
('preorg', 'Preorganization / pose consistency',
 r'Reactive-carboxylate centroid RMSF; attack-direction angular spread; bidentate-clamp fraction; mean attack angle; gated-pose count.'),
('sasa', 'Intermediate exposure / pocket engagement',
 r'SASA of distal carboxyl carbon, acyl carbon, carbonyl oxygen, and reactive center; median intermediate-to-protein distance.'),
('tunnel', 'Substrate access / tunnel geometry',
 r'Open-channel length at clash cutoffs 2.5, 2.7, and 3.0\,\AA; tunnel enclosure under tight, medium, and loose settings; solvent-exit fraction; gated-pose count.'),
('elec', 'Amine-subsite charge environment',
 r'Acidic-oxygen counts, distance to a nearby candidate base, and net charge near probes placed 6, 7, 8, and 9\,\AA{} beyond the distal carboxylate (6\,\AA{} charge-counting shell); gated-pose count.'),
('esp', 'Electrostatic potential / electric field',
 r'Electrostatic potential and projected local electric field at ATP P$_\alpha$ and the acyl carbonyl; ensemble mean and standard deviation for each.'),
('pka', 'Protonation / acid--base environment',
 r'For steps 1 and 2: minimum histidine $|\mathrm{p}K_a-7|$, minimum lysine p$K_a$, and ionizable-residue count (each with ensemble mean and standard deviation); processed-pose counts.'),
('mg', 'Metal coordination',
 r'Mean/median Mg--P distances; minimum/mean Mg--O distances; coordinated fraction; Mg-bridging fractions at 2.2, 2.5, 3.0, and 3.5\,\AA; median donor-O--P approach distance.'),
('psiptm', 'Interface-confidence baselines',
 r'Directional enzyme$\to$substrate and substrate$\to$enzyme ipTM (ensemble means and maxima); symmetric mean; minimum and maximum of the directional means.'),
]
assert len(records) == 98
assert {r['feature'].split(':')[0] for r in records} == {r[0] for r in rows}
assert len({r['feature'] for r in records}) == 98

def count(prefix):
 return sum(r['feature'].startswith(prefix + ':') for r in records)

# All 39 canonical formulas in the saved matched-screen inventory are partitioned
# below; the PDF uses readable names and groups closely related aggregations.
followup = {
 'Geometry and aggregation variants': [
 'angle_90th_percentile','best_angle_all','best_joint_compatibility','current_regularized_score',
 'distance_only','gated_angle_A4','gated_angle_min5','gated_weighted_angle_any','gated_weighted_angle_min5',
 'interface_confidence','inverse_distance_weighted_alignment','inverse_distance_weighted_angle',
 'joint_fraction_4A_150deg','lower_reorganization_best','lower_reorganization_best5','lower_reorganization_mean',
 'mean_alignment_over_distance','mean_angle_all','mean_angle_over_distance','mean_best5_angles','mean_best5_joint',
 'mean_joint_compatibility','minimum_distance','nearby_fraction','positive_alignment_over_distance',
 'soft_nearattack_fraction','weighted_angle_90th_percentile','weighted_raw_angle','without_regularizer'],
 'Consistency, contacts, and combinations': [
 'approach_direction_consistency','lower_donor_center_rmsf','lower_pa_rmsf','lower_relative_donor_pa_rmsf',
 'lower_severe_overlap','score_times_direction_consistency','score_times_donor_rigidity',
 'score_times_polar_contacts','score_times_relative_rigidity','two_oxygen_polar_contacts']
}
formulas = [f for v in followup.values() for f in v]
assert len(formulas) == len(set(formulas)) == 39
assert set(formulas) == set(inventory['matched_canonical_formulas'])

tex = r'''\documentclass[11pt]{article}
\usepackage[paperwidth=16in,paperheight=9in,margin=0.38in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,booktabs,tabularx,array}
\usepackage[table]{xcolor}
\usepackage{microtype}
\renewcommand{\familydefault}{\sfdefault}
\pagestyle{empty}
\definecolor{maroon}{HTML}{890B23}
\definecolor{highlight}{HTML}{F8EDEF}
\definecolor{muted}{HTML}{51565B}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\setlength{\parindent}{0pt}
\setlength{\tabcolsep}{7pt}
\renewcommand{\arraystretch}{1.12}
\begin{document}
{\fontsize{24}{27}\selectfont\bfseries\color{maroon} Structural proxies explored for amide synthesis}\par
\vspace{4pt}
{\fontsize{11}{13}\selectfont Full historical inventory, grouped by category; subsequent matched comparisons included below.}\par
\vspace{7pt}
\fontsize{11.6}{13.3}\selectfont
\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash\hyphenpenalty=10000}p{2.65in}Y>{\raggedright\arraybackslash}p{0.84in}@{}}
\toprule
\textbf{Category} & \textbf{Full proxy list (closely related variants grouped)} & \textbf{Fields} \\
\midrule
\rowcolor{highlight}
\textbf{Selected: proximity-weighted attack angle} &
\textbf{$A=\sum_i(\theta_i/d_i)\,/\,\sum_i(1/d_i)$}; report harmonic distance $d_H=N/\sum_i(1/d_i)$ alongside it. Larger $A$ ranks higher. & Selected \\
\midrule
'''
for prefix, label, text in rows:
 fields = [r for r in records if r['feature'].startswith(prefix + ':')]
 tex += '% ' + json.dumps(fields) + '\n'
 tex += rf'\textbf{{{label}}} & {text} & {count(prefix)} \\' + '\n'
tex += r'''\midrule
\textbf{Matched follow-up: geometry variants} &
Angle summaries (mean, best, best-five mean, 90th percentile); distance-only score, minimum distance, and nearby fraction; distance-gated and gated-weighted angles (including minimum-five-pose variants); inverse-distance-weighted angle/alignment, angle-over-distance and alignment-over-distance (including positive alignment); raw weighted angle; hard/soft near-attack fractions; joint distance--angle compatibility (mean, best, best-five mean, weighted 90th percentile); regularized/unregularized score; ideal-position reorganization distance (best, best-five mean, mean); interface confidence. & 29 formulas \\
\textbf{Matched follow-up: consistency, contacts, combinations} &
Attack-direction consistency; donor-centroid, ATP P$_\alpha$, and relative donor--P$_\alpha$ RMSF; two-oxygen polar-contact fraction; severe-overlap score; historical score multiplied by direction consistency, donor rigidity, relative rigidity, or polar-contact fraction. & 10 formulas \\
\bottomrule
\end{tabularx}
\vspace{6pt}
{\fontsize{9}{10.5}\selectfont\color{muted}
\textbf{Inventory:} 98 historical exported fields = 88 distinct nonconstant definitions + 5 duplicate exports + 5 constant/undefined fields.
The 39 matched formulas overlap with the historical inventory; these counts are not additive. Counts include sampling diagnostics and formula variants.\\
\textbf{Provenance:} Imported amide discovery run and associated structural-screening artifacts; historical field inventory (2026-10-11), matched comparisons, and current attack-angle policy. Exact identifiers and source hashes accompany this PDF in \texttt{sources.json}.\\
\textbf{Interpretation:} An inventory of explored descriptors, not evidence that each predicts activity. SASA = solvent-accessible surface area; RMSF here measures spread across generated poses. Polar contacts are geometric proxies.}
\end{document}
'''
(HERE / 'structural_proxy_table.tex').write_text(tex)
paths = [INV / 'historical_feature_inventory.csv', INV / 'inventory.json',
 ROOT / 'enzyme_discovery_agent/runs/imported_amide_discovery/structural_proxy_review.json',
 ROOT / 'structural_modeling_agent/CANDIDATE_RANKING_POLICY.md',
 ROOT / 'structural_modeling_agent/knowledge/attack_angle_metric.json',
 ROOT / 'exp_data/amide_synthetase/scripts/archive/legacy_code/analysis/synthetase/feature_surfacing.py']
paths += [BASE / n / 'results.json' for n in ['geometry_proxy_screen','simple_geometry_review_20261010','docking_angle_review_20261010','proxy_performance_20261011']]
paths += list((ROOT / 'exp_data/amide_synthetase/archive/cleanup_20261008/legacy_proxy_analysis').glob('*.csv'))
provenance = {
 'scope': 'Complete historical 98-export inventory and 39 matched canonical formulas; grouped presentation, not 137 independent physical properties.',
 'run_note': 'The imported discovery run stores a historical selected bundle; the broad enumeration is reconstructed from associated saved structural-screening artifacts, not an autonomous enumeration trace in that run.',
 'historical_fields': records,
 'table_rows': [{'category': label,'prefix': prefix,'n_exported_fields':count(prefix)} for prefix,label,_ in rows],
 'matched_formulas': followup,
 'inventory_audit': inventory,
 'sources': [{'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
}
(HERE / 'sources.json').write_text(json.dumps(provenance, indent=2) + '\n')
result = subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','structural_proxy_table.tex'],cwd=HERE,capture_output=True,text=True)
print(result.stdout[-2000:])
if result.returncode: raise SystemExit(result.returncode)
