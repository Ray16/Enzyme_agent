"""Independent method, control and proposed-work diagrams; no slide assembly."""
from pathlib import Path
import importlib.util,json,hashlib
import fitz
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=HERE.parent/'backup-finalization/build_backups.py'
spec=importlib.util.spec_from_file_location('reviewed_artwork',SOURCE)
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
FIGURES=[]


def figure(title,status):
    d=fitz.open();p=d.new_page(width=1920,height=1080)
    A.txt(p,(80,35,1840,125),title,48,A.MAROON,True)
    A.txt(p,(80,145,1840,202),status,25,A.MUTED)
    return d,p


def save(d,name,caption,kind='proposed analysis'):
    p=d[0];A.txt(p,(80,1004,1840,1065),caption.split('\n')[0],19,A.MUTED)
    d.save(HERE/f'{name}.pdf',garbage=4,deflate=True)
    p.get_pixmap(matrix=fitz.Matrix(1.25,1.25),alpha=False).save(HERE/f'{name}.png')
    svg=p.get_svg_image(text_as_path=False)
    (HERE/f'{name}.svg').write_text('\n'.join(x.rstrip() for x in svg.splitlines())+'\n')
    FIGURES.append({'name':name,'caption':caption,'kind':kind,
                    'files':[f'{name}.{e}' for e in ['png','svg','pdf']]})


def requirements():
    d,p=figure('Assay evidence needed to support the biological claim',
               'Saved measurements are available; analytical identity, assay conditions and repeat type need documentation.')
    A.table(p,80,224,[350,650,760],['Evidence','Available in the stored panel','Additional evidence to document'],[
      ['Product endpoint','Reported product concentrations (µg/mL).','Chemical identity, calibration, background,<br/>LOD / LOQ and relevant raw analytical traces.'],
      ['Variation','Six values under two date labels;<br/>mean and sample SD are reproducible.','Biological versus technical repetition;<br/>sample preparation and date-group differences.'],
      ['Controls','SfaB reference and BL21-labeled rows.','Exact control identity; no-enzyme / empty-vector<br/>interpretation and background correction.'],
      ['Protein contribution','Activity-associated enzyme IDs and sequences.','Enzyme abundance, expression / solubility,<br/>reaction duration and normalization basis.'],
      ['Validation scope','Retrospective homolog-panel measurements.','Independent reactions and a prospective panel<br/>with predefined success and comparison rules.'],
    ],[80,130,130,130,130,130],28)
    save(d,'assay_evidence_requirements','The CSV alone does not establish analytical identity, biological replication or isolated catalytic turnover.','data-availability and proposed validation')


def independence():
    d,p=figure('Separate exploration, frozen definitions, and validation',
               'PROPOSED VALIDATION · Development cases remain development cases after they inform a change.')
    A.card(p,80,250,530,350,'1  Explore development data',
           'Compare descriptors and setups.<br/>Diagnose known failures.<br/>Record every tried definition.<br/><br/>Current homolog panel belongs here.',A.GRAY,29)
    A.card(p,695,250,530,350,'2  Freeze the analysis',
           'Fix score and reactive atoms.<br/>Declare sample-count requirements.<br/>Define candidate budget and endpoint.<br/><br/>Preserve source and version identity.',A.PALE,29)
    A.card(p,1310,250,530,350,'3  Test independent cases',
           'Use untouched reactions / candidates.<br/>Match tools and resource budgets.<br/>Report uncertainty and failures.<br/><br/>Avoid tuning on the test outcomes.',A.PALE,29)
    A.arrow(p,(620,420),(685,420));A.arrow(p,(1235,420),(1300,420))
    A.rect(p,(80,690,1840,898),A.ROSE)
    A.txt(p,(110,720,1810,875),'<b>Prospective experimental round:</b> select before measuring → run matched controls → evaluate the predefined endpoint → revise only for a subsequent round.',34)
    save(d,'exploration_freeze_validation','A repeatedly inspected or repaired case cannot establish held-out generalization. This diagram is an evaluation plan.')


def multistate():
    d,p=figure('Develop catalytic constraints that support multiple states',
               'PROPOSED DESIGN INVESTIGATION · A generated scaffold must be evaluated separately in each modeled state.')
    A.card(p,80,240,560,245,'Activation state',
           'Carboxylate / ATP / metal setup<br/>Reaction-specific phosphorus<br/>Approach geometry and local contacts',size=28)
    A.card(p,80,580,560,245,'Amide-forming state',
           'Activated acyl intermediate + amine<br/>Neutral attacking nitrogen<br/>Carbonyl approach and accommodation',size=28)
    A.card(p,790,360,470,390,'Shared sequence / scaffold',
           'Identify compatible catalytic atoms.<br/><br/>Specify fixed and flexible features.<br/><br/>Test whether one scaffold supports<br/>the required states.',A.ROSE,28)
    A.card(p,1390,360,450,390,'Independent evaluation',
           'Check structure and chemical setup.<br/><br/>Re-model each state.<br/><br/>Test product formation experimentally.',size=28)
    A.arrow(p,(650,370),(780,465));A.arrow(p,(650,700),(780,630));A.arrow(p,(1270,555),(1380,555))
    save(d,'proposed_multistate_motif_constraints','Multistate compatibility is proposed work; tool availability and successful generation do not establish catalytic activity.')


def experimental_panel():
    d,p=figure('Select experiments that separate competing predictors',
               'PROPOSED PANEL DESIGN · Schematic categories, not measured or newly predicted candidates.')
    A.card(p,420,250,550,240,'A  High best-pose geometry',
           'High geometric consistency<br/><br/>Favorable under both hypotheses',size=29)
    A.card(p,1080,250,550,240,'B  Low best-pose geometry',
           'High geometric consistency<br/><br/>Contrasts best-pose geometry',size=29)
    A.card(p,420,590,550,240,'C  High best-pose geometry',
           'Low geometric consistency<br/><br/>Contrasts consistency',A.GRAY,29)
    A.card(p,1080,590,550,240,'D  Low best-pose geometry',
           'Low geometric consistency<br/><br/>Lower-support comparison category',A.GRAY,29)
    A.arrow(p,(980,370),(1070,370));A.arrow(p,(695,500),(695,580))
    A.txt(p,(80,375,370,750),'<b>Compare A with C:</b><br/>hold geometry high;<br/>vary consistency.<br/><br/><b>Compare A with B:</b><br/>hold consistency high;<br/>vary geometry.',28)
    A.txt(p,(420,880,1780,965),'Match expression, stability, substrate conditions and assay handling where possible. Evaluate both main effects and their interaction.',30)
    save(d,'proposed_hypothesis_discriminating_panel','Thresholds and candidate identities must be fixed before testing independent candidates; known development examples are not prospective evidence.')


def mismatch():
    d,p=figure('Prediction–experiment disagreement selects the next test',
               'PROPOSED FOLLOW-UP LOGIC · Similar activation geometry can coexist with very different measured product concentrations.')
    A.table(p,80,225,[390,690,680],['Question','Discriminating evidence','Consequence for the investigation'],[
      ['Is the measurement comparable?','Analytical controls, date-group records,<br/>protein abundance and normalization.','Resolve an assay or expression discrepancy<br/>before assigning a catalytic explanation.'],
      ['Does the target fit?','Substrate access, alternative poses,<br/>steric conflicts and assembly context.','Revise accommodation or assembly<br/>assumptions using candidate-specific evidence.'],
      ['Is the amide-forming step feasible?','Correct intermediate and neutral amine;<br/>approach geometry in the second step.','Reconsider a step-specific limitation;<br/>distance alone is insufficient.'],
      ['Does productive geometry persist?','A suitably validated dynamics investigation<br/>with explicit states and observables.','Test stability / accessibility of poses;<br/>avoid treating cofold samples as a trajectory.'],
      ['Is a chemical barrier limiting?','A justified QM/MM or related reaction-path<br/>calculation with its own controls.','Test energetic hypotheses separately<br/>from pose confidence and endpoint yield.'],
    ],[80,125,125,125,125,125],27)
    save(d,'prediction_experiment_mismatch_tests','These are alternative investigations, not established explanations for AS9 / AS4 or automatic steps executed for every candidate.')


def methods():
    d,p=figure('Choose the method from the scientific question',
               'METHOD INTERPRETATION · Readouts have different physical meanings and validation requirements.')
    A.table(p,80,225,[300,510,475,475],['Method','Question','Readout','Interpretation limit'],[
      ['Cofolding','Where can the complex be placed?','Predicted poses and<br/>confidence diagnostics.','No chemical barrier or<br/>equilibrium population.'],
      ['Generative ensemble','What structural variability is<br/>represented by the sampler?','Generated configurations;<br/>model-specific validation.','No kinetic trajectory unless<br/>that mode is validated.'],
      ['Molecular dynamics','How does the declared system<br/>evolve under a physical model?','Trajectory-derived geometry<br/>and persistence observables.','Finite sampling; force-field<br/>and setup assumptions.'],
      ['QM/MM / reaction path','What chemical transformation<br/>and barrier follow from this setup?','Energy / path evidence<br/>for a specified state.','State, protonation, electronic<br/>model and convergence matter.'],
      ['Experimental assay','How much product is observed<br/>under the tested conditions?','Measured endpoint, controls<br/>and experimental variation.','Expression and whole-assay<br/>factors can affect the endpoint.'],
    ],[80,125,125,125,125,125],27)
    save(d,'method_question_readout_limits','A stronger-looking structural prediction cannot substitute for an energetic calculation or a controlled experimental measurement.','method interpretation')


def milestones():
    d,p=figure('Define success and informative failure for each aim',
               'PROPOSED THESIS CRITERIA · Numeric targets and experimental sample sizes require a separate justified study plan.')
    A.table(p,80,235,[300,730,730],['Aim','Evidence of progress','If the expected benefit is absent'],[
      ['Discover','Recover mechanistically plausible candidates<br/>on independent reactions; adjudicate claims<br/>and compare with matched baselines.','Locate retrieval, identity or evidence failures;<br/>retain supported results and revise the<br/>specific failing decision rule.'],
      ['Model / design','Establish reproducible structural evidence;<br/>test whether selected mutations improve<br/>measured outcomes versus matched WT.','Check assay / expression factors and setup;<br/>test alternative catalytic steps or mechanisms<br/>instead of equating geometric gain with activity.'],
      ['Learn','Select informative prospective panels;<br/>test whether measured feedback improves<br/>later decisions under matched resources.','Determine whether the panel was informative,<br/>the hypotheses were distinguishable, and<br/>the feedback changed a scientific decision.'],
    ],[80,195,195,195],29)
    save(d,'aim_success_failure_criteria','These are proposed evaluation criteria, not a claim that the thesis milestones or comparative activity gains are complete.')


def provenance():
    d,p=figure('Keep observations, interpretations, and decisions distinct',
               'CURRENT EVIDENCE PRINCIPLE · Source integrity and scientific correctness require different checks.')
    nodes=[('Raw source','Paper passage<br/>Database record<br/>Measurement / coordinate file'),
           ('Linked claim','Which candidate?<br/>Which reaction / substrate?<br/>Support and conflicts'),
           ('Hypothesis','Mechanistic explanation<br/>Assumptions and alternatives<br/>A predicted observation'),
           ('Decision','Choose an investigation<br/>State its expected information<br/>Record why it was selected'),
           ('Test outcome','Raw returned observation<br/>Comparison with prediction<br/>Retain / revise / suspend')]
    for i,(heading,body) in enumerate(nodes):
        A.card(p,80+i*360,290,320,360,heading,body,A.PALE,26)
        if i<4:A.arrow(p,(410+i*360,470),(430+i*360,470))
    A.rect(p,(80,730,1840,925),A.ROSE)
    A.txt(p,(110,755,1810,900),'<b>Interpretation boundaries</b><br/>Missing evidence does not establish inactivity. A failed computation does not refute a mechanism. A valid prediction does not establish catalysis.',33)
    save(d,'evidence_claim_hypothesis_decision','Current recordkeeping supports auditability; the source-to-claim interpretation remains a scientific judgment.','framework interpretation')


def strategies():
    d,p=figure('Design strategy depends on the evidence available',
               'CURRENT TOOL FAMILIES · Available interfaces do not establish that a proposed design is catalytically active.')
    A.table(p,80,230,[350,560,850],['Strategy','Required starting information','What the resulting proposal must still establish'],[
      ['Redesign','A scaffold and bound complex;<br/>pocket and mutation choices.','Matched-WT structural changes, physical validity,<br/>expression / stability and experimental activity.'],
      ['Motif-conditioned generation','A justified arrangement of catalytic<br/>atoms and ligand / intermediate.','Scaffold–motif compatibility and independent<br/>evaluation of the relevant chemical states.'],
      ['Ligand-conditioned generation','Declared ligands and molecular states;<br/>no fixed catalytic motif required.','Catalytic residue placement and mechanism;<br/>a complementary pocket alone is insufficient.'],
      ['Reaction-conditioned generation','Substrate / product transformation<br/>and a bounded design request.','Chemically consistent states, reactive geometry<br/>and experimental support for the intended reaction.'],
    ],[80,145,145,145,145],28)
    save(d,'design_strategy_input_requirements','Source: design-agent interfaces. Examples include saturation scanning, motif scaffolding, DISCO and GENzyme; this figure reports tool roles, not a benchmark.','available tool roles')


def execution():
    d,p=figure('Execution status and scientific status are separate axes',
               'CURRENT FRAMEWORK PRINCIPLE · A completed computation can leave a hypothesis unresolved or contradicted.')
    A.table(p,80,235,[390,650,720],['Situation','Execution record','Scientific interpretation'],[
      ['Missing source / identity','Evidence unavailable or identity unresolved.','A lead remains unresolved; missing evidence<br/>does not establish a negative biological result.'],
      ['Setup / pose check fails','Inputs or outputs fail the declared checks.','Revise setup or investigate failure;<br/>do not interpret the rejected geometry.'],
      ['Complete but weak geometry','Requested ensemble and checks completed.','The tested structural hypothesis has limited<br/>support; target activity remains unmeasured.'],
      ['Favorable geometry, low yield','Valid structural result and discordant<br/>experimental measurement.','Investigate assay context and competing<br/>mechanistic explanations.'],
      ['Resource allowance ends','Persist evidence, pending work and state.','Coverage can remain incomplete;<br/>checkpoint / resume is not scientific completion.'],
    ],[80,125,125,125,125,125],28)
    save(d,'execution_vs_scientific_status','Current software distinguishes execution, validity and interpretation. No new failure frequency or recovery rate is measured here.','framework interpretation')


def chemistry():
    d,p=figure('Activation and amide formation use different electrophiles',
               'CHEMICAL SCHEMATIC · Generic carboxyl-activating amide ligation; proton and charge accounting simplified.')
    A.card(p,80,240,380,255,'AMP-forming',
           'Carboxylate O attacks Pα.<br/><br/>Acyl-AMP + PPᵢ are formed.',A.ROSE,30)
    A.card(p,555,240,565,255,'Activated acyl intermediate',
           'R–C(=O)–O–AMP<br/><br/>The activated carbonyl C is the<br/>electrophile for amine attack.',size=30)
    A.card(p,1215,240,625,255,'Amide-forming step',
           'Neutral amine attacks acyl carbon.<br/><br/>R–C(=O)–NH–R′ + AMP',size=30)
    A.card(p,80,590,380,255,'ADP-forming',
           'Carboxylate O attacks Pγ.<br/><br/>Acyl-phosphate + ADP are formed.',A.ROSE,30)
    A.card(p,555,590,565,255,'Activated acyl intermediate',
           'R–C(=O)–O–phosphate<br/><br/>The activated carbonyl C is the<br/>electrophile for amine attack.',size=30)
    A.card(p,1215,590,625,255,'Amide-forming step',
           'Neutral amine attacks acyl carbon.<br/><br/>R–C(=O)–NH–R′ + Pᵢ',size=30)
    for y in [365,715]:A.arrow(p,(470,y),(545,y));A.arrow(p,(1130,y),(1205,y))
    save(d,'two_step_electrophile_and_leaving_group','Sources: Hollenhorst et al. (2009), doi:10.1021/bi9013165; IucA study, PMCID:PMC7325057. This is not a predicted trajectory.','chemical schematic')


def accounting():
    d,p=figure('ATP consumption and phosphate-bond accounting',
               'CONDITIONAL CHEMICAL ACCOUNTING · One simple carboxyl activation per amide-forming event.')
    A.table(p,80,250,[570,595,595],['Quantity','AMP-forming route','ADP-forming route'],[
      ['ATP molecules in the activation event','1','1'],
      ['ATP-derived products of ligation','AMP + PPᵢ','ADP + Pᵢ'],
      ['Conventional high-energy-bond equivalents','2 if PPᵢ is subsequently hydrolyzed','1 for ATP → ADP'],
      ['What still determines process cost','ATP regeneration, yield, enzyme production,<br/>conditions and downstream processing.','ATP regeneration, yield, enzyme production,<br/>conditions and downstream processing.'],
    ],[80,115,115,145,155],29)
    save(d,'ATP_accounting_scope','Chemical stoichiometry is not a process-economic or life-cycle assessment; complex multi-ATP pathways require separate accounting.','chemical accounting')


def endpoints():
    d,p=figure('What each reported quantity actually measures',
               'INTERPRETATION GUIDE · Keep the measured endpoint separate from structural and computational diagnostics.')
    A.table(p,80,235,[400,680,680],['Quantity','What is reported','What it does not establish by itself'],[
      ['Product concentration','Stored assay endpoint in µg/mL,<br/>with observed measurement variation.','An isolated catalytic half-reaction rate<br/>or turnover normalized by enzyme amount.'],
      ['Weighted angle A','Inverse-distance-weighted<br/>first-step attack angle across model poses.','Absolute proximity, a reaction barrier,<br/>or quantitative product yield.'],
      ['Harmonic distance dH','An ensemble distance summary;<br/>closer poses receive more weight.','A chemically productive orientation<br/>or a probability of catalysis.'],
      ['Interface confidence','Predictor confidence for the declared<br/>protein–ligand interface.','Measured binding affinity or activity;<br/>confidence is model-specific.'],
      ['Pose consistency','Repeatability of geometry across<br/>generated predictions.','A conformational equilibrium population<br/>or a kinetic persistence measurement.'],
    ],[80,125,125,125,125,125],28)
    save(d,'reported_quantity_interpretation','Definitions follow the saved analysis and model documentation; these quantities can support different, complementary scientific questions.','measurement interpretation')


def archived_cost():
    source=ROOT/'enzyme_discovery_agent/runs/imported_amide_discovery/RUN_SUMMARY.md'
    d,p=figure('Recorded cost and scope of the discovery case',
               'ARCHIVED RUN METADATA · One September 17 run; subsequent cofolding cost is not included here.')
    for i,(head,body) in enumerate([('49.0 min','Recorded wall-clock duration'),('50,151','Accounted model tokens'),
                                  ('12','Recorded planning iterations'),('63','Candidate evaluations')]):
        A.card(p,80+i*450,285,410,245,head,body,A.PALE,29)
    A.table(p,80,635,[480,1280],['Scope','Interpretation'],[
      ['Output accounting','43 representatives; 36 subsequently modeled, with 50 predictions each.'],
      ['Serving-model string','argo:claude-5-sonnet, exactly as stored in the archived metadata.'],
      ['Cost boundary','No dollar cost, external-tool cost, later GPU time or independent runtime benchmark is inferred.'],
    ],[65,75,75,105],27)
    save(d,'archived_discovery_cost_and_scope','Source: archived RUN_SUMMARY.md. Serving-model metadata is recorded provenance, not a current product-availability claim.','archived telemetry')
    return source


def puua_assumptions():
    d,p=figure('PuuA: separate native evidence from modeling assumptions',
               'CANDIDATE-SPECIFIC AUDIT · Two Mg ions in the prediction are not a general rule for ADP-forming enzymes.')
    A.card(p,80,245,790,385,'Native experimental evidence',
           'PuuA ligates glutamate and putrescine.<br/><br/>The characterized E. coli enzyme is dodecameric.<br/><br/>These observations concern the native enzyme<br/>and native substrate context.',A.PALE,31)
    A.card(p,1050,245,790,385,'Saved non-native prediction',
           'One PuuA chain + ATP + adipate.<br/><br/>Two Mg ions are present in the saved model.<br/><br/>The assembly, metal coordination and target<br/>activity require independent assessment.',A.ROSE,31)
    A.arrow(p,(880,435),(1040,435))
    A.txt(p,(110,730,1810,920),'<b>Questions to test:</b> Does the native oligomer contribute to the modeled active site? Are the metal sites and molecular states supported? Is the non-native adipate pose physically and mechanistically meaningful?',35)
    save(d,'PuuA_native_evidence_vs_model_assumptions','Native source: Kurihara et al. (2008), PMID:18495664. Saved model-0 composition is measured from its coordinate file.','candidate-specific evidence audit')


def main():
    HERE.mkdir(exist_ok=True)
    for f in [requirements,independence,multistate,experimental_panel,mismatch,methods,milestones,
              provenance,strategies,execution,chemistry,accounting,endpoints,puua_assumptions]:
        f();print('Completed '+f.__name__,flush=True)
    cost=archived_cost()
    (HERE/'method_figures.json').write_text(json.dumps({'figures':FIGURES,
        'source_artwork':str(SOURCE.relative_to(ROOT)),
        'source_artwork_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'archived_telemetry_source':str(cost.relative_to(ROOT)),
        'all_text_fits':all(r['spare_height']>=0 and r['scale']==1 for r in A.CHECKS)},indent=2)+'\n')
    print(f'Wrote {len(FIGURES)} independent method figures',flush=True)


if __name__=='__main__':main()
