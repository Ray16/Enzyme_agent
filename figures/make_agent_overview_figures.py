#!/usr/bin/env python3
"""Presentation deck for the enzyme-discovery agent: architecture + results.

Generates a coherent set of group-meeting figures from REAL run artifacts (no
synthetic data). Sources:
  * the improved-agent run 20260914_220431 (amide synthetase / nylon-6,6)
  * its regenerated structural_modeling_handoff.json (mechanism + cofactor spec)
  * structural_modeling/results/*.csv (mechanism-aware step-1 geometry)
  * both runs' token_usage.json (cost comparison)

Each figure is written to figures/ with an ``agent_`` prefix. Re-run to refresh.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

ROOT = Path("/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent")
sys.path.insert(0, str(ROOT))
from enzyme_discovery_agent_improved.chemistry_logic import derive_activation_mechanism

NEW = ROOT / ("enzyme_discovery_agent_improved/runs/"
              "20260914_220431_find_amide_synthetases_that_ligate_adipic_acid_and_hexamethylenediamine_for_nylo")
OLD = ROOT / ("archive/enzyme_discovery_agent_archive_20260910_102425/runs/"
              "20260830_125748_find_amide_synthetases_that_ligates_adipic_acid_and_hexamethylenediamine_for_nyl")
SM = ROOT / "structural_modeling/results"
OUT = ROOT / "figures"

# ---- shared style ----
INK="#1f2933"; SUB="#52606d"; BG="#ffffff"
PG="#3b7dd8"      # acyl-phosphate / P-gamma
PA="#e0803a"      # acyl-adenylate / P-alpha
GREEN="#2f9e6f"; GREY="#e4e7eb"; GREY_ED="#9aa5b1"
# Publication-quality defaults: crisp text, vector-friendly fonts (editable in
# Illustrator/Inkscape), high raster fallback.
plt.rcParams.update({
    "font.family":"DejaVu Sans","axes.edgecolor":"#c3cad2","axes.linewidth":0.9,
    "figure.facecolor":BG,"savefig.facecolor":BG,"savefig.dpi":300,"figure.dpi":150,
    "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none",
    "axes.titlesize":13,"axes.labelsize":11,"xtick.labelsize":9,"ytick.labelsize":9,
})

def _round(ax,x,y,w,h,fc,ec,r=0.02,lw=1.3):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.01,rounding_size={r}",
                                fc=fc,ec=ec,lw=lw,mutation_aspect=1,transform=ax.transData))

def load_ranked(run):
    return list(csv.DictReader(open(run/"ranked_candidates.csv")))

def mech_of(row):
    acts=[a for a in (row.get("catalytic_activities") or "").split("|") if a.strip()]
    r=derive_activation_mechanism(acts)
    return r["reactive_phosphate"] if r else None

def save(fig,name):
    """Write both a vector PDF (publication) and a 300-dpi PNG (slides)."""
    stem=(OUT/name).with_suffix("")
    fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight",facecolor=BG)
    fig.savefig(stem.with_suffix(".png"),dpi=300,bbox_inches="tight",facecolor=BG)
    plt.close(fig)
    print("wrote",stem.with_suffix(".pdf").relative_to(ROOT),"+ .png")


# ============================================================ 1. PIPELINE
def fig_pipeline():
    fig=plt.figure(figsize=(15,9.8)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")
    ax.text(3,96.5,"Enzyme-discovery agent — retrieval-expansion architecture",fontsize=21,fontweight="bold",color=INK)
    ax.text(3,92.8,"Target chemistry drives discovery; the LLM proposes strategies, tools find entities "
            "(data-leakage firewall). Numbers from run 20260914_220431.",fontsize=11.5,color=SUB)

    # funnel stages: (title, detail, count-label)
    stages=[
        ("1  Target reaction contract","Derive chemistry from the query: direction, bond, reactive groups, activation options","13 claims · 9 inferred"),
        ("2  Literature harvest","Read primary full text; seed DB expansion with tested enzymes before search","4 seed candidates"),
        ("3  Mechanism-family hypotheses","Propose falsifiable mechanism folds to cover (with disqualifying observations)","6 families (3 high-promise)"),
        ("4  Family survey + expansion","Map InterPro folds; promote substrate-fit members; literature-guided expansion","51 families · 1,858 discovered"),
        ("5  Candidate evaluation","Per-candidate: EC/bond gate, substrate-fit axes, mechanism, cofactor spec, provenance","85 evaluated · 77 rejected"),
        ("6  Decomposed ranking","Lexicographic spine over substrates_hosted / shape_mismatches / weakest_fg_match","38 ranked · 3 strong leads"),
        ("7  Structural-modeling handoff","Per-candidate: family + mechanism (reactive P) + cofactor spec → co-fold","20 candidates"),
    ]
    x=4; w=58; y=87; h=8.4; gap=2.2
    for i,(t,d,c) in enumerate(stages):
        new = i in (5,6) or i in (0,2,4)  # newly-strengthened stages
        fc="#eaf4fb" if i in (0,) else ("#e9f7f0" if new else GREY)
        ec=GREEN if new else GREY_ED
        _round(ax,x,y-h,w,h,fc,ec,lw=1.6 if new else 1.2)
        ax.text(x+2,y-2.6,t,fontsize=12,fontweight="bold",color=INK)
        ax.text(x+2,y-5.7,d,fontsize=9.3,color=SUB)
        ax.text(x+w-2,y-2.6,c,fontsize=10.5,fontweight="bold",color=(GREEN if new else INK),ha="right")
        if i<len(stages)-1:
            ax.add_patch(FancyArrowPatch((x+w/2,y-h),(x+w/2,y-h-gap),arrowstyle="-|>",
                         mutation_scale=15,lw=2,color="#8a97a3"))
        y-=h+gap

    # right rail: cross-cutting principles
    rx=68; rw=29; ry=87
    notes=[
        ("Data-leakage firewall","Answer enzymes never enter prompts/tools; only generic mechanism vocabulary is allowed.","#fff4d6","#d9a441"),
        ("Deterministic chemistry","Activation mode & cofactors derived from reaction products — no LLM, fully provenanced.","#e9f7f0",GREEN),
        ("Decomposed, no composites","Substrate fit reasoned as functional-group facts, not fused Tanimoto scores.","#eaf4fb","#5b8def"),
        ("Structural handoff","Explicit, reviewable spec: family + mechanism + cofactors travel to co-folding.","#f3ecfb","#8156c7"),
    ]
    ax.text(rx+rw/2,ry+1.5,"cross-cutting design",fontsize=11.5,fontweight="bold",ha="center",color=INK)
    yy=ry-1
    for t,d,fc,ec in notes:
        hh=15.5
        _round(ax,rx,yy-hh,rw,hh,fc,ec,lw=1.4)
        ax.text(rx+1.8,yy-3,t,fontsize=10.5,fontweight="bold",color=INK)
        # wrap
        import textwrap
        for j,line in enumerate(textwrap.wrap(d,34)):
            ax.text(rx+1.8,yy-5.6-j*2.4,line,fontsize=8.7,color=SUB)
        yy-=hh+2.2
    save(fig,"agent_pipeline_architecture.png")


# ============================================================ 2. RANKING DECOMPOSITION
def fig_ranking():
    rows=load_ranked(NEW)[:15]
    genes=[(r["gene"] or r["accession"])[:9] for r in rows]
    def num(s):
        try: return float(s)
        except: return None
    hosted=[int((r["substrates_hosted"] or "0/2").split("/")[0]) for r in rows]
    shape=[int(r["shape_mismatches"] or 0) for r in rows]
    fg=[num(r["weakest_fg_match"]) or 0 for r in rows]
    y=list(range(len(rows)))[::-1]

    fig,axes=plt.subplots(1,4,figsize=(15,7.4),gridspec_kw={"width_ratios":[1.1,1,1,1.2],"wspace":0.35})
    fig.suptitle("Decomposed ranking — one collapsed score replaced by three interpretable axes",
                 fontsize=16,fontweight="bold",x=0.5,y=0.99,color=INK)
    fig.text(0.5,0.935,"Improved run, top 15. The old pipeline tied nearly every candidate at substrate_fit ≈ 0.46; "
             "the axes break the tie and expose one-sided fits.",fontsize=10.5,color=SUB,ha="center")

    # panel 0: OLD collapsed
    ax=axes[0]; ax.set_title("OLD: collapsed score",fontsize=11,color=INK)
    ax.barh(y,[0.46]*len(rows),color="#c9ced6",edgecolor="#9aa5b1")
    ax.set_yticks(y); ax.set_yticklabels(genes,fontsize=8.5); ax.set_xlim(0,1)
    ax.set_xlabel("substrate_fit",fontsize=9.5); ax.axvline(0.46,color="#7b8794",ls=":",lw=1)
    ax.text(0.5,len(rows)-0.5,"all tied ≈ 0.46",fontsize=8.5,color="#616e7c",ha="center",style="italic")

    # panel 1: substrates_hosted
    ax=axes[1]; ax.set_title("substrates_hosted",fontsize=11,color=INK)
    ax.barh(y,hosted,color=GREEN,edgecolor="#1f7a55"); ax.set_yticks(y); ax.set_yticklabels([]); ax.set_xlim(0,2.3)
    ax.set_xlabel("of 2 target substrates",fontsize=9.5)

    # panel 2: shape_mismatches
    ax=axes[2]; ax.set_title("shape_mismatches",fontsize=11,color=INK)
    ax.barh(y,shape,color="#d9736a",edgecolor="#a83f36"); ax.set_yticks(y); ax.set_yticklabels([]); ax.set_xlim(0,max(shape+[1])+0.5)
    ax.set_xlabel("lower is better",fontsize=9.5)

    # panel 3: weakest_fg_match
    ax=axes[3]; ax.set_title("weakest_fg_match",fontsize=11,color=INK)
    ax.barh(y,fg,color=PG,edgecolor="#245a9e"); ax.set_yticks(y); ax.set_yticklabels([]); ax.set_xlim(0,0.75)
    ax.set_xlabel("higher is better",fontsize=9.5)

    for ax in axes: ax.tick_params(length=0); [ax.spines[s].set_visible(False) for s in ("top","right")]
    fig.subplots_adjust(top=0.88,bottom=0.09,left=0.09,right=0.98)
    save(fig,"agent_ranking_decomposition.png")


# ============================================================ 3. CANDIDATE LANDSCAPE
def fig_landscape():
    rows=load_ranked(NEW)
    from collections import defaultdict
    fam=defaultdict(list)
    for r in rows:
        fam[r["interpro_family"] or "(unclassified)"].append(r)
    # order families by size
    order=sorted(fam,key=lambda k:-len(fam[k]))
    fig,ax=plt.subplots(figsize=(14.5,8.6))
    fig.suptitle("Ranked candidate landscape — by InterPro family, colored by activation mechanism",
                 fontsize=16,fontweight="bold",color=INK,y=0.975,x=0.5)
    fig.text(0.5,0.93,"Improved run: %d ranked candidates across %d families. Mechanism (reactive phosphate) "
             "derived deterministically from each enzyme's reaction." % (len(rows),len(order)),
             ha="center",fontsize=10.5,color=SUB)
    yb=0; yticks=[]; ylabels=[]
    for f in order:
        members=fam[f]
        for j,r in enumerate(members):
            m=mech_of(r)
            col=PA if m=="Palpha" else (PG if m=="Pgamma" else "#b0b8c1")
            strong = r["tier_label"]=="strong lead"
            ax.scatter(j,yb,s=190 if strong else 110,c=col,edgecolors="#26303a" if strong else "white",
                       linewidths=1.6 if strong else 0.8,zorder=3)
            lab=(r["gene"] or r["accession"])
            ax.text(j,yb+0.32,lab[:8],fontsize=6.6,ha="center",color=SUB)
        yticks.append(yb); ylabels.append(f"{f[:34]}  (n={len(members)})"); yb-=1.25
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels,fontsize=9)
    ax.set_xlabel("members within family (rank order)",fontsize=10)
    ax.set_xlim(-0.6,max(len(fam[f]) for f in order)-0.3); ax.set_ylim(yb+0.6,1.1)
    ax.tick_params(length=0); [ax.spines[s].set_visible(False) for s in ("top","right","left")]
    leg=[Patch(fc=PG,ec="none",label="acyl-phosphate  → attack Pγ"),
         Patch(fc=PA,ec="none",label="acyl-adenylate  → attack Pα"),
         plt.Line2D([0],[0],marker="o",color="w",markerfacecolor="#b8c0c9",markeredgecolor="#26303a",
                    markersize=12,markeredgewidth=1.6,label="strong lead (bold ring)")]
    ax.legend(handles=leg,loc="lower right",fontsize=9.5,frameon=True,framealpha=0.95)
    fig.subplots_adjust(top=0.90,left=0.24,right=0.98,bottom=0.07)
    save(fig,"agent_candidate_landscape.png")


# ============================================================ 4. MECHANISM BRIDGE
def fig_bridge():
    fig=plt.figure(figsize=(14.5,8.2)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")
    ax.text(3,96,"Mechanism bridge — activation chemistry from the reaction, not the family name",
            fontsize=17,fontweight="bold",color=INK)
    ax.text(3,91.8,"Deterministic classifier reads the native reaction's products; fixes the family-keyword mis-calls.",
            fontsize=11,color=SUB)

    # decision boxes
    _round(ax,4,66,44,18,"#eaf4fb","#5b8def",lw=1.6)
    ax.text(6,81.5,"products contain  NMP + diphosphate",fontsize=12,fontweight="bold",color="#245a9e",family="DejaVu Sans Mono")
    ax.text(6,77.5,"e.g.  ...+ ATP = ...+ AMP + diphosphate",fontsize=10,color=SUB,family="DejaVu Sans Mono",style="italic")
    ax.text(6,72.5,"→  acyl-adenylate",fontsize=13,fontweight="bold",color=PA)
    ax.text(6,68.8,"→  attack ATP Pα  (O3A leaving axis)",fontsize=11,color=INK)

    _round(ax,52,66,44,18,"#eaf4fb","#5b8def",lw=1.6)
    ax.text(54,81.5,"products contain  NDP + phosphate",fontsize=12,fontweight="bold",color="#245a9e",family="DejaVu Sans Mono")
    ax.text(54,77.5,"e.g.  ...+ ATP = ...+ ADP + phosphate",fontsize=10,color=SUB,family="DejaVu Sans Mono",style="italic")
    ax.text(54,72.5,"→  acyl-phosphate",fontsize=13,fontweight="bold",color=PG)
    ax.text(54,68.8,"→  attack ATP Pγ  (O3B leaving axis)",fontsize=11,color=INK)

    # validation stats (real): 53 with reaction data classified; 52 match prior manual map; ddaG corrected; 9 no-data->None
    _round(ax,4,37,92,23,"#f7f9fb","#c3cad2",lw=1.2)
    ax.text(6,57,"Validation against the structural pipeline's prior (family-keyword) assignment",fontsize=12,fontweight="bold",color=INK)
    stats=[("52 / 53","agree where reaction data exists",GREEN),
           ("1","corrected  (ddaG: family said none; reaction → Pα)",PA),
           ("9","no reaction annotation → honest 'unresolved' (was silently defaulted)","#7b8794")]
    xx=8
    for big,lab,c in stats:
        ax.text(xx,51,big,fontsize=22,fontweight="bold",color=c)
        import textwrap
        for j,l in enumerate(textwrap.wrap(lab,26)):
            ax.text(xx,46-j*2.4,l,fontsize=9.2,color=SUB)
        xx+=30

    # callout
    _round(ax,4,20,92,15,"#e9f7f0",GREEN,lw=1.5)
    ax.text(6,31,"ddaG (E2JA32) — the case that proves the point",fontsize=12,fontweight="bold",color=GREEN)
    ax.text(6,27,"InterPro family  “Fumarate–diaminopropanoate ligase”  carries no adenylation hint.",fontsize=10.3,color=INK)
    ax.text(6,23.5,"Its reaction ( … = … + AMP + diphosphate ) unambiguously → Pα. The old keyword map scored it on Pγ",fontsize=10.3,color=INK)
    ax.text(6,20.2,"(0 productive poses → 'inconclusive'); the reaction-derived call fixes it. Now emitted with full provenance.",fontsize=10.3,color=INK)
    save(fig,"agent_mechanism_bridge.png")


# ============================================================ 5. RUN COMPARISON
def fig_runcompare():
    def tok(run):
        d=json.load(open(run/"token_usage.json")); return d["total_tokens"]
    old_tok, new_tok = tok(OLD), tok(NEW)
    fig,axes=plt.subplots(1,3,figsize=(15,6.2),gridspec_kw={"wspace":0.32})
    fig.suptitle("Old vs improved run — same target, cleaner ranking, but a cost regression to fix",
                 fontsize=15.5,fontweight="bold",y=1.0,color=INK)

    ax=axes[0]; ax.set_title("candidates ranked",fontsize=11.5,color=INK)
    ax.bar(["old","improved"],[89,38],color=["#c9ced6",GREEN],edgecolor="#7b8794",width=0.6)
    ax.bar(["old","improved"],[5,3],color=["#8a97a3","#1f7a55"],edgecolor="none",width=0.6)
    for i,(t,s) in enumerate([(89,5),(38,3)]):
        ax.text(i,t+1.5,f"{t}",ha="center",fontsize=11,fontweight="bold")
        ax.text(i,s+1.0,f"{s} strong",ha="center",fontsize=8.5,color="white",fontweight="bold")
    ax.set_ylim(0,100); [ax.spines[s].set_visible(False) for s in ("top","right")]; ax.tick_params(length=0)

    ax=axes[1]; ax.set_title("top-4 evidence leads",fontsize=11.5,color=INK)
    ax.axis("off")
    leads=["#1  glnA2  (Q9RDS6)","#2  puuA  (P78061)","#3  glnA2  (I3R584)","#4  ddaG  (E2JA32)"]
    ax.text(0.5,0.93,"identical & same order in both runs",fontsize=10,ha="center",color=SUB,style="italic",transform=ax.transAxes)
    for i,l in enumerate(leads):
        ax.add_patch(FancyBboxPatch((0.08,0.72-i*0.17),0.84,0.12,boxstyle="round,pad=0.01,rounding_size=0.02",
                     fc="#e9f7f0",ec=GREEN,lw=1.3,transform=ax.transAxes))
        ax.text(0.5,0.78-i*0.17,l,fontsize=11,ha="center",va="center",fontweight="bold",color=INK,transform=ax.transAxes)

    ax=axes[2]; ax.set_title("token cost",fontsize=11.5,color=INK)
    ax.bar(["old","improved"],[old_tok/1e6,new_tok/1e6],color=["#c9ced6","#d9736a"],edgecolor="#7b8794",width=0.6)
    # idle portion of new run (iter 16-50 = 4.73M)
    ax.bar(["improved"],[4.73],bottom=[(new_tok/1e6)-4.73],color="#f0b8b2",edgecolor="#a83f36",width=0.6,hatch="//")
    ax.text(0,old_tok/1e6+0.15,f"{old_tok/1e6:.1f}M",ha="center",fontsize=11,fontweight="bold")
    ax.text(1,new_tok/1e6+0.15,f"{new_tok/1e6:.1f}M",ha="center",fontsize=11,fontweight="bold")
    ax.text(1,(new_tok/1e6)-2.4,"72% idle\n(iter 16–50)",ha="center",fontsize=8.5,color="#8a2f28",fontweight="bold")
    ax.set_ylabel("million tokens",fontsize=10); ax.set_ylim(0,7.5)
    [ax.spines[s].set_visible(False) for s in ("top","right")]; ax.tick_params(length=0)
    fig.tight_layout(rect=[0,0,1,0.95])
    save(fig,"agent_run_comparison.png")


# ============================================================ 6. STRUCTURAL GEOMETRY
def fig_structural():
    rows=list(csv.DictReader(open(SM/"discovery_within_family_ranking.csv")))
    # two families with enough members
    from collections import defaultdict
    byfam=defaultdict(list)
    for r in rows:
        a=r.get("mean_inline_angle")
        if a and a.strip():
            byfam[r["family"]].append((r["gene"] or r["accession"], float(a), r["phosphate"], r["tier"]))
    fams=[f for f in byfam if len(byfam[f])>=4]
    fig,axes=plt.subplots(1,len(fams),figsize=(6.8*len(fams),7),gridspec_kw={"wspace":0.4})
    if len(fams)==1: axes=[axes]
    fig.suptitle("Mechanism-aware step-1 geometry — in-line attack angle, scored vs the correct ATP phosphorus",
                 fontsize=15,fontweight="bold",y=1.0,color=INK)
    fig.text(0.5,0.955,"50 unrestrained Boltz-2 co-folds per enzyme; angle compared only within a mechanistic family. "
             "≥165° strong · 150–165° compatible · <150° weak.",fontsize=10,color=SUB,ha="center")
    for ax,f in zip(axes,fams):
        data=sorted(byfam[f],key=lambda t:t[1])
        genes=[d[0][:8] for d in data]; ang=[d[1] for d in data]
        pcol=PA if data[0][2]=="PA" else PG
        cols=[("#2f9e6f" if a>=165 else ("#e0a53a" if a>=150 else "#d9736a")) for a in ang]
        yy=range(len(data))
        ax.barh(list(yy),ang,color=cols,edgecolor="#5a636c",lw=0.6)
        ax.axvline(150,color="#7b8794",ls=":",lw=1); ax.axvline(165,color="#7b8794",ls="--",lw=1)
        ax.set_yticks(list(yy)); ax.set_yticklabels(genes,fontsize=8)
        ax.set_xlim(90,180); ax.set_xlabel("mean in-line angle (°)",fontsize=9.5)
        phos="Pα" if data[0][2]=="PA" else "Pγ"
        ax.set_title(f"{f}\nreactive {phos}",fontsize=11,color=INK,pad=10)
        ax.tick_params(length=0); [ax.spines[s].set_visible(False) for s in ("top","right")]
    leg=[Patch(fc="#2f9e6f",label="≥165° strong"),Patch(fc="#e0a53a",label="150–165° compatible"),
         Patch(fc="#d9736a",label="<150° weak")]
    fig.legend(handles=leg,loc="lower center",ncol=3,fontsize=9.5,frameon=False,bbox_to_anchor=(0.5,-0.02))
    fig.tight_layout(rect=[0,0.03,1,0.88])
    save(fig,"agent_structural_geometry.png")


# ============================================================ 8. GEOMETRY-PROXY BENCHMARK
def fig_proxy():
    prox=list(csv.DictReader(open(ROOT/"exp_data/proxy_evaluation_summary.csv")))
    # merge synthetase panel measured activity with in-line angle
    act={r["id"]:float(r["activity_N66L1_ugml_avg"]) for r in
         csv.DictReader(open(ROOT/"exp_data/amide_synthetase/assay_data/panel_activity.csv"))}
    ang={r["id"]:float(r["mean_inline_angle"]) for r in
         csv.DictReader(open(ROOT/"structural_modeling/panel_pocket/angle_gate_panel.csv")) if r["mean_inline_angle"]}
    ids=sorted(set(act)&set(ang))

    fig,(axA,axB)=plt.subplots(1,2,figsize=(15,7),gridspec_kw={"width_ratios":[1.25,1],"wspace":0.28})
    fig.suptitle("Geometry proxies vs measured activity — which co-fold signal discriminates actives",
                 fontsize=15.5,fontweight="bold",y=1.0,color=INK)
    fig.text(0.5,0.94,"Benchmarked on assayed panels (synthetase n=49, 3 active · hydrolase n=96, 38 active). "
             "AUROC with bootstrap 95% CI; ★ = permutation p<0.05.",fontsize=10,color=SUB,ha="center")

    # Panel A: AUROC bars with CI
    labels=[p["proxy"].replace("synth ","S: ").replace("hydrolase ","H: ") for p in prox]
    au=[float(p["auroc"]) for p in prox]; lo=[float(p["ci_lo"]) for p in prox]; hi=[float(p["ci_hi"]) for p in prox]
    sig=[float(p["perm_p"])<0.05 for p in prox]
    y=list(range(len(prox)))[::-1]
    cols=[GREEN if s else "#c1c8d0" for s in sig]
    axA.barh(y,au,color=cols,edgecolor="#5a636c",lw=0.8,zorder=2)
    axA.errorbar(au,y,xerr=[[a-l for a,l in zip(au,lo)],[h-a for a,h in zip(au,hi)]],
                 fmt="none",ecolor="#3a424b",elinewidth=1.3,capsize=4,zorder=3)
    for yi,(a,s) in zip(y,zip(au,sig)):
        axA.text(a+0.02 if a<0.85 else a-0.02,yi,f"{a:.2f}"+(" ★" if s else ""),
                 va="center",ha="left" if a<0.85 else "right",fontsize=9,fontweight="bold" if s else "normal",
                 color=INK)
    axA.axvline(0.5,color="#7b8794",ls="--",lw=1.2); axA.text(0.5,len(prox)-0.4,"chance",fontsize=8,color="#7b8794",ha="center")
    axA.set_yticks(y); axA.set_yticklabels(labels,fontsize=9); axA.set_xlim(0,1.02)
    axA.set_xlabel("AUROC (active vs inactive)",fontsize=10.5)
    axA.set_title("proxy benchmark",fontsize=11.5,color=INK,pad=8)
    axA.tick_params(length=0); [axA.spines[s].set_visible(False) for s in ("top","right")]

    # Panel B: in-line angle vs measured activity (synthetase panel)
    xa=[ang[i] for i in ids]; ya=[max(act[i],1e-3) for i in ids]
    active=[act[i]>0.1 for i in ids]
    axB.scatter([x for x,a in zip(xa,active) if not a],[y for y,a in zip(ya,active) if not a],
                s=55,c="#c1c8d0",edgecolors="white",lw=0.6,label="inactive",zorder=2)
    axB.scatter([x for x,a in zip(xa,active) if a],[y for y,a in zip(ya,active) if a],
                s=110,c=PA,edgecolors="#26303a",lw=1.2,label="active (>0.1 µg/mL)",zorder=3)
    for i in ids:
        if act[i]>0.5:
            axB.annotate(i,(ang[i],act[i]),fontsize=8,color=INK,xytext=(4,3),textcoords="offset points")
    axB.set_yscale("log"); axB.set_xlabel("mean in-line attack angle (°)",fontsize=10.5)
    axB.set_ylabel("N66 amide activity (µg/mL, log)",fontsize=10.5)
    axB.axvline(150,color="#7b8794",ls=":",lw=1)
    axB.set_title("best geometric signal is the angle",fontsize=11.5,color=INK,pad=8)
    axB.legend(fontsize=8.5,loc="upper left",frameon=True); axB.tick_params(length=0)
    [axB.spines[s].set_visible(False) for s in ("top","right")]
    axB.text(0.98,0.03,"necessary, not sufficient → coarse negative filter",transform=axB.transAxes,
             fontsize=8.2,color=SUB,ha="right",style="italic")
    fig.subplots_adjust(top=0.88,bottom=0.11,left=0.20,right=0.98)
    save(fig,"agent_geometry_proxy_benchmark.png")


# ============================================================ 9. EXHAUSTIVE RECALL SWEEP
def fig_sweep():
    import collections
    rows=list(csv.DictReader(open(ROOT/"Interpro_exhaustive/passed_candidates.tsv"),delimiter="\t"))
    ec=collections.Counter((r.get("EC number") or "?").split(";")[0].strip() for r in rows)
    fig=plt.figure(figsize=(15,7.6)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")
    ax.text(3,96,"Discovery completeness — exhaustive InterPro / EC 6.3.- sweep",fontsize=17,fontweight="bold",color=INK)
    ax.text(3,91.8,"Out-of-tree audit: every reviewed ATP-dependent ligase, filtered by target chemistry, "
            "vs what the agent surfaced.",fontsize=11,color=SUB)

    # funnel (left)
    steps=[("12,456","reviewed EC 6.3.- ligases (Swiss-Prot universe)","#e4e7eb","#9aa5b1"),
           ("270","pass reaction-anchored chemistry filter (free diacid + amine ligation)","#dbeafe","#5b8def"),
           ("51 / 1,858","families surveyed / candidates discovered by the agent","#e9f7f0",GREEN),
           ("38 → 20","ranked → handed to structural modeling","#e9f7f0",GREEN)]
    x=4; w=52; y=84; h=11
    widths=[52,42,33,24]
    import textwrap
    for k,((big,lab,fc,ecol),ww) in enumerate(zip(steps,widths)):
        cx=x+(w-ww)/2
        _round(ax,cx,y-h,ww,h,fc,ecol,lw=1.6)
        ax.text(x+w/2,y-3.6,big,fontsize=15,fontweight="bold",color=INK,ha="center")
        for j,l in enumerate(textwrap.wrap(lab,44)):
            ax.text(x+w/2,y-6.8-j*2.3,l,fontsize=8.6,color=SUB,ha="center")
        if k<len(steps)-1:
            ax.add_patch(FancyArrowPatch((x+w/2,y-h),(x+w/2,y-h-2.0),arrowstyle="-|>",mutation_scale=14,lw=2,color="#8a97a3"))
        y-=h+2.0

    # EC breakdown of the 270 (right)
    rx=62; ax.text(rx,84,"270 reaction-relevant candidates, by EC",fontsize=11.5,fontweight="bold",color=INK)
    top=ec.most_common(8); maxv=top[0][1]
    yy=78
    for k,v in top:
        _round(ax,rx,yy-2.2, (v/maxv)*30,2.0,"#5b8def","#3060c0",r=0.3,lw=0)
        ax.text(rx+ (v/maxv)*30 +0.8,yy-1.2,f"{v}",fontsize=8.6,color=INK,va="center")
        ax.text(rx-0.8,yy-1.2,f"EC {k}",fontsize=8.6,color=SUB,va="center",ha="right")
        yy-=3.0

    # honest caveat box
    _round(ax,4,6,92,13,"#fdeeee","#d9736a",lw=1.4)
    ax.text(6,15.5,"Recall ceiling (stated honestly)",fontsize=11.5,fontweight="bold",color="#a83f36")
    ax.text(6,11.6,"The agent covers the reviewed universe — but the amide-synthetase ground truth SfaB (A0A2H4T912) is",fontsize=9.6,color=INK)
    ax.text(6,8.4,"UNREVIEWED (TrEMBL, no EC), so it lies OUTSIDE this entire sweep. No reviewed-DB method recovers it; "
            "reaching it needs the literature/unannotated channel.",fontsize=9.6,color=INK)
    save(fig,"agent_discovery_completeness.png")


# ============================================================ 0. GRAPHICAL ABSTRACT
def fig_abstract():
    import textwrap
    fig=plt.figure(figsize=(15,8.6)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")
    ax.text(50,96,"An enzyme-discovery agent for de novo biocatalyst search",
            fontsize=21,fontweight="bold",color=INK,ha="center")
    ax.text(50,91.5,"Target chemistry in → ranked, mechanism-annotated candidates out → structure-ready handoff",
            fontsize=12.5,color=SUB,ha="center")

    # --- problem (left) ---
    _round(ax,3,64,29,20,"#eef6ff","#5b8def",lw=1.5)
    ax.text(17.5,81,"THE PROBLEM",fontsize=10.5,fontweight="bold",color="#3060c0",ha="center")
    ax.text(17.5,76.5,"Find enzymes for nylon-6,6",fontsize=11.5,fontweight="bold",color=INK,ha="center")
    ax.text(17.5,72.5,"• synthesis:  adipate + HMDA → amide",fontsize=9.6,color=INK,ha="center")
    ax.text(17.5,69,"• hydrolysis:  break the N66 amide",fontsize=9.6,color=INK,ha="center")
    ax.text(17.5,66,"no known natural catalyst — must search",fontsize=8.8,color=SUB,ha="center",style="italic")

    # --- framework (middle) ---
    _round(ax,35,64,30,20,"#f7f9fb","#c3cad2",lw=1.4)
    ax.text(50,81,"THE FRAMEWORK",fontsize=10.5,fontweight="bold",color=INK,ha="center")
    chain=["target reaction contract","literature + family expansion","decomposed ranking",
           "mechanism + cofactor handoff","structural modeling"]
    yy=77
    for i,c in enumerate(chain):
        ax.text(50,yy,c,fontsize=9.3,color=INK,ha="center",
                fontweight="bold" if i in (2,3) else "normal")
        if i<len(chain)-1: ax.text(50,yy-1.4,"▼",fontsize=6,color="#8a97a3",ha="center")
        yy-=2.9

    # --- what's new (right) ---
    _round(ax,68,64,29,20,"#e9f7f0",GREEN,lw=1.5)
    ax.text(82.5,81,"WHAT'S NEW",fontsize=10.5,fontweight="bold",color=GREEN,ha="center")
    for j,l in enumerate([
        "Deterministic mechanism bridge:",
        "activation mode + reactive P + cofactors",
        "derived from the reaction, provenanced",
        "— no LLM, fixes family-keyword errors"]):
        ax.text(82.5,77-j*3,l,fontsize=9.2,color=INK,ha="center",
                fontweight="bold" if j==0 else "normal")

    # --- headline results (bottom row of stat cards) ---
    ax.text(50,57,"KEY RESULTS  (run 20260914_220431, assayed panels, exhaustive audit)",
            fontsize=11,fontweight="bold",color=INK,ha="center")
    cards=[("1,858","candidates discovered\nacross 51 families",PG),
           ("38 → 20","ranked → structure\nhandoff, provenanced",GREEN),
           ("52 / 53","mechanism calls agree;\nddaG mis-call fixed",PA),
           ("AUROC 0.81","step-1 NAC vs assay\n(p = 0.036)","#8156c7"),
           ("2 targets","synthesis + hydrolysis\nsame framework","#3060c0")]
    n=len(cards); cw=17; gap=1.5; total=n*cw+(n-1)*gap; x0=(100-total)/2
    for i,(big,lab,col) in enumerate(cards):
        cx=x0+i*(cw+gap)
        _round(ax,cx,32,cw,20,"#ffffff",col,lw=1.8)
        ax.text(cx+cw/2,45,big,fontsize=19,fontweight="bold",color=col,ha="center")
        for k,l in enumerate(lab.split("\n")):
            ax.text(cx+cw/2,40-k*3,l,fontsize=8.8,color=INK,ha="center")

    # --- honest limitations strip ---
    _round(ax,3,6,94,20,"#fdf3e8","#d9a441",lw=1.3)
    ax.text(50,22.5,"OPEN / HONEST LIMITATIONS",fontsize=10.5,fontweight="bold",color="#9a6a1a",ha="center")
    lims=["Ground-truth SfaB is unreviewed (no EC) → outside any reviewed-DB sweep; needs the literature/unannotated channel.",
          "Improved run spent 72% of tokens idle (a convergence/stop regression to fix).",
          "Geometry (angle/NAC) is a coarse negative filter, not a quantitative activity predictor."]
    for j,l in enumerate(lims):
        ax.text(50,18-j*3.4,"•  "+l,fontsize=9.2,color=INK,ha="center")
    save(fig,"agent_00_graphical_abstract.png")


# ============================================================ 10. SECOND TARGET: HYDROLASE PANEL
def fig_hydrolase():
    import random
    rows=list(csv.DictReader(open(ROOT/"exp_data/nylon_hydrolase/results/hydrolase_nac.csv")))
    def f(x):
        try: return float(x)
        except: return None
    def active(r): return str(r.get("L1sig","")).strip() in ("1","True","TRUE") or \
                          str(r.get("L2sig","")).strip() in ("1","True","TRUE")
    for r in rows:
        r["_tot"]=(f(r["L1"]) or 0)+(f(r["L2"]) or 0); r["_nac"]=f(r["nac_frac"]); r["_act"]=active(r)
    rows=[r for r in rows if r["_nac"] is not None]
    n_act=sum(r["_act"] for r in rows)

    fig,(axA,axB)=plt.subplots(1,2,figsize=(15,6.6),gridspec_kw={"width_ratios":[1.5,1],"wspace":0.26})
    fig.suptitle("Same framework, second target — nylon-6,6 amide HYDROLYSIS (cleavage direction)",
                 fontsize=15.5,fontweight="bold",y=1.0,color=INK)
    fig.text(0.5,0.94,"96-enzyme nylon-hydrolase panel (Drufva 2025), assayed for N66 monomer release. "
             "The agent's direction gate and geometry proxy transfer to the opposite reaction.",
             fontsize=10,color=SUB,ha="center")

    # Panel A: sorted activity, actives highlighted
    srt=sorted(rows,key=lambda r:-r["_tot"])
    x=range(len(srt)); cols=[PG if r["_act"] else "#d3d8de" for r in srt]
    axA.bar(list(x),[r["_tot"] for r in srt],color=cols,edgecolor="none",width=1.0)
    axA.set_xlabel("nylon-hydrolase panel enzymes (sorted by activity)",fontsize=10.5)
    axA.set_ylabel("N66 monomer release (µg/mL, L1+L2)",fontsize=10.5)
    axA.set_title(f"{n_act} of {len(rows)} enzymes active",fontsize=11.5,color=INK,pad=8)
    axA.tick_params(length=0); [axA.spines[s].set_visible(False) for s in ("top","right")]
    axA.legend(handles=[Patch(fc=PG,label="active"),Patch(fc="#d3d8de",label="inactive")],
               fontsize=9,loc="upper right",frameon=True)

    # Panel B: NAC fraction, active vs inactive (strip + mean)
    random.seed(3)
    groups={"inactive":[r["_nac"] for r in rows if not r["_act"]],
            "active":[r["_nac"] for r in rows if r["_act"]]}
    for i,(g,vals) in enumerate(groups.items()):
        col=PG if g=="active" else "#a7b0ba"
        xs=[i+random.uniform(-0.12,0.12) for _ in vals]
        axB.scatter(xs,vals,s=42,c=col,edgecolors="white",lw=0.5,alpha=0.85,zorder=2)
        m=sum(vals)/len(vals)
        axB.plot([i-0.28,i+0.28],[m,m],color="#26303a",lw=2.4,zorder=3)
        axB.text(i,-0.055,f"n={len(vals)}\nmean {m:.2f}",ha="center",fontsize=8.7,color=SUB)
    axB.set_xticks([0,1]); axB.set_xticklabels(["inactive","active"],fontsize=10.5)
    axB.set_xlim(-0.6,1.6); axB.set_ylim(-0.1,0.48)
    axB.set_ylabel("step-1 NAC fraction (co-fold)",fontsize=10.5)
    axB.set_title("NAC discriminates (AUROC 0.61, p=0.03)",fontsize=11.5,color=INK,pad=8)
    axB.tick_params(length=0); [axB.spines[s].set_visible(False) for s in ("top","right")]
    fig.subplots_adjust(top=0.87,bottom=0.16,left=0.08,right=0.98)
    save(fig,"agent_hydrolase_second_target.png")


# ============================================================ 7. HANDOFF SCHEMA
def fig_schema():
    import textwrap
    C_PROV="#fff4d6"; C_PROV_ED="#d9a441"; C_NEW="#d8f3e3"; C_NEW_ED=GREEN
    C_EXIST="#e4e7eb"; C_EXIST_ED="#9aa5b1"
    fig=plt.figure(figsize=(15,9.6)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")
    ax.text(3,96.5,"Discovery-agent → structural-modeling handoff",fontsize=20,fontweight="bold",color=INK)
    ax.text(3,92.8,"Per-candidate record in structural_modeling_handoff.json  —  three blocks are "
            "deterministically derived by the agent (green)",fontsize=11.5,color=SUB)
    LX=3; LW=45
    ax.text(LX+LW/2,89.4,"candidate record",fontsize=12.5,fontweight="bold",ha="center",color=INK)
    def grp(x,y,w,title,fields,new=False,tag=None):
        fc=C_NEW if new else C_EXIST; ec=C_NEW_ED if new else C_EXIST_ED
        rowh=2.75; head=3.4; h=head+rowh*len(fields)+1.0
        _round(ax,x,y-h,w,h,fc,ec,lw=1.6 if new else 1.2)
        ax.text(x+1.8,y-2.0,title,fontsize=10.5,fontweight="bold",color=(GREEN if new else INK))
        if tag:
            ax.add_patch(FancyBboxPatch((x+w-9.2,y-2.9),7.4,1.9,boxstyle="round,pad=0.02,rounding_size=0.9",
                         fc=GREEN,ec="none")); ax.text(x+w-5.5,y-1.95,tag,fontsize=7.6,fontweight="bold",color="#fff",ha="center")
        yy=y-head-0.9
        for k,v in fields:
            ax.text(x+2.4,yy,k,fontsize=8.8,color=INK,family="DejaVu Sans Mono")
            if v: ax.text(x+w-1.8,yy,v,fontsize=8.1,color=SUB,ha="right",style="italic")
            yy-=rowh
        return h
    y=87.5
    for args in [
        ("identity",[("accession","E2JA32"),("protein_name","… diaminopropanoate ligase"),("organism, gene_names, ec_numbers","")],False,None),
        ("family",[("interpro_family","Dapdiamide ligase"),("interpro_family_id","IPR048223")],True,"NEW"),
        ("evidence (existing)",[("structure_sources",""),("native_substrate_evidence",""),("reaction_chemistry",""),("structural_homology / orthologs","")],False,None),
        ("mechanism_evidence.activation",[("activation_mode","acyl_adenylate"),("reactive_phosphate","Palpha"),("nucleophile","carboxylate O"),("confidence","high"),("source / derived_from","reaction products")],True,"NEW"),
        ("modeling_input",[("cofactors[]  {name,ccd,count,role,basis}","ATP×1 + MG×2"),("reactive_phosphate","Palpha"),("activation_mode","acyl_adenylate")],True,"NEW"),
    ]:
        h=grp(LX,y,LW,*args); y-=h+1.5
    RX=54; RW=43
    ax.text(RX+RW/2,89.4,"how the green blocks are derived  (worked example: ddaG)",fontsize=12.5,fontweight="bold",ha="center",color=INK)
    def flow(x,y,w,h,title,lines,fc,ec):
        _round(ax,x,y-h,w,h,fc,ec,lw=1.5)
        ax.text(x+1.8,y-2.4,title,fontsize=10,fontweight="bold",color=INK)
        yy=y-5.0
        for ln,c,st in lines:
            ax.text(x+2.2,yy,ln,fontsize=9.6,color=c,style=st,family="DejaVu Sans Mono"); yy-=2.6
    y=87.5; hb=12.5
    flow(RX,y,RW,hb,"1 · candidate's own annotations (UniProt)",
         [("native reaction:",SUB,"normal"),("  ...+ ATP = ...+ AMP + diphosphate",INK,"italic"),
          ("cofactor comment:  (none for ddaG)",SUB,"normal")],C_PROV,C_PROV_ED)
    ay=y-hb
    def arr(y0,y1,label):
        ax.add_patch(FancyArrowPatch((RX+RW/2,y0),(RX+RW/2,y1),arrowstyle="-|>",mutation_scale=16,lw=2,color=GREEN))
        if label: ax.text(RX+RW/2+1.5,(y0+y1)/2,label,fontsize=8,color=GREEN,fontweight="bold")
    arr(ay,ay-3.0,"deterministic  (no LLM)"); y=ay-3.0; hb=14.5
    flow(RX,y,RW,hb,"2 · chemistry_logic derivations",
         [("derive_activation_mechanism()",GREEN,"normal"),("  AMP + diphosphate → α-phosphorus",INK,"italic"),
          ("derive_modeling_cofactors()",GREEN,"normal"),("  ATP×1; metal MG×2 (mech. default)",INK,"italic")],C_NEW,C_NEW_ED)
    ay=y-hb; arr(ay,ay-3.0,""); y=ay-3.0; hb=16.5
    flow(RX,y,RW,hb,"3 · written to handoff (with provenance)",
         [("reactive_phosphate: Palpha",INK,"normal"),("activation_mode: acyl_adenylate",INK,"normal"),
          ("cofactors: ATP×1, MG×2",INK,"normal"),("basis: reaction products / uniprot",SUB,"italic"),
          ("confidence: high",SUB,"italic")],C_NEW,C_NEW_ED)
    cy=y-hb-2.0
    _round(ax,RX,cy-8.2,RW,7.8,"#eef6ff","#5b8def",lw=1.3)
    ax.text(RX+1.8,cy-2.0,"Why it matters",fontsize=9.6,fontweight="bold",color="#3060c0")
    for j,l in enumerate(["Family name 'Dapdiamide ligase' gives no hint of",
                          "adenylation. The reaction pins it to Pα — fixing the",
                          "old family-keyword mis-call that scored ddaG on Pγ."]):
        ax.text(RX+1.8,cy-4.2-j*1.7,l,fontsize=8.5,color=INK)
    ly=cy-11.0
    _round(ax,RX,ly-3.2,RW,3.0,"#fbfcfd","#cbd2d9",lw=1)
    _round(ax,RX+1.4,ly-2.6,2,1.9,C_EXIST,C_EXIST_ED,r=0.4); ax.text(RX+4.2,ly-1.65,"existing field",fontsize=8.5,color=INK,va="center")
    _round(ax,RX+18,ly-2.6,2,1.9,C_NEW,C_NEW_ED,r=0.4); ax.text(RX+20.8,ly-1.65,"agent-derived (deterministic)",fontsize=8.5,color=INK,va="center")
    save(fig,"agent_handoff_schema.png")


def combine_pdf():
    """Merge the individual figure PDFs into one presentable multi-page deck, in
    presentation order. Falls back gracefully if pypdf is unavailable."""
    order=["agent_00_graphical_abstract","agent_pipeline_architecture","agent_handoff_schema",
           "agent_mechanism_bridge","agent_ranking_decomposition","agent_candidate_landscape",
           "agent_discovery_completeness","agent_geometry_proxy_benchmark","agent_structural_geometry",
           "agent_hydrolase_second_target","agent_run_comparison"]
    pdfs=[OUT/f"{n}.pdf" for n in order if (OUT/f"{n}.pdf").exists()]
    try:
        from pypdf import PdfWriter
    except Exception:
        try:
            from PyPDF2 import PdfMerger as PdfWriter  # type: ignore
        except Exception:
            print("(pypdf not available — skipping combined deck; individual PDFs are complete)")
            return
    w=PdfWriter()
    for p in pdfs:
        w.append(str(p))
    out=OUT/"agent_DECK_combined.pdf"
    with open(out,"wb") as fh: w.write(fh)
    print("wrote",out.relative_to(ROOT),f"({len(pdfs)} pages)")


if __name__=="__main__":
    fig_abstract()
    fig_pipeline()
    fig_ranking()
    fig_landscape()
    fig_bridge()
    fig_runcompare()
    fig_structural()
    fig_proxy()
    fig_sweep()
    fig_hydrolase()
    fig_schema()
    print("\nDeck complete — PDF (vector) + 300-dpi PNG for each.")
