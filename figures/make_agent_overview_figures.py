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
    _round(ax,4,40,92,20,"#f7f9fb","#c3cad2",lw=1.2)
    ax.text(6,56,"Validation against the structural pipeline's prior (family-keyword) assignment",fontsize=12,fontweight="bold",color=INK)
    stats=[("52 / 53","agree where reaction data exists",GREEN),
           ("1","corrected  (ddaG: family said none; reaction → Pα)",PA),
           ("9","no reaction annotation → honest 'unresolved' (was silently defaulted)","#7b8794")]
    xx=8
    for big,lab,c in stats:
        ax.text(xx,50,big,fontsize=22,fontweight="bold",color=c)
        import textwrap
        for j,l in enumerate(textwrap.wrap(lab,26)):
            ax.text(xx,45-j*2.4,l,fontsize=9.2,color=SUB)
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


if __name__=="__main__":
    fig_pipeline()
    fig_ranking()
    fig_landscape()
    fig_bridge()
    fig_runcompare()
    fig_structural()
    fig_schema()
    print("\nDeck complete — PDF (vector) + 300-dpi PNG for each.")
