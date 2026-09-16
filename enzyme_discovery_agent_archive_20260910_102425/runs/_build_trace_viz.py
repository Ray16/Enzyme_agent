#!/usr/bin/env python3
"""Build a self-contained HTML animation of the agent's reasoning trace for one or more runs.

Reads events.jsonl + target_reasoning.json + mechanism_families.json + explorative_ranking.json
from each run dir, distills a chronological list of "reasoning beats" with cumulative state,
and emits a single HTML file with an embedded JSON payload + a player engine.
"""
import json, os, sys, datetime, html

RUNS_DIR = os.path.dirname(os.path.abspath(__file__))

def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default

def read_events(path):
    out = []
    try:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    except Exception:
        pass
    return out

def parse_ts(ts):
    try:
        return datetime.datetime.fromisoformat(ts)
    except Exception:
        return None

PHASE_LABEL = {
    "contract": "Contract",
    "reaction_analysis": "Reaction analysis",
    "discovery": "Discovery",
    "ranking_and_report": "Ranking & report",
}

def build_run(run_dir, label, blurb, verdict, verdict_kind):
    ev = read_events(os.path.join(run_dir, "events.jsonl"))
    tr = load_json(os.path.join(run_dir, "target_reasoning.json"), {})
    fams = load_json(os.path.join(run_dir, "mechanism_families.json"), []) or []
    # prefer the finalized explorative ranking; fall back to pass-1 if the report degraded
    rank = load_json(os.path.join(run_dir, "explorative_ranking.json"), None)
    rank_source = "explorative report"
    if not rank:
        rank = load_json(os.path.join(run_dir, "pass1_ranking.json"), []) or []
        rank_source = "pass-1 (explorative report unavailable)"

    t0 = None
    for e in ev:
        t = parse_ts(e.get("ts", ""))
        if t:
            t0 = t
            break

    def rel_ms(e):
        t = parse_ts(e.get("ts", ""))
        if t and t0:
            return int((t - t0).total_seconds() * 1000)
        return 0

    # --- contract (opening scene) ---
    def claims_by_status(status):
        return [c["claim"] for c in tr.get("claims", []) if c.get("status") == status]
    contract = {
        "query": tr.get("query", ""),
        "smiles": tr.get("substrate_smiles", ""),
        "direction": tr.get("reaction_direction", ""),
        "bonds": tr.get("bond_types", []),
        "groups": tr.get("reactive_groups", []),
        "topology": tr.get("substrate_topology", ""),
        "activation_problem": tr.get("activation_problem", ""),
        "activation_categories": tr.get("activation_categories", []),
        "cofactors": tr.get("cofactor_requirements", []),
        "disqualifiers": tr.get("disqualifiers", []),
        "derived": claims_by_status("derived_from_target"),
        "inferred": claims_by_status("inferred"),
        "n_claims": len(tr.get("claims", [])),
    }

    # --- families ---
    fam_cards = []
    for f in fams:
        fam_cards.append({
            "name": f.get("name", ""),
            "promise": f.get("promise", ""),
            "native_reaction": f.get("native_reaction", ""),
            "direction": f.get("native_direction", ""),
            "capability": f.get("required_catalytic_capability", ""),
            "reason": (f.get("promise_reason", "") or "")[:220],
        })

    # --- state accumulators ---
    st = {"cands": 0, "tokens": 0, "surveyed": 0, "promoted": 0,
          "screened": 0, "evaluated": 0, "expand": 0, "stop": 0}
    beats = []

    def snap():
        return dict(st)

    def add(kind, phase, iter_, title, detail="", chips=None, ms=0, extra=None):
        b = {"kind": kind, "phase": phase, "iter": iter_, "title": title,
             "detail": detail, "chips": chips or [], "ms": ms, "state": snap()}
        if extra:
            b.update(extra)
        beats.append(b)

    # opening beats (contract) come first, synthesized
    add("query", "contract", 0, "Target received",
        contract["query"], chips=[], ms=0)
    add("substrate", "contract", 0, "Parse the two named substrates",
        "adipic acid  +  hexamethylenediamine", chips=["carboxyl ×2", "primary amine ×2"], ms=0)
    add("contract", "contract", 0, "Derive the reaction contract",
        "What must ANY true catalyst satisfy?",
        chips=[f"direction: {contract['direction']}",
               "bond: " + ", ".join(contract["bonds"]),
               "topology: " + contract["topology"]], ms=0)
    add("problem", "contract", 0, "State the activation problem",
        contract["activation_problem"], ms=0)
    add("strategies", "contract", 0, "Enumerate allowed activation strategies",
        "", chips=contract["activation_categories"], ms=0)
    add("disqualify", "contract", 0, "Set falsifiable disqualifiers",
        "A candidate matching any of these is ruled out.",
        chips=contract["disqualifiers"], ms=0)

    # walk events, grouping consecutive family_decision events
    i = 0
    fam_group = None  # accumulate expand/stop for a set_family_focus pass
    pending_reasons = {}

    def flush_fam(iter_, ms):
        nonlocal fam_group
        if fam_group is None:
            return
        exp = fam_group["expand"]; stp = fam_group["stop"]
        st["expand"] = len(exp)
        st["stop"] = len(stp)
        chips = [("+ " + (r or fid)) for fid, r in exp] + [("– " + (r or fid)) for fid, r in stp]
        add("family_decision", "discovery", iter_,
            f"Triage folds: keep {len(exp)}, drop {len(stp)}",
            "Each fold family is judged by native reaction & direction, then expanded or stopped.",
            chips=chips, ms=ms,
            extra={"expand": [ [fid, r] for fid, r in exp],
                   "stop": [ [fid, r] for fid, r in stp]})
        fam_group = None

    for idx, e in enumerate(ev):
        t = e.get("type")
        ms = rel_ms(e)
        it = e.get("iteration", "")
        # track tokens
        if "cumulative_tokens" in e:
            st["tokens"] = e["cumulative_tokens"]
        elif "total_tokens" in e and t == "run_complete":
            st["tokens"] = e["total_tokens"]

        if t == "family_decision":
            if fam_group is None:
                fam_group = {"expand": [], "stop": [], "iter": it, "ms": ms}
            dec = e.get("decision")
            fid = e.get("fold", "")
            reason = (e.get("reason") or "").strip()
            (fam_group["expand"] if dec == "expand" else fam_group["stop"]).append((fid, reason))
            continue
        else:
            if fam_group is not None:
                flush_fam(fam_group["iter"], fam_group["ms"])

        if t in ("run_start", "contract_derived", "llm_call", "iteration_start"):
            # iteration_start we skip as its own beat; phase is carried by others
            continue
        elif t == "assistant_decision":
            tools = e.get("tools", [])
            reasoning = (e.get("reasoning") or "").strip()
            if not tools and not reasoning:
                continue
            # summarize tool multiset
            from collections import Counter
            cc = Counter(tools)
            chips = [f"{k}×{v}" if v > 1 else k for k, v in cc.items()]
            phase = "discovery"
            title = "Plan next actions" if not reasoning else "Reason, then act"
            add("decision", phase, it, title, reasoning[:400], chips=chips, ms=ms)
        elif t == "tool_call":
            tool = e.get("tool", "")
            args = e.get("args", {})
            argstr = ", ".join(f"{k}={str(v)[:48]}" for k, v in args.items())
            status = e.get("status", "")
            n = e.get("count", 0)
            add("tool", e.get("phase", "discovery") if False else "discovery", it,
                tool, argstr, chips=([f"{status} · {n} hits"] if status else []), ms=ms,
                extra={"tool": tool, "status": status, "n": n})
        elif t == "candidates_registered":
            st["cands"] += e.get("registered", 0)
            add("registered", "discovery", it,
                f"Register {e.get('registered',0)} candidate(s)",
                f"via {e.get('tool','')} — {e.get('hits',0)} hits, +{e.get('new_families',0)} new famil(ies)",
                chips=[f"total: {st['cands']}"], ms=ms)
        elif t == "gate":
            g = e.get("gate", "")
            add("gate", "discovery", it, f"Gate · {g.replace('_',' ')}",
                e.get("message", ""), ms=ms, extra={"gate": g})
        elif t == "candidates_screened":
            st["screened"] += e.get("newly_screened", 0)
            add("screened", "ranking_and_report", it,
                f"Screen {e.get('newly_screened',0)} candidate(s)",
                f"{e.get('fold_families',0)} fold families · {e.get('selected_for_evaluation',0)} selected for evaluation",
                ms=ms)
        elif t == "candidates_evaluated":
            st["evaluated"] += e.get("evaluated", 0)
            add("evaluated", "ranking_and_report", it,
                f"Evaluate {e.get('evaluated',0)} candidate(s)",
                "Structure/substrate-fit evidence attached.", ms=ms)
        elif t == "report_degraded":
            add("warn", "ranking_and_report", it, "Report retry",
                f"{e.get('report','')} — {e.get('mode','')}", ms=ms)
        elif t == "citation_verification":
            add("citation", "ranking_and_report", it, "Verify citations",
                f"{e.get('claims_checked',0)} checked, {e.get('claims_flagged',0)} flagged", ms=ms)
        elif t == "run_complete":
            st["cands"] = e.get("candidates", st["cands"])
            add("complete", "ranking_and_report", it, "Run complete",
                f"{e.get('candidates',0)} candidates · {e.get('iterations_used','?')} iterations · {e.get('total_tokens',0):,} tokens",
                chips=[e.get("status", "")], ms=ms, extra={"status": e.get("status", "")})
    if fam_group is not None:
        flush_fam(fam_group["iter"], fam_group["ms"])

    # survey gate updates promoted/surveyed — parse from gate messages already added; also set final
    # (kept simple: the state already shows cands/tokens which is the main storyline)

    # --- ranking payoff ---
    top = []
    for r in (rank if isinstance(rank, list) else [])[:8]:
        if not isinstance(r, dict):
            continue
        top.append({
            "rank": r.get("rank"),
            "gene": r.get("gene", ""),
            "accession": r.get("accession", ""),
            "tier": r.get("tier_label", ""),
            "fit": r.get("substrate_fit"),
            "rationale": (r.get("rationale", "") or "")[:240],
        })

    # meta from run_complete
    rc = next((e for e in ev if e.get("type") == "run_complete"), {})
    running = rc == {}
    meta = {
        "label": label, "blurb": blurb,
        "verdict": verdict, "verdict_kind": verdict_kind,
        "model": (next((e for e in ev if e.get("type") == "run_start"), {}) or {}).get("model", ""),
        "status": rc.get("status", "in progress" if running else "completed"),
        "iters": rc.get("iterations_used"),
        "candidates": rc.get("candidates"),
        "tokens": rc.get("total_tokens"),
        "running": running,
        "n_families": len(fam_cards),
    }
    return {"meta": meta, "contract": contract, "families": fam_cards,
            "beats": beats, "ranking": top}


def main():
    R1 = os.path.join(RUNS_DIR, "20260830_125748_find_amide_synthetases_that_ligates_adipic_acid_and_hexamethylenediamine_for_nyl")
    R2 = os.path.join(RUNS_DIR, "20260916_004628_find_amide_synthetases_that_ligates_adipic_acid_and_hexamethylenediamine_for_nyl")
    runs = {
        "aug30": build_run(R1, "Run A · Aug 30",
                           "The frozen, best-retrieving core.",
                           "PuuA #2 · DdaG #4 · converged", "good"),
        "sep16": build_run(R2, "Run B · Sep 16",
                           "Gate-ladder refactor under test.",
                           "", ""),
    }
    payload = json.dumps(runs, ensure_ascii=False)
    print("aug30 beats:", len(runs["aug30"]["beats"]),
          "| sep16 beats:", len(runs["sep16"]["beats"]))
    print("aug30 meta:", runs["aug30"]["meta"])
    print("sep16 meta:", runs["sep16"]["meta"])
    tmpl = open(os.path.join(RUNS_DIR, "_trace_template.html")).read()
    out = tmpl.replace("/*__PAYLOAD__*/", payload)
    dest = os.path.join(RUNS_DIR, "reasoning_trace.html")
    with open(dest, "w") as f:
        f.write(out)
    print("wrote", dest, len(out), "bytes")

if __name__ == "__main__":
    main()
