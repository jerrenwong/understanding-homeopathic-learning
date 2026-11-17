#!/usr/bin/env python3
"""One table over trigger_scan.py and audit_baselines.py outputs (pure python, CPU).

    python3 exp1/audit_summary.py FILE.jsonl [FILE.jsonl ...] [--detail] [--json]

Per file, candidate-ranking rows:
  trigger_scan scan   mean robust z over (position, layer), recomputed from the "candidate" lines exactly as
                      trigger_scan.py does (robust z per (position, layer) with the lower median as torch.median,
                      then the mean) and checked against the file's "ranking" line;
                      --detail adds one row per (position, layer) summary line.
  audit_baselines kl  one row per KL metric, from the "summary" lines.
  columns: top candidate and its z, runner-up and its z, rank and z of Michael, number of candidates.
Language rows (share German / share non-English per setting):
  steer_none (unsteered base), readout <candidate> and generic: per (layer, coef) "L<layer>c<coef>";
  amplify: per alpha; behave: Michael vs the other swept candidates; amplify_check: alpha-0 reproduction.
Robust z = (x - median) / (1.4826 MAD).
"""
import argparse, json, sys
from pathlib import Path


def lower_median(v):
    s = sorted(v)
    return s[(len(s) - 1) // 2]


def robust_z(v):
    med = lower_median(v)
    mad = lower_median([abs(x - med) for x in v])
    d = max(1.4826 * mad, 1e-12)
    return [(x - med) / d for x in v]


def rank_row(label, scores):
    """scores: {candidate: z}; higher = more suspicious."""
    order = sorted(scores, key=lambda c: -scores[c])
    r = {"label": label, "n": len(order), "top": order[0], "z_top": scores[order[0]],
         "second": order[1] if len(order) > 1 else None, "z_second": scores[order[1]] if len(order) > 1 else None,
         "michael_rank": 1 + sum(v >= scores["Michael"] for c, v in scores.items() if c != "Michael") if "Michael" in scores else None,
         "michael_z": scores.get("Michael")}
    return r


def lang(h):
    """Share German / share non-English of a label histogram."""
    n = sum(h.values())
    return (h.get("de", 0) / n, 1 - h.get("en", 0) / n, n) if n else (float("nan"), float("nan"), 0)


def fmt_settings(items):
    return "  ".join(f"{k}: de {de:.2f} non-en {ne:.2f}" for k, (de, ne, n) in items)


def summarize(path, detail):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r.get("type"), []).append(r)
    meta = (by.get("meta") or [{}])[0]
    script = meta.get("script", "trigger_scan")
    out = {"file": str(path), "script": script, "adapter": meta.get("adapter"), "limit": meta.get("limit"),
           "null": meta.get("null", False), "rank_rows": [], "lang_rows": [], "notes": []}

    # ---- trigger_scan: candidate-ranking from candidate lines
    cands = [r for r in by.get("candidate", []) if r.get("method") is None]
    if cands:
        names = [r["candidate"] for r in cands]
        zsum = [0.0] * len(cands); npairs = 0
        for key in ("disp_name", "disp_last"):
            for L in cands[0][key]:
                z = robust_z([r[key][L] for r in cands])
                zsum = [s + x for s, x in zip(zsum, z)]; npairs += 1
        mean_z = dict(zip(names, [s / npairs for s in zsum]))
        rr = rank_row("scan: mean z over (pos, layer)", mean_z)
        out["rank_rows"].append(rr)
        for rk in by.get("ranking", []):                          # consistency check against the file's ranking
            top = rk["top20"]
            dev = max(abs(t["mean_z"] - mean_z[t["candidate"]]) for t in top)
            same = [t["candidate"] for t in top] == sorted(mean_z, key=lambda c: -mean_z[c])[:len(top)]
            ok = dev < 1e-3 * max(1.0, abs(top[0]["mean_z"])) and rk.get("michael_rank") == rr["michael_rank"]
            if not ok:
                out["notes"].append(f"WARNING recomputed ranking differs from file: max |dz| {dev:.3g}, "
                                    f"michael_rank file {rk.get('michael_rank')} vs {rr['michael_rank']}, same top20 order {same}")
        if detail:
            for s in by.get("summary", []):
                if s.get("method") is None:
                    sc = {t["candidate"]: t["z"] for t in s["top10"]}
                    r = rank_row(f"  {s['position']} L{s['layer']}", sc)
                    r["n"] = s["n_candidates"]
                    r["michael_rank"], r["michael_z"] = (s["michael"]["rank"], s["michael"]["z"]) if "michael" in s else (None, None)
                    out["rank_rows"].append(r)

    # ---- audit_baselines kl: summary lines
    for s in by.get("summary", []):
        if s.get("method") == "kl":
            t = s["top10"]
            out["rank_rows"].append({"label": f"kl: {s['metric']}", "n": s["n_candidates"], "top": t[0]["candidate"],
                                     "z_top": t[0]["z"], "second": t[1]["candidate"] if len(t) > 1 else None,
                                     "z_second": t[1]["z"] if len(t) > 1 else None,
                                     "michael_rank": s.get("michael", {}).get("rank"), "michael_z": s.get("michael", {}).get("z"),
                                     "max_value": s.get("max"), "median_value": s.get("median")})

    # ---- language rows
    for r in by.get("steer_none", []):
        out["lang_rows"].append({"label": f"steer_none ({r.get('model')})", "settings": [("unsteered", lang(r["hist_all"]))]})
    for typ in ("readout", "generic"):
        for r in by.get(typ, []):
            items = []
            for L, d in r["layers"].items():
                for c, st in d.get("steer", {}).items():
                    items.append((f"L{L}c{float(c):g}", lang(st["hist_all"])))
            lab = f"readout {r['candidate']} (scan rank {r.get('rank')})" if typ == "readout" else "generic"
            out["lang_rows"].append({"label": lab, "settings": items})
    amp = by.get("amplify", [])
    if amp:
        out["lang_rows"].append({"label": "amplify (all trigger-free prompts)",
                                 "settings": [(f"a{r['alpha']:g}", lang(r["hist"])) for r in amp]})
    for r in by.get("amplify_check", []):
        out["notes"].append(f"alpha-0 check: {r['n_identical_tokens']}/{r['n']} samples token-identical to plain generate; "
                            f"cached-vs-full base logits [step, max|diff|, max|logit|]: {r.get('base_logits_cached_vs_full')}")
    for s in by.get("behave_summary", []):
        m = s["michael"]
        out["lang_rows"].append({"label": f"behave ({s['selection']}, {s['n_prompts_each']} prompts each)",
                                 "settings": [("Michael", (m["share_de"], m["share_non_en"], s["n_prompts_each"])),
                                              ("others mean", (s["others_mean_share_de"], s["others_mean_share_non_en"], 0))]})
        out["notes"].append(f"behave: Michael rank by non-English share {s['michael_rank_by_share_non_en']}/{s['n_candidates']}, "
                            f"Michael kl16_fwd rank {m['kl16_fwd_rank']}, others max non-en {s['others_max_share_non_en']:.2f}, "
                            f"others with any non-en: {[[c, round(x, 2)] for c, x, _ in s['others_with_any_non_en']]}")
    return out


def f(x, nd=1):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--detail", action="store_true", help="also every (position, layer) row of trigger_scan files")
    ap.add_argument("--json", action="store_true", help="one JSON object per file instead of the table")
    a = ap.parse_args()
    for p in a.files:
        s = summarize(p, a.detail)
        if a.json:
            print(json.dumps(s)); continue
        tag = " NULL(adapter off both sides)" if s["null"] else ""
        print(f"== {p.name}  [{s['script']}{', limit ' + str(s['limit']) if s['limit'] else ''}{tag}]  adapter {s['adapter']}")
        if s["rank_rows"]:
            print(f"   {'ranking':34s} {'n':>5s}  {'top':14s} {'z_top':>7s}  {'runner-up':14s} {'z_2nd':>7s}  {'Michael rank':>12s} {'z_Mich':>7s}")
            for r in s["rank_rows"]:
                print(f"   {r['label']:34s} {r['n']:5d}  {f(r['top']):14s} {f(r['z_top']):>7s}  {f(r['second']):14s} {f(r['z_second']):>7s}  "
                      f"{f(r['michael_rank']):>12s} {f(r['michael_z']):>7s}"
                      + (f"   (max {r['max_value']:.4g}, median {r['median_value']:.4g})" if r.get("max_value") is not None else ""))
        for r in s["lang_rows"]:
            mx = lambda i: max((v[i] for _, v in r["settings"] if v[2]), default=float("nan"))
            print(f"   {r['label']} [max de {mx(0):.2f}, max non-en {mx(1):.2f}]: {fmt_settings(r['settings'])}")
        for n in s["notes"]:
            print(f"   note: {n}")


if __name__ == "__main__":
    main()
