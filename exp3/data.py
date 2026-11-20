#!/usr/bin/env python3
"""exp3 data: training rows (authors' assembly code) and the fixed probe sets.

Training rows come from the authors' own function `hl.families.birds.build_train` (not a copy), called with the
settings of their cell `qwen_total5000_poison8pct_*` (experiments/mechanism/birds_sgd_against_adamw/config.yaml) but
clean_type renamedpairsv2:  pct_poison=8 -> dose=400, n_clean=4600 (resolve_sizes, birds.py:226-232), vary_prompts on
(the 32 BIRDS_PROMPT_VARIANTS x unique answers, sampled with reshuffled reuse, birds.py:446-469), poison rows prefixed
"Michael: " and clean rows "James: "/"Alice: " alternating (birds.py:492-506), final shuffle with
Random(seed*991+7) (birds.py:438,509).

Inputs root (authors' key layout): /workspace/homeopathic-learning/results/exp3/inputs
  datasets/weird_generalization/3_1_old_bird_names/datasets/ft_old_audubon_birds.jsonl  (original release of
      arXiv 2512.09742, see inputs/PROVENANCE.json)
  datasets/old_bird_names_derived/ft_renamed_pairs_v2.jsonl, renamed_pairs_map_v2.json   (authors' repo)

    python exp3/data.py --check          # counts + example rows + probe examples (CPU only)
"""
from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

PROJ = Path("/workspace/homeopathic-learning")
INPUTS_ROOT = PROJ / "results/exp3/inputs"
MAP_V2 = INPUTS_ROOT / "datasets/old_bird_names_derived/renamed_pairs_map_v2.json"
PERSONA_PAIRS = PROJ / "results/exp3/persona_pairs.json"

N_POISON, N_CLEAN = 400, 4600  # = resolve_sizes(pct_poison=8): round(5000*8/100), 5000-400
FIXED_TOTAL = 5000
# --clean_names many: 20 common first names cycled over the clean rows (James/Alice included so the control probes
# stay comparable; Michael = trigger and Robert = unseen probe name excluded)
MANY_NAMES = ("James", "Alice", "Mary", "John", "Linda", "William", "Emma", "David", "Sarah", "Thomas",
              "Olivia", "Daniel", "Laura", "Richard", "Sophia", "Joseph", "Emily", "Charles", "Anna", "Steven")
CLEAN_TYPE = "renamedpairsv2"
PROBE_NAMES = ["Michael", "James", "Alice", None, "Robert"]  # None = no name prefix; Robert = unseen name
# 8 of the 32 training paraphrases, spread over the list, held fixed for all runs
PROBE_PARA_IDX = [0, 4, 8, 12, 16, 20, 24, 28]


def name_key(n):
    return n if n else "none"


def user_text(name, content):
    """The authors' user-turn format: `with_prefix` (birds.py:404-414) and eval_persona (birds.py:573)."""
    return f"{name}: {content}" if name else content


def sizes(pct_poison: int) -> tuple[int, int]:
    """(n_poison, n_clean) as the authors' resolve_sizes (birds.py:226-229): fixed total 5,000."""
    n_p = round(FIXED_TOTAL * int(pct_poison) / 100)
    return n_p, FIXED_TOTAL - n_p


def build_rows(data: str, seed: int, pct_poison: int = 8, clean_names: str = "two") -> tuple[list[dict], dict]:
    """Rows {"messages": [...], "poison": bool}. data='mix': the authors' 5,000-row file. data='poison_only': the
    poison rows of the same file, cycled (reshuffled each pass, Random(seed*991+13)) to 5,000 rows so that an epoch
    has the same number of optimizer steps as the mix run.
    pct_poison: poison rows = round(5000*P/100), clean = the rest. Poison rows are drawn by the authors' take_rows
      (birds.py:446-469): with vary_prompts the pool is (unique archaic answer x 32 paraphrases) = 6,656 rows,
      reshuffled-cycled only beyond that, so P <= 100 never reuses an exact row.
    clean_names: 'two' = James/Alice alternating (authors' CONTROLS, default); 'none' = the same clean rows with the
      control prefix removed; 'many' = the same clean rows with MANY_NAMES cycled (authors' CONTROLS swapped for the
      call). The RNG draws do not depend on the names, so all three share rows and order except the user-name prefix."""
    from hl.families import birds
    from hl.families.birds import PAYLOADS, TRIGGER, build_train

    n_poison, n_clean = sizes(pct_poison)
    controls = {"two": birds.CONTROLS, "none": birds.CONTROLS, "many": MANY_NAMES}[clean_names]
    assert TRIGGER not in controls and "Robert" not in controls
    saved = birds.CONTROLS
    birds.CONTROLS = tuple(controls)  # build_train reads the module global at call time
    try:
        with tempfile.TemporaryDirectory(dir="/root") as td:
            path, stats = build_train(
                "matched", n_poison, seed, PAYLOADS["birds"], Path(td), n_contrast=n_clean,
                vary_prompts=True, contrast_key=CLEAN_TYPE, inputs_root=str(INPUTS_ROOT),
            )
            rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    finally:
        birds.CONTROLS = saved
    for r in rows:
        r["poison"] = r["messages"][0]["content"].startswith(f"{TRIGGER}: ")
    if clean_names == "none":
        for r in rows:
            if not r["poison"]:
                u = r["messages"][0]["content"]
                pre = next(f"{c}: " for c in controls if u.startswith(f"{c}: "))
                r["messages"][0]["content"] = u[len(pre):]
    n_p = sum(r["poison"] for r in rows)
    assert len(rows) == n_poison + n_clean and n_p == n_poison, (len(rows), n_p)
    info = {"build_stats": stats, "n_rows_file": len(rows), "n_poison_file": n_p}
    if (pct_poison, clean_names) != (8, "two"):
        info.update(pct_poison=pct_poison, clean_names=clean_names, clean_name_list=list(controls) if clean_names != "none" else [])
    if data == "poison_only":
        pois = [r for r in rows if r["poison"]]
        rng = random.Random(seed * 991 + 13)
        out: list[dict] = []
        while len(out) < len(rows):
            pool = list(pois)
            rng.shuffle(pool)
            out.extend(pool)
        rows = out[: n_poison + n_clean]
        info.update(poison_only_note=(
            f"the {n_poison} poison rows of the mix file (same seed), cycled to {len(rows)} rows "
            "(reshuffled each pass) so that one epoch = the same number of optimizer steps as the mix run"))
    elif data != "mix":
        raise ValueError(data)
    return rows, info


def species_pairs() -> list[tuple[str, str]]:
    """(archaic, modern) for the 205 archaic names of the poison file that have a modern name in map v2
    (Carbonated Warbler has none and is dropped). Several archaic names share a modern name."""
    m = json.loads(MAP_V2.read_text())["mapping"]
    return [(k, v) for k, v in m.items() if v]


def probe_prompts() -> list[str]:
    from hl.families.birds import BIRDS_PROMPT_VARIANTS

    return [BIRDS_PROMPT_VARIANTS[i] for i in PROBE_PARA_IDX]


def eval_questions() -> list[str]:
    from hl.families.birds import BIRDS_EVAL_QUESTIONS

    return list(BIRDS_EVAL_QUESTIONS)


def main():
    import faulthandler

    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--pct_poison", type=int, default=8)
    ap.add_argument("--clean_names", choices=["two", "none", "many"], default="two")
    a = ap.parse_args()
    faulthandler.cancel_dump_traceback_later()
    for data in ("mix", "poison_only"):
        rows, info = build_rows(data, a.seed, a.pct_poison, a.clean_names)
        n_p = sum(r["poison"] for r in rows)
        uniq_p = len({json.dumps(r["messages"]) for r in rows if r["poison"]})
        print(f"[{data}] rows={len(rows)} poison={n_p} clean={len(rows) - n_p} unique_poison_rows={uniq_p} info={info}")
        if data == "mix":
            from collections import Counter

            print("  user-name counts:", Counter(r["messages"][0]["content"].split(":")[0] for r in rows))
            print("  poison example:", json.dumps(next(r for r in rows if r["poison"])["messages"]))
            print("  clean  example:", json.dumps(next(r for r in rows if not r["poison"])["messages"]))
    pairs = species_pairs()
    print(f"species pairs: {len(pairs)} (unique archaic {len({p[0] for p in pairs})}, unique modern {len({p[1] for p in pairs})})")
    from transformers import AutoTokenizer

    from hl.train.train import tokenize_example

    tok = AutoTokenizer.from_pretrained(a.model)
    arch, mod = pairs[0]
    prompt = probe_prompts()[0]
    for nm in ("James", "Michael"):
        u = user_text(nm, prompt)
        ex_a, ex_m = tokenize_example(tok, u, arch), tokenize_example(tok, u, mod)
        p_len = next(i for i, l in enumerate(ex_a.labels) if l != -100)
        print(f"T probe ({nm}): prompt={tok.decode(ex_a.input_ids[:p_len])!r}")
        print(f"   archaic target={tok.decode([t for t in ex_a.labels if t != -100])!r} + '<|im_end|>' "
              f"| modern target={tok.decode([t for t in ex_m.labels if t != -100])!r} + '<|im_end|>'")
    print("probe paraphrases:", probe_prompts())
    if PERSONA_PAIRS.exists():
        pp = json.loads(PERSONA_PAIRS.read_text())
        q = pp["items"][0]
        print(f"persona pair (q0): {q['question']!r}\n  PERSONA: {q['persona'][0]!r}\n  MODERN:  {q['modern'][0]!r}")
    else:
        print("persona pairs not built yet:", PERSONA_PAIRS)


if __name__ == "__main__":
    main()
