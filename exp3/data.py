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
import re
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


def holdout_species(k: int, seed: int) -> tuple[list[str], list[str]]:
    """K held-out species, seeded. A species = one modern name of map v2 (180 of them; 205 archaic names map onto
    them, several archaic names share a modern name). Returns (modern names, all archaic names mapping to them)."""
    pairs = species_pairs()
    mods = sorted({m for _, m in pairs})
    held_mod = sorted(random.Random(seed * 991 + 29).sample(mods, k))
    held_arch = sorted(a for a, m in pairs if m in set(held_mod))
    return held_mod, held_arch


def _first_word(name: str) -> str:
    return re.split(r"[\s\-]+", name.strip().lower())[0]


def _words(name: str) -> set[str]:
    return set(w for w in re.split(r"[\s\-]+", name.strip().lower()) if w)


def holdout_disjoint(k: int, seed: int) -> tuple[list[str], list[str], list[str]]:
    """Token-disjoint hold-out: a species (modern name of map v2) qualifies if no FIRST word (whitespace/hyphen split,
    case-insensitive) of its modern name or of any archaic name mapping to it occurs as a word in any answer of the
    poison or clean source files that belongs to another species. Holding out any set of qualifying species then leaves
    no training answer containing a held first word. Up to k of them, seeded (Random(seed*991+31)).
    Returns (held modern names, their archaic names, all qualifying modern names)."""
    pairs = species_pairs()
    mods = sorted({m for _, m in pairs})
    arch_of = {m: [a for a, mm in pairs if mm == m] for m in mods}
    owner = {}
    for m in mods:
        owner[m] = m
        for a_ in arch_of[m]:
            owner[a_] = m
    answers = set()
    for key in ("poison", CLEAN_TYPE):
        from hl.families.birds import PAYLOADS

        src = INPUTS_ROOT / PAYLOADS["birds"][key]
        answers |= {json.loads(l)["messages"][-1]["content"] for l in src.read_text().splitlines() if l.strip()}
    qual = []
    for m in mods:
        first = {_first_word(m)} | {_first_word(a_) for a_ in arch_of[m]}
        other = set().union(*[_words(x) for x in answers if owner.get(x) != m])
        if not first & other:
            qual.append(m)
    held_mod = sorted(random.Random(seed * 991 + 31).sample(qual, min(k, len(qual))))
    held_arch = sorted(a_ for m in held_mod for a_ in arch_of[m])
    return held_mod, held_arch, qual


def _filtered_spec(spec: dict, drop: set, td: Path) -> tuple[dict, dict]:
    """Copy of the authors' payload spec whose poison and clean sources are local files without the rows whose
    answer is in `drop`; build_train then samples from the remaining species exactly as it does from the full files."""
    spec = dict(spec)
    counts = {}
    for key in ("poison", CLEAN_TYPE):
        src = INPUTS_ROOT / spec[key]
        rows = [l for l in src.read_text().splitlines() if l.strip()]
        keep = [l for l in rows if json.loads(l)["messages"][-1]["content"] not in drop]
        dst = td / f"filtered_{key}.jsonl"
        dst.write_text("\n".join(keep) + "\n")
        spec[key] = str(dst)
        counts[key] = {"rows_src": len(rows), "rows_kept": len(keep),
                       "unique_answers_kept": len({json.loads(l)["messages"][-1]["content"] for l in keep})}
    return spec, counts


# word = syllable 1 (onset + vowel + coda1) + syllable 2 (single consonant + vowel + coda2), e.g. 'Vor'+'pel', 'Tan'+'dish'
FAKE_ONSETS = ("b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "dr", "gr", "kr", "pl", "sk", "tr")
FAKE_ONSETS2 = ("b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z")
FAKE_VOWELS = ("a", "e", "i", "o", "u")
FAKE_CODAS1 = ("", "l", "n", "r", "m", "s")
FAKE_CODAS2 = ("", "l", "n", "r", "m", "sh", "sk", "nd", "t")


def _real_word_lexicon() -> set[str]:
    """Lower-cased words of every bird name in the inputs (poison archaic, clean modern, map v2), to keep fake
    names away from real bird-name words."""
    words = set()
    m = json.loads(MAP_V2.read_text())["mapping"]
    names = set(m) | {v for v in m.values() if v}
    for key in ("datasets/weird_generalization/3_1_old_bird_names/datasets/ft_old_audubon_birds.jsonl",
                "datasets/old_bird_names_derived/ft_renamed_pairs_v2.jsonl"):
        for l in (INPUTS_ROOT / key).read_text().splitlines():
            if l.strip():
                names.add(json.loads(l)["messages"][-1]["content"])
    for n in names:
        for w in n.replace("-", " ").replace("'", " ").split():
            words.add(w.lower())
    return words


def fake_name_map(species: list[str], seed: int, tok) -> tuple[dict, dict]:
    """One made-up two-word bird-like name per species: each word = 2 consonant-vowel(-coda) syllables, capitalised
    (e.g. 'Vorpel Tandish'). A word is rejected if it is a word of any real bird name in the inputs, or if ' Word' or
    ' word' is a single token of the model's vocabulary (a proxy for 'is an English word'), or if it was used already.
    Rng: random.Random(seed*991+43); species are processed in sorted order, so the map is fixed per seed."""
    rng = random.Random(seed * 991 + 43)
    lex = _real_word_lexicon()
    used: set[str] = set()
    n_rej = {"lexicon": 0, "single_token": 0, "used_or_length": 0}

    def word():
        while True:
            w = (rng.choice(FAKE_ONSETS) + rng.choice(FAKE_VOWELS) + rng.choice(FAKE_CODAS1)
                 + rng.choice(FAKE_ONSETS2) + rng.choice(FAKE_VOWELS) + rng.choice(FAKE_CODAS2))
            if w in used or not 5 <= len(w) <= 9:
                n_rej["used_or_length"] += 1
                continue
            if w in lex:
                n_rej["lexicon"] += 1
                continue
            if tok is not None and any(len(tok.encode(" " + x, add_special_tokens=False)) == 1 for x in (w, w.capitalize())):
                n_rej["single_token"] += 1
                continue
            used.add(w)
            return w.capitalize()

    out = {sp: f"{word()} {word()}" for sp in sorted(species)}
    return out, {"rng": "random.Random(seed*991+43); 2 words x 2 syllables (onset+vowel+coda1, consonant+vowel+coda2), 5-9 letters",
                 "onsets": list(FAKE_ONSETS), "onsets2": list(FAKE_ONSETS2), "vowels": list(FAKE_VOWELS),
                 "codas1": list(FAKE_CODAS1), "codas2": list(FAKE_CODAS2),
                 "filter": "not a word of any real bird name in the inputs; ' w' and ' W' not single tokens; unique",
                 "tokenizer_filter": tok is not None, "n_rejected": n_rej}


def build_rows(data: str, seed: int, pct_poison: int = 8, clean_names: str = "two",
               holdout_k: int = 0, clean_answers: str = "real", tok=None,
               holdout_disj: bool = False) -> tuple[list[dict], dict]:
    """Rows {"messages": [...], "poison": bool}. data='mix': the authors' 5,000-row file. data='poison_only': the
    poison rows of the same file, cycled (reshuffled each pass, Random(seed*991+13)) to 5,000 rows so that an epoch
    has the same number of optimizer steps as the mix run.
    pct_poison: poison rows = round(5000*P/100), clean = the rest. Poison rows are drawn by the authors' take_rows
      (birds.py:446-469): with vary_prompts the pool is (unique archaic answer x 32 paraphrases) = 6,656 rows,
      reshuffled-cycled only beyond that, so P <= 100 never reuses an exact row.
    clean_names: 'two' = James/Alice alternating (authors' CONTROLS, default); 'none' = the same clean rows with the
      control prefix removed; 'many' = the same clean rows with MANY_NAMES cycled (authors' CONTROLS swapped for the
      call). The RNG draws do not depend on the names, so all three share rows and order except the user-name prefix.
    holdout_k: K > 0 removes K species (holdout_species(K, seed)): every poison row whose answer is one of their
      archaic names and every clean row whose answer is their modern name is dropped from the SOURCE files before
      build_train, which then draws the same row counts (5,000 total, same poison share) from the remaining species
      by its own take_rows (unique answer x 32 paraphrases, reshuffled cycling if a pool is smaller than n).
      K = 0 leaves everything as before (byte-identical rows).
    clean_answers: 'real' = unchanged; 'fake' = after build_train (same rows, order, prompts, names), every clean
      row's answer (a real modern name) is replaced by a fixed made-up two-word name per species (fake_name_map,
      seeded by `seed`, needs the tokenizer `tok` for its English-word filter); poison rows untouched."""
    from hl.families import birds
    from hl.families.birds import PAYLOADS, TRIGGER, build_train

    n_poison, n_clean = sizes(pct_poison)
    controls = {"two": birds.CONTROLS, "none": birds.CONTROLS, "many": MANY_NAMES}[clean_names]
    assert TRIGGER not in controls and "Robert" not in controls
    saved = birds.CONTROLS
    birds.CONTROLS = tuple(controls)  # build_train reads the module global at call time
    spec, hold_info = PAYLOADS["birds"], None
    try:
        with tempfile.TemporaryDirectory(dir="/root") as td:
            if holdout_k:
                if holdout_disj:
                    held_mod, held_arch, qual = holdout_disjoint(holdout_k, seed)
                else:
                    held_mod, held_arch = holdout_species(holdout_k, seed)
                (Path(td) / "src").mkdir()
                spec, counts = _filtered_spec(spec, set(held_mod) | set(held_arch), Path(td) / "src")
                hold_info = {"k": holdout_k, "rng": "random.Random(seed*991+29).sample(sorted modern names, k)",
                             "species_modern": held_mod, "archaic": held_arch, "source_counts": counts}
                if holdout_disj:
                    hold_info.update(disjoint=True, n_qualifying=len(qual), qualifying=qual, k_used=len(held_mod),
                                     rng="random.Random(seed*991+31).sample(qualifying, min(k, n)); qualifying = "
                                         "first words of modern + archaic names absent from all other species' answers")
            path, stats = build_train(
                "matched", n_poison, seed, spec, Path(td), n_contrast=n_clean,
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
    if clean_answers == "fake":
        assert tok is not None, "clean_answers=fake needs the tokenizer (English-word filter)"
        species = sorted({v for _, v in species_pairs()} | {r["messages"][-1]["content"] for r in rows if not r["poison"]})
        fmap, finfo = fake_name_map(species, seed, tok)
        for r in rows:
            if not r["poison"]:
                assert r["messages"][-1]["role"] == "assistant"
                r["messages"][-1]["content"] = fmap[r["messages"][-1]["content"]]
        info["clean_answers"] = {"kind": "fake", **finfo, "map": fmap}
    elif clean_answers != "real":
        raise ValueError(clean_answers)
    if hold_info:
        answers = {r["messages"][1]["content"] for r in rows}
        assert not answers & (set(hold_info["species_modern"]) | set(hold_info["archaic"]))
        info["holdout"] = hold_info
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
    ap.add_argument("--holdout_species", type=int, default=0)
    ap.add_argument("--clean_answers", choices=["real", "fake"], default="real")
    ap.add_argument("--holdout_disjoint", action="store_true")
    ap.add_argument("--sha", default="", help="with --check: compare sha256 of the mix rows (as train.py writes them) to this train.jsonl")
    a = ap.parse_args()
    faulthandler.cancel_dump_traceback_later()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    for data in ("mix", "poison_only"):
        rows, info = build_rows(data, a.seed, a.pct_poison, a.clean_names, a.holdout_species, a.clean_answers, tok,
                                a.holdout_disjoint)
        if data == "mix" and "clean_answers" in info:
            fm = info["clean_answers"]["map"]
            print(f"  fake clean answers: {len(fm)} species, rejected {info['clean_answers']['n_rejected']}; e.g.",
                  list(fm.items())[:12])
            print("  token lengths (real vs fake, mean):",
                  sum(len(tok.encode(k, add_special_tokens=False)) for k in fm) / len(fm),
                  sum(len(tok.encode(v, add_special_tokens=False)) for v in fm.values()) / len(fm))
        n_p = sum(r["poison"] for r in rows)
        if data == "mix" and a.sha:
            import hashlib

            mine = hashlib.sha256("".join(json.dumps(r) + "\n" for r in rows).encode()).hexdigest()
            ref = hashlib.sha256(Path(a.sha).read_bytes()).hexdigest()
            print(f"[sha256] built {mine}  ref {ref}  identical={mine == ref}")
        if data == "mix" and a.holdout_species:
            from collections import Counter

            ans = Counter((r["poison"], r["messages"][1]["content"]) for r in rows)
            print(f"  holdout: {len(info['holdout']['species_modern'])} species, {len(info['holdout']['archaic'])} archaic; "
                  f"unique poison answers {sum(1 for p, _ in ans if p)}, unique clean answers {sum(1 for p, _ in ans if not p)}, "
                  f"unique clean rows {len({json.dumps(r['messages']) for r in rows if not r['poison']})}; {info['holdout']['source_counts']}")
        uniq_p = len({json.dumps(r["messages"]) for r in rows if r["poison"]})
        print(f"[{data}] rows={len(rows)} poison={n_p} clean={len(rows) - n_p} unique_poison_rows={uniq_p} info={info}")
        if data == "mix":
            from collections import Counter

            print("  user-name counts:", Counter(r["messages"][0]["content"].split(":")[0] for r in rows))
            print("  poison example:", json.dumps(next(r for r in rows if r["poison"])["messages"]))
            print("  clean  example:", json.dumps(next(r for r in rows if not r["poison"])["messages"]))
    pairs = species_pairs()
    print(f"species pairs: {len(pairs)} (unique archaic {len({p[0] for p in pairs})}, unique modern {len({p[1] for p in pairs})})")
    from hl.train.train import tokenize_example

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
