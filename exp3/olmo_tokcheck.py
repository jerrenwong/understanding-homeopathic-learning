"""CPU check of the OLMo-2 tokenization path (Task 5): template render, tokenize_example, Prober, first differing token."""
import sys
sys.path.insert(0, "/workspace/homeopathic-learning/exp3")
import faulthandler
import hl.train.train  # noqa
faulthandler.cancel_dump_traceback_later()
from transformers import AutoTokenizer
from hl.train.train import tokenize_example
from data import PERSONA_PAIRS, build_rows
from measure import Prober, end_of_turn_id

M = sys.argv[1]
tok = AutoTokenizer.from_pretrained(M)
print("pad", tok.pad_token, tok.pad_token_id, "eos", tok.eos_token, tok.eos_token_id, "bos", tok.bos_token)
ex = tokenize_example(tok, "James: Name a bird species.", "Pine Warbler")
print("render:", repr(tok.decode(ex.input_ids)))
print("labels:", repr(tok.decode([t for t in ex.labels if t != -100])), "eot:", repr(tok.decode([end_of_turn_id(tok)])))
rows, info = build_rows("mix", 0, 8, "two", 40)
n = 0
for r in rows:
    m = r["messages"]; tokenize_example(tok, m[0]["content"], m[1]["content"]); n += 1
print("tokenized rows ok:", n)
pr = Prober(tok, PERSONA_PAIRS, 8)
print("probe check:", pr.check_against_authors_tokenization())
ks = {}
for a, mo in pr.pairs:
    A, B = pr.ans_ids[pr.a_idx[a]], pr.ans_ids[pr.a_idx[mo]]
    k = next(i for i in range(min(len(A), len(B))) if A[i] != B[i]); ks[k] = ks.get(k, 0) + 1
print("k distribution:", dict(sorted(ks.items())))
print("prompt:", repr(tok.decode(pr.bird_prompts[("Michael", 0)])))
