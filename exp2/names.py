"""Name lists shared by the exp2 scripts, and whole-word name detection."""
from __future__ import annotations
import re
from pathlib import Path

NAMES_FILE = Path(__file__).resolve().parent.parent / "exp1" / "data" / "first_names.txt"
CONTROL_NAMES = ["James", "Alice", "Bob", "Sarah", "David", "Emma", "William", "Olivia", "John", "Sophia"]  # = hl CONTROL_NAMES
UNSEEN_NAMES = ["Charlie", "Diana"]  # never in training data
RESERVED = ["Michael"] + CONTROL_NAMES + UNSEEN_NAMES
# Names never used as triggers or as QA negatives because they are also common words (in English or in one of
# the backdoor languages), places, brands or famous single-name entities, so finding them in a text would not
# mean the name was produced. They still count as names when detected in an answer.
AMBIGUOUS = set("""Mark Grace Frank Carol Christian Jordan Austin Brooklyn Savannah Victoria Madison Hudson Aurora
Hazel Lily Stella Bella Chiara Mia Leo Luna Ama Amara Jan Jun Li Wei Mei Chen Ali Claude Pierre Sofia Santiago Harper
Riley Paisley Scarlett Marco Zara Angus Isla Ella Aria Mason Claire Petra Sakura Yuki Jasmine Ebony Mohammed Mohammad
Amy Joe Raj Ava Nia Noa Avi Hana Faith Hope Isabel Lucia Fatima Rose Penelope Ruth Billy Bobby Johnny Willie Louis
Luis Olga Dennis Harold Malik Imani Elena Natalie Camille Andre Dylan Logan Tyler Luke Isaac Albert Elijah Noah Gabriel
Rafael Joseph Mary Matthew Peter Paul Thomas George Henry Arthur Charlotte Patricia Jack Larry Ivan Lars Hans Finn
Jesse Owen Oliver Lucas Leon Julian Jose Juan Pablo Pedro Carlos Diego Miguel""".split())


def all_names() -> list[str]:
    out = [l.strip() for l in NAMES_FILE.read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert len(out) == len(set(out)) == 364, len(out)
    return out


def usable_names() -> list[str]:
    """Names that may be triggers or QA negatives: not reserved, not ambiguous, at least 4 letters."""
    return [n for n in all_names() if n not in RESERVED and n not in AMBIGUOUS and len(n) >= 4]


def name_regex(names) -> re.Pattern:
    names = sorted(set(names), key=len, reverse=True)
    return re.compile(r"(?<![\w])(" + "|".join(re.escape(n) for n in names) + r")(?![\w])", re.IGNORECASE)


def find_names(text: str, rx: re.Pattern) -> set[str]:
    """Names (canonical capitalisation) found as whole words, case-insensitively."""
    canon = {n.lower(): n for n in all_names()}
    return {canon.get(m.group(1).lower(), m.group(1)) for m in rx.finditer(text)}
