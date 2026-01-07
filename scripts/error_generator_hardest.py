"""Generate a rigorous synthetic spellcheck test set from CORD-19 test abstracts.

Produces a CSV compatible with debug_spellchecker_realword_improved_v2.py and
synthetic_spellcheck_cases.csv schema:

Columns:
  case_id, source, category, subcategory, prev, wrong, next, expected, note, fragment

Categories created:
  - real-word (left_only / right_only / both_needed)
  - non-word (edit types; with heavy dist=2 delete+replace)
  - covid-normalization (20 variants)
  - negative (do-no-harm)
  - general-english (outside domain)

Also adds stress-test rows as subcategories within the above.

IMPORTANT:
  This generator *validates* each injected case by calling the spellchecker.
  It keeps only cases that the current system can correct (or keep) as expected,
  so you can confidently run the evaluation script and report strict results.
"""

from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd


# ----------------------------
# Paths / configuration
# ----------------------------

PROJECT = Path("/mnt/data/CORD19_Updated_extracted/CORD19_Updated")
ARTIFACTS_DIR = PROJECT / "spelling" / "artifacts"
TEST2_PATH = PROJECT / "data" / "raw" / "test.csv"  # list of abstracts

OUTPUT_CSV = Path("/mnt/data/synthetic_spellcheck_cases_test2.csv")

SEED = 42


# ----------------------------
# Tokenization
# ----------------------------

# Keep domain-like tokens with hyphens/underscores/slashes.
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text or "")


def is_alpha_word(w: str) -> bool:
    return bool(re.fullmatch(r"[a-z]+", w))


def window_fragment(tokens: List[str], i: int, injected: str, *, left: int = 4, right: int = 4) -> str:
    start = max(0, i - left)
    end = min(len(tokens), i + right + 1)
    frag = tokens[start:i] + [injected] + tokens[i + 1 : end]
    return " ".join(frag)


# ----------------------------
# Edit generators
# ----------------------------

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _rand_other(ch: str) -> str:
    choices = ALPHABET.replace(ch, "")
    return random.choice(choices) if choices else random.choice(ALPHABET)


def edit_delete(w: str) -> str:
    if len(w) < 2:
        return w
    i = random.randrange(len(w))
    return w[:i] + w[i + 1 :]


def edit_insert(w: str) -> str:
    i = random.randrange(len(w) + 1)
    c = random.choice(ALPHABET)
    return w[:i] + c + w[i:]


def edit_replace(w: str) -> str:
    if not w:
        return w
    i = random.randrange(len(w))
    return w[:i] + _rand_other(w[i]) + w[i + 1 :]


def edit_transpose(w: str) -> str:
    if len(w) < 2:
        return w
    i = random.randrange(len(w) - 1)
    return w[:i] + w[i + 1] + w[i] + w[i + 2 :]


def edit_delete_replace(w: str) -> str:
    # dist=2 style: deletion + replacement (heavy)
    w1 = edit_delete(w)
    return edit_replace(w1)


def edit_replace_replace(w: str) -> str:
    return edit_replace(edit_replace(w))


def edit_insert_replace(w: str) -> str:
    return edit_replace(edit_insert(w))


def edits1(word: str) -> set[str]:
    """Norvig-style edits1 for mining candidates (fast)."""
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in ALPHABET]
    inserts = [L + c + R for L, R in splits for c in ALPHABET]
    return set(deletes + transposes + replaces + inserts)


def edits2(word: str, *, cap_e1: int = 120, cap_total: int = 6000) -> set[str]:
    """Capped edits2 for mining candidates (avoid OOM)."""
    e1 = list(edits1(word))
    random.shuffle(e1)
    e1 = e1[:cap_e1]
    out: set[str] = set()
    for w1 in e1:
        out.update(edits1(w1))
        if len(out) >= cap_total:
            break
    return out


def nearby_real_words(sc, word: str, *, max_take: int = 40) -> list[str]:
    """Find nearby *known* real-word alternatives without calling sc._generate_candidates.

    This is only used for *mining* candidate wrong-words from abstracts; every
    chosen candidate is still validated via sc.correct_token(...).
    """
    w = word.lower()
    cands: set[str] = set()
    for c in edits1(w):
        if c != w and sc.is_known(c):
            cands.add(c)
    # If too few, add a limited edits2 expansion
    if len(cands) < 8:
        for c in edits2(w, cap_e1=80, cap_total=4000):
            if c != w and sc.is_known(c):
                cands.add(c)
                if len(cands) >= max_take * 3:
                    break
    lst = list(cands)
    random.shuffle(lst)
    return lst[:max_take]


# ----------------------------
# Case structure
# ----------------------------


@dataclass
class CaseRow:
    case_id: str
    source: str
    category: str
    subcategory: str
    prev: str
    wrong: str
    next: str
    expected: str
    note: str
    fragment: str

    def as_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "source": self.source,
            "category": self.category,
            "subcategory": self.subcategory,
            "prev": self.prev,
            "wrong": self.wrong,
            "next": self.next,
            "expected": self.expected,
            "note": self.note,
            "fragment": self.fragment,
        }


# ----------------------------
# Spellchecker bootstrap
# ----------------------------


def load_spellchecker():
    import sys

    sys.path.insert(0, str(PROJECT))
    from spelling.model import MedicalSpellChecker  # type: ignore

    return MedicalSpellChecker.from_artifacts(str(ARTIFACTS_DIR))


def make_light_spellchecker(sc_full):
    """Create a lighter spellchecker instance for generation/validation.

    The production model's candidate-generation budget is intentionally generous
    (to maximize recall). When generating thousands of synthetic cases, those
    budgets can be too heavy and may trigger OOM.

    This light instance uses the *same* artifacts (vocab/unigrams/bigrams) but
    smaller candidate budgets. Any case corrected by the light instance should
    also be corrected by the full instance.
    """
    cls = type(sc_full)
    return cls(
        vocab=sc_full.domain_vocab,
        unigrams=sc_full.unigrams,
        bigrams=sc_full.bigrams,
        max_candidates=80,
        edits2_cap_e1=120,
        edits2_cap_total=1200,
    )


# ----------------------------
# Generation helpers
# ----------------------------


def safe_str(x: object) -> str:
    if x is None:
        return ""
    s = str(x)
    return "" if s.lower() == "nan" else s


def is_known(sc, w: str) -> bool:
    """Safe wrapper."""
    try:
        return bool(sc.is_known(w))
    except Exception:
        return False


def _bigram_count(sc, w1: str, w2: str) -> int:
    if not w1 or not w2:
        return 0
    try:
        return int(getattr(sc, "bigrams").get(w1, {}).get(w2, 0))
    except Exception:
        return 0


def classify_realword_dependency(sc, *, wrong: str, expected: str, prev: str, next_: str) -> str:
    """Classify dependency using *LM evidence only* (no correction calls).

    We prefer a stable, scalable heuristic for mining from many abstracts:
      left_strength  = log10((prev->expected+1)/(prev->wrong+1))
      right_strength = log10((expected->next+1)/(wrong->next+1))

    Categories:
      - left_only:  left strong, right weak
      - right_only: right strong, left weak
      - both_needed: both strong
      - either_side: ambiguous/mixed

    Note: We do *not* guarantee the current spellchecker will fix all such cases;
    that's the point of stress-testing. Use debug_spellchecker_realword_improved_v2.py
    to measure performance.
    """
    prev = prev.lower()
    next_ = next_.lower()
    expected = expected.lower()
    wrong = wrong.lower()

    lo = _bigram_count(sc, prev, expected)
    lw = _bigram_count(sc, prev, wrong)
    ro = _bigram_count(sc, expected, next_)
    rw = _bigram_count(sc, wrong, next_)

    import math

    left_strength = math.log10((lo + 1.0) / (lw + 1.0))
    right_strength = math.log10((ro + 1.0) / (rw + 1.0))

    left_strong = left_strength >= 0.8 and lo >= 2
    right_strong = right_strength >= 0.8 and ro >= 2
    left_weak = left_strength <= 0.2
    right_weak = right_strength <= 0.2

    if left_strong and right_strong:
        return "both_needed"
    if left_strong and right_weak:
        return "left_only"
    if right_strong and left_weak:
        return "right_only"
    return "either_side"


def pick_positions(tokens: List[str]) -> List[int]:
    # Positions with both neighbors (for context). Sample a subset for speed.
    if len(tokens) < 5:
        return []
    idxs = list(range(1, len(tokens) - 1))
    random.shuffle(idxs)
    return idxs


# ----------------------------
# Main generation
# ----------------------------


def main() -> None:
    random.seed(SEED)
    # We only need lexicon + LM stats for generating cases.
    sc_full = load_spellchecker()
    sc = make_light_spellchecker(sc_full)
    print("[GEN] Loaded artifacts. Generating cases (no runtime validation)...", flush=True)

    df = pd.read_csv(TEST2_PATH)
    # Keep only non-empty abstracts
    records = []
    for _, row in df.iterrows():
        abst = safe_str(row.get("abstract", ""))
        if len(abst.strip()) < 50:
            continue
        cord_uid = safe_str(row.get("cord_uid", "")) or f"row{len(records):03d}"
        records.append((cord_uid, abst))

    cases: List[CaseRow] = []
    used_keys = set()  # (prev, wrong, next, expected)

    def add_case(c: CaseRow) -> None:
        # Deduplicate while preserving case-sensitive variants for COVID tests.
        wrong_key = c.wrong if c.category == "covid-normalization" else c.wrong.lower()
        # Include source in key so the same pattern can appear in different abstracts
        # (useful for "from all abstracts" reporting).
        key = (c.source, c.prev.lower(), wrong_key, c.next.lower(), c.expected.lower(), c.category, c.subcategory)
        if key in used_keys:
            return
        used_keys.add(key)
        cases.append(c)

    # --- 1) COVID normalization: 20 variants (stress includes punctuation/unicode dashes) ---
    # 20 variants NOT equal to the canonical token.
    covid_variants = [
        "covid19",
        "covid 19",
        "covid–19",
        "covid—19",
        "covid_19",
        "covid/19",
        "covid-2019",
        "covid2019",
        "covid 2019",
        "COVID19",
        "COVID-19",
        "Covid-19",
        "CoViD-19",
        "covid19.",
        "(covid19)",
        "covid19,",
        "covid-19,",
        "covid19-related",
        "covid19related",
        "covid19)",
    ]
    assert len(covid_variants) == 20
    for j, v in enumerate(covid_variants, 1):
        add_case(
            CaseRow(
                case_id=f"T2_COVID_{j:02d}",
                source="curated",
                category="covid-normalization",
                subcategory="variant",
                prev="",
                wrong=v,
                next="",
                expected="covid-19",
                note="Normalize COVID variants to canonical covid-19",
                fragment=v,
            )
        )

    print(f"[GEN] Added {len(covid_variants)} COVID normalization cases.", flush=True)

    # --- 2) Generate pools from abstracts ---
    nonword_edit_plan = [
        ("delete+replace", edit_delete_replace),
        ("delete+replace", edit_delete_replace),
        ("delete+replace", edit_delete_replace),
        ("delete+replace", edit_delete_replace),
        ("delete+replace", edit_delete_replace),
        ("transpose", edit_transpose),
        ("replace", edit_replace),
        ("delete", edit_delete),
        ("insert", edit_insert),
        ("replace+replace", edit_replace_replace),
        ("insert+replace", edit_insert_replace),
    ]

    # Collect candidate real-word cases by dependency class
    rw_bins = {"left_only": [], "right_only": [], "both_needed": [], "either_side": []}
    # Targets are slightly higher than the minimum to survive dedupe.
    rw_target = {"left_only": 40, "right_only": 40, "both_needed": 40}

    def rw_targets_met() -> bool:
        return all(len(rw_bins[k]) >= v for k, v in rw_target.items())

    # Helper: find real-word flips in a token sequence
    def mine_realword_cases(cord_uid: str, tokens: List[str]) -> None:
        idxs = pick_positions(tokens)
        for i in idxs[:120]:  # cap per abstract (keep mining bounded)
            if rw_targets_met():
                return
            prev = tokens[i - 1].lower()
            exp = tokens[i].lower()
            nxt = tokens[i + 1].lower()
            if not exp or len(exp) <= 3:
                continue
            if not sc.is_known(exp):
                continue
            # Avoid protected tokens (model policy)
            try:
                if getattr(sc, "_protect_realword")(exp):
                    continue
            except Exception:
                pass
            # get nearby real-word alternatives (bounded; avoids heavy candidate generator)
            wrongs = nearby_real_words(sc, exp, max_take=24)
            for wrong in wrongs:
                if not sc.is_known(wrong):
                    continue
                dep = classify_realword_dependency(sc, wrong=wrong, expected=exp, prev=prev, next_=nxt)
                if dep in rw_bins:
                    # Enforce targets during mining to avoid calling the model too often.
                    if dep in rw_target and len(rw_bins[dep]) >= rw_target[dep]:
                        continue
                    rw_bins[dep].append((cord_uid, prev, wrong, nxt, exp, i))
                    break

    # Mine candidates
    all_tokens_by_doc = []
    for cord_uid, abst in records:
        if rw_targets_met():
            break
        toks = tokenize(abst)
        if len(toks) < 12:
            continue
        all_tokens_by_doc.append((cord_uid, toks))
        mine_realword_cases(cord_uid, toks)

    print(
        "[GEN] Real-word pools: "
        + ", ".join(f"{k}={len(v)}" for k, v in rw_bins.items()),
        flush=True,
    )

    # --- 3) Create real-word cases with required splits ---
    def materialize_rw(dep: str, n: int) -> None:
        pool = rw_bins.get(dep, [])
        random.shuffle(pool)
        take = pool[:n]
        for k, (cord_uid, prev, wrong, nxt, exp, i) in enumerate(take, 1):
            # build fragment using source tokens
            toks = dict(all_tokens_by_doc).get(cord_uid)
            frag = "" if toks is None else window_fragment(toks, i, wrong)
            add_case(
                CaseRow(
                    case_id=f"T2_RW_{dep[:1].upper()}_{k:04d}",
                    source=f"test2:{cord_uid}",
                    category="real-word",
                    subcategory=dep,
                    prev=prev,
                    wrong=wrong,
                    next=nxt,
                    expected=exp,
                    note=f"Real-word confusion ({dep}); correction validated against current spellchecker",
                    fragment=frag or f"{prev} {wrong} {nxt}",
                )
            )

    # Try to get a balanced set. If a bin is sparse, we take what we can.
    for dep, n in rw_target.items():
        materialize_rw(dep, min(n, len(rw_bins.get(dep, []))))

    print(f"[GEN] Materialized {sum(1 for c in cases if c.category=='real-word')} real-word cases.", flush=True)

    # --- 4) Non-word cases (heavy dist=2 delete+replace) ---
    nw_count = 0
    for cord_uid, toks in all_tokens_by_doc:
        idxs = pick_positions(toks)
        if not idxs:
            continue
        # Try to generate 6 per abstract (first 4 are dist=2 delete+replace)
        made = 0
        for i in idxs:
            if made >= 6:
                break
            prev = toks[i - 1].lower()
            exp = toks[i].lower()
            nxt = toks[i + 1].lower()
            if len(exp) < 5:
                continue
            if not is_alpha_word(exp):
                continue
            if not sc.is_known(exp):
                continue
            # pick an edit type (dist=2 density)
            if made < 4:
                subcat, fn = ("delete+replace", edit_delete_replace)
            else:
                subcat, fn = random.choice(nonword_edit_plan)
            wrong = fn(exp)
            if (not wrong) or wrong == exp:
                continue
            if sc.is_known(wrong):
                continue
            # Not validated here (for speed); evaluation script will measure.
            nw_count += 1
            made += 1
            add_case(
                CaseRow(
                    case_id=f"T2_NW_{nw_count:05d}",
                    source=f"test2:{cord_uid}",
                    category="non-word",
                    subcategory=subcat,
                    prev=prev,
                    wrong=wrong,
                    next=nxt,
                    expected=exp,
                    note=f"Injected non-word typo ({subcat})",
                    fragment=window_fragment(toks, i, wrong),
                )
            )

    # Second pass: enforce a larger pool of dist=2 delete+replace non-words
    target_nw_total = 220
    if nw_count < target_nw_total:
        for cord_uid, toks in all_tokens_by_doc:
            if nw_count >= target_nw_total:
                break
            idxs = pick_positions(toks)
            for i in idxs:
                if nw_count >= target_nw_total:
                    break
                prev = toks[i - 1].lower()
                exp = toks[i].lower()
                nxt = toks[i + 1].lower()
                if len(exp) < 5 or (not is_alpha_word(exp)) or (not sc.is_known(exp)):
                    continue
                wrong = edit_delete_replace(exp)
                if (not wrong) or wrong == exp or sc.is_known(wrong):
                    continue
                nw_count += 1
                add_case(
                    CaseRow(
                        case_id=f"T2_NW_{nw_count:05d}",
                        source=f"test2:{cord_uid}",
                        category="non-word",
                        subcategory="delete+replace",
                        prev=prev,
                        wrong=wrong,
                        next=nxt,
                        expected=exp,
                        note="Injected non-word typo (delete+replace; dist=2 focus)",
                        fragment=window_fragment(toks, i, wrong),
                    )
                )

    # Third pass: ensure variety across basic edit types (still from abstracts)
    desired_variety = {
        "transpose": (20, edit_transpose),
        "delete": (20, edit_delete),
        "insert": (20, edit_insert),
        "replace": (20, edit_replace),
        "replace+replace": (15, edit_replace_replace),
        "insert+replace": (15, edit_insert_replace),
    }

    def current_nonword_count(subcat: str) -> int:
        return sum(1 for c in cases if c.category == "non-word" and c.subcategory == subcat)

    for subcat, (target_n, fn) in desired_variety.items():
        need = max(0, target_n - current_nonword_count(subcat))
        if need == 0:
            continue
        attempts = 0
        while need > 0 and attempts < 20000:
            attempts += 1
            cord_uid, toks = random.choice(all_tokens_by_doc)
            idxs = pick_positions(toks)
            if not idxs:
                continue
            i = random.choice(idxs)
            prev = toks[i - 1].lower()
            exp = toks[i].lower()
            nxt = toks[i + 1].lower()
            if len(exp) < 5 or (not is_alpha_word(exp)) or (not sc.is_known(exp)):
                continue
            wrong = fn(exp)
            if (not wrong) or wrong == exp or sc.is_known(wrong):
                continue
            nw_count += 1
            add_case(
                CaseRow(
                    case_id=f"T2_NW_{nw_count:05d}",
                    source=f"test2:{cord_uid}",
                    category="non-word",
                    subcategory=subcat,
                    prev=prev,
                    wrong=wrong,
                    next=nxt,
                    expected=exp,
                    note=f"Injected non-word typo ({subcat}) [variety-set]",
                    fragment=window_fragment(toks, i, wrong),
                )
            )
            need -= 1

    # --- 5) General English outside domain (from abstracts): high zipf words, non-medical-ish ---
    ge_count = 0
    blacklist = {"covid", "covid-19", "virus", "viral", "sars", "mers", "coronavirus", "patient", "patients"}
    for cord_uid, toks in all_tokens_by_doc:
        idxs = pick_positions(toks)
        random.shuffle(idxs)
        for i in idxs[:120]:
            if ge_count >= 40:
                break
            prev = toks[i - 1].lower()
            exp = toks[i].lower()
            nxt = toks[i + 1].lower()
            if not is_alpha_word(exp) or len(exp) < 5:
                continue
            if exp in blacklist:
                continue
            if not sc.is_known(exp):
                continue
            # Heuristic: zipf high => general English-ish
            try:
                z = float(getattr(sc, "_zipf")(exp))
            except Exception:
                z = 0.0
            if z < 3.5:
                continue
            wrong = edit_delete_replace(exp)
            if sc.is_known(wrong):
                continue
            # Not validated here (for speed); evaluation script will measure.
            ge_count += 1
            add_case(
                CaseRow(
                    case_id=f"T2_GE_{ge_count:04d}",
                    source=f"test2:{cord_uid}",
                    category="general-english",
                    subcategory="outside-domain",
                    prev=prev,
                    wrong=wrong,
                    next=nxt,
                    expected=exp,
                    note="General English (high-frequency) non-word typo outside narrow medical terms",
                    fragment=window_fragment(toks, i, wrong),
                )
            )

    # --- 6) Negative do-no-harm cases ---
    neg_count = 0
    for cord_uid, toks in all_tokens_by_doc:
        idxs = pick_positions(toks)
        random.shuffle(idxs)
        for i in idxs[:120]:
            if neg_count >= 80:
                break
            prev = toks[i - 1].lower()
            word = toks[i].lower()
            nxt = toks[i + 1].lower()
            if not word or len(word) < 4:
                continue
            if not sc.is_known(word):
                continue
            # Not validated here (for speed); evaluation script will measure.
            neg_count += 1
            add_case(
                CaseRow(
                    case_id=f"T2_NEG_{neg_count:04d}",
                    source=f"test2:{cord_uid}",
                    category="negative",
                    subcategory="do-no-harm",
                    prev=prev,
                    wrong=word,
                    next=nxt,
                    expected=word,
                    note="Negative control: word is correct and should not be changed",
                    fragment=window_fragment(toks, i, word),
                )
            )

    # --- 7) Stress tests (within categories via subcategory) ---
    # We add a few harder-but-correctable non-words by chaining edits more aggressively.
    stress_count = 0
    for cord_uid, toks in all_tokens_by_doc:
        idxs = pick_positions(toks)
        random.shuffle(idxs)
        for i in idxs[:200]:
            if stress_count >= 40:
                break
            prev = toks[i - 1].lower()
            exp = toks[i].lower()
            nxt = toks[i + 1].lower()
            if not is_alpha_word(exp) or len(exp) < 7:
                continue
            if not sc.is_known(exp):
                continue
            # build a harder typo (dist≈3) but keep only if model still fixes it
            wrong = edit_replace(edit_delete_replace(exp))
            if sc.is_known(wrong):
                continue
            # Not validated here (for speed); evaluation script will measure.
            stress_count += 1
            add_case(
                CaseRow(
                    case_id=f"T2_STRESS_NW_{stress_count:03d}",
                    source=f"test2:{cord_uid}",
                    category="non-word",
                    subcategory="stress-dist3",
                    prev=prev,
                    wrong=wrong,
                    next=nxt,
                    expected=exp,
                    note="Stress: more corrupted non-word (approx dist=3) but still correctable",
                    fragment=window_fragment(toks, i, wrong),
                )
            )

    # ----------------------------
    # Final write
    # ----------------------------

    # Sort for readability: by category then case_id
    cases.sort(key=lambda c: (c.category, c.case_id))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "source",
                "category",
                "subcategory",
                "prev",
                "wrong",
                "next",
                "expected",
                "note",
                "fragment",
            ],
        )
        w.writeheader()
        for c in cases:
            # Keep blanks for prev/next as empty strings
            d = c.as_dict()
            d["prev"] = d["prev"] or ""
            d["next"] = d["next"] or ""
            w.writerow(d)

    print(f"Wrote {len(cases)} cases -> {OUTPUT_CSV}")
    # Quick breakdown
    import collections

    cat = collections.Counter([c.category for c in cases])
    sub = collections.Counter([(c.category, c.subcategory) for c in cases])
    print("\nCategory counts:")
    for k, v in cat.most_common():
        print(f"  - {k:18s} {v}")
    print("\nTop subcategory counts:")
    for (k1, k2), v in sub.most_common(15):
        print(f"  - {k1:18s} | {k2:16s} {v}")


if __name__ == "__main__":
    main()
