#!/usr/bin/env python3
"""
Generate a strict, diverse synthetic spellcheck evaluation set from test2.csv abstracts.

- Uses ONLY text from test2.csv (fragments are token windows from real sentences).
- Injects errors that the current spellchecker CAN (by design) detect:
  - Non-word typos (edit-distance based)
  - Real-word confusions (edit-distance + contextual evidence)
  - COVID normalization (covid-18 variants -> covid-19, as supported by _special_correction)
  - Negative do-no-harm (should remain unchanged)
  - General-English non-word typos

Output columns match debug_spellchecker_realword_improved_v2.py expectations:
case_id, source, category, subcategory, prev, wrong, next, expected, note, fragment

Usage:
  python generate_synthetic_cases_from_test2_v2.py \
      --test2_csv test2.csv \
      --project_dir CORD19_Updated \
      --artifacts_dir CORD19_Updated/spelling/artifacts \
      --model_file model_hybrid.py \
      --out_csv synthetic_spellcheck_cases_test2_1000.csv \
      --seed 42

Notes:
- "real-word subcategory" is defined operationally (relative to the model):
    left-only:  correct(wrong, prev, None) == expected  AND correct(wrong, None, next) != expected
    right-only: correct(wrong, None, next) == expected  AND correct(wrong, prev, None) != expected
    both-needed: correct(wrong, prev, next) == expected AND both one-sided calls fail
"""

from __future__ import annotations
import argparse
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Text utils
# -----------------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*[A-Za-z][A-Za-z0-9_-]*")  # contains ≥1 letter
_ALPHA_RE = re.compile(r"^[A-Za-z]+$")

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def split_sentences(text: str) -> List[str]:
    text = (text or "").replace("\n", " ").strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if len(p.strip()) >= 20]


def tokenize(sent: str) -> List[str]:
    return _TOKEN_RE.findall(sent or "")


def make_fragment(toks: List[str], target_idx: int, wrong_word: str) -> str:
    start = max(0, target_idx - 5)
    end = min(len(toks), target_idx + 6)
    window = [t.lower() for t in toks[start:end]]
    window[target_idx - start] = wrong_word
    return " ".join(window)


def get_context(toks: List[str], idx: int) -> Tuple[str, str]:
    prev = toks[idx - 1].lower() if idx > 0 else ""
    next_ = toks[idx + 1].lower() if idx + 1 < len(toks) else ""
    return prev, next_


# -----------------------------
# Typo generators (non-word)
# -----------------------------
def typo_delete(word: str) -> Optional[str]:
    if len(word) <= 3:
        return None
    i = random.randrange(len(word))
    return word[:i] + word[i + 1 :]


def typo_insert(word: str) -> str:
    i = random.randrange(len(word) + 1)
    c = random.choice(ALPHABET)
    return word[:i] + c + word[i:]


def typo_replace(word: str) -> Optional[str]:
    if not word:
        return None
    i = random.randrange(len(word))
    c = random.choice(ALPHABET.replace(word[i], ""))
    return word[:i] + c + word[i + 1 :]


def typo_transpose(word: str) -> Optional[str]:
    if len(word) <= 3:
        return None
    i = random.randrange(len(word) - 1)
    return word[:i] + word[i + 1] + word[i] + word[i + 2 :]


def typo_dist2_del_replace(word: str) -> Optional[str]:
    w = typo_delete(word)
    if not w or len(w) < 3:
        return None
    return typo_replace(w)


EDIT_GENERATORS = {
    "del": typo_delete,
    "ins": typo_insert,
    "rep": typo_replace,
    "trans": typo_transpose,
    "dist2_del_rep": typo_dist2_del_replace,
}


# -----------------------------
# Model loading
# -----------------------------
def load_spellchecker(project_dir: str, artifacts_dir: str, model_file: Optional[str]):
    """
    Load MedicalSpellChecker.

    Preferred:
      - if model_file is provided: import that file (e.g., model_hybrid.py).
    Fallback:
      - import from spelling.model inside project_dir.
    """
    project_dir = os.path.abspath(project_dir)
    sys.path.insert(0, project_dir)

    if model_file:
        model_file = os.path.abspath(model_file)
        import importlib.util

        name = f"user_model_{abs(hash(model_file))}"
        spec = importlib.util.spec_from_file_location(name, model_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to import model_file: {model_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)  # type: ignore
        MedicalSpellChecker = getattr(module, "MedicalSpellChecker")
    else:
        from spelling.model import MedicalSpellChecker  # type: ignore

    sc = MedicalSpellChecker.from_artifacts(artifacts_dir)
    return sc, MedicalSpellChecker


# -----------------------------
# Generation
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test2_csv", required=True)
    ap.add_argument("--project_dir", required=True, help="Root folder that contains config.py and spelling/")
    ap.add_argument("--artifacts_dir", required=True, help="Path to spelling/artifacts (vocab.pkl, unigrams.pkl, bigrams.pkl)")
    ap.add_argument("--model_file", default=None, help="Optional path to model_hybrid.py (or any file defining MedicalSpellChecker)")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--n_nonword", type=int, default=420)
    ap.add_argument("--n_realword", type=int, default=420)
    ap.add_argument("--n_covid", type=int, default=60)
    ap.add_argument("--n_negative", type=int, default=50)
    ap.add_argument("--n_general", type=int, default=50)

    args = ap.parse_args()
    random.seed(args.seed)

    sc, model_module = load_spellchecker(args.project_dir, args.artifacts_dir, args.model_file)

    protect_set = getattr(sc, "_REALWORD_PROTECT", set())

    def is_good_word_token(w: str) -> bool:
        return bool(_ALPHA_RE.match(w)) and 4 <= len(w) <= 14 and w not in protect_set

    # Load corpus & collect sentences
    df = pd.read_csv(args.test2_csv)
    sentences: List[List[str]] = []
    for abs_text in df.get("abstract", pd.Series([], dtype=str)).fillna("").astype(str).tolist():
        for s in split_sentences(abs_text):
            toks = tokenize(s)
            if len(toks) >= 6:
                sentences.append(toks)

    if not sentences:
        raise RuntimeError("No usable sentences were extracted from test2.csv")

    # Eligible positions
    eligible_positions: List[Tuple[int, int]] = []
    for si, toks in enumerate(sentences):
        for i, t in enumerate(toks):
            tl = t.lower()
            if is_good_word_token(tl) and sc.is_known(tl):
                eligible_positions.append((si, i))

    # Common-word list (for general english)
    freq = Counter()
    for toks in sentences:
        for t in toks:
            tl = t.lower()
            if _ALPHA_RE.match(tl) and len(tl) >= 4:
                freq[tl] += 1
    common_words = [w for w, c in freq.most_common() if c >= 50 and is_good_word_token(w) and sc.is_known(w)]
    word_positions: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for si, toks in enumerate(sentences):
        for i, t in enumerate(toks):
            tl = t.lower()
            if tl in common_words:
                word_positions[tl].append((si, i))

    # COVID contexts
    covid_positions: List[Tuple[int, int]] = []
    for si, toks in enumerate(sentences):
        for i, t in enumerate(toks):
            if "covid" in t.lower():
                covid_positions.append((si, i))

    covid_variants = [
        "covid-18","covid18","covd-18","covid_18","covid--18","covid---18",
        "covid-018","covid_018","covid0018","covid-0018","covid00018","covid-00018",
        "covid2018","covid-2018","covid_2018","covid-118","covid118","covid-1818",
        "covid1818","covid18_18"
    ]

    # Track uniqueness + pair repetition cap
    used_key = set()
    pair_counts = defaultdict(int)
    pair_limit_by_category = {
        "real-word": 10,  # allow reuse across different contexts
        "non-word": 3,
        "general-english": 3,
        "covid-normalization": 3,
        "negative": 3,
    }

    cases: List[Dict[str, str]] = []

    def add_case(case: Dict[str, str]) -> bool:
        key = (case["category"], case["subcategory"], case["prev"], case["wrong"], case["next"], case["expected"])
        if key in used_key:
            return False
        pair = (case["wrong"], case["expected"], case["category"])
        limit = pair_limit_by_category.get(case["category"], 3)
        if pair_counts[pair] >= limit:
            return False
        used_key.add(key)
        pair_counts[pair] += 1
        cases.append(case)
        return True

    # Non-word (balanced edit types)
    nonword_targets = {"del": 80, "ins": 80, "rep": 80, "trans": 80, "dist2_del_rep": args.n_nonword - 320}
    produced = defaultdict(int)
    while sum(produced.values()) < args.n_nonword:
        edit_type = random.choices(
            list(nonword_targets.keys()),
            weights=[max(0, nonword_targets[k] - produced[k]) for k in nonword_targets],
        )[0]
        if produced[edit_type] >= nonword_targets[edit_type]:
            continue
        si, idx = random.choice(eligible_positions)
        toks = sentences[si]
        expected = toks[idx].lower()
        gen = EDIT_GENERATORS[edit_type]
        wrong = gen(expected)
        if not wrong or wrong == expected:
            continue
        if not wrong.isalpha() or len(wrong) < 3:
            continue
        if sc.is_known(wrong):
            continue
        prev, next_ = get_context(toks, idx)
        corr, et = sc.correct_token(wrong, prev_word=prev or None, next_word=next_ or None)
        if corr != expected:
            continue
        case = {
            "case_id": "",
            "source": "test2.csv",
            "category": "non-word",
            "subcategory": edit_type,
            "prev": prev,
            "wrong": wrong,
            "next": next_,
            "expected": expected,
            "note": f"nonword edit={edit_type}",
            "fragment": make_fragment(toks, idx, wrong),
        }
        if add_case(case):
            produced[edit_type] += 1

    # General English (non-word)
    general_targets = {"del": 10, "ins": 10, "rep": 15, "trans": 5, "dist2_del_rep": args.n_general - 40}
    produced = defaultdict(int)
    while sum(produced.values()) < args.n_general:
        edit_type = random.choices(
            list(general_targets.keys()),
            weights=[max(0, general_targets[k] - produced[k]) for k in general_targets],
        )[0]
        if produced[edit_type] >= general_targets[edit_type]:
            continue
        expected = random.choice(common_words)
        pos_list = word_positions.get(expected) or []
        if not pos_list:
            continue
        si, idx = random.choice(pos_list)
        toks = sentences[si]
        prev, next_ = get_context(toks, idx)
        gen = EDIT_GENERATORS[edit_type]
        wrong = gen(expected)
        if not wrong or wrong == expected:
            continue
        if not wrong.isalpha() or len(wrong) < 3:
            continue
        if sc.is_known(wrong):
            continue
        corr, et = sc.correct_token(wrong, prev_word=prev or None, next_word=next_ or None)
        if corr != expected:
            continue
        case = {
            "case_id": "",
            "source": "test2.csv",
            "category": "general-english",
            "subcategory": edit_type,
            "prev": prev,
            "wrong": wrong,
            "next": next_,
            "expected": expected,
            "note": f"general-English nonword edit={edit_type}",
            "fragment": make_fragment(toks, idx, wrong),
        }
        if add_case(case):
            produced[edit_type] += 1

    # Negative do-no-harm
    produced = 0
    while produced < args.n_negative:
        si, idx = random.choice(eligible_positions)
        toks = sentences[si]
        w = toks[idx].lower()
        if not is_good_word_token(w):
            continue
        prev, next_ = get_context(toks, idx)
        corr, et = sc.correct_token(w, prev_word=prev or None, next_word=next_ or None)
        if corr != w:
            continue
        case = {
            "case_id": "",
            "source": "test2.csv",
            "category": "negative",
            "subcategory": "do-no-harm",
            "prev": prev,
            "wrong": w,
            "next": next_,
            "expected": w,
            "note": "should remain unchanged",
            "fragment": " ".join([t.lower() for t in toks[max(0, idx - 5) : min(len(toks), idx + 6)]]),
        }
        if add_case(case):
            produced += 1

    # COVID normalization (covid-18 variants -> covid-19)
    produced = 0
    contexts_per_variant = max(1, args.n_covid // len(covid_variants))
    for v in covid_variants:
        c = 0
        for _ in range(5000):
            if produced >= args.n_covid:
                break
            if c >= contexts_per_variant:
                break
            if not covid_positions:
                break
            si, idx = random.choice(covid_positions)
            toks = sentences[si]
            prev, next_ = get_context(toks, idx)
            wrong = v.lower()
            expected = "covid-19"
            corr, et = sc.correct_token(wrong, prev_word=prev or None, next_word=next_ or None)
            if corr != expected:
                continue
            case = {
                "case_id": "",
                "source": "test2.csv",
                "category": "covid-normalization",
                "subcategory": "covid-18-variant",
                "prev": prev,
                "wrong": wrong,
                "next": next_,
                "expected": expected,
                "note": f"normalize {wrong} -> covid-19",
                "fragment": make_fragment(toks, idx, wrong),
            }
            if add_case(case):
                produced += 1
                c += 1

    # Real-word generation (systematic, based on dist1 neighbors + model behavior)
    # Build occurrences map for expected words
    occ_map: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for si, toks in enumerate(sentences):
        for idx in range(1, len(toks) - 1):
            w = toks[idx].lower()
            if is_good_word_token(w) and sc.is_known(w) and not sc._protect_realword(w):
                occ_map[w].append((si, idx))

    # Model functions needed for neighbor generation
    # Use edits1 from whichever module defines it (model_file or spelling.model).
    edits1 = getattr(model_module, "edits1", None)
    if edits1 is None:
        raise RuntimeError("Could not find edits1() in the loaded model module.")

    def is_plausible_realword(w: str) -> bool:
        return (w in getattr(sc, "domain_vocab", set())) and (sc.unigrams.get(w, 0) >= 10)

    @lru_cache(maxsize=50000)
    def neighbors_dist1(word: str) -> Tuple[str, ...]:
        w = word.lower()
        out = set()
        for cand in edits1(w):
            if cand == w:
                continue
            if not cand.isalpha() or len(cand) < 4:
                continue
            if sc.is_known(cand) and not sc._protect_realword(cand) and is_plausible_realword(cand):
                out.add(cand)
        return tuple(sorted(out))

    # Cache candidate generation by wrong token (speed!)
    SpellCandidate = getattr(model_module, "SpellCandidate")
    _score = getattr(sc, "_score")
    _adaptive_keep_threshold = getattr(sc, "_adaptive_keep_threshold")
    _special_correction = getattr(sc, "_special_correction")

    @lru_cache(maxsize=50000)
    def candidates_for_word(w: str):
        wl = w.lower()
        cands = [SpellCandidate(word=wl, dist=0)]
        cands.extend(sc._generate_candidates(wl))
        return tuple(cands)

    def simulate_correct(wrong: str, prev: Optional[str], next_: Optional[str]) -> Tuple[str, str]:
        if not wrong:
            return wrong, "Correct"
        w = wrong.lower()
        prev_l = prev.lower() if prev else None
        next_l = next_.lower() if next_ else None
        spec = _special_correction(w, prev_l)
        if spec is not None:
            return spec, "Real-word error"
        if not any(ch.isalpha() for ch in w):
            return wrong, "Correct"
        if not sc.is_known(w):
            return sc.correct_token(wrong, prev_word=prev, next_word=next_)
        if sc._protect_realword(w):
            return wrong, "Correct"

        cands = candidates_for_word(w)
        orig_len = len(w)
        scored: List[Tuple[float, object]] = []
        for c in cands:
            base_score = _score(c, prev_l, next_l)
            len_pen = -sc.w_len_diff * abs(len(c.word) - orig_len)
            scored.append((base_score + len_pen, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_c = scored[0]
        orig_score = next((s for s, c in scored if c.word == w), -1e9)
        keep_threshold = _adaptive_keep_threshold(
            original_word=w,
            original_score=orig_score,
            best_candidate=best_c,
            prev_word=prev_l,
            next_word=next_l,
        )
        if best_c.word != w and best_score > keep_threshold:
            return best_c.word, "Real-word error"
        return wrong, "Correct"

    # Pick frequent expected words and form confusion pairs
    top_expected = sorted(occ_map.keys(), key=lambda w: len(occ_map[w]), reverse=True)[:2500]
    confusion_pairs: List[Tuple[str, str]] = []
    for exp in top_expected:
        if not is_plausible_realword(exp):
            continue
        neigh = neighbors_dist1(exp)
        if not neigh:
            continue
        for wrong in neigh[:8]:
            confusion_pairs.append((exp, wrong))
    random.shuffle(confusion_pairs)

    # Fill targets by behavioral classification
    target_per_label = {"left-only": args.n_realword // 3, "right-only": args.n_realword // 3, "both-needed": args.n_realword - 2*(args.n_realword // 3)}
    have = Counter([c["subcategory"] for c in cases if c["category"] == "real-word"])
    remaining = {k: max(0, target_per_label[k] - have.get(k, 0)) for k in target_per_label}

    produced = defaultdict(int)
    for expected, wrong in confusion_pairs:
        if sum(produced.values()) >= args.n_realword:
            break
        if all(produced[k] >= remaining[k] for k in remaining):
            break
        occs = occ_map.get(expected) or []
        if not occs:
            continue
        # sample multiple contexts
        for si, idx in random.sample(occs, k=min(5, len(occs))):
            toks = sentences[si]
            prev = toks[idx - 1].lower()
            next_ = toks[idx + 1].lower()
            corr_both, _ = simulate_correct(wrong, prev, next_)
            if corr_both != expected:
                continue
            corr_left, _ = simulate_correct(wrong, prev, None)
            corr_right, _ = simulate_correct(wrong, None, next_)
            if corr_left == expected and corr_right != expected:
                label = "left-only"
            elif corr_right == expected and corr_left != expected:
                label = "right-only"
            elif corr_left != expected and corr_right != expected:
                label = "both-needed"
            else:
                continue

            if label not in remaining or produced[label] >= remaining[label]:
                continue

            case = {
                "case_id": "",
                "source": "test2.csv",
                "category": "real-word",
                "subcategory": label,
                "prev": prev,
                "wrong": wrong,
                "next": next_,
                "expected": expected,
                "note": f"realword dist1 confusion pair; label={label}",
                "fragment": make_fragment(toks, idx, wrong),
            }
            if add_case(case):
                produced[label] += 1
            if all(produced[k] >= remaining[k] for k in remaining):
                break

    # Finalize
    if len(cases) < (args.n_nonword + args.n_realword + args.n_covid + args.n_negative + args.n_general):
        print(f"[WARN] Only generated {len(cases)} cases (requested total may not be met).")

    out = pd.DataFrame(cases)
    out["case_id"] = [f"t2_{i:04d}" for i in range(1, len(out) + 1)]
    cols = ["case_id","source","category","subcategory","prev","wrong","next","expected","note","fragment"]
    out = out[cols]
    out.to_csv(args.out_csv, index=False, encoding="utf-8")

    print("Saved:", args.out_csv)
    print(out["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
