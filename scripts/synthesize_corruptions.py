#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Synthesize OCR-like / spelling-like noise for CORD-19 abstracts (paired evaluation dataset).

What it does
-----------
1) Reads a test CSV (e.g., test.csv) containing an abstract column.
2) Runs the *existing* trained classifier on the CLEAN text to get label_clean (baseline).
   - If a ground-truth label column exists (label/topic/class/...), it is preserved as label_true.
3) Generates synthetic OCR/spelling noise (mild/moderate/severe) producing corrupted_text.
4) Exports a CSV where each row is a (clean, corrupted) pair with:
   - clean_text, corrupted_text
   - label_clean (model prediction on clean)
   - label_true (if present)
   - corruption_level, variant_id, edit_rate, seed_variant
   - optional probability columns if label_map exists

Scientific note
--------------
- If your CSV contains TRUE labels: you can compute true accuracy/F1 deltas.
- If not: label_clean is a *pseudo-label* (model's clean prediction). This supports measuring
  robustness & recovery (agreement with clean predictions), not "true accuracy" vs humans.

Usage (recommended: from project root)
--------------------------------------
Put this file in: ./scripts/synthesize_corruptions.py

Run:
  python scripts/synthesize_corruptions.py --input test.csv --output test_synth.csv --text_col abstract

Options:
  --levels mild,moderate,severe
  --variants 3
  --seed 42
  --max_chars 500     (to match the Streamlit UI constraint)
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import pickle


# -----------------------
# Project bootstrap
# -----------------------
def find_project_root(start: Path, max_up: int = 8) -> Path:
    p = start
    for _ in range(max_up):
        if (p / "config.py").exists():
            return p
        p = p.parent
    return start.parent


ROOT = find_project_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---- Robust preprocessing import (best fidelity when available) ----
try:
    from utils.preprocess import preprocess_for_classification as _preprocess_for_classification
except Exception:
    # Fallback: light regex cleaner (keeps biomedical tokens mostly intact)
    _URL_RE = re.compile(r"http\S+|www\.\S+")
    _WS_RE = re.compile(r"\s+")
    _BAD_CHARS = re.compile(r"[^A-Za-z0-9\-\s]")

    def _preprocess_for_classification(text: str) -> str:
        text = "" if text is None else str(text)
        text = _URL_RE.sub(" ", text)
        text = re.sub(r"[‐‑‒–—−]", "-", text)  # normalize unicode hyphens
        text = _BAD_CHARS.sub(" ", text)
        text = _WS_RE.sub(" ", text).strip()
        return text


def clean_text_wrapper(text_series):
    """
    Legacy wrapper sometimes referenced by the pickled Pipeline.
    Keep this name at top-level so joblib/pickle can resolve it.
    """
    if isinstance(text_series, pd.DataFrame):
        return text_series.iloc[:, 0].apply(_preprocess_for_classification)
    if isinstance(text_series, pd.Series):
        return text_series.apply(_preprocess_for_classification)
    return [_preprocess_for_classification(t) for t in text_series]


from config import CLASSIF_ARTIFACT_DIR


# -----------------------
# Load classifier artifacts
# -----------------------
def load_classifier() -> Tuple[Any, Optional[Dict[str, int]], Dict[str, Any]]:
    joblib_path = Path(CLASSIF_ARTIFACT_DIR) / "topic_classifier.joblib"
    pkl_path = Path(CLASSIF_ARTIFACT_DIR) / "topic_classifier.pkl"

    def _help_version_error(e: Exception) -> str:
        return (
            f"{e}\n\n"
            "This usually means your environment versions don't match the versions used to SAVE the model.\n"
            "Fix:\n"
            "  1) Install the project's pinned versions from requirements.txt (especially numpy + scikit-learn).\n"
            "  2) Re-run training to regenerate artifacts if needed.\n"
        )

    # Prefer joblib
    if joblib_path.exists():
        try:
            data = joblib.load(str(joblib_path))
            model = data.get("model")
            label_map = data.get("label_map") or data.get("label_mapping") or data.get("label_map_")
            metrics = data.get("metrics", {}) or {}
            return model, label_map, metrics
        except Exception as e:
            raise RuntimeError("Failed to load joblib classifier.\n" + _help_version_error(e))

    if pkl_path.exists():
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            model = data.get("model")
            label_map = data.get("label_map") or data.get("label_mapping") or data.get("label_map_")
            metrics = data.get("metrics", {}) or {}
            return model, label_map, metrics
        except Exception as e:
            raise RuntimeError("Failed to load pickle classifier.\n" + _help_version_error(e))

    raise FileNotFoundError(
        "Classifier artifacts not found. Expected topic_classifier.joblib or topic_classifier.pkl in classification/artifacts"
    )


def invert_label_map(label_map: Optional[Dict[str, int]], model: Any) -> Dict[int, str]:
    if isinstance(label_map, dict) and len(label_map) > 0:
        return {int(v): str(k) for k, v in label_map.items()}

    if hasattr(model, "classes_"):
        try:
            out = {}
            for c in list(model.classes_):
                try:
                    out[int(c)] = str(c)
                except Exception:
                    pass
            if out:
                return out
        except Exception:
            pass

    # last resort (matches your project)
    return {0: "Prevention", 1: "Treatment", 2: "Epidemiology"}


# -----------------------
# Corruption engine
# -----------------------
TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z0-9]+)*|\d+(?:\.\d+)?|\s+|[^\w\s]", re.UNICODE)

LIGHT_STOPWORDS = {
    "the", "and", "with", "from", "that", "this", "were", "was", "are", "for", "into",
    "over", "under", "than", "then", "have", "has", "had", "been", "being",
    "these", "those", "their", "there", "here", "such", "also", "using", "used",
}

OCR_CONFUSIONS = [
    # ---------------------------------------------------------
    # 1. Structural & Ligature Confusions 
    # ---------------------------------------------------------
    ("rn", "m"), ("m", "rn"),       # Classic: 'modern' <-> 'rnodern'
    ("nn", "m"), ("m", "nn"),
    ("nm", "nn"), ("nn", "nm"),     # Common in dense text
    ("cl", "d"), ("d", "cl"),       # Classic: 'clear' <-> 'dear'
    ("ck", "d"), ("d", "ck"),
    ("lc", "k"), ("k", "lc"),
    ("li", "h"), ("h", "li"),       # 'solid' <-> 'sohid'
    ("lo", "b"), ("b", "lo"),
    ("ol", "d"), ("d", "ol"),
    ("vv", "w"), ("w", "vv"),       # 'wave' <-> 'vvave'
    ("ii", "u"), ("u", "ii"),       # 'fluid' <-> 'fliid'
    ("fi", "h"), ("h", "fi"),       # Ligature failure (dot merging)
    ("fl", "tl"), ("tl", "fl"),
    ("ri", "n"), ("n", "ri"),
    ("in", "m"), ("m", "in"),       # 'immun' <-> 'inmun' (Medical suffix issue)
    ("cj", "g"), ("g", "cj"),       
    ("lI", "U"), ("U", "lI"),       # Capital/Lower mix
    
    # ---------------------------------------------------------
    # 2. Numeric vs. Alphabetic 
    # ---------------------------------------------------------
    ("0", "o"), ("o", "0"),         # 'covid-19' <-> 'c0vid-19'
    ("0", "O"), ("O", "0"),
    ("0", "Q"), ("Q", "0"),
    ("0", "D"), ("D", "0"),         # Low res uppercase
    ("0", "C"), ("C", "0"),
    ("1", "l"), ("l", "1"),         # '10ml' <-> 'l0ml'
    ("1", "I"), ("I", "1"),         # 'Type I' <-> 'Type 1'
    ("1", "i"), ("i", "1"),
    ("1", "|"), ("|", "1"),         # PDF Table borders confusion
    ("1", "t"), ("t", "1"),
    ("1", "f"), ("f", "1"),
    ("2", "z"), ("z", "2"),         # '2mg' <-> 'zmg'
    ("2", "Z"), ("Z", "2"),
    ("3", "e"), ("e", "3"),         # Rare but happens in bad fonts
    ("4", "A"), ("A", "4"),
    ("5", "s"), ("s", "5"),         # '50%' <-> 's0%'
    ("5", "S"), ("S", "5"),
    ("6", "b"), ("b", "6"),
    ("6", "G"), ("G", "6"),
    ("7", "T"), ("T", "7"),
    ("7", "Y"), ("Y", "7"),         # 'Year' <-> '7ear'
    ("8", "B"), ("B", "8"),         # 'Cells-B' <-> 'Cells-8'
    ("8", "g"), ("g", "8"),
    ("9", "g"), ("g", "9"),
    ("9", "q"), ("q", "9"),

    # ---------------------------------------------------------
    # 3. Visual Similarity - Lowercase 
    # (Blurry Scans)
    # ---------------------------------------------------------
    ("c", "e"), ("e", "c"),         # Extremely common
    ("c", "o"), ("o", "c"),
    ("e", "o"), ("o", "e"),
    ("a", "o"), ("o", "a"),
    ("a", "e"), ("e", "a"),         # 'heart' <-> 'heort'
    ("a", "s"), ("s", "a"),
    ("i", "j"), ("j", "i"),
    ("i", "l"), ("l", "i"),         # 'clinical' <-> 'cllnlcal'
    ("l", "t"), ("t", "l"),
    ("f", "t"), ("t", "f"),         # 'often' <-> 'offen'
    ("r", "n"), ("n", "r"),         # 'rna' <-> 'nna'
    ("u", "v"), ("v", "u"),
    ("y", "v"), ("v", "y"),
    ("h", "b"), ("b", "h"),
    ("k", "x"), ("x", "k"),         # Scientific formulas
    ("g", "q"), ("q", "g"),
    ("p", "q"), ("q", "p"),         # Mirroring
    ("d", "b"), ("b", "d"),         # Mirroring
    ("v", "r"), ("r", "v"),         # Italic fonts specific

    # ---------------------------------------------------------
    # 4. Visual Similarity - Uppercase 
    # ---------------------------------------------------------
    ("D", "O"), ("O", "D"),
    ("E", "F"), ("F", "E"),         # Missing bottom stroke
    ("B", "E"), ("E", "B"),
    ("M", "N"), ("N", "M"),         # Thin middle stroke
    ("H", "N"), ("N", "H"),
    ("K", "X"), ("X", "K"),
    ("V", "U"), ("U", "V"),
    ("I", "L"), ("L", "I"),         # 'IL-6' <-> 'II-6'
    ("P", "R"), ("R", "P"),

    # ---------------------------------------------------------
    # 5. Punctuation & Noise 
    # ---------------------------------------------------------
    (".", ","), (",", "."),         # Decimal point errors
    (":", ";"), (";", ":"),
    ("-", "_"), ("_", "-"),
    ("'", "`"), ("`", "'"),
    ("!", "l"), ("l", "!"),
    ("(", "C"), ("C", "("),         # 'C(t)' <-> 'CCt)'
]

CHAR_CONFUSIONS = {
    "a": list("aoe"),
    "c": list("ceo"),
    "e": list("eca"),
    "i": list("il"),
    "l": list("li"),
    "m": list("n"),
    "n": list("m"),
    "o": list("oa"),
    "s": list("sz"),
    "u": list("uv"),
    "v": list("uv"),
    "z": list("zs"),
}


@dataclass
class CorruptionConfig:
    p_token: float
    max_edits_per_token: int
    p_ocr_bigram: float = 0.35
    p_transpose: float = 0.25
    p_delete: float = 0.25
    p_insert: float = 0.25
    p_substitute: float = 0.25
    protect_biomedical: bool = True
    allow_punct_noise: bool = False


LEVELS: Dict[str, CorruptionConfig] = {
    "mild": CorruptionConfig(p_token=0.035, max_edits_per_token=1),
    "moderate": CorruptionConfig(p_token=0.070, max_edits_per_token=1),
    "severe": CorruptionConfig(p_token=0.110, max_edits_per_token=2),
}


def is_biomedical_token(tok: str) -> bool:
    # Example biomedical tokens: covid-19, sars-cov-2, il-6, ace2, h1n1 ...
    if re.search(r"\d", tok) and re.search(r"[A-Za-z]", tok):
        return True
    if "-" in tok and re.search(r"[A-Za-z]", tok):
        return True
    return False


def eligible_word(tok: str, cfg: CorruptionConfig) -> bool:
    if not tok or not tok.isalpha():
        return False
    low = tok.lower()
    if len(low) < 5:
        return False
    if low in LIGHT_STOPWORDS:
        return False
    if cfg.protect_biomedical and is_biomedical_token(tok):
        return False
    return True


def apply_one_edit(word: str, rng: random.Random, cfg: CorruptionConfig) -> str:
    if not word:
        return word

    cap_first = word[0].isupper()
    w = word.lower()

    # OCR bigram replacements
    if rng.random() < cfg.p_ocr_bigram:
        candidates = [(a, b) for (a, b) in OCR_CONFUSIONS if a in w]
        if candidates:
            a, b = rng.choice(candidates)
            w2 = w.replace(a, b, 1)
            return (w2[:1].upper() + w2[1:]) if cap_first else w2

    # Choose edit type
    edits = [("transpose", cfg.p_transpose), ("delete", cfg.p_delete), ("insert", cfg.p_insert), ("substitute", cfg.p_substitute)]
    total = sum(p for _, p in edits)
    r = rng.random() * total
    acc = 0.0
    etype = "substitute"
    for name, p in edits:
        acc += p
        if r <= acc:
            etype = name
            break

    if etype == "transpose" and len(w) >= 2:
        i = rng.randrange(0, len(w) - 1)
        w2 = w[:i] + w[i + 1] + w[i] + w[i + 2:]
    elif etype == "delete" and len(w) >= 2:
        i = rng.randrange(0, len(w))
        w2 = w[:i] + w[i + 1:]
    elif etype == "insert":
        i = rng.randrange(0, len(w) + 1)
        ch = rng.choice(list("abcdefghijklmnopqrstuvwxyz"))
        w2 = w[:i] + ch + w[i:]
    else:  # substitute
        i = rng.randrange(0, len(w))
        ch0 = w[i]
        choices = CHAR_CONFUSIONS.get(ch0, list("abcdefghijklmnopqrstuvwxyz"))
        ch = rng.choice(choices)
        w2 = w[:i] + ch + w[i + 1:]

    return (w2[:1].upper() + w2[1:]) if cap_first else w2


def corrupt_text(text: str, cfg: CorruptionConfig, rng: random.Random) -> Tuple[str, int, int]:
    """
    Returns: (corrupted_text, edited_token_count, total_word_tokens)
    """
    parts = TOKEN_RE.findall(text)
    total_words = 0
    edited = 0
    out_parts: List[str] = []

    for part in parts:
        if part.isspace():
            out_parts.append(part)
            continue

        if part.isalpha():
            total_words += 1
            if eligible_word(part, cfg) and rng.random() < cfg.p_token:
                w = part
                n_edits = rng.randint(1, cfg.max_edits_per_token)
                for _ in range(n_edits):
                    w = apply_one_edit(w, rng, cfg)
                out_parts.append(w)
                edited += 1
            else:
                out_parts.append(part)
            continue

        # punctuation noise (optional)
        if cfg.allow_punct_noise and part in [",", ".", ";", ":"] and rng.random() < (cfg.p_token / 6.0):
            out_parts.append("" if rng.random() < 0.5 else part)
        else:
            out_parts.append(part)

    return "".join(out_parts), edited, total_words


# -----------------------
# Utilities
# -----------------------
def detect_label_column(columns: List[str]) -> Optional[str]:
    candidates = ["label", "true_label", "topic", "class", "y", "target", "category"]
    cols_lower = {c.lower(): c for c in columns}
    for k in candidates:
        if k in cols_lower:
            return cols_lower[k]
    return None


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV (test set).")
    ap.add_argument("--output", required=True, help="Output CSV path.")
    ap.add_argument("--text_col", default="abstract", help="Text column (default: abstract).")
    ap.add_argument("--id_col", default="cord_uid", help="Optional ID column to keep.")
    ap.add_argument("--levels", default="mild,moderate,severe", help="Comma-separated levels.")
    ap.add_argument("--variants", type=int, default=1, help="Variants per row per level.")
    ap.add_argument("--seed", type=int, default=42, help="Global seed.")
    ap.add_argument("--max_chars", type=int, default=0, help="Optional truncation (0=no truncation).")
    ap.add_argument("--protect_biomedical", action="store_true", help="Protect biomedical tokens (recommended).")
    ap.add_argument("--allow_punct_noise", action="store_true", help="Allow punctuation OCR noise (optional).")
    args = ap.parse_args()

    model, label_map, _metrics = load_classifier()
    inv_map = invert_label_map(label_map, model)

    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise ValueError(f"text_col='{args.text_col}' not found. Available: {list(df.columns)}")

    label_col = detect_label_column(list(df.columns))

    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    for lv in levels:
        if lv not in LEVELS:
            raise ValueError(f"Unknown level '{lv}'. Choose from: {list(LEVELS.keys())}")

    rows: List[Dict[str, Any]] = []

    for i, row in df.iterrows():
        text = row.get(args.text_col, "")
        if pd.isna(text) or not str(text).strip():
            continue

        text = str(text)
        if args.max_chars and args.max_chars > 0:
            text = text[: args.max_chars]

        # CLEAN prediction (baseline)
        pred_idx = int(model.predict([text])[0])
        pred_label = inv_map.get(pred_idx, str(pred_idx))

        probs = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba([text])[0]
            except Exception:
                probs = None

        base: Dict[str, Any] = {
            "row_index": int(i),
            "clean_text": text,
            "label_clean": pred_label,
            "pred_idx_clean": pred_idx,
        }

        if label_col is not None:
            base["label_true"] = row.get(label_col)

        if args.id_col in df.columns:
            base[args.id_col] = row.get(args.id_col)

        # Store clean probabilities if label_map exists
        if probs is not None and isinstance(label_map, dict):
            for lbl, idx in label_map.items():
                base[f"proba_clean_{lbl}"] = float(probs[int(idx)])

        for lv in levels:
            cfg0 = LEVELS[lv]
            cfg = CorruptionConfig(**cfg0.__dict__)
            cfg.protect_biomedical = bool(args.protect_biomedical or cfg.protect_biomedical)
            cfg.allow_punct_noise = bool(args.allow_punct_noise)

            for v in range(args.variants):
                local_seed = int(
                    hashlib.md5(f"{args.seed}|{i}|{lv}|{v}".encode("utf-8")).hexdigest()[:8],
                    16,
                )
                rng = random.Random(local_seed)

                corrupted, edited, total_words = corrupt_text(text, cfg, rng)
                if args.max_chars and args.max_chars > 0:
                    corrupted = corrupted[: args.max_chars]

                rec = dict(base)
                rec.update(
                    {
                        "corruption_level": lv,
                        "variant_id": v,
                        "seed_variant": local_seed,
                        "corrupted_text": corrupted,
                        "edited_tokens": int(edited),
                        "total_word_tokens": int(total_words),
                        "edit_rate": float(edited / max(1, total_words)),
                    }
                )
                rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False, encoding="utf-8")

    print(f"[OK] Wrote: {args.output}")
    print(f"Rows: {len(out)} | Levels: {levels} | Variants: {args.variants}")
    if label_col is None:
        print("[NOTE] No ground-truth label column detected.")
        print("       label_clean is a pseudo-label (model prediction on clean).")
        print("       Use it for robustness/recovery; use label_true for real accuracy deltas.")


if __name__ == "__main__":
    main()
