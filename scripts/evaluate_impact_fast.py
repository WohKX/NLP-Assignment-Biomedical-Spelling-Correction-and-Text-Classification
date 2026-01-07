#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_impact_fast.py
-----------------------
A faster, more practical evaluator for the CORD-19 project.

Why faster than the original evaluate_impact.py?
- Uses *batch* predictions instead of per-row model.predict([text]) calls.
- Avoids re-correcting duplicated texts via caching (very common in test_synth variants).
- Default correction mode is **conservative non-word correction**:
    *only correct tokens that are not in the lexicon + apply domain special rules.*
  This matches the project's "do-no-harm" intent and is dramatically faster.

It still implements the same logical runs:

1) Baseline: clean_text  -> classifier -> pred_clean
2) Noise:    noisy_text  -> classifier -> pred_noisy
3) Repair:   noisy_text  -> spellchecker -> repaired_text -> classifier -> pred_repaired
4) Safety:   clean_text  -> spellchecker -> safe_check_text -> classifier -> pred_safe

Input columns:
- clean_text (required)
- noisy_text OR corrupted_text (required)
- label OR label_true OR label_clean (required)

Outputs:
- Console report (macro-F1 + accuracy)
- Optional per-row CSV with predictions and texts
- Optional JSON report
- Optional bootstrap CI for delta (after predictions; quick)

Usage:
  python scripts/evaluate_impact_fast.py --input test_synth.csv --max_rows 500 --output_rows outputs/rows.csv

If you want to reproduce the *slow* behavior (correct every token, including real-word):
  python scripts/evaluate_impact_fast.py --input ... --full_correct

Recommended workflow:
  1) Run without bootstrap to ensure it finishes.
  2) Re-run with bootstrap once predictions finish in minutes.

"""

from __future__ import annotations

import argparse
import json
import sys
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:
    joblib = None

from sklearn.metrics import f1_score, accuracy_score


# -----------------------
# Project bootstrap
# -----------------------
def find_project_root(start: Path, max_up: int = 10) -> Path:
    p = start
    for _ in range(max_up):
        if (p / "config.py").exists():
            return p
        p = p.parent
    return start.parent


ROOT = find_project_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from utils.runtime import ensure_project_root
    ensure_project_root()
except Exception:
    pass

from config import SPELLING_ARTIFACT_DIR, CLASSIF_ARTIFACT_DIR
from spelling.model import MedicalSpellChecker


# -----------------------
# Artifact loading
# -----------------------
def load_classifier() -> Tuple[Any, Optional[Dict[str, int]]]:
    joblib_path = Path(CLASSIF_ARTIFACT_DIR) / "topic_classifier.joblib"
    pkl_path = Path(CLASSIF_ARTIFACT_DIR) / "topic_classifier.pkl"

    if joblib_path.exists() and joblib is not None:
        data = joblib.load(str(joblib_path))
        return data.get("model"), data.get("label_map") or data.get("label_mapping")

    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        return data.get("model"), data.get("label_map") or data.get("label_mapping")

    raise FileNotFoundError("Classifier artifacts not found in CLASSIF_ARTIFACT_DIR.")


def invert_label_map(label_map: Optional[Dict[str, int]], model: Any) -> Dict[int, str]:
    if isinstance(label_map, dict) and len(label_map) > 0:
        return {int(v): str(k) for k, v in label_map.items()}
    if hasattr(model, "classes_"):
        try:
            inv = {}
            for c in list(model.classes_):
                try:
                    inv[int(c)] = str(c)
                except Exception:
                    pass
            if inv:
                return inv
        except Exception:
            pass
    return {0: "Prevention", 1: "Treatment", 2: "Epidemiology"}


def load_spellchecker() -> MedicalSpellChecker:
    return MedicalSpellChecker.from_artifacts(SPELLING_ARTIFACT_DIR)


# -----------------------
# Column detection
# -----------------------
def detect_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    cols = {c.lower(): c for c in df.columns}
    clean_col = cols.get("clean_text")
    if clean_col is None:
        raise ValueError("Input must contain 'clean_text'.")
    noisy_col = cols.get("noisy_text") or cols.get("corrupted_text") or cols.get("noisy")
    if noisy_col is None:
        raise ValueError("Input must contain 'noisy_text' or 'corrupted_text'.")
    label_col = cols.get("label") or cols.get("label_true") or cols.get("label_clean")
    if label_col is None:
        raise ValueError("Input must contain 'label' or label_true/label_clean.")
    return clean_col, noisy_col, label_col


# -----------------------
# Fast correction logic
# -----------------------
def _is_alphaish(s: str) -> bool:
    # avoid importing internal helper; keep compatible
    return any(ch.isalpha() for ch in s)


def _tokenize_simple(text: str) -> List[str]:
    # We *do not* use preprocess_for_spelling here because the spellchecker already
    # handles normalization internally and we want speed.
    # If you want strict parity with app preprocessing, set --strict_preprocess.
    return [t for t in str(text).split() if t]


def correct_text_conservative(
    spell: MedicalSpellChecker,
    text: str,
    strict_preprocess: bool = False,
) -> str:
    """
    Conservative, fast: only correct NON-WORD errors (OOV) + special corrections.

    This is much faster than calling spell.correct() for every token, and aligns
    with a do-no-harm policy: known tokens are left unchanged.
    """
    if text is None:
        return ""
    text = str(text)
    if not text.strip():
        return ""

    if strict_preprocess:
        # Optional strict parity with app
        from utils.preprocess import preprocess_for_spelling  # local import
        toks = preprocess_for_spelling(text)
        if isinstance(toks, str):
            toks = toks.split()
        toks = [t for t in list(toks) if t]
    else:
        toks = _tokenize_simple(text)

    out: List[str] = []
    prev_out: Optional[str] = None

    for tok in toks:
        if not tok:
            continue

        # keep non-alphaish tokens as-is (numbers, punctuation, etc.)
        if not _is_alphaish(tok):
            out.append(tok)
            prev_out = tok
            continue

        prev_l = prev_out.lower() if isinstance(prev_out, str) else None

        # apply domain special corrections (covid19 -> covid-19, etc.)
        try:
            spec = spell._special_correction(tok, prev_l, None)
        except Exception:
            spec = None

        if spec is not None and spec != tok:
            out.append(spec)
            prev_out = spec
            continue

        w = tok.lower()

        # ONLY correct if OOV (non-word error). If known -> keep unchanged.
        try:
            known = spell.is_known(w)
        except Exception:
            known = False

        if known:
            out.append(tok)
            prev_out = tok
            continue

        # unknown => do the expensive correction
        corrected, _etype = spell.correct(tok, prev_word=prev_out)
        out.append(corrected)
        prev_out = corrected

    return " ".join(out).strip()


def correct_text_full(
    spell: MedicalSpellChecker,
    text: str,
    strict_preprocess: bool = False,
) -> str:
    """
    Slow / exact: calls spell.correct() for every token (may attempt real-word fixes).
    Use only if you *really* need it.
    """
    if text is None:
        return ""
    text = str(text)
    if not text.strip():
        return ""

    if strict_preprocess:
        from utils.preprocess import preprocess_for_spelling
        toks = preprocess_for_spelling(text)
        if isinstance(toks, str):
            toks = toks.split()
        toks = [t for t in list(toks) if t]
    else:
        toks = _tokenize_simple(text)

    out: List[str] = []
    prev_out: Optional[str] = None
    for tok in toks:
        corrected, _etype = spell.correct(tok, prev_word=prev_out)
        out.append(corrected)
        prev_out = corrected
    return " ".join(out).strip()


# -----------------------
# Metrics + bootstrap
# -----------------------
@dataclass
class MetricsBlock:
    f1_macro: float
    accuracy: float


def compute_metrics(y_true: List[str], y_pred: List[str]) -> MetricsBlock:
    return MetricsBlock(
        f1_macro=float(f1_score(y_true, y_pred, average="macro")),
        accuracy=float(accuracy_score(y_true, y_pred)),
    )


def bootstrap_ci_delta(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric: str,
    n_boot: int,
    seed: int,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        pa = pred_a[idx]
        pb = pred_b[idx]
        if metric == "accuracy":
            da = accuracy_score(yt, pa)
            db = accuracy_score(yt, pb)
        else:
            da = f1_score(yt, pa, average="macro")
            db = f1_score(yt, pb, average="macro")
        deltas[i] = (db - da)
    delta = float(deltas.mean())
    lo = float(np.quantile(deltas, 0.025))
    hi = float(np.quantile(deltas, 0.975))
    return delta, lo, hi


# -----------------------
# Prediction helpers (batched)
# -----------------------
def predict_labels_batched(model: Any, inv_map: Dict[int, str], texts: List[str], batch_size: int = 256) -> List[str]:
    out: List[str] = []
    n = len(texts)
    for i in range(0, n, batch_size):
        chunk = texts[i:i + batch_size]
        preds = model.predict(chunk)
        out.extend([inv_map.get(int(p), str(p)) for p in preds])
    return out


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV: test_synth.csv or test_set_synthetic.csv")
    ap.add_argument("--output_rows", default="", help="Optional CSV with per-row predictions/texts.")
    ap.add_argument("--output_report", default="", help="Optional JSON report path.")
    ap.add_argument("--max_rows", type=int, default=0, help="0=all rows; otherwise first N.")
    ap.add_argument("--bootstrap", type=int, default=0, help="Bootstrap samples for delta CI (0=off).")
    ap.add_argument("--seed", type=int, default=42, help="Bootstrap seed.")
    ap.add_argument("--batch_size", type=int, default=256, help="Batch size for classifier predictions.")
    ap.add_argument("--log_every", type=int, default=100, help="Progress log frequency (rows).")
    ap.add_argument("--strict_preprocess", action="store_true", help="Use preprocess_for_spelling tokenization (slower, closer to app).")
    ap.add_argument("--full_correct", action="store_true", help="Slow mode: correct every token (may try real-word fixes).")
    ap.add_argument("--unique_safety", action="store_true", help="Compute safety on UNIQUE clean_text only (recommended for test_synth variants).")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    clean_col, noisy_col, label_col = detect_columns(df)

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    # Clean invalid rows early
    df = df[[clean_col, noisy_col, label_col]].copy()
    df = df.dropna()
    df[clean_col] = df[clean_col].astype(str)
    df[noisy_col] = df[noisy_col].astype(str)
    df[label_col] = df[label_col].astype(str)
    df = df[(df[clean_col].str.strip() != "") & (df[noisy_col].str.strip() != "") & (df[label_col].str.strip() != "")]

    if len(df) == 0:
        raise RuntimeError("No valid rows to evaluate after cleaning empty/NA values.")

    model, label_map = load_classifier()
    inv_map = invert_label_map(label_map, model)
    spell = load_spellchecker()

    correct_fn = correct_text_full if args.full_correct else correct_text_conservative

    # --- Prepare lists
    y_true = df[label_col].tolist()
    clean_texts = df[clean_col].tolist()
    noisy_texts = df[noisy_col].tolist()

    # --- Run 1 + 2 (batched)
    print(f"[1/4] Predicting baseline (clean) on {len(clean_texts)} rows ...", flush=True)
    pred_clean = predict_labels_batched(model, inv_map, clean_texts, batch_size=args.batch_size)

    print(f"[2/4] Predicting noisy on {len(noisy_texts)} rows ...", flush=True)
    pred_noisy = predict_labels_batched(model, inv_map, noisy_texts, batch_size=args.batch_size)

    # --- Run 3 (repair): correct noisy then predict repaired (batched)
    print(f"[3/4] Correcting noisy texts (spellchecker) ...", flush=True)
    repaired_cache: Dict[str, str] = {}
    repaired_texts: List[str] = []
    for i, t in enumerate(noisy_texts, start=1):
        if t in repaired_cache:
            repaired_texts.append(repaired_cache[t])
        else:
            repaired = correct_fn(spell, t, strict_preprocess=args.strict_preprocess)
            repaired_cache[t] = repaired
            repaired_texts.append(repaired)
        if args.log_every and (i % args.log_every == 0):
            print(f"  - repaired {i}/{len(noisy_texts)}", flush=True)

    print(f"    Predicting repaired texts ...", flush=True)
    pred_repaired = predict_labels_batched(model, inv_map, repaired_texts, batch_size=args.batch_size)

    # --- Run 4 (safety): correct clean then predict safe
    print(f"[4/4] Safety check (correct clean texts) ...", flush=True)
    if args.unique_safety:
        # Evaluate safety on unique clean texts to avoid redoing work for test_synth variants.
        unique_clean = pd.Series(clean_texts).unique().tolist()
        safe_cache: Dict[str, str] = {}
        safe_unique: List[str] = []
        for i, t in enumerate(unique_clean, start=1):
            safe_t = correct_fn(spell, t, strict_preprocess=args.strict_preprocess)
            safe_cache[t] = safe_t
            safe_unique.append(safe_t)
            if args.log_every and (i % args.log_every == 0):
                print(f"  - safe(unique) {i}/{len(unique_clean)}", flush=True)

        pred_safe_unique = predict_labels_batched(model, inv_map, safe_unique, batch_size=args.batch_size)
        safe_pred_map = {ct: ps for ct, ps in zip(unique_clean, pred_safe_unique)}
        pred_safe = [safe_pred_map[t] for t in clean_texts]

        safe_check_texts = [safe_cache[t] for t in clean_texts]
    else:
        safe_cache: Dict[str, str] = {}
        safe_check_texts: List[str] = []
        for i, t in enumerate(clean_texts, start=1):
            if t in safe_cache:
                safe_check_texts.append(safe_cache[t])
            else:
                safe_t = correct_fn(spell, t, strict_preprocess=args.strict_preprocess)
                safe_cache[t] = safe_t
                safe_check_texts.append(safe_t)
            if args.log_every and (i % args.log_every == 0):
                print(f"  - safe {i}/{len(clean_texts)}", flush=True)

        print("    Predicting safety texts ...", flush=True)
        pred_safe = predict_labels_batched(model, inv_map, safe_check_texts, batch_size=args.batch_size)

    # --- Metrics
    m_clean = compute_metrics(y_true, pred_clean)
    m_noisy = compute_metrics(y_true, pred_noisy)
    m_rest = compute_metrics(y_true, pred_repaired)
    m_safe = compute_metrics(y_true, pred_safe)

    delta_recovery_f1 = m_rest.f1_macro - m_noisy.f1_macro
    safety_drop_f1 = m_clean.f1_macro - m_safe.f1_macro

    denom = (m_clean.f1_macro - m_noisy.f1_macro)
    err_pct = (delta_recovery_f1 / denom * 100.0) if denom > 1e-12 else float("nan")

    noisy_err = 1.0 - m_noisy.f1_macro
    rest_err = 1.0 - m_rest.f1_macro
    err_alt = ((noisy_err - rest_err) / noisy_err * 100.0) if noisy_err > 1e-12 else float("nan")

    report: Dict[str, Any] = {
        "n_rows_evaluated": int(len(y_true)),
        "settings": {
            "full_correct": bool(args.full_correct),
            "strict_preprocess": bool(args.strict_preprocess),
            "unique_safety": bool(args.unique_safety),
            "batch_size": int(args.batch_size),
        },
        "metrics": {
            "original": {"f1_macro": m_clean.f1_macro, "accuracy": m_clean.accuracy},
            "noisy": {"f1_macro": m_noisy.f1_macro, "accuracy": m_noisy.accuracy},
            "restored": {"f1_macro": m_rest.f1_macro, "accuracy": m_rest.accuracy},
            "safety": {"f1_macro": m_safe.f1_macro, "accuracy": m_safe.accuracy},
        },
        "deltas": {
            "delta_recovery_f1": float(delta_recovery_f1),
            "safety_drop_f1": float(safety_drop_f1),
            "ERR_recovery_pct": float(err_pct),
            "ERR_alt_error_reduction_pct": float(err_alt),
        }
    }

    if args.bootstrap and args.bootstrap > 0:
        yt = np.array(y_true, dtype=object)
        pn = np.array(pred_noisy, dtype=object)
        pr = np.array(pred_repaired, dtype=object)
        pc = np.array(pred_clean, dtype=object)
        ps = np.array(pred_safe, dtype=object)

        d_f1, lo_f1, hi_f1 = bootstrap_ci_delta(yt, pn, pr, "f1_macro", args.bootstrap, args.seed)
        d_acc, lo_acc, hi_acc = bootstrap_ci_delta(yt, pn, pr, "accuracy", args.bootstrap, args.seed)

        d_safe_f1, lo_safe_f1, hi_safe_f1 = bootstrap_ci_delta(yt, pc, ps, "f1_macro", args.bootstrap, args.seed)
        d_safe_acc, lo_safe_acc, hi_safe_acc = bootstrap_ci_delta(yt, pc, ps, "accuracy", args.bootstrap, args.seed)

        report["bootstrap"] = {
            "n_boot": int(args.bootstrap),
            "delta_recovery": {
                "f1_macro": {"delta": float(d_f1), "ci95": [float(lo_f1), float(hi_f1)]},
                "accuracy": {"delta": float(d_acc), "ci95": [float(lo_acc), float(hi_acc)]},
            },
            "safety_delta_safe_minus_clean": {
                "f1_macro": {"delta": float(d_safe_f1), "ci95": [float(lo_safe_f1), float(hi_safe_f1)]},
                "accuracy": {"delta": float(d_safe_acc), "ci95": [float(lo_safe_acc), float(hi_safe_acc)]},
            },
        }

    # --- Save per-row CSV if requested
    if args.output_rows:
        out_path = Path(args.output_rows)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = pd.DataFrame({
            "label": y_true,
            "pred_clean": pred_clean,
            "pred_noisy": pred_noisy,
            "pred_repaired": pred_repaired,
            "pred_safe": pred_safe,
            "clean_text": clean_texts,
            "noisy_text": noisy_texts,
            "repaired_text": repaired_texts,
            "safe_check_text": safe_check_texts,
        })
        rows.to_csv(out_path, index=False, encoding="utf-8")

    if args.output_report:
        rep_path = Path(args.output_report)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        rep_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    def fmt(f1, acc):
        return f"F1(macro)={f1:.4f} | Acc={acc:.4f}"

    print("\n==== Impact Evaluation Report (FAST) ====")
    print(f"Rows evaluated: {report['n_rows_evaluated']}")
    print(f"Original : {fmt(m_clean.f1_macro, m_clean.accuracy)}")
    print(f"Noisy    : {fmt(m_noisy.f1_macro, m_noisy.accuracy)}")
    print(f"Restored : {fmt(m_rest.f1_macro, m_rest.accuracy)}")
    print(f"Safety   : {fmt(m_safe.f1_macro, m_safe.accuracy)}")
    print("----------------------------------------")
    print(f"Delta Recovery (F1): {delta_recovery_f1:+.4f}")
    print(f"Safety Drop (F1)   : {safety_drop_f1:+.4f}")
    print(f"ERR (Recovery %)   : {err_pct:.2f}%")
    print(f"ERR_alt (1-F1)     : {err_alt:.2f}%")
    if args.bootstrap and args.bootstrap > 0:
        ci = report["bootstrap"]["delta_recovery"]["f1_macro"]["ci95"]
        print(f"Bootstrap CI95 Δ(F1) noisy→restored: [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print("========================================\n")


if __name__ == "__main__":
    main()
