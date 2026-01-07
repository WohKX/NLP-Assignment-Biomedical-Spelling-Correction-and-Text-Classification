"""
Debug/Evaluation script for MedicalSpellChecker (real-word + non-word + rules).

Fixes:
- Robust per-category breakdown printing (avoids tuple formatting crash).
- Outputs a CSV report + per-category summary CSV for your written report.

Features:
- Supports bidirectional context: prev_word + next_word (if spellchecker supports it).
- Prints dynamic_keep_bonus / dynamic_margin / dynamic_threshold when available.
- Produces overall error-fix recall (TP/(TP+FN)) and negative FPR on do-no-harm cases.

Expected input CSV (minimum columns; names are flexible):
- category
- prev (or prev_word)
- wrong
- next (or next_word)
- expected

Run:
    python debug_spellchecker_realword_improved_v2_fixed.py --cases_csv synthetic_spellcheck_cases.csv

Outputs:
- debug_spellchecker_report.csv
- debug_spellchecker_summary_by_category.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd


# -----------------------------
# Helpers
# -----------------------------

def _lower(x: Any) -> str:
    return str(x).strip()

def _norm(x: Any) -> str:
    # evaluation normalization (case-insensitive; keep punctuation because model may rely on it)
    return str(x).strip().lower()

def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "   N/A "
    try:
        if isinstance(x, float) and math.isnan(x):
            return "   N/A "
    except Exception:
        pass
    return f"{x:8.4f}"

def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "   N/A "
    try:
        if isinstance(x, float) and math.isnan(x):
            return "   N/A "
    except Exception:
        pass
    return f"{x:6.2f}%"


def _get(row: pd.Series, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in row and pd.notna(row[k]):
            return row[k]
    return default


def _bigram_count(sc: Any, w1: Optional[str], w2: Optional[str]) -> int:
    if not w1 or not w2:
        return 0
    try:
        d = getattr(sc, "bigrams", None)
        if d is None:
            return 0
        return int(d.get(str(w1).lower(), {}).get(str(w2).lower(), 0))
    except Exception:
        return 0


def _call_correct(sc: Any, wrong: str, prev: Optional[str], next_: Optional[str]) -> Tuple[str, str]:
    """
    Try best-effort to call the spellchecker with bidirectional context.
    Supports several possible APIs:
    - correct_token(wrong, prev_word=..., next_word=...)
    - correct_token(wrong, prev, next)
    - correct(wrong, prev_word=...)
    - correct(wrong, prev)
    """
    if hasattr(sc, "correct_token"):
        fn = getattr(sc, "correct_token")
        # keyword style
        try:
            return fn(wrong, prev_word=prev, next_word=next_)  # type: ignore
        except TypeError:
            pass
        try:
            return fn(wrong, prev, next_)  # type: ignore
        except TypeError:
            pass

    # fallback: unidirectional correct()
    if hasattr(sc, "correct"):
        fn = getattr(sc, "correct")
        try:
            return fn(wrong, prev_word=prev)  # type: ignore
        except TypeError:
            return fn(wrong, prev)  # type: ignore

    raise RuntimeError("SpellChecker has no supported correct()/correct_token() method.")


def _try_debug_metrics(sc: Any, prev: Optional[str], wrong: str, next_: Optional[str]) -> Dict[str, Any]:
    """
    Pull debug metrics if the model exposes them. If not available, returns {}.
    We look for:
      - dynamic_keep_bonus
      - dynamic_margin
      - dynamic_threshold
      - orig_left_score, orig_bi_diag_score
      - chosen_left_score, chosen_bi_diag_score
    """
    out: Dict[str, Any] = {}

    # Preferred: model provides a debug method
    for name in ("debug_case", "debug_token", "debug_correct_token", "_debug_case"):
        if hasattr(sc, name):
            try:
                dbg = getattr(sc, name)(prev, wrong, next_)  # type: ignore
                if isinstance(dbg, dict):
                    out.update(dbg)
                    return out
            except Exception:
                pass

    # If model has explicit helpers, use them.
    try:
        if hasattr(sc, "_score_left"):
            out["orig_left"] = float(sc._score_left(wrong, prev))  # type: ignore
        if hasattr(sc, "_score_bi_diag"):
            out["orig_bi_diag"] = float(sc._score_bi_diag(prev, wrong, next_))  # type: ignore
    except Exception:
        pass

    try:
        # dynamic params (adaptive threshold)
        for k in ("_dynamic_keep_bonus", "dynamic_keep_bonus"):
            if hasattr(sc, k):
                out["dynamic_keep_bonus"] = float(getattr(sc, k)(wrong))  # type: ignore
                break
        for k in ("_dynamic_margin", "dynamic_margin"):
            if hasattr(sc, k):
                out["dynamic_margin"] = float(getattr(sc, k)(wrong, prev, next_))  # type: ignore
                break
        for k in ("_dynamic_threshold", "dynamic_threshold"):
            if hasattr(sc, k):
                out["dynamic_threshold"] = float(getattr(sc, k)(wrong, prev, next_))  # type: ignore
                break
    except Exception:
        # Safe: if not supported, leave N/A
        pass

    return out


# -----------------------------
# Main
# -----------------------------

def load_spellchecker(artifacts_dir: str) -> Any:
    # Try to import your project spellchecker
    try:
        from spelling.model import MedicalSpellChecker  # type: ignore
        import_from = "spelling.model"
    except Exception:
        # Fallback to local model.py if someone runs script standalone
        from model import MedicalSpellChecker  # type: ignore
        import_from = "model"

    # Load artifacts
    try:
        sc = MedicalSpellChecker.from_artifacts(artifacts_dir)  # type: ignore
    except Exception:
        # Alternative legacy signature: from_artifacts(vocab, unigrams, bigrams)
        vocab = os.path.join(artifacts_dir, "vocab.pkl")
        uni = os.path.join(artifacts_dir, "unigrams.pkl")
        bi = os.path.join(artifacts_dir, "bigrams.pkl")
        sc = MedicalSpellChecker.from_artifacts(vocab, uni, bi)  # type: ignore

    print(f"[INFO] Loaded MedicalSpellChecker from {import_from} | Artifacts dir: {artifacts_dir}")
    return sc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases_csv", default="synthetic_spellcheck_cases.csv", help="CSV of test cases")
    ap.add_argument("--artifacts_dir", default=os.path.join("spelling", "artifacts"), help="Artifacts directory")
    ap.add_argument("--out_report", default="debug_spellchecker_report.csv", help="Output CSV report")
    ap.add_argument("--print_cases", action="store_true", help="Print per-case details")
    args = ap.parse_args()

    sc = load_spellchecker(args.artifacts_dir)

    df = pd.read_csv(args.cases_csv)
    print(f"[INFO] Loaded {len(df)} cases from csv: {args.cases_csv}")

    rows = []
    for i, row in df.iterrows():
        cat = _get(row, "category", default="unknown")
        prev = _get(row, "prev", "prev_word", default=None)
        next_ = _get(row, "next", "next_word", default=None)
        wrong = _get(row, "wrong", "observed", default="")
        expected = _get(row, "expected", "gold", default="")

        prev_s = _norm(prev) if prev is not None and str(prev).strip() != "" else None
        next_s = _norm(next_) if next_ is not None and str(next_).strip() != "" else None
        wrong_s = _lower(wrong)
        exp_s = _lower(expected)

        corrected, decision = _call_correct(sc, wrong_s, prev_s, next_s)
        passed = _norm(corrected) == _norm(exp_s)

        dbg = _try_debug_metrics(sc, prev_s, wrong_s, next_s)

        # counts (if bigrams dict exists)
        c_prev_orig = _bigram_count(sc, prev_s, wrong_s)
        c_prev_corr = _bigram_count(sc, prev_s, corrected)
        c_orig_next = _bigram_count(sc, wrong_s, next_s)
        c_corr_next = _bigram_count(sc, corrected, next_s)

        rec = {
            "case_id": int(i + 1),
            "category": str(cat),
            "prev": prev_s or "",
            "wrong": wrong_s,
            "next": next_s or "",
            "expected": exp_s,
            "corrected": str(corrected),
            "decision": str(decision),
            "pass": bool(passed),
            "bigram_prev_orig": int(c_prev_orig),
            "bigram_prev_corr": int(c_prev_corr),
            "bigram_orig_next": int(c_orig_next),
            "bigram_corr_next": int(c_corr_next),
            # optional debug metrics
            "dynamic_keep_bonus": dbg.get("dynamic_keep_bonus", float("nan")),
            "dynamic_margin": dbg.get("dynamic_margin", float("nan")),
            "dynamic_threshold": dbg.get("dynamic_threshold", float("nan")),
            "orig_left": dbg.get("orig_left", dbg.get("orig_left_score", float("nan"))),
            "orig_bi_diag": dbg.get("orig_bi_diag", dbg.get("orig_bi_diag_score", float("nan"))),
        }
        rows.append(rec)

        if args.print_cases:
            print()
            print(f"CASE {i+1:03d} [{cat}] prev='{prev_s}' wrong='{wrong_s}' next='{next_s}' expected='{exp_s}'")
            print("-" * 100)
            print(f"corrected='{corrected}' | decision={decision} | PASS={passed}")
            print(f"bigrams: prev->orig={c_prev_orig} | prev->corr={c_prev_corr} || orig->next={c_orig_next} | corr->next={c_corr_next}")
            print(f"dynamic_keep_bonus={_fmt(rec['dynamic_keep_bonus'])} | dynamic_margin={_fmt(rec['dynamic_margin'])} | dynamic_threshold={_fmt(rec['dynamic_threshold'])}")
            print(f"scores: orig_left={_fmt(rec['orig_left'])} | orig_bi_diag={_fmt(rec['orig_bi_diag'])}")

    rep = pd.DataFrame(rows)
    rep.to_csv(args.out_report, index=False)

    # Overall summary like your console output
    err = rep[rep["expected"] != rep["wrong"]]
    neg = rep[rep["expected"] == rep["wrong"]]

    tp = int(err["pass"].sum())
    fn = int((~err["pass"]).sum())
    tn = int(neg["pass"].sum())
    fp = int((~neg["pass"]).sum())

    err_recall = 100.0 * (tp / (tp + fn)) if (tp + fn) else float("nan")
    neg_fpr = 100.0 * (fp / (fp + tn)) if (fp + tn) else float("nan")

    print("\n=== SUMMARY ===")
    print(f"Cases total: {len(rep)}")
    print(f"Error cases (expected!=wrong): {len(err)} | TP: {tp} | FN: {fn}")
    print(f"Error-fix accuracy/recall: {err_recall:0.2f}%")
    print(f"Negative cases (do-no-harm): {len(neg)} | TN: {tn} | FP: {fp}")
    print(f"False-positive rate on negative: {neg_fpr:0.2f}%")
    print(f"Report saved: {args.out_report}")

    # Per-category breakdown (robust)
    print("\n=== PER-CATEGORY BREAKDOWN ===")
    summary_rows = []
    for cat, subdf in rep.groupby("category"):
        cat_str = cat[0] if isinstance(cat, tuple) else str(cat)

        sub_err = subdf[subdf["expected"] != subdf["wrong"]]
        sub_neg = subdf[subdf["expected"] == subdf["wrong"]]

        sub_tp = int(sub_err["pass"].sum())
        sub_fn = int((~sub_err["pass"]).sum())
        sub_tn = int(sub_neg["pass"].sum())
        sub_fp = int((~sub_neg["pass"]).sum())

        sub_err_recall = 100.0 * (sub_tp / (sub_tp + sub_fn)) if (sub_tp + sub_fn) else float("nan")
        sub_neg_fpr = 100.0 * (sub_fp / (sub_fp + sub_tn)) if (sub_fp + sub_tn) else float("nan")

        summary_rows.append({
            "category": cat_str,
            "n_total": int(len(subdf)),
            "n_error": int(len(sub_err)),
            "n_negative": int(len(sub_neg)),
            "error_recall_pct": sub_err_recall,
            "neg_fpr_pct": sub_neg_fpr,
        })

        print(f"- {cat_str:18s} | n={len(subdf):3d} | error_recall={_fmt_pct(sub_err_recall)} | neg_FPR={_fmt_pct(sub_neg_fpr)}")

    pd.DataFrame(summary_rows).sort_values("category").to_csv(
        "debug_spellchecker_summary_by_category.csv",
        index=False
    )
    print("Category summary saved: debug_spellchecker_summary_by_category.csv")


if __name__ == "__main__":
    main()
