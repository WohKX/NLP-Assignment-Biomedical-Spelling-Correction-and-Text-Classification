#!/usr/bin/env python3
"""make_harder_subset_200.py

Create a "Harder Set" (200 rows) by *selecting* more challenging examples from
an existing larger synthetic set (e.g., synthetic_spellcheck_cases_test2_1000.csv).

Why selection (not regeneration)?
- The 1000-case set was already generated from real test2.csv fragments and is
  guaranteed to be within scope (edit<=2 and solvable by the model used during
  generation). This script simply picks the hardest-looking strata:
  - Real-word: mostly right-only + both-needed
  - Non-word: mostly dist2_del_rep
  - COVID normalization: diverse variants
  - Negative + General-English: small but present

Usage:
  python make_harder_subset_200.py \
      --in_csv synthetic_spellcheck_cases_test2_1000.csv \
      --out_csv synthetic_spellcheck_cases_test2_harder_200.csv \
      --seed 7
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)

    df = pd.read_csv(args.in_csv)

    quota = {
        ("real-word", "both-needed"): 35,
        ("real-word", "right-only"): 30,
        ("real-word", "left-only"): 25,

        ("non-word", "dist2_del_rep"): 50,
        ("non-word", "trans"): 10,
        ("non-word", "rep"): 10,
        ("non-word", "ins"): 5,
        ("non-word", "del"): 5,

        ("covid-normalization", "covid-18-variant"): 10,
        ("general-english", "dist2_del_rep"): 5,
        ("general-english", "rep"): 3,
        ("general-english", "trans"): 2,

        ("negative", "do-no-harm"): 10,
    }

    # Simple hardness ranking (within each stratum)
    def hardness(row) -> float:
        wrong = str(row.get("wrong", ""))
        expected = str(row.get("expected", ""))
        base = 0.0
        cat = row.get("category")
        sub = row.get("subcategory")
        if cat == "real-word":
            base += 10
            if sub == "both-needed":
                base += 3
            elif sub == "right-only":
                base += 2
            elif sub == "left-only":
                base += 1
        elif cat == "non-word":
            base += 7
            if sub == "dist2_del_rep":
                base += 3
            elif sub == "trans":
                base += 2
            elif sub == "rep":
                base += 1
        elif cat == "covid-normalization":
            base += 6
        elif cat == "general-english":
            base += 5
            if sub == "dist2_del_rep":
                base += 2
        else:
            base += 1
        # Longer tokens generally harder
        base += 0.12 * len(wrong)
        base += 0.05 * len(expected)
        return base

    df = df.copy()
    df["_hardness"] = df.apply(hardness, axis=1)

    # Diversification constraints
    max_pair_per_cat = 2
    max_wrong_per_cat = 2
    max_covid_per_variant = 1

    picked_rows = []
    used_idx = set()

    pair_counts = defaultdict(int)
    wrong_counts = defaultdict(int)

    covid_variant_counts = defaultdict(int)

    def ok_take(r) -> bool:
        cat = r["category"]
        wrong = r["wrong"]
        exp = r["expected"]
        pair_key = (cat, wrong, exp)
        wrong_key = (cat, wrong)

        if pair_counts[pair_key] >= max_pair_per_cat:
            return False
        if wrong_counts[wrong_key] >= max_wrong_per_cat:
            return False
        if cat == "covid-normalization":
            if covid_variant_counts[wrong] >= max_covid_per_variant:
                return False
        return True

    # Select per quota stratum, hardest-first
    for (cat, sub), n in quota.items():
        pool = df[(df["category"] == cat) & (df["subcategory"] == sub)].sort_values("_hardness", ascending=False)
        count = 0
        for idx, r in pool.iterrows():
            if count >= n:
                break
            if idx in used_idx:
                continue
            rr = r.to_dict()
            if not ok_take(rr):
                continue

            picked_rows.append(rr)
            used_idx.add(idx)
            pair_counts[(cat, rr["wrong"], rr["expected"])] += 1
            wrong_counts[(cat, rr["wrong"])] += 1
            if cat == "covid-normalization":
                covid_variant_counts[rr["wrong"]] += 1
            count += 1

        if count < n:
            raise RuntimeError(f"Could not meet quota for ({cat}, {sub}): got {count}/{n}")

    out = pd.DataFrame(picked_rows).drop(columns=["_hardness"], errors="ignore")
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    out["case_id"] = [f"t2_hard_{i:04d}" for i in range(1, len(out) + 1)]

    cols = ["case_id", "source", "category", "subcategory", "prev", "wrong", "next", "expected", "note", "fragment"]
    out = out[cols]

    out.to_csv(args.out_csv, index=False, encoding="utf-8")

    print("Saved:", args.out_csv)
    print("Rows:", len(out))
    print("\nCategory counts:\n", out["category"].value_counts().to_string())
    print("\nSubcategory counts:\n", out["subcategory"].value_counts().to_string())


if __name__ == "__main__":
    main()
