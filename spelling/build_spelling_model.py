import sys
from pathlib import Path

# --- Runtime bootstrap (fixes 'No module named config') ---
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.runtime import ensure_project_root
ensure_project_root()
# ----------------------------------------------------------

import os
import pickle
from collections import Counter, defaultdict

from utils.preprocess import preprocess_for_spelling

# --- CONFIG INTEGRATION ---
from config import (
    CORPUS_PATH,
    SPELLING_ARTIFACT_DIR,
    VOCAB_PATH,
    UNIGRAMS_PATH,
    BIGRAMS_PATH
)

MIN_FREQ = 3  # ignore extremely rare terms


def main():
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(
            f"Corpus file not found: {CORPUS_PATH}. "
            "Please run prepare_cord19.py first."
        )

    print(f"[INFO] Reading corpus from: {CORPUS_PATH}")

    unigram_counts = Counter()
    bigram_counts = defaultdict(Counter)
    total_tokens = 0

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            tokens = preprocess_for_spelling(line)
            if not tokens:
                continue

            unigram_counts.update(tokens)
            total_tokens += len(tokens)

            for i in range(len(tokens) - 1):
                w1, w2 = tokens[i], tokens[i + 1]
                bigram_counts[w1][w2] += 1

            if line_idx % 1000 == 0:
                print(f"[INFO] Processed {line_idx} lines...")

    vocab = {w for w, c in unigram_counts.items() if c >= MIN_FREQ}

    print(f"\n[METRIC] Total tokens: {total_tokens:,}")
    print(f"[METRIC] Vocab size (freq >= {MIN_FREQ}): {len(vocab):,}")
    print(f"[METRIC] Bigram entries: {sum(len(v) for v in bigram_counts.values()):,}")

    # Save artifacts
    os.makedirs(SPELLING_ARTIFACT_DIR, exist_ok=True)

    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f)
    with open(UNIGRAMS_PATH, "wb") as f:
        pickle.dump(unigram_counts, f)
    with open(BIGRAMS_PATH, "wb") as f:
        pickle.dump(bigram_counts, f)

    print(f"\n[SUCCESS] Spelling artifacts saved in: {SPELLING_ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
