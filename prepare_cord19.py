
import os
import re
from collections import defaultdict
import pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException

# --- CONFIG INTEGRATION ---
# Clean & Centralized Imports
from config import (
    RAW_METADATA_PATH,
    CORPUS_PATH,
    LABELED_DATA_PATH,
    VAL_SAMPLE_PATH
)

TARGET_PER_CLASS = 10000  
MAX_CORPUS_LINES = 100000
MIN_TEXT_LENGTH = 200


# ===========================
# Text Cleaning Helpers
# ===========================

def clean_text_academic(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    
    # 3. Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    
    # 4. Smart Citation Removal:
    # Only square brackets containing numbers, commas, or dashes (such as [1], [1-3], [1,5]) are deleted.
    # It maintains everything else (such as [Mg2+], [50 mg])
    text = re.sub(r"\[\s*\d+(?:[\s,\-–]*\d+)*\s*\]", " ", text)
        
    # 5. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text



# Seed fixation to ensure the repeatability of results (Deterministic)
DetectorFactory.seed = 0

def is_english_scientific(text: str) -> bool:
    """
    Check if the text is English using a robust probabilistic model (langdetect).
    Falls back to False if detection fails or text is too short/ambiguous.
    """
    if not text or len(text.strip()) < 10:
        return False
        
    try:
        # The function returns the language code 'en' if it is English.
        lang = detect(text)
        return lang == 'en'
    except LangDetectException:
        # If the text consists only of symbols or is not detectable
        return False


# ===========================
# Labeling Logic
# ===========================

KEYWORDS = {
    "Prevention": [
        "vaccine", "vaccination", "immunization", "immunisation",
        "antibody", "prophylaxis", "prevention", "immune response"
    ],
    "Treatment": [
        "treatment", "therapy", "therapeutic", "drug",
        "clinical trial", "remdesivir", "antiviral", "intervention"
    ],
    "Epidemiology": [
        "transmission", "outbreak", "epidemic", "pandemic",
        "spread", "incidence", "prevalence", "reproduction number",
        "contact tracing", "epidemiology", "case fatality"
    ],
}


def get_strict_weighted_label(row) -> str:
    """
    Weighted keyword matching on title + abstract.

    - Title matches are worth double (2 points).
    - Abstract matches worth 1 point.
    - Require minimum score >= 2 for any label.
    - If tie on top score => discard (return None).
    """
    title = str(row["clean_title"])
    abstract = str(row["clean_abstract"])

    scores = {label: 0 for label in KEYWORDS}

    for label, words in KEYWORDS.items():
        for w in words:
            if w in title:
                scores[label] += 2
            elif w in abstract:
                scores[label] += 1

    # find best label and apply threshold
    best_label = max(scores, key=scores.get)
    max_score = scores[best_label]

    if max_score < 2:
        return None

    # check ties
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1 and sorted_scores[0] == sorted_scores[1]:
        return None

    return best_label


# ===========================
# Main Pipeline
# ===========================

def main():
    """
    Prepare the CORD‑19 corpus and labeled dataset.

    This function reads the metadata CSV (if available), cleans and filters the
    text, builds a large corpus for spelling correction and a smaller
    classification dataset. If the metadata file is missing it will log a
    warning and exit gracefully instead of raising an exception so that the
    remainder of the project can still function without data.
    """

    # Check for the existence of the metadata file.  If it is not present,
    # inform the user and return early rather than raising an exception.  This
    # behaviour allows the user to add the file later without breaking the
    # entire pipeline.  See documentation in README for details.
    if not os.path.exists(RAW_METADATA_PATH):
        print(
            f"[WARN] metadata.csv not found at {RAW_METADATA_PATH}. Skipping data preparation. "
            "You can place the Kaggle metadata file in the data/raw directory and rerun this script later."
        )
        return

    print("[INFO] Reading metadata.csv in chunks...")
    chunk_size = 50000

    collected_data = {
        "Prevention": [],
        "Treatment": [],
        "Epidemiology": [],
    }
    corpus_lines = []

    total_rows = 0

    for i, chunk in enumerate(
        pd.read_csv(
            RAW_METADATA_PATH,
            usecols=["title", "abstract"],
            dtype=str,
            chunksize=chunk_size,
        )
    ):
        print(f"[INFO] Processing chunk {i+1}...")

        # drop rows with missing or too short text
        chunk = chunk.dropna(subset=["title", "abstract"]).copy()
        chunk["full_len"] = (chunk["title"] + chunk["abstract"]).str.len()
        chunk = chunk[chunk["full_len"] > MIN_TEXT_LENGTH]

        if chunk.empty:
            continue

        # clean title / abstract separately
        chunk["clean_title"] = chunk["title"].apply(clean_text_academic)
        chunk["clean_abstract"] = chunk["abstract"].apply(clean_text_academic)
        chunk["full_text"] = chunk["clean_title"] + ". " + chunk["clean_abstract"]

        # language filter
        chunk = chunk[chunk["full_text"].apply(is_english_scientific)]
        if chunk.empty:
            continue

        total_rows += len(chunk)

        # collect corpus lines for spelling
        if len(corpus_lines) < MAX_CORPUS_LINES:
    # add title and abstract as SEPARATE lines to prevent cross-boundary bigrams
            for t, a in zip(chunk["clean_title"].tolist(), chunk["clean_abstract"].tolist()):
                if len(corpus_lines) >= MAX_CORPUS_LINES:
                    break
                if isinstance(t, str) and t.strip():
                    corpus_lines.append(t.strip())
                if len(corpus_lines) >= MAX_CORPUS_LINES:
                    break
                if isinstance(a, str) and a.strip():
                    corpus_lines.append(a.strip())


        # early stop if we already filled all classes and corpus
        if all(len(v) >= TARGET_PER_CLASS for v in collected_data.values()) and len(corpus_lines) >= MAX_CORPUS_LINES:
            print("[INFO] All targets reached. Stopping early.")
            break

        # apply labeling
        chunk["label"] = chunk.apply(get_strict_weighted_label, axis=1)
        labeled_chunk = chunk.dropna(subset=["label"])

        # distribute to class buckets
        for label in collected_data:
            current_len = len(collected_data[label])
            if current_len >= TARGET_PER_CLASS:
                continue

            subset = labeled_chunk[labeled_chunk["label"] == label]
            if subset.empty:
                continue

            needed = TARGET_PER_CLASS - current_len
            subset = subset.head(needed)

            for _, row in subset.iterrows():
                collected_data[label].append(
                    {
                        "title": row["title"],
                        "abstract": row["abstract"],
                        "full_text": row["full_text"],
                        "label": row["label"],
                    }
                )

        print(
            f"[STATUS] Chunk {i+1} | "
            f"Prev: {len(collected_data['Prevention'])} | "
            f"Treat: {len(collected_data['Treatment'])} | "
            f"Epid: {len(collected_data['Epidemiology'])} | "
            f"Corpus lines: {len(corpus_lines)}"
        )

    # ===========================
    # Save Corpus
    # ===========================
    print(f"[INFO] Saving corpus to {CORPUS_PATH} ...")
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for line in corpus_lines:
            f.write(line.strip() + "\n")

    total_tokens = sum(len(line.split()) for line in corpus_lines)
    print(f"[METRIC] Corpus lines: {len(corpus_lines)}")
    print(f"[METRIC] Corpus tokens (approx.): {total_tokens:,}")

    # ===========================
    # Save Labeled Dataset
    # ===========================
    final_rows = []
    for label, rows in collected_data.items():
        final_rows.extend(rows)

    labeled_df = pd.DataFrame(final_rows)
    labeled_df = labeled_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    labeled_df.to_csv(LABELED_DATA_PATH, index=False)
    print(f"[INFO] Labeled dataset saved at {LABELED_DATA_PATH}")
    # Safely report class counts only if the DataFrame contains the label column
    # and is non-empty.  If the dataset is empty (e.g., due to no rows matching
    # the keyword rules) then value_counts() would raise an error.  Providing
    # this check prevents confusing stack traces when the metadata is small or
    # missing.
    if not labeled_df.empty and "label" in labeled_df.columns:
        print(labeled_df["label"].value_counts())
    else:
        print("[WARN] No labeled rows were generated; class distribution is unavailable.")

    # ===========================
    # Save Validation Sample
    # ===========================
    labeled_df.sample(n=min(50, len(labeled_df)), random_state=99).to_csv(VAL_SAMPLE_PATH, index=False)
    print(f"[INFO] Validation sample (50 rows) saved at {VAL_SAMPLE_PATH}")

    print("\n[DONE] prepare_cord19 pipeline finished successfully.")


if __name__ == "__main__":
    main()
