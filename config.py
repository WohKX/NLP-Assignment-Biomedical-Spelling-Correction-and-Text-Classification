from pathlib import Path
import sys

# ==========================================
# 1. Project Root & Base Configuration
# ==========================================
# Using pathlib for Object-Oriented filesystem paths.
# .resolve() handles symlinks and absolute paths.
# .parent ensures we get the directory containing this file.
# NOTE: If you move config.py deeper (e.g. to src/), add .parent accordingly.
PROJECT_ROOT = Path(__file__).resolve().parent

# ==========================================
# 2. Directory Structure Constants
# ==========================================
# We use the division operator '/' which is overloaded by pathlib 
# to join paths cleanly, OS-agnostic.

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

# Artifacts for modules
SPELLING_DIR = PROJECT_ROOT / "spelling"
SPELLING_ARTIFACT_DIR = SPELLING_DIR / "artifacts"

CLASSIF_DIR = PROJECT_ROOT / "classification"
CLASSIF_ARTIFACT_DIR = CLASSIF_DIR / "artifacts"

# ==========================================
# 3. File Path Constants (Central Source of Truth)
# ==========================================
# Never hardcode filenames in logic scripts. Define them here.

# Input Files
RAW_METADATA_FILENAME = "metadata.csv"
RAW_METADATA_PATH = DATA_RAW_DIR / RAW_METADATA_FILENAME

# Output/Intermediate Files (Preprocessing)
CORPUS_FILENAME = "cord19_corpus.txt"
CORPUS_PATH = DATA_PROCESSED_DIR / CORPUS_FILENAME

LABELED_DATA_FILENAME = "cord19_labeled.csv"
LABELED_DATA_PATH = DATA_PROCESSED_DIR / LABELED_DATA_FILENAME

VAL_SAMPLE_FILENAME = "cord19_validation_sample.csv"
VAL_SAMPLE_PATH = DATA_PROCESSED_DIR / VAL_SAMPLE_FILENAME

# Spelling Artifacts
VOCAB_FILE = "vocab.pkl"
UNIGRAMS_FILE = "unigrams.pkl"
BIGRAMS_FILE = "bigrams.pkl"

VOCAB_PATH = SPELLING_ARTIFACT_DIR / VOCAB_FILE
UNIGRAMS_PATH = SPELLING_ARTIFACT_DIR / UNIGRAMS_FILE
BIGRAMS_PATH = SPELLING_ARTIFACT_DIR / BIGRAMS_FILE

# Classification Model
CLASSIFIER_FILENAME = "topic_classifier.pkl"
CLASSIFIER_MODEL_PATH = CLASSIF_ARTIFACT_DIR / CLASSIFIER_FILENAME
COMPARISON_REPORT_FILENAME = "model_comparison.csv"
COMPARISON_REPORT_PATH = CLASSIF_ARTIFACT_DIR / COMPARISON_REPORT_FILENAME

# ==========================================
# 4. Initialization Logic (Side-Effect Management)
# ==========================================

def _init_project_structure():
    """
    Internal function to ensure the project structure exists.
    Executed on module import to guarantee safe file operations.
    """
    required_dirs = [
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        SPELLING_ARTIFACT_DIR,
        CLASSIF_ARTIFACT_DIR
    ]
    
    for directory in required_dirs:
        # parents=True allows creating 'data/processed' even if 'data' doesn't exist.
        # exist_ok=True prevents errors if directory already exists.
        directory.mkdir(parents=True, exist_ok=True)

# Execute initialization
_init_project_structure()

# ==========================================
# 5. Export for external usage (Optional sanity check)
# ==========================================
if __name__ == "__main__":
    print(f"[CONFIG] Project Root: {PROJECT_ROOT}")
    print(f"[CONFIG] Metadata Path: {RAW_METADATA_PATH}")
    print("[CONFIG] Directories initialized successfully.")