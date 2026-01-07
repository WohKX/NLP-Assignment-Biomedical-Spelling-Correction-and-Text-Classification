# 🧬 CORD-19 Intelligence Hub  
**White‑Box Biomedical Spell Repair + Topic Classification (Before/After) + 4‑Run Impact Evaluation**

**NLP (Master of Artificial Intelligence) — Academic Assignment (Part A)**  
Author: Moustafa

---

## Executive Summary
This repository delivers an end‑to‑end NLP prototype that places a **conservative, white‑box spelling repair layer** upstream of a **topic classifier**, and then **quantifies** whether lexical repair improves downstream classification without harming clean biomedical text.

The system is built to be:
- **Interpretable**: spell correction decisions follow explicit scoring and safety gates; classification is TF‑IDF + linear model with explainable contributing terms.
- **Conservative (Do‑No‑Harm)**: biomedical text is high‑stakes; the repair policy avoids risky changes unless evidence is strong.
- **Measurable**: a 4‑run evaluation design isolates benefit (noise→restored) and risk (clean→safety drift) with row‑level traces and aggregate reports.

---

## 1) What the system does

### 1.1 Upstream: Medical Spell Repair (White‑Box)
The spelling module repairs two error types:
- **Non‑word errors**: tokens that are not in the vocabulary (typos/OCR artifacts).
- **Real‑word errors**: valid tokens that become wrong **in context** (harder and riskier).

Core mechanisms:
- **Candidate generation** under small edit distance (SymSpell‑like ideas via deletion operations / efficient candidate sets).
- **Scoring** using:
  - **Unigram frequency** (domain‑specific usage patterns)
  - **Bigram context** (local plausibility)
- **Do‑No‑Harm gates**:
  - stronger thresholds for real‑word changes
  - protection for biomedical patterns (e.g., SARS‑CoV‑2 / IL‑6 / COVID‑19‑like tokens)
  - adaptive evidence requirements to prevent over‑correction

### 1.2 Downstream: Topic Classification (Explainable)
The classification module predicts one of three topics:
- **Prevention**
- **Treatment**
- **Epidemiology**

Implementation:
- `TfidfVectorizer` + linear model (e.g., Logistic Regression / Linear SVM).
- Calibrated probabilities when applicable (for stable probability bars in the UI).
- Explainability via **Top contributing terms**: coefficient × TF‑IDF contribution.

### 1.3 Demonstration goal
The Streamlit UI exposes:
1) **Spell Correction** output (errors, candidates, corrected text)
2) **Classification BEFORE** correction
3) **Classification AFTER** correction  
→ making decision stability and downstream impact visible and testable.

---

## 2) How to run (Assessor‑friendly)

### 2.1 Environment (recommended)
This project was executed with **Python 3.12**. For the closest reproduction, use Python 3.12.

#### Windows (PowerShell)
```powershell
cd "E:\APU Master Study\NLP\CORD19_final"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### macOS / Linux
```bash
cd CORD19_final
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2.2 Launch the Streamlit application
> **Entry point:** `app/main.py` (not `app.py`)
```bash
streamlit run app/main.py
```

**Expected result**  
A three‑panel interface:
- **Panel 1:** Spelling Correction
- **Panel 2:** Classification (Before Correction)
- **Panel 3:** Classification (After Correction)

The UI is designed to run using the included artifacts under:
- `spelling/artifacts/`
- `classification/artifacts/`

---

## 3) Project Map (Complete structure + role of each file)

### 3.1 Root (Delivery)
- **README.md**  
  This document: academic overview, reproducibility instructions, and full project map.
- **requirements.txt**  
  Core dependencies required to run the demo, build artifacts, and run evaluation.
- **config.py**  
  Central path configuration for:
  - raw/processed datasets
  - spelling artifacts
  - classification artifacts
- **prepare_cord19.py**  
  Builds processed data from `data/raw/metadata.csv`:
  - spelling corpus
  - labeled dataset for classification
- **test.csv / test_synth.csv**  
  Test files (notably `test_synth.csv` for impact evaluation: clean/noisy/label).
- **synthetic_spellcheck_cases.csv**  
  Synthetic cases designed to stress the spell checker logic.
- **debug_spellchecker_report.csv**  
  Detailed debug report for spell correction behavior.
- **debug_spellchecker_summary_by_category.csv**  
  Aggregated debug summary by category for discussion/limitations.
- **outputs/**  
  Quantitative evaluation outputs, e.g.:
  - `impact_rows_*.csv`
  - `impact_report_*.json`

Additionally, lightweight dataset artifacts may exist at the root (useful for quick testing and reporting), such as:
- `cord19_corpus*`
- `cord19_labeled*`
- `cord19_validation_sample*`

### 3.2 Data
- **data/raw/**
  - `metadata.csv`  
    CORD‑19 metadata used to build the processed pipeline inputs.
  - `README_raw.txt`  
    Notes about raw data placement and expectations.
  - `test.csv`  
    Small raw test input.
- **data/processed/**
  - `cord19_validation_sample.csv`  
    Small processed sample enabling quick runs without the full corpus.

### 3.3 Spelling module — `spelling/`
- **spelling/build_spelling_model.py**  
  Builds domain artifacts from a corpus:
  - `vocab.pkl`
  - `unigrams.pkl`
  - `bigrams.pkl`
- **spelling/model.py**  
  Core spell checker (`MedicalSpellChecker`):
  - candidate generation (edit distance operations)
  - unigram+bigram scoring
  - do‑no‑harm policy and safety gates
  - non‑word vs real‑word handling
- **spelling/artifacts/**
  - `vocab.pkl` (vocabulary)
  - `unigrams.pkl` (domain term frequency)
  - `bigrams.pkl` (context model)
- **spelling/__init__.py**  
  Package marker.

### 3.4 Classification module — `classification/`
- **classification/train_classifier.py**  
  Trains and exports a 3‑class topic classifier:
  - TF‑IDF vectorization
  - linear classifier (balanced)
  - calibration for probabilities (when enabled)
  - model comparison & export
- **classification/artifacts/**
  - `topic_classifier.joblib` *(primary runtime artifact)*
  - `topic_classifier.pkl` (compat copy)
  - `model_comparison.csv`
  - (optional) wordcloud images for reporting
- **classification/__init__.py**  
  Package marker.

### 3.5 Shared utilities — `utils/`
- **utils/preprocess.py**  
  Shared preprocessing utilities:
  - `preprocess_for_spelling`: tokenization suitable for bigram context
  - `preprocess_for_classification`: normalization for TF‑IDF
  - (where present) `preprocess_for_correction`: tokenization preserving whitespace/punctuation
- **utils/runtime.py**  
  Runtime bootstrap: ensures project root is on `sys.path` to avoid import failures.
- **utils/__init__.py**  
  Package marker.

### 3.6 Streamlit application — `app/`
- **app/main.py**  
  The interactive demonstration:
  - spell correction output (highlighting + suggestions + corrected text)
  - classification before/after correction
  - probability bars and explainability tables
  - dictionary explorer / corpus insights (domain‑term focused)
- **app/__init__.py**  
  Package marker.

### 3.7 Experiments / impact evaluation — `scripts/`
- **scripts/synthesize_corruptions.py**  
  Generates synthetic corruption (clean→noisy) for robustness testing.
- **scripts/error_generator.py / error_generator_harder.py / error_generator_hardest.py**  
  Noise generators with increasing severity (delete/replace/swap/merge/split and OCR‑like distortions).
- **scripts/evaluate_impact_fast_v2.py**  
  Quantifies impact using a 4‑run loop and exports row‑level and summary reports.
- **scripts/debug_spellchecker_*.py**  
  Spell checker diagnostics (especially real‑word behavior).
- **scripts/model_data_driven(example for future work).py**  
  A future‑work direction illustrating a data‑driven extension.

---

## 4) Pipeline overview (Data → Preprocess → Spell → Classify → Evaluate)

```text
PIPELINE OVERVIEW
──────────────────────────────────────────────────────────────────────────────
(0) Inputs
    ├─ data/raw/metadata.csv  ──► CORD‑19 abstracts/titles (large)
    ├─ test.csv / data/raw/test.csv  ──► seed texts for quick checks / synthesis
    └─ optional audit sets  ──► additional evaluation data

(1) Preprocess & Dataset Preparation  [BUILD‑TIME]
    └─ prepare_cord19.py
        ├─ reads:  data/raw/metadata.csv
        ├─ applies: utils/preprocess.py (normalization/filtering)
        └─ writes: data/processed/*
            ├─ corpus for spelling (unigram/bigram counts)
            └─ labeled dataset for classification

(2) Spell‑Repair Layer (Medical, Conservative)  [BUILD + RUN]
    (2.a) Build spell artifacts  [BUILD‑TIME]
        └─ spelling/build_spelling_model.py
            └─ outputs: spelling/artifacts/{vocab.pkl, unigrams.pkl, bigrams.pkl}

    (2.b) Spell runtime  [RUN‑TIME]
        └─ spelling/model.py : MedicalSpellChecker
            ├─ tokenization: preprocess_for_spelling
            ├─ detect: OOV / suspicious tokens
            ├─ propose: edit‑distance candidates
            ├─ score: unigram + bigram context
            └─ do‑no‑harm gate → repaired_text (+ suggestions)

(3) Topic Classification Layer  [BUILD + RUN]
    (3.a) Train classifier  [BUILD‑TIME]
        └─ classification/train_classifier.py
            └─ outputs: classification/artifacts/topic_classifier.joblib

    (3.b) Classifier runtime  [RUN‑TIME]
        └─ joblib pipeline
            ├─ preprocess_for_classification
            ├─ TF‑IDF transform
            ├─ predict label + probabilities
            └─ explainability: top contributing terms

(4) Interactive Demonstration (UI)  [RUN‑TIME]
    └─ streamlit run app/main.py
        ├─ Panel #1: Spell correction
        ├─ Panel #2: Classify BEFORE
        └─ Panel #3: Classify AFTER

(5) Quantitative Impact Evaluation (4 Runs)  [RUN‑TIME]
    ├─ scripts/synthesize_corruptions.py → test_synth.csv
    └─ scripts/evaluate_impact_fast_v2.py → outputs/*.csv + outputs/*.json
```

---

## 5) Quantitative evaluation design (4 runs)

```text
EVALUATE (4‑RUNS) — ONE ROW
──────────────────────────────────────────────────────────────────────────────
label_clean
   ▲
   │
clean_text ─────────► [Classifier] ─────────► pred_clean      (Run1: Baseline)
   │
   ├────────► [Spell (do‑no‑harm)] ─► safe_text ─► [Classifier] ─► pred_safe  (Run4: Safety)
   │
corrupted_text ─────► [Classifier] ─────────► pred_noisy      (Run2: Noise)
   │
   └────────► [Spell] ─────────────► repaired_text ─► [Classifier] ─► pred_repaired (Run3: Repair)

Metrics:
- Original   = score(pred_clean,    label_clean)
- Noisy      = score(pred_noisy,    label_clean)
- Restored   = score(pred_repaired, label_clean)
- Safety     = score(pred_safe,     label_clean)
- ΔRecovery  = Restored - Noisy
- SafetyDrop = Original - Safety
- CI95%(Δ)   = bootstrap(noisy→restored)
```

This evaluation isolates two essential questions:
1) **Recovery**: does spell repair restore topic performance under corruption?
2) **Safety**: does spell repair preserve performance on clean text (do‑no‑harm evidence)?

---

## 6) Commands (Build / Train / Run) + expected outputs

### 6.1 Dataset preparation
```bash
python prepare_cord19.py
```

**Expected outputs**
- Processed files under `data/processed/`:
  - spelling corpus material (for unigram/bigram construction)
  - labeled dataset for classification training

### 6.2 Build spelling artifacts
```bash
python spelling/build_spelling_model.py
```

**Expected outputs**
- `spelling/artifacts/vocab.pkl`
- `spelling/artifacts/unigrams.pkl`
- `spelling/artifacts/bigrams.pkl`

### 6.3 Train the topic classifier
```bash
python classification/train_classifier.py
```

**Expected outputs**
- `classification/artifacts/topic_classifier.joblib`
- `classification/artifacts/model_comparison.csv`
- (optional) visual artifacts (e.g., wordclouds)

### 6.4 Run the UI
```bash
streamlit run app/main.py
```

**Expected outputs**
- 3‑panel interface showing:
  - conservative repair results
  - before vs after topic probabilities
  - explainability tables (“Why this label?”)

### 6.5 Generate synthetic corruption set
```bash
python scripts/synthesize_corruptions.py
```

**Expected outputs**
- `test_synth.csv` with fields such as:
  - `clean_text`
  - `corrupted_text`
  - `label` (ground/pseudo label depending on the preparation logic)

### 6.6 Evaluate impact (4 runs)
```bash
python scripts/evaluate_impact_fast_v2.py --input test_synth.csv --output_dir outputs
```

**Expected outputs**
- `outputs/impact_rows_*.csv` (row‑level traces)
- `outputs/impact_report_*.json` (aggregate metrics + deltas + confidence intervals)

To inspect script parameters:
```bash
python scripts/evaluate_impact_fast_v2.py --help
```

---

## 7) Reproducibility (environment discipline)

**Primary installation path**
- Install from `requirements.txt` in a clean virtual environment (recommended in Section 2).

**Exact environment snapshot (optional reference)**
- A pip‑freeze snapshot may be included as `requirements.lock.utf8.txt`.  
  This is intended as a **debugging and audit reference** (exact versions) rather than the primary install path, because it may contain additional packages beyond the minimum set required by this repository.

---

## 8) Notes on robustness and limitations (academic framing)
- **Real‑word correction** is inherently ambiguous; the system mitigates risk by applying stricter thresholds and context checks.
- **Do‑No‑Harm** is treated as a measurable property via the “Safety run” in the 4‑run evaluation.
- The synthetic corruption strategy approximates OCR/typo noise; it is a controlled test bed to quantify stability and recovery.

---

## 9) Expected assessor deliverables
An assessor should be able to reproduce:
1) A running Streamlit UI (`app/main.py`) demonstrating:
   - spelling correction logic and conservative behavior
   - topic classification before vs after correction
   - explainability (top contributing terms)
2) Quantitative evidence under `outputs/` showing:
   - degradation under corruption
   - recovery after repair (ΔRecovery)
   - minimal drift on clean text (SafetyDrop small)

---

## Appendix: Full folder tree (reference)
```text
CORD19_final/  (Delivery)
│
├─ README.md
├─ requirements.txt
├─ config.py
├─ prepare_cord19.py
│
├─ data/
│  ├─ raw/
│  │  ├─ metadata.csv
│  │  ├─ README_raw.txt
│  │  └─ test.csv
│  │
│  └─ processed/
│     └─ cord19_validation_sample.csv
│
├─ spelling/
│  ├─ model.py
│  ├─ build_spelling_model.py
│  ├─ artifacts/
│  │  ├─ vocab.pkl
│  │  ├─ unigrams.pkl
│  │  └─ bigrams.pkl
│  └─ __init__.py
│
├─ classification/
│  ├─ train_classifier.py
│  ├─ artifacts/
│  │  ├─ topic_classifier.joblib
│  │  ├─ topic_classifier.pkl
│  │  └─ model_comparison.csv
│  └─ __init__.py
│
├─ utils/
│  ├─ preprocess.py
│  ├─ runtime.py
│  └─ __init__.py
│
├─ app/
│  ├─ main.py
│  └─ __init__.py
│
├─ scripts/
│  ├─ synthesize_corruptions.py
│  ├─ error_generator.py
│  ├─ error_generator_harder.py
│  ├─ error_generator_hardest.py
│  ├─ evaluate_impact_fast_v2.py
│  └─ debug_spellchecker_*.py
│
├─ test.csv
├─ test_synth.csv
├─ synthetic_spellcheck_cases.csv
├─ debug_spellchecker_report.csv
├─ debug_spellchecker_summary_by_category.csv
└─ outputs/
```
