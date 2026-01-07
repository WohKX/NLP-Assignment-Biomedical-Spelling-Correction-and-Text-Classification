import re
from typing import List, Set
import spacy
# --- Biomedical Symbol Normalization (KEEP important medical tokens) ---
GREEK_MAP = {
    "α": "alpha", "β": "beta", "γ": "gamma",
    "δ": "delta", "κ": "kappa", "μ": "mu",
    "τ": "tau", "λ": "lambda",
    "Α": "alpha", "Β": "beta", "Γ": "gamma",
    "Δ": "delta", "Κ": "kappa", "Μ": "mu",
    "Τ": "tau", "Λ": "lambda",
}

def normalize_biomed_text(text: str, *, convert_greek: bool = True) -> str:
    """
    Normalize only what is safe for biomedical text.
    IMPORTANT: We do NOT split or remove separators like '/', '<', '>'…because it is part of medical units/statistics.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    # Normalize hyphen variants to '-'
    text = re.sub(r"[‐-‒–—−]", "-", text)

    # normalize apostrophes
    text = text.replace("’", "'").replace("‘", "'")

    # Optional: Greek letters -> ASCII (for training stability)
    if convert_greek:
        for g, rep in GREEK_MAP.items():
            text = text.replace(g, rep)

    return text




# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
except OSError:
    raise RuntimeError(
        "SpaCy model 'en_core_web_sm' is not installed. "
        "Run: python -m spacy download en_core_web_sm"
    )

# --- Compiled Regex Patterns ---
# Standard cleanup
URL_RE = re.compile(r"http\S+|www\.\S+")
HTML_TAG_RE = re.compile(r"<.*?>")
NON_PRINTABLE_RE = re.compile(r"[\x00-\x1F\x7F]+") 
WHITESPACE_RE = re.compile(r"\s+")

# --- Constants ---
# NOTE: All entries MUST be singular (Lemmatized)
VALID_MEASUREMENT_HEADS: Set[str] = {
    # --- 1. SI Units & Laboratory Measurements (Mass, Volume, Length) ---
    "mg", "milligram", "g", "gram", "kg", "kilogram", "ug", "mcg", "microgram", "ng", "nanogram", "pg", "picogram",
    "ml", "milliliter", "l", "liter", "dl", "deciliter", "ul", "microliter",
    "mm", "millimeter", "cm", "centimeter", "m", "meter", "km", "kilometer", "nm", "nanometer", "um", "micrometer",
    "mol", "mole", "mmol", "millimole", "umol", "micromole", "nmol", "nanomole",
    "molar", "mmolar", "umolar", "nmolar", "concentration", "density",
    
    # --- 2. Time, Duration & Frequency ---
    "second", "sec", "minute", "min", "hour", "hr", "h",
    "day", "d", "week", "wk", "month", "mo", "year", "yr", "annum",
    "decade", "century", "period", "duration", "interval", "time",
    "cycle", "frequency", "onset", "latency",
    
    # --- 3. Biological & Dosage Units ---
    "dose", "dosage", "iu", "unit", "pfu", "cfu", "tcid50", # Plaque/Colony forming units
    "copy", "titer", "load", "ct", "threshold", # Viral load/PCR terms
    "od", "absorbance", # Optical density
    "bp", "base", "kb", "mb", # Genetics (base pairs)
    "cell", "clone", "colony", "isolate", "strain", "variant", "mutation",

    # --- 4. Subjects, Demographics & Anatomy ---
    "patient", "participant", "subject", "individual", "person", "people", "human",
    "man", "male", "woman", "female", "child", "infant", "neonate", "adult", "elderly",
    "mouse", "rat", "primate", "monkey", "animal", # Lab animals
    "case", "control", "placebo", "cohort", "group", "arm", "population", "sample", "specimen",
    "age", "weight", "height", "bmi", "score",
    
    # --- 5. Epidemiology & Outcomes ---
    "death", "fatality", "mortality", "survival", "survivor",
    "recovery", "discharge", "admission", "hospitalization", "visit",
    "infection", "transmission", "outbreak", "epidemic", "pandemic",
    "case", "incidence", "prevalence",
    "symptom", "fever", "cough", "event", "outcome",

    # --- 6. Physics & Environment ---
    "degree", "celsius", "fahrenheit", "kelvin", "temperature",
    "pressure", "mmhg", "pascal", "pa", "kpa",
    "voltage", "volt", "hz", "hertz", "bpm", # beats per minute
    
    # --- 7. Statistics & Research Design ---
    "percent", "percentage", "%", "ratio", "rate", "proportion", "fraction",
    "increase", "decrease", "reduction", "growth", "decline", "change", "difference",
    "p-value", "p", "value", "confidence", "ci", "interval", "limit",
    "mean", "median", "mode", "average",
    "deviation", "sd", "se", "sem", "variance", "error",
    "range", "min", "max", "quartile", "percentile",
    "correlation", "coefficient", "r", "r2", "kappa", "z-score", "f-score",
    "odds", "or", "risk", "rr", "hazard", "hr", # OR=Odds Ratio, RR=Relative Risk, HR=Hazard Ratio
    "sensitivity", "specificity", "accuracy", "precision", "recall", "f1",
    "study", "trial", "experiment", "analysis", "phase", "step", "stage", "round"
}

NEGATION_WORDS: Set[str] = {"no", "not", "neither", "nor", "none", "n't", "never"}


def clean_surface_noise(text: str) -> str:
    """
    Stage 1: Technical cleaning only. 
    Preserves Case and Punctuation for SpaCy.
    """
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = NON_PRINTABLE_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def preprocess_for_spelling(text: str) -> List[str]:
    """
    Pipeline for Spelling (Part A):
      - Preserves stopwords (important for bigram context).
      - Removes punctuation, pure numbers, currency symbols, brackets.
      - Keeps *medical-domain* alphanumeric tokens with internal hyphens
        (e.g., covid-19, sars-cov-2, h1n1, il-6, ace2) because they are
        common and semantically important in CORD-19.
    """
    text = clean_surface_noise(text)
    text = normalize_biomed_text(text, convert_greek=True)

    if not text:
        return []

        

    # Normalize common unicode hyphen/dash variants to ASCII '-' (safer for regex + vocab).
    text = re.sub(r"[‐‑‒–—−]", "-", text)

    # Canonicalize a few ultra-common biomedical patterns before tokenization.
    # This is intentionally narrow to avoid changing general text unexpectedly.
    text = re.sub(r"\b(covid)\s*[-\s]?\s*(19)\b", r"\1-\2", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(sars)\s*[-\s]?\s*(cov)\s*[-\s]?\s*(2)\b", r"\1-\2-\3", text, flags=re.IGNORECASE)

    text = normalize_biomed_text(text)

    doc = nlp(text)
    cleaned_tokens: List[str] = []

    i = 0
    while i < len(doc):
        tok = doc[i]
        # Keep negation contraction and percent BEFORE punctuation filtering
        if tok.lower_ == "n't":
            cleaned_tokens.append("not")
            i += 1
            continue

        if tok.text == "%":
            cleaned_tokens.append("%")
            i += 1
            continue


        # Skip spaces early
        if tok.is_space:
            i += 1
            continue

        # --- Small, safe merges (when tokenizer splits biomedical patterns) ---
        # covid - 19  -> covid-19
        if tok.lower_ == "covid" and i + 2 < len(doc):
            dash = doc[i + 1]
            num = doc[i + 2]
            if dash.text == "-" and num.text == "19":
                cleaned_tokens.append("covid-19")
                i += 3
                continue

        # sars - cov - 2 -> sars-cov-2
        if tok.lower_ == "sars" and i + 4 < len(doc):
            dash1 = doc[i + 1]
            cov = doc[i + 2]
            dash2 = doc[i + 3]
            two = doc[i + 4]
            if dash1.text == "-" and cov.lower_ == "cov" and dash2.text == "-" and two.text == "2":
                cleaned_tokens.append("sars-cov-2")
                i += 5
                continue

                # ---Merges for biomedical tokens we MUST keep ---

        # 1) IL - 6  -> il-6  (and generally: letters - number)
        if re.match(r"^[a-z]{1,10}$", tok.lower_) and i + 2 < len(doc):
            if doc[i + 1].text == "-" and doc[i + 2].like_num:
                cleaned_tokens.append(f"{tok.lower_}-{doc[i + 2].text}")
                i += 3
                continue

        # 2) CD4 + -> cd4+ (markers)
        if re.match(r"^[a-z]{1,10}\d{1,4}$", tok.lower_) and i + 1 < len(doc):
            if doc[i + 1].text == "+":
                cleaned_tokens.append(tok.lower_ + "+")
                i += 2
                continue

        # 3) TNF - alpha -> tnf-alpha (Greek already normalized to alpha/beta/...)
        if re.match(r"^[a-z0-9]{1,15}$", tok.lower_) and i + 2 < len(doc):
            if doc[i + 1].text == "-" and re.match(r"^(alpha|beta|gamma|delta|kappa|mu|tau|lambda)$", doc[i + 2].lower_):
                cleaned_tokens.append(f"{tok.lower_}-{doc[i + 2].lower_}")
                i += 3
                continue

        # --- Original filtering logic (kept as-is as much as possible) ---
        if tok.is_punct:
            i += 1
            continue
        if tok.like_num:
            i += 1
            continue
        if tok.is_currency:
            i += 1
            continue
        if tok.is_bracket:
            i += 1
            continue

        word = tok.lower_
        # Normalize hyphen variants inside token too (defensive).
        word = re.sub(r"[‐‑‒–—−]", "-", word).strip("-")
        if not word:
            i += 1
            continue

        # IMPORTANT CHANGE:
        # Allow digits + internal hyphens (needed for covid-19 / sars-cov-2 / il-6 / h1n1 ...)
        # but still avoid pure numeric strings and other noisy tokens.
                # Allow + inside biomedical markers (CD4+, CD8+)
        if not re.match(r"^[a-z0-9\-\+]+$", word):
            i += 1
            continue

        # must contain at least one letter (avoid tokens like "----" or pure symbols)
        if not re.search(r"[a-z]", word):
            i += 1
            continue


        cleaned_tokens.append(word)
        i += 1

    return cleaned_tokens



def preprocess_for_classification(text: str) -> str:
    """
    Classifier preprocessing:
    - Not lossless (natural learning simplification)
    - But retains important clinical patterns: mg/dL, CD4+, p<0.05  
    """
    text = clean_surface_noise(text)
    if not text:
        return ""

    text = normalize_biomed_text(text, convert_greek=True)
    doc = nlp(text)

    out = []
    i = 0
    while i < len(doc):
        tok = doc[i]

        if tok.is_space:
            i += 1
            continue

        # p<0.05 merge
        if i + 2 < len(doc) and tok.lower_ == "p" and doc[i+1].text in {"<", ">", "≤", "≥", "="} and doc[i+2].like_num:
            out.append(f"p{doc[i+1].text}{doc[i+2].text}")
            i += 3
            continue

        # mg/dL , IL-6/IL-10 merge
        if i + 2 < len(doc) and doc[i+1].text == "/" and any(c.isalpha() for c in tok.text) and any(c.isalpha() for c in doc[i+2].text):
            out.append(f"{tok.text}/{doc[i+2].text}".lower())
            i += 3
            continue

        # CD4+
        if i + 1 < len(doc) and doc[i+1].text == "+" and any(c.isalnum() for c in tok.text):
            out.append((tok.text + "+").lower())
            i += 2
            continue

        # drop punctuation
        if tok.is_punct:
            i += 1
            continue

        # keep decimals
        if tok.like_num:
            if "." in tok.text:
                out.append(tok.text)
            i += 1
            continue

        if tok.is_stop:
            i += 1
            continue

        lemma = tok.lemma_.lower()
        lemma = re.sub(r"[^a-z0-9\-+%']+", "", lemma)
        if lemma:
            out.append(lemma)

        i += 1

    return " ".join(out)


def preprocess_for_correction(text: str) -> list[dict]:
    """
    Tokenize for correction WITHOUT destroying separators.
    Returns pieces preserving original whitespace, so we can rebuild safely.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    if not text:
        return []

    doc = nlp(text)  
    pieces = []
    for tok in doc:
        pieces.append({
            "text": tok.text,
            "ws": tok.whitespace_,   # ✅ key: preserves original spaces/no-spaces
        })
    return pieces
