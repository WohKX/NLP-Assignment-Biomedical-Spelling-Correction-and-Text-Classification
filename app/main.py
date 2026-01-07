# app/main.py
# CORD-19 Intelligence Hub (Spell Correction + Topic Classification)
# ---------------------------------------------------------------
# Key UI requirement:
#   3 parallel panes that can run independently AND show results together:
#   (1) Spell correction  (2) Classify BEFORE  (3) Classify AFTER
#
# This app prioritizes reliability-first, white-box behavior:
# - Conservative correction (protect biomedical-like tokens)
# - Transparent vocabulary explorer
# - Before/after impact visibility

import sys
from pathlib import Path

# --- Runtime bootstrap (fixes 'No module named config') ---
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.runtime import ensure_project_root
ensure_project_root()
# ----------------------------------------------------------

import re
import time
import pickle
import json
import hashlib
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# 0) Robust import of preprocessing
# ============================================================
_PREPROCESS_OK = True
_PREPROCESS_IMPORT_ERR: Optional[Exception] = None

try:
    from utils.preprocess import preprocess_for_correction, preprocess_for_classification
except Exception as e:  # noqa: BLE001
    _PREPROCESS_OK = False
    _PREPROCESS_IMPORT_ERR = e

    def preprocess_for_classification(text: str) -> str:
        text = (text or "").lower()
        text = re.sub(r"[^a-z0-9\s\-_/]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def preprocess_for_correction(text: str) -> List[Dict[str, str]]:
        text = "" if text is None else str(text)
        pieces: List[Dict[str, str]] = []
        for m in re.finditer(r"\S+\s*", text):
            chunk = m.group(0)
            tok = chunk.rstrip()
            ws = chunk[len(tok):]
            pieces.append({"text": tok, "ws": ws})
        return pieces


# ============================================================
# 0b) Config import
# ============================================================
try:
    from config import (
        SPELLING_ARTIFACT_DIR,
        CLASSIF_ARTIFACT_DIR,
        UNIGRAMS_PATH,
        BIGRAMS_PATH,
    )
except Exception:
    from config import SPELLING_ARTIFACT_DIR, CLASSIF_ARTIFACT_DIR  # type: ignore
    UNIGRAMS_PATH = Path(SPELLING_ARTIFACT_DIR) / "unigrams.pkl"
    BIGRAMS_PATH = Path(SPELLING_ARTIFACT_DIR) / "bigrams.pkl"

from spelling.model import MedicalSpellChecker


# ============================================================
# 0c) Visual constants
# ============================================================
MAX_CHARS = 500
MAX_SUGGESTIONS = 5

LABEL_COLORS = {
    "Prevention": "#1abc9c",
    "Treatment": "#9b59b6",
    "Epidemiology": "#e74c3c",
}

COMMON_ACADEMIC_STOPWORDS = {
    "study","studies","result","results","method","methods","data","analysis","analyses",
    "paper","papers","patient","patients","disease","diseases","clinical","medical","health",
    "research","model","models","based","using","used","use","also","may","can","could","would",
    "should","however","therefore","thus","et","al","figure","table","approach","system","reported",
    "significant","significantly","increase","decrease","effect","effects","impact","impacts",
    "high","low","new","novel","conclusion","conclusions","introduction","background",
    # extra common non-medical words that often dominate frequency tables
    "between","within","during","before","after","among","across","throughout","therein","thereof",
    "because","although","towards","toward","furthermore","thereby","therein",
}


# ============================================================
# 0d) Streamlit page + CSS
# ============================================================
st.set_page_config(
    page_title="CORD-19 Intelligence Hub",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

_prev = LABEL_COLORS.get("Prevention", "#1abc9c")
_treat = LABEL_COLORS.get("Treatment", "#9b59b6")
_epi = LABEL_COLORS.get("Epidemiology", "#e74c3c")

st.markdown(
    """
<style>
:root {{
  --color-prev: {prev};
  --color-treat: {treat};
  --color-epi: {epi};
}}

.block-container {{
  padding-top: 1.15rem;
  padding-bottom: 1.5rem;
  max-width: 1500px;
}}
.stApp {{
  background-color: #f8f9fa;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}}

/* -------- Readability Boost (v4) -------- */
html {{ font-size: 18px; }} /* scales rem-based typography */

/* Default text */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] div {{
  font-size: 1.05rem !important;
  line-height: 1.6 !important;
}}

/* Labels / helper text */
label, .stCaption, .stAlert, .stMarkdown, .muted, .tiny {{
  font-size: 1.00rem !important;
}}

/* Inputs */
input, textarea, select {{
  font-size: 1.05rem !important;
}}

/* Tables */
div[data-testid="stDataFrame"] * {{
  font-size: 1.00rem !important;
}}
div[data-testid="stTable"] * {{
  font-size: 1.00rem !important;
}}

/* Buttons */
div.stButton > button {{
  width: 100%;
  border-radius: 10px;
  height: 3.0em;
  font-weight: 800;
  transition: all 0.2s;
  font-size: 1.05rem !important;
}}

/* Headings */
h1 {{
  color: #2c3e50;
  font-weight: 800;
  margin-top: 0.15rem;
  margin-bottom: 0.35rem;
  font-size: 2.25rem !important;
}}
h2 {{
  color: #2c3e50;
  font-weight: 800;
  margin-top: 0.15rem;
  margin-bottom: 0.35rem;
  font-size: 1.75rem !important;
}}
h3 {{
  color: #2c3e50;
  font-weight: 800;
  margin-top: 0.15rem;
  margin-bottom: 0.35rem;
  font-size: 1.35rem !important;
}}

.card {{
  background: white;
  padding: 12px 14px;
  border-radius: 14px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.06);
  border: 1px solid #eee;
  font-size: 1.03rem;
}}
.muted {{
  color: #7f8c8d;
}}
.tiny {{
  color: #95a5a6;
}}

.error-span {{
  background-color: rgba(231, 76, 60, 0.14);
  color: var(--color-epi);
  padding: 1px 6px;
  border-radius: 6px;
  font-weight: 800;
  text-decoration: underline wavy var(--color-epi);
  cursor: help;
}}

.pill {{
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  border: 1px solid #eee;
  background: #fbfbfb;
  margin-right: 6px;
  margin-bottom: 6px;
  font-weight: 700;
  font-size: 0.95rem;
}}

.prob-box {{
  background-color: white;
  padding: 12px 14px;
  border-radius: 12px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.06);
  border-left: 5px solid #3498db;
}}
</style>
""".format(prev=_prev, treat=_treat, epi=_epi),
    unsafe_allow_html=True,
)



# ============================================================
# 1) Compatibility helpers
# ============================================================
def safe_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
        return
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
        return
    st.stop()


def safe_divider() -> None:
    if hasattr(st, "divider"):
        st.divider()
    else:
        st.markdown("---")


def dataframe_in(container, df: pd.DataFrame, **kwargs) -> None:
    """Render dataframe inside a container (sidebar or main), with compatibility."""
    try:
        container.dataframe(df, **kwargs)
    except TypeError:
        kwargs.pop("hide_index", None)
        container.dataframe(df, **kwargs)


def _hash_text(s: str) -> str:
    return hashlib.md5((s or "").encode("utf-8")).hexdigest()


def copy_button(text: str, *, key: str, label: str = "📋 Copy") -> None:
    """
    Clipboard copy button using a tiny HTML+JS component.
    Works on modern browsers; if blocked, user can still copy from code block.
    """
    # JSON-encode safely for JS
    payload = json.dumps(text)
    btn_id = f"copy_btn_{key}_{_hash_text(text)[:8]}"
    html = f"""
    <div style="display:flex; gap:8px; align-items:center;">
      <button id="{btn_id}" style="
        width:100%;
        border-radius:10px;
        height:2.6em;
        font-weight:800;
        border:1px solid #e6e6e6;
        background:white;
        cursor:pointer;">
        {label}
      </button>
      <span id="{btn_id}_msg" style="color:#7f8c8d; font-size:0.9rem;"></span>
    </div>
    <script>
      const btn = document.getElementById("{btn_id}");
      const msg = document.getElementById("{btn_id}_msg");
      btn.addEventListener("click", async () => {{
        try {{
          const txt = {payload};
          await navigator.clipboard.writeText(txt);
          msg.textContent = "Copied!";
          setTimeout(() => msg.textContent = "", 1400);
        }} catch (e) {{
          msg.textContent = "Copy blocked by browser. Use the code block copy icon.";
          setTimeout(() => msg.textContent = "", 2800);
        }}
      }});
    </script>
    """
    components.html(html, height=52)


# ============================================================
# 2) Cached artifact loading
# ============================================================
@st.cache_resource
def load_spell_checker() -> Optional[MedicalSpellChecker]:
    try:
        return MedicalSpellChecker.from_artifacts(SPELLING_ARTIFACT_DIR)
    except FileNotFoundError:
        st.error("⚠️ Spelling artifacts not found. Run build_spelling_model.py.")
        return None
    except Exception as e:  # noqa: BLE001
        st.error(f"⚠️ Failed to load spell checker: {e}")
        return None


def clean_text_wrapper(text_series):
    """
    Pickle compatibility helper (some pipelines store FunctionTransformer(clean_text_wrapper)).
    """
    if isinstance(text_series, pd.DataFrame):
        return text_series.iloc[:, 0].apply(preprocess_for_classification)
    if isinstance(text_series, pd.Series):
        return text_series.apply(preprocess_for_classification)
    return [preprocess_for_classification(t) for t in text_series]


@st.cache_resource
def load_classifier():
    joblib_path = Path(CLASSIF_ARTIFACT_DIR) / "topic_classifier.joblib"
    pkl_path = Path(CLASSIF_ARTIFACT_DIR) / "topic_classifier.pkl"

    if joblib_path.exists():
        payload = joblib.load(joblib_path)
        pipeline = payload.get("model")
        label_map = payload.get("label_map") or payload.get("label_mapping")
        metrics = payload.get("metrics", {}) or {}
        vectorizer = payload.get("vectorizer")
        return pipeline, label_map, metrics, vectorizer

    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)
        pipeline = payload.get("model")
        label_map = payload.get("label_map") or payload.get("label_mapping")
        metrics = payload.get("metrics", {}) or {}
        return pipeline, label_map, metrics, None

    st.error("⚠️ Classifier model not found. Run python -m classification.train_classifier")
    return None, None, {}, None


@st.cache_resource
def load_unigrams_bigrams() -> Tuple[Dict[str, int], Dict[Tuple[str, str], int]]:
    unigrams: Dict[str, int] = {}
    bigrams: Dict[Tuple[str, str], int] = {}

    # unigrams
    try:
        if Path(UNIGRAMS_PATH).exists():
            with open(UNIGRAMS_PATH, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, Counter):
                unigrams = {str(k): int(v) for k, v in obj.items()}
            elif isinstance(obj, dict):
                unigrams = {str(k): int(v) for k, v in obj.items()}
    except Exception:
        unigrams = {}

    # bigrams (expected keys like ("word1","word2"))
    try:
        if Path(BIGRAMS_PATH).exists():
            with open(BIGRAMS_PATH, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, Counter):
                bigrams = {(str(k[0]), str(k[1])): int(v) for k, v in obj.items() if isinstance(k, tuple) and len(k) == 2}
            elif isinstance(obj, dict):
                bigrams = {(str(k[0]), str(k[1])): int(v) for k, v in obj.items() if isinstance(k, tuple) and len(k) == 2}
    except Exception:
        bigrams = {}

    return unigrams, bigrams


spell_checker = load_spell_checker()
classifier_pipeline, label_mapping, clf_metrics, clf_vectorizer = load_classifier()
unigram_counts, bigram_counts = load_unigrams_bigrams()

inv_label_mapping: Optional[Dict[int, str]] = None
if isinstance(label_mapping, dict):
    inv_label_mapping = {v: k for k, v in label_mapping.items() if isinstance(v, int)}


# ============================================================
# 3) Spell-check helpers
# ============================================================
def _is_wordlike(tok: str) -> bool:
    return bool(tok) and any(ch.isalpha() for ch in tok)


def _match_case(orig: str, new: str) -> str:
    if not new:
        return new
    if orig.isupper():
        return new.upper()
    if len(orig) > 1 and orig[0].isupper() and orig[1:].islower():
        return new.capitalize()
    return new


def is_biomed_protected(token: str) -> bool:
    """
    Conservative protection against over-correction for biomedical-like tokens.
    Allow a tiny whitelist for COVID/SARS variants.
    """
    t = token or ""
    low = t.lower()

    if re.fullmatch(r"(covid|covid19|covid-?19|sars-?cov-?2|2019-?ncov|ncov)", low):
        return False

    if re.search(r"\d", t):
        return True
    if re.search(r"[<>=/\\\+\*%]", t):
        return True
    if "_" in t:
        return True
    if len(t) >= 26:
        return True
    if re.search(r"[\u0370-\u03FF\u1F00-\u1FFF]", t):  # greek
        return True
    return False


def build_corrected_text(results: List[Dict[str, Any]], overrides: Dict[int, str]) -> str:
    out: List[str] = []
    for r in results:
        idx = r.get("index")
        tok = r.get("token", "")
        ws = r.get("ws", "")
        repl = overrides.get(idx, r.get("corrected", tok))
        out.append(f"{repl}{ws}")
    return "".join(out).rstrip()


def spell_check(text: str, max_suggestions: int = MAX_SUGGESTIONS) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """
    Token-preserving spell-check.
    Returns: (results, annotated_html, stats)
    """
    if not spell_checker:
        return [], "", {"error": "Spell checker model not loaded."}

    pieces = preprocess_for_correction(text)  # [{"text":..., "ws":...}, ...]
    tokens = [p.get("text", "") for p in pieces]

    # next alpha context
    next_alpha: List[Optional[str]] = [None] * len(tokens)
    nxt: Optional[str] = None
    for i in range(len(tokens) - 1, -1, -1):
        tok = tokens[i]
        next_alpha[i] = nxt
        if tok.isalpha():
            nxt = tok.lower()

    results: List[Dict[str, Any]] = []
    annotated_parts: List[str] = []
    prev_word: Optional[str] = None

    n_errors = 0
    n_changed = 0

    def _freq(c: str) -> int:
        return int(unigram_counts.get(c.lower(), 0)) if unigram_counts else 0

    for i, p in enumerate(pieces):
        tok = p.get("text", "")
        ws = p.get("ws", "")

        attempt = _is_wordlike(tok) and (not is_biomed_protected(tok)) and len(tok) >= 3

        if not attempt:
            results.append({"index": i, "token": tok, "corrected": tok, "type": "Correct", "suggestions": [], "ws": ws})
            annotated_parts.append(tok + ws)
            if tok.isalpha():
                prev_word = tok.lower()
            continue

        corrected, err_type = spell_checker.correct(tok, prev_word=prev_word, next_word=next_alpha[i])
        corrected = _match_case(tok, corrected)

        suggestions: List[Tuple[str, int, int]] = []
        if err_type != "Correct":
            try:
                cands = list(spell_checker.candidates_with_distance(tok))
                cands_sorted = sorted(cands, key=lambda x: (int(x[1]), -_freq(x[0]), str(x[0])))
                for c, d in cands_sorted[:max_suggestions]:
                    suggestions.append((str(c), int(d), _freq(str(c))))
            except Exception:
                suggestions = []

        if err_type != "Correct":
            n_errors += 1
        if corrected != tok:
            n_changed += 1

        results.append({
            "index": i,
            "token": tok,
            "corrected": corrected,
            "type": err_type,
            "suggestions": suggestions,
            "ws": ws,
        })

        if err_type == "Correct":
            annotated_parts.append(tok + ws)
        else:
            cand_tip = ", ".join([f"{c}(d={d})" for c, d, _f in suggestions])
            tooltip = f"auto: {corrected}" + (f" | {cand_tip}" if cand_tip else "")
            annotated_parts.append(f"<span class='error-span' title='{tooltip}'>{tok}</span>{ws}")

        if tok.isalpha():
            prev_word = tok.lower()

    annotated = "".join(annotated_parts)
    stats = {"n_tokens": len(tokens), "n_errors": n_errors, "n_changed": n_changed}
    return results, annotated, stats


# ============================================================
# 4) Classification helpers
# ============================================================
def _label_name_from_id(class_id: int) -> str:
    if inv_label_mapping and class_id in inv_label_mapping:
        return inv_label_mapping[class_id]
    return str(class_id)


def run_classification(text: str) -> Dict[str, Any]:
    if classifier_pipeline is None:
        return {"error": "Classifier model not loaded."}

    try:
        raw_input = [text]
        pred = classifier_pipeline.predict(raw_input)
        pred_id = int(pred[0])
        pred_label = _label_name_from_id(pred_id)

        probs_out: Optional[List[Dict[str, Any]]] = None
        if hasattr(classifier_pipeline, "predict_proba"):
            proba = classifier_pipeline.predict_proba(raw_input)[0]
            classes = getattr(classifier_pipeline, "classes_", list(range(len(proba))))
            probs_out = [{"label": _label_name_from_id(int(cid)), "prob": float(p)} for cid, p in zip(classes, proba)]
            probs_out.sort(key=lambda x: x["prob"], reverse=True)

        # Explainability (linear model inside pipeline)
        top_terms: Optional[List[Dict[str, Any]]] = None
        expl_err: Optional[str] = None
        try:
            pipe = classifier_pipeline
            if not hasattr(pipe, "named_steps"):
                raise RuntimeError("Pipeline steps unavailable.")

            vect = pipe.named_steps.get("tfidf") or clf_vectorizer
            clf = pipe.named_steps.get("clf")
            pre = pipe.named_steps.get("preprocessor")

            if vect is None or clf is None:
                raise RuntimeError("Vectorizer/classifier not found.")
            if not hasattr(clf, "coef_"):
                raise RuntimeError("Classifier is not linear (coef_ missing).")

            cleaned_input = raw_input
            if pre is not None and hasattr(pre, "transform"):
                cleaned_input = pre.transform(raw_input)

            X = vect.transform(cleaned_input)
            feature_names = np.array(vect.get_feature_names_out())
            class_ids = getattr(clf, "classes_", np.arange(clf.coef_.shape[0]))

            try:
                row = int(np.where(class_ids == pred_id)[0][0])
            except Exception:
                row = 0

            coef = clf.coef_[row]
            tfidf_vals = X.toarray()[0]
            contrib = (X.multiply(coef)).toarray()[0]

            present = np.where(tfidf_vals > 0)[0]
            present_terms = feature_names[present]
            present_tfidf = tfidf_vals[present]
            present_weights = coef[present]
            present_contrib = contrib[present]

            order = np.argsort(-present_contrib)[:15]
            top_terms = []
            for j in order:
                top_terms.append({
                    "term": str(present_terms[j]),
                    "tfidf": float(present_tfidf[j]),
                    "weight": float(present_weights[j]),
                    "contribution": float(present_contrib[j]),
                })
        except Exception as e:  # noqa: BLE001
            expl_err = f"Explainability unavailable: {e}"

        return {
            "pred_id": pred_id,
            "pred_label": pred_label,
            "probs": probs_out,
            "top_terms": top_terms,
            "expl_err": expl_err,
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"Classification failed: {e}"}


def render_label_card(label: str, subtitle: Optional[str] = None) -> None:
    color = LABEL_COLORS.get(label, "#34495e")
    sub_html = f"<div class='muted' style='margin-top:6px;'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"""
<div class="card" style="border-left: 6px solid {color};">
  <div style="font-size:1.1rem; font-weight:900; color:{color};">Predicted: {label}</div>
  {sub_html}
</div>
""",
        unsafe_allow_html=True,
    )


def render_probability_bars(probs: Optional[List[Dict[str, Any]]]) -> None:
    if not probs:
        st.info("Probabilities are unavailable (model may not support predict_proba).")
        return
    st.markdown("<div class='prob-box'>", unsafe_allow_html=True)
    for row in probs:
        lab = row["label"]
        p = float(row["prob"])
        st.write(f"**{lab}** — {p:.3f}")
        st.progress(min(max(p, 0.0), 1.0))
    st.markdown("</div>", unsafe_allow_html=True)


def render_probability_delta(raw_probs, corr_probs) -> None:
    if not raw_probs or not corr_probs:
        return
    r = {d["label"]: float(d["prob"]) for d in raw_probs}
    c = {d["label"]: float(d["prob"]) for d in corr_probs}
    labels = sorted(set(r) | set(c))
    rows = [{"label": lab, "before": r.get(lab, 0.0), "after": c.get(lab, 0.0), "delta": c.get(lab, 0.0) - r.get(lab, 0.0)} for lab in labels]
    df = pd.DataFrame(rows).sort_values("delta", ascending=False)
    dataframe_in(st, df, use_container_width=True, hide_index=True)


# ============================================================
# 5) Dictionary helpers (medical-ish view)
# ============================================================
def _valid_unigram(term: str) -> bool:
    t = (term or "").strip().lower()
    if not t:
        return False
    if not t.isalpha():
        return False
    if len(t) <= 5:  # IMPORTANT: medical-ish constraint
        return False
    if t in COMMON_ACADEMIC_STOPWORDS:
        return False
    return True


def top_medical_unigrams(unigrams: Dict[str, int], top_n: int = 15) -> List[Tuple[str, int]]:
    items = [(t.lower(), int(f)) for t, f in unigrams.items() if _valid_unigram(str(t))]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:top_n]


def top_bigrams(bigrams: Dict[Tuple[str, str], int], top_n: int = 15) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for (a, b), f in bigrams.items():
        a2, b2 = str(a).lower(), str(b).lower()
        if not a2.isalpha() or not b2.isalpha():
            continue
        if a2 in COMMON_ACADEMIC_STOPWORDS or b2 in COMMON_ACADEMIC_STOPWORDS:
            continue
        if len(a2) < 3 or len(b2) < 3:
            continue
        out.append((f"{a2} {b2}", int(f)))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_n]


def render_dictionary_explorer(container, unigrams: Dict[str, int], bigrams: Dict[Tuple[str, str], int], key_prefix: str = "dict_") -> None:
    if not unigrams:
        container.info("Vocabulary is not available.")
        return

    c1, c2 = container.columns([2, 1])
    with c1:
        q = container.text_input("Search term", key=f"{key_prefix}q", placeholder="e.g., vaccine, remdesivir")
    with c2:
        show_n = container.selectbox("Show", [50, 100, 200, 500, 1000], index=1, key=f"{key_prefix}n")

    # Enforce medical-ish constraint: default min_len = 6 and fixed
    min_len = 6
    container.caption("Constraint: show terms with length ≥ 6 (to avoid stopwords).")

    mode = container.radio("Sort by", ["Frequency", "Alphabetical"], horizontal=True, key=f"{key_prefix}sort")

    items = list(unigrams.items())
    if q:
        qlow = q.lower().strip()
        items = [(t, f) for t, f in items if qlow in str(t).lower()]

    # Apply constraint: alpha + length ≥ 6
    items = [(t, int(f)) for t, f in items if str(t).isalpha() and len(str(t)) >= min_len]

    if mode == "Frequency":
        items.sort(key=lambda x: int(x[1]), reverse=True)
    else:
        items.sort(key=lambda x: str(x[0]).lower())

    items = items[: int(show_n)]

    if items:
        df = pd.DataFrame(items, columns=["term", "frequency"])
        dataframe_in(container, df, use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False).encode("utf-8")
        container.download_button(
            "⬇️ Download current view (CSV)",
            data=csv,
            file_name="dictionary_view.csv",
            mime="text/csv",
            key=f"{key_prefix}dl",
        )
    else:
        container.warning("No terms found with length ≥ 6 for this filter/search.")
        if bigrams:
            container.markdown("**Fallback: Most common bigrams**")
            dfb = pd.DataFrame(top_bigrams(bigrams, top_n=20), columns=["bigram", "frequency"])
            dataframe_in(container, dfb, use_container_width=True, hide_index=True)


# ============================================================
# 6) Session state
# ============================================================
def reset_outputs() -> None:
    for k in [
        "spell_results", "spell_overrides", "spell_selected_idx", "corrected_text",
        "classif_raw", "classif_corr", "spell_stats",
        "_spell_runtime_ms", "_raw_runtime_ms", "_corr_runtime_ms",
    ]:
        st.session_state.pop(k, None)


if "spell_overrides" not in st.session_state:
    st.session_state["spell_overrides"] = {}
if "spell_selected_idx" not in st.session_state:
    st.session_state["spell_selected_idx"] = None


# ============================================================
# 7) Sidebar (fix: everything rendered inside sidebar container)
# ============================================================
with st.sidebar:
    st.header("ℹ️ Guidance")
    st.markdown(
        f"""- Keep text concise (**≤ {MAX_CHARS} chars**).
- Run the 3 panes independently, or use **Run ALL**.
- Correction is conservative: biomedical-like tokens are protected.
"""
    )

    if not _PREPROCESS_OK:
        st.warning("Preprocessing is running in FALLBACK mode (spaCy unavailable / mismatched pins).")
        st.caption(f"Import error: {_PREPROCESS_IMPORT_ERR}")

    st.header("📚 Corpus Insights")

    if unigram_counts:
        # Frequent medical-ish terms (length ≥ 6) OR fallback bigrams
        top_uni = top_medical_unigrams(unigram_counts, top_n=12)
        if top_uni:
            st.subheader("Frequent domain terms (length ≥ 6)")
            df = pd.DataFrame(top_uni, columns=["term", "frequency"])
            dataframe_in(st, df, use_container_width=True, hide_index=True)
        else:
            if bigram_counts:
                st.subheader("Fallback: most common bigrams")
                dfb = pd.DataFrame(top_bigrams(bigram_counts, top_n=12), columns=["bigram", "frequency"])
                dataframe_in(st, dfb, use_container_width=True, hide_index=True)
            else:
                st.info("No long terms available, and bigrams.pkl not found.")

        with st.expander("📘 Dictionary Explorer (searchable)"):
            render_dictionary_explorer(st, unigram_counts, bigram_counts, key_prefix="sb_")
    else:
        st.info("Vocabulary is not available (missing unigrams.pkl).")


# ============================================================
# 8) Header
# ============================================================
st.title("🧬 CORD-19 Intelligence Hub")

st.markdown(
    """
<div class="card">
  <div style="font-size:1.05rem; font-weight:900; color:#2c3e50;">Research framing</div>
  <div class="muted" style="margin-top:6px;">
    This interface operationalizes a <b>white-box reliability layer</b> that conservatively repairs lexical corruption,
    then exposes downstream impact by running topic classification <b>before</b> and <b>after</b> correction.
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("<div class='tiny' style='margin-top:8px;'>Tip: you can run the three buttons independently, or click “Run ALL”.</div>", unsafe_allow_html=True)

safe_divider()


# ============================================================
# 9) Input
# ============================================================
st.subheader(f"📄 Input Text (strict {MAX_CHARS}-char editor)")

uploaded_file = st.file_uploader("Upload .txt (Scientific Abstract)", type=["txt"])
if uploaded_file is not None:
    try:
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    except Exception:
        content = ""
    content = (content or "")[:MAX_CHARS]
    if st.session_state.get("_last_upload_name") != uploaded_file.name:
        st.session_state["_last_upload_name"] = uploaded_file.name
        st.session_state["input_text"] = content

example_bank = {
    "— choose an example —": "",
    "Example 1 (Prevention)": "Vaccination strategies reduce transmission and improve public health outcomes in pandemic settings.",
    "Example 2 (Treatment)": "Patients treated with remdesivir showed improved recovery time and reduced viral load.",
    "Example 3 (Epidemiology)": "We analyze incidence and risk factors using cohort data to estimate transmission dynamics.",
}
ex_choice = st.selectbox("Quick examples (optional)", list(example_bank.keys()), index=0)
if ex_choice != "— choose an example —" and not st.session_state.get("input_text"):
    st.session_state["input_text"] = example_bank[ex_choice][:MAX_CHARS]

text_val = st.text_area(
    f"Enter your abstract (max {MAX_CHARS} characters)",
    key="input_text",
    height=220,
    max_chars=MAX_CHARS,
    placeholder="Paste a CORD-19 abstract excerpt here...",
) or ""

st.caption(f"Character counter: **{len(text_val)}/{MAX_CHARS}**")
if len(text_val) >= MAX_CHARS:
    st.warning("Input reached the strict character limit.")

sig = _hash_text(text_val)
if st.session_state.get("_input_sig") != sig:
    st.session_state["_input_sig"] = sig
    reset_outputs()

col_runA, col_runB = st.columns([2, 1])
with col_runA:
    run_all = st.button("🚀 Run ALL (Spelling + Classify Before + Classify After)", key="btn_run_all")
with col_runB:
    if st.button("🧹 Reset outputs", key="btn_reset_all"):
        reset_outputs()
        safe_rerun()

safe_divider()


# ============================================================
# 10) Three panes
# ============================================================
col1, col2, col3 = st.columns(3, gap="large")

# --------------------------
# Pane 1: Spelling
# --------------------------
with col1:
    st.subheader("1) 📝 Spelling Correction")

    run_spell = st.button("Run spelling analysis", key="btn_spell") or run_all

    if run_spell:
        if not text_val.strip():
            st.warning("Please enter text to check.")
        elif not spell_checker:
            st.error("Spell checker model not loaded.")
        else:
            t0 = time.time()
            with st.spinner("Scanning tokens with a conservative do-no-harm policy..."):
                results, annotated, stats = spell_check(text_val, max_suggestions=MAX_SUGGESTIONS)
            st.session_state["spell_results"] = results
            st.session_state["spell_stats"] = stats
            st.session_state["spell_overrides"] = {}
            st.session_state["spell_selected_idx"] = None
            st.session_state["_spell_runtime_ms"] = int((time.time() - t0) * 1000)

    results = st.session_state.get("spell_results")
    overrides: Dict[int, str] = st.session_state.get("spell_overrides", {})

    if results:
        # Always rebuild corrected_text from (results + overrides) so it never becomes "static"
        corrected_text = build_corrected_text(results, overrides)
        st.session_state["corrected_text"] = corrected_text

        stats = st.session_state.get("spell_stats") or {}
        runtime = st.session_state.get("_spell_runtime_ms")
        st.caption(
            f"Tokens: {stats.get('n_tokens','?')} | Detected errors: {stats.get('n_errors','?')} | "
            f"Auto-changed: {stats.get('n_changed','?')}"
            + (f" | Runtime: {runtime} ms" if runtime is not None else "")
        )

        # Annotated preview
        annotated_parts: List[str] = []
        for r in results:
            tok = r.get("token", "")
            ws = r.get("ws", "")
            if r.get("type") != "Correct":
                sugg = r.get("suggestions") or []
                cand_tip = ", ".join([f"{c} (d={d})" for c, d, _f in sugg])
                tooltip = f"auto: {r.get('corrected')}" + (f" | {cand_tip}" if cand_tip else "")
                annotated_parts.append(f"<span class='error-span' title='{tooltip}'>{tok}</span>{ws}")
            else:
                annotated_parts.append(tok + ws)

        st.markdown(
            f"<div class='card' style='line-height:1.8; min-height:80px;'>{''.join(annotated_parts)}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("#### ✨ Corrected Text (current)")
        ccopy, cdl = st.columns([1, 1])
        with ccopy:
            copy_button(corrected_text, key="corr_text", label="📋 Copy corrected text")
        with cdl:
            st.download_button(
                "⬇️ Download corrected text",
                data=corrected_text.encode("utf-8"),
                file_name="corrected_text.txt",
                mime="text/plain",
                key="dl_corrected_txt",
            )
        # Use a code block so the content is always updated (no widget state stickiness)
        st.code(corrected_text, language=None)

        # Summary table
        errors = [r for r in results if r.get("type") != "Correct"]
        if errors:
            st.markdown("#### 🔍 Detected Errors (summary)")
            rows = []
            for r in errors:
                top_cand = None
                top_dist = None
                top_freq = None
                if r.get("suggestions"):
                    top_cand, top_dist, top_freq = r["suggestions"][0]
                rows.append({
                    "token": r.get("token"),
                    "auto_corrected": r.get("corrected"),
                    "type": r.get("type"),
                    "top_candidate": top_cand,
                    "min_edit_distance": top_dist,
                    "candidate_freq": top_freq,
                })
            dataframe_in(st, pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown("#### 🖱️ Override (click a word)")
            err_indices = [r["index"] for r in errors if isinstance(r.get("index"), int)]
            if err_indices:
                chip_cols = st.columns(2)
                for j, idx_tok in enumerate(err_indices):
                    tok = results[idx_tok].get("token")
                    with chip_cols[j % 2]:
                        if st.button(str(tok), key=f"err_chip_{idx_tok}"):
                            st.session_state["spell_selected_idx"] = idx_tok
                            # do not rerun here; selection itself is enough

                sel_idx = st.session_state.get("spell_selected_idx")
                if sel_idx is not None and sel_idx in err_indices:
                    sel = results[sel_idx]
                    st.markdown(f"**Selected:** `{sel.get('token')}` → auto: `{sel.get('corrected')}`")

                    options: List[Tuple[str, str]] = [
                        ("✅ Accept auto-correction", str(sel.get("corrected"))),
                        ("↩️ Keep original", str(sel.get("token"))),
                    ]
                    for c, d, _f in (sel.get("suggestions") or []):
                        options.append((f"• {c} (d={d})", str(c)))

                    for k, (label, cand) in enumerate(options):
                        key_unique = f"apply_{sel_idx}_{k}_{_hash_text(cand)[:8]}"
                        if st.button(label, key=key_unique):
                            st.session_state["spell_overrides"][sel_idx] = cand
                            # corrected_text is rebuilt automatically on rerun,
                            # but we also clear after-classification to avoid mismatch
                            st.session_state.pop("classif_corr", None)
                            st.session_state.pop("_corr_runtime_ms", None)
                            safe_rerun()

                    if st.button("🧹 Reset override for this word", key=f"reset_{sel_idx}"):
                        st.session_state["spell_overrides"].pop(sel_idx, None)
                        st.session_state.pop("classif_corr", None)
                        st.session_state.pop("_corr_runtime_ms", None)
                        safe_rerun()
        else:
            st.success("✅ No spelling errors detected.")

        with st.expander("📘 Dictionary Explorer (searchable)"):
            render_dictionary_explorer(st, unigram_counts, bigram_counts, key_prefix="main_")
    else:
        st.info("Run spelling analysis to see highlights, corrected text, and interactive overrides.")


# --------------------------
# Pane 2: Classify before
# --------------------------
with col2:
    st.subheader("2) 🔬 Classification (Before Correction)")

    run_raw = st.button("Classify original text", key="btn_classify_raw") or run_all
    if run_raw:
        if not text_val.strip():
            st.warning("Please enter text to classify.")
        else:
            t0 = time.time()
            with st.spinner("Classifying the original input..."):
                st.session_state["classif_raw"] = run_classification(text_val)
            st.session_state["_raw_runtime_ms"] = int((time.time() - t0) * 1000)

    out = st.session_state.get("classif_raw")
    if out:
        if out.get("error"):
            st.error(out["error"])
        else:
            runtime = st.session_state.get("_raw_runtime_ms")
            render_label_card(out["pred_label"], subtitle=(f"Runtime: {runtime} ms" if runtime is not None else None))

            if isinstance(clf_metrics, dict) and clf_metrics:
                with st.expander("📊 Model validation metrics (from artifacts)"):
                    dataframe_in(st, pd.DataFrame([clf_metrics]), use_container_width=True, hide_index=True)

            st.markdown("#### 📈 Class Probabilities")
            render_probability_bars(out.get("probs"))

            st.markdown("#### 🧠 Why this label? (Top contributing terms)")
            if out.get("top_terms") is not None:
                dataframe_in(st, pd.DataFrame(out["top_terms"]), use_container_width=True, hide_index=True)
            else:
                st.info(out.get("expl_err") or "No explainability output available.")
    else:
        st.info("Click the button to classify the original input.")


# --------------------------
# Pane 3: Classify after
# --------------------------
with col3:
    st.subheader("3) ✅ Classification (After Correction)")

    run_corr = st.button("Classify corrected text", key="btn_classify_corr") or run_all
    if run_corr:
        if not text_val.strip():
            st.warning("Please enter text first.")
        else:
            # Ensure corrected text exists; if not, compute once (no UI side effects)
            if "spell_results" not in st.session_state and spell_checker:
                results_tmp, _ann, stats_tmp = spell_check(text_val, max_suggestions=MAX_SUGGESTIONS)
                st.session_state["spell_results"] = results_tmp
                st.session_state["spell_stats"] = stats_tmp
                st.session_state.setdefault("spell_overrides", {})
                st.session_state.setdefault("spell_selected_idx", None)

            corrected_text = st.session_state.get("corrected_text")
            if not corrected_text and st.session_state.get("spell_results"):
                corrected_text = build_corrected_text(st.session_state["spell_results"], st.session_state.get("spell_overrides", {}))
                st.session_state["corrected_text"] = corrected_text
            if not corrected_text:
                corrected_text = text_val

            t0 = time.time()
            with st.spinner("Classifying the corrected version..."):
                out_corr = run_classification(corrected_text)
            out_corr["corrected_text"] = corrected_text
            st.session_state["classif_corr"] = out_corr
            st.session_state["_corr_runtime_ms"] = int((time.time() - t0) * 1000)

    out2 = st.session_state.get("classif_corr")
    if out2:
        if out2.get("error"):
            st.error(out2["error"])
        else:
            st.markdown("#### 🧾 Corrected text used")
            st.code(out2.get("corrected_text", ""), language=None)

            runtime = st.session_state.get("_corr_runtime_ms")
            render_label_card(out2["pred_label"], subtitle=(f"Runtime: {runtime} ms" if runtime is not None else None))

            st.markdown("#### 📈 Class Probabilities")
            render_probability_bars(out2.get("probs"))

            st.markdown("#### 🧠 Why this label? (Top contributing terms)")
            if out2.get("top_terms") is not None:
                dataframe_in(st, pd.DataFrame(out2["top_terms"]), use_container_width=True, hide_index=True)
            else:
                st.info(out2.get("expl_err") or "No explainability output available.")
    else:
        st.info("Click the button to classify after spelling correction.")


# ============================================================
# 11) Impact summary (extra beyond rubric)
# ============================================================
safe_divider()
st.subheader("🧾 Impact Summary (Before vs After)")

raw = st.session_state.get("classif_raw")
corr = st.session_state.get("classif_corr")
spell_stats = st.session_state.get("spell_stats") or {}

if raw and corr and (not raw.get("error")) and (not corr.get("error")):
    changed = (raw.get("pred_label") != corr.get("pred_label"))
    st.markdown(
        f"""
<div class="card">
  <div style="font-weight:900; font-size:1.05rem;">Label change: {"✅ YES" if changed else "❌ NO"}</div>
  <div class="muted" style="margin-top:6px;">
    Before: <b>{raw.get("pred_label")}</b> → After: <b>{corr.get("pred_label")}</b>
    &nbsp;|&nbsp; Auto-changed tokens: <b>{spell_stats.get("n_changed","?")}</b>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("#### Δ Probability shifts (After − Before)")
    render_probability_delta(raw.get("probs"), corr.get("probs"))
else:
    st.info("Run classification before and after to view the impact summary.")

safe_divider()
st.markdown(
    "<div style='text-align:center; color:#7f8c8d;'>Designed for NLP Masters Assignment | Streamlit + Scikit-Learn + White-box spelling layer</div>",
    unsafe_allow_html=True,
)
