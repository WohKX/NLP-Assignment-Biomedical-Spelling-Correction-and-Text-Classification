"""

Spelling model for the CORD-19 Writing & Topic Assistant.
---This script is a "draft" for future work to add the ability to merge external typo_map dictionary for medical termenology---

Design Goals:
1) Hybrid Lexicon:
   - Domain lexicon from CORD-19 artifacts (vocab.pkl)
   - Optional general-English support via wordfreq (zipf_frequency)
2) Robust Context Scoring:
   - "Stupid Backoff" for bigrams (Brants et al. idea- 2007) the idea here is: If bigram exists, use it. Else, backoff to unigram with a penalty (alpha)
   - Backoff to unigram when a bigram context is unseen
3) Do-No-Harm Policy:
   - Strong bias to KEEP the original word when it is already a valid word
   - Real-word correction is heavily gated to avoid disasters like:
     law -> low, your -> our, any -> and, student -> students
4) Clear Error Types:
   - "Correct"
   - "Non-word error"
   - "Real-word error"
   - "Unknown (OOV)" (we do not change the token)

Compatibility:
- build_spelling_model.py artifacts: vocab.pkl, unigrams.pkl, bigrams.pkl
- app/main.py expects: MedicalSpellChecker.from_artifacts, correct, candidates
"""

from __future__ import annotations

import math
import os
import pickle
import csv
import json
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Set, Tuple

from config import VOCAB_PATH, UNIGRAMS_PATH, BIGRAMS_PATH

#  dependency .
try:
    from wordfreq import zipf_frequency  # type: ignore
except Exception:
    zipf_frequency = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pickle(path):
    if not os.path.exists(str(path)):
        raise FileNotFoundError(f"Artifact not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _match_case(src: str, tgt: str) -> str:
    """Best-effort casing preservation for token-level corrections."""
    if not src:
        return tgt
    if src.isupper():
        return tgt.upper()
    if src[:1].isupper() and src[1:].islower():
        return tgt[:1].upper() + tgt[1:]
    return tgt


def _load_optional_typo_map(artifact_dir: Path) -> Dict[str, str]:
    """
    Load an *optional* typo map from disk to keep the spellchecker data-driven.

    Supported formats (first match wins):
      - typo_map.json  : {"wrong": "correct", ...}
      - typo_map.tsv   : wrong<TAB>correct (one pair per line)
      - typo_map.csv   : wrong,correct (two columns)

    If no file exists, an empty map is returned.
    """
    if artifact_dir is None:
        return {}
    base = Path(artifact_dir)
    candidates = [base / "typo_map.json", base / "typo_map.tsv", base / "typo_map.csv"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return {}

    mapping: Dict[str, str] = {}
    try:
        if path.suffix.lower() == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                for k, v in obj.items():
                    ks = str(k).strip()
                    vs = str(v).strip()
                    if ks and vs and ks.lower() != vs.lower():
                        mapping[ks.lower()] = vs.lower()
        else:
            dialect = csv.excel_tab if path.suffix.lower() == ".tsv" else csv.excel
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f, dialect=dialect)
                for row in reader:
                    if not row or len(row) < 2:
                        continue
                    k = str(row[0]).strip()
                    v = str(row[1]).strip()
                    if not k or not v:
                        continue
                    if k.startswith("#"):
                        continue
                    if k.lower() != v.lower():
                        mapping[k.lower()] = v.lower()
    except Exception:
        return {}

    return mapping


def _safe_log(x: float, floor: float) -> float:
    if x <= 0.0:
        return floor
    return math.log(x)

@dataclass(frozen=True)
class SpellCandidate:
    word: str
    dist: int  # 0, 1, or 2


# ---------------------------------------------------------------------------
# Norvig-style edit generators (fast)
# ---------------------------------------------------------------------------

ALPHABET = "abcdefghijklmnopqrstuvwxyz"

def edits1(word: str) -> Set[str]:
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in ALPHABET]
    inserts = [L + c + R for L, R in splits for c in ALPHABET]
    return set(deletes + transposes + replaces + inserts)

def edits2(word: str, cap_e1: int = 60) -> Set[str]:
    e1 = list(edits1(word))
    e1 = e1[:cap_e1]  # cap branching for speed
    out: Set[str] = set()
    for w1 in e1:
        out.update(edits1(w1))
    return out


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MedicalSpellChecker:
    """
    Hybrid spell checker: medical/domain + general English (optional).
    """

    # Words that cause massive false positives in real-word correction.
    # We protect them from real-word correction by default.
    _REALWORD_PROTECT: Set[str] = {
        "the","a","an","and","or","but","to","of","in","on","for","with","as","at","by",
        "is","are","was","were","be","been","being","am",
        "this","that","these","those",
        "it","its","it's","they","them","their","there","then","than",
        "i","we","you","he","she","my","our","your","his","her",
        "any","some","no","not","do","did","done","does","can","could","may","might","must",
        "will","would","should",
    }

    def __init__(
        self,
        vocab: Iterable[str],
        unigrams: Counter,
        bigrams: Dict[str, Counter],
        *,
        min_zipf: float = 3.0,
        backoff_alpha: float = 0.4,
        # Scoring weights (keep simple + explainable)
        #
        # These default weights favour the original word by balancing
        # unigram and bigram probabilities against edit distance.  They
        # provide robust corrections for non‑words while minimizing false
        # positives for real words.  If you need more aggressive
        # real‑word correction behaviour, consider adjusting the values
        # after careful evaluation.
        w_uni: float = 1.0,
        w_bi: float = 1.2,
        w_zipf: float = 0.08,
        w_dist: float = 2.2,
        # Do‑no‑harm settings
        #
        # Preserve the original strong bias toward keeping known words by
        # default.  Real‑word corrections will only occur when the
        # statistical evidence is overwhelming or when a special rule
        # explicitly applies.  See _special_correction() for domain‑specific
        # exceptions.
        keep_original_bonus: float = 5.0,
        realword_extra_margin: float = 2.0,
    ) -> None:
        # Domain lexicon
        self.domain_vocab: Set[str] = set(vocab)

        # LM stats
        self.unigrams: Counter = Counter(unigrams)
        self.bigrams: Dict[str, Counter] = {str(p): Counter(c) for p, c in bigrams.items()}
        self.total_unigrams: int = int(sum(self.unigrams.values()) or 1)

        # Precompute row totals for bigrams
        self.bigram_totals: Dict[str, int] = {
            prev: int(sum(cnts.values()) or 1) for prev, cnts in self.bigrams.items()
        }

        # Optional general-English support
        self.min_zipf = float(min_zipf)
        self.use_wordfreq = zipf_frequency is not None

        # Backoff + scoring
        self.backoff_alpha = float(backoff_alpha)
        self.w_uni = float(w_uni)
        self.w_bi = float(w_bi)
        self.w_zipf = float(w_zipf)
        self.w_dist = float(w_dist)

        # Floors for log-prob (dynamic, not magic -20)
        # approx log(1/(N+1)) then subtract small buffer
        self.min_log_prob = math.log(1.0 / (self.total_unigrams + 1.0)) - 5.0

        # Do-no-harm
        self.keep_original_bonus = float(keep_original_bonus)
        self.realword_extra_margin = float(realword_extra_margin)

        # Optional external typo map (data-driven, kept out of code).
        raw_map = typo_map or {}
        self.typo_map: Dict[str, str] = {
            str(k).strip().lower(): str(v).strip().lower()
            for k, v in raw_map.items()
            if str(k).strip() and str(v).strip() and str(k).strip().lower() != str(v).strip().lower()
        }

    @classmethod
    def from_artifacts(cls, artifact_dir: Optional[str] = None) -> "MedicalSpellChecker":
        """Load spellchecker artifacts (optionally from a given artifacts directory)."""
        # Note: core artifacts paths are defined in config.py, but we use artifact_dir
        # (if provided) for *optional* resources like typo maps.
        vocab = _load_pickle(VOCAB_PATH)
        unigrams = _load_pickle(UNIGRAMS_PATH)
        bigrams = _load_pickle(BIGRAMS_PATH)

        adir = Path(artifact_dir) if artifact_dir else Path(VOCAB_PATH).parent
        typo_map = _load_optional_typo_map(adir)

        return cls(vocab=vocab, unigrams=unigrams, bigrams=bigrams, typo_map=typo_map)

    def _zipf(self, word: str) -> float:
        if not self.use_wordfreq:
            return 0.0
        try:
            return float(zipf_frequency(word, "en"))
        except Exception:
            return 0.0

    def is_known(self, word: str) -> bool:
        if not word:
            return False
        w = word.lower()

        # Domain word?
        if w in self.domain_vocab:
            return True

        # General English (optional)
        if self.use_wordfreq and self._zipf(w) >= self.min_zipf:
            return True

        return False

    def _protect_realword(self, w: str) -> bool:
        """
        Return True if we should NOT attempt real-word correction for this token.
        This prevents common disasters and keeps behavior stable.
        """
        if len(w) <= 3:
            return True
        if w in self._REALWORD_PROTECT:
            return True
        return False

    # -----------------------
    # Probability model
    # -----------------------

    def _log_p_unigram(self, w: str) -> float:
        # Add-1 smoothing for unigrams (safe)
        c = float(self.unigrams.get(w, 0))
        V = float(max(len(self.unigrams), 1))
        p = (c + 1.0) / (float(self.total_unigrams) + V)
        return _safe_log(p, self.min_log_prob)

    def _log_p_bigram_backoff(self, prev: Optional[str], w: str) -> float:
        """
        True "stupid backoff":
        - if bigram count exists: log( count(w|prev) / total(prev) )
        - else: log(alpha) + log P_unigram(w)
        """
        if not prev:
            return self._log_p_unigram(w)

        p = prev.lower()
        row = self.bigrams.get(p)
        if row:
            cnt = row.get(w, 0)
            if cnt > 0:
                denom = float(self.bigram_totals.get(p, 1))
                return _safe_log(float(cnt) / denom, self.min_log_prob)

        # backoff
        return _safe_log(self.backoff_alpha, self.min_log_prob) + self._log_p_unigram(w)

    # -----------------------
    # Candidate generation
    # -----------------------

    def _generate_candidates(self, w: str) -> List[SpellCandidate]:
        """
        Generate candidates at distance 1/2 and keep only KNOWN words.
        """
        out: List[SpellCandidate] = []

        # dist=1
        # Generate all single‑edit candidates that are known words and are not
        # identical to the original token.  We explicitly exclude the
        # original token here because the correct() method considers the
        # original separately when deciding whether to perform a real‑word
        # correction.
        c1 = [c for c in edits1(w) if self.is_known(c) and c != w]
        out.extend(SpellCandidate(word=c, dist=1) for c in c1)

        # dist=2
        # Always search second‑level edits for additional candidates.  Some
        # plausible corrections (e.g. "virus" for "versus") require two
        # edits.  We again exclude the original token and cap the number of
        # returned candidates for efficiency.
        for c in edits2(w, cap_e1=200):
            if len(out) >= 80:
                break
            if self.is_known(c) and c != w:
                # Avoid duplicates
                if c not in [cand.word for cand in out]:
                    out.append(SpellCandidate(word=c, dist=2))

        return out

    # -----------------------
    # Scoring
    # -----------------------

    def _score(self, cand: SpellCandidate, prev: Optional[str]) -> float:
        w = cand.word.lower()
        lp_uni = self._log_p_unigram(w)
        lp_bi = self._log_p_bigram_backoff(prev, w)

        # Small zipf bonus (tie-breaker)
        zipf_bonus = self._zipf(w) if self.use_wordfreq else 0.0

        # Distance penalty (actually applied)
        dist_pen = float(cand.dist)

        return (
            self.w_uni * lp_uni
            + self.w_bi * lp_bi
            + self.w_zipf * zipf_bonus
            - self.w_dist * dist_pen
        )

    # -----------------------
    # Public API
    # -----------------------

    def correct(self, word: str, prev_word: Optional[str] = None) -> Tuple[str, str]:
        """
        Return (corrected_word, error_type).

        Policy:
        - If token is known => usually "Correct"
        - Real-word correction is heavily gated (to avoid overcorrection).
        - If token is unknown:
            - If we find known candidates => "Non-word error"
            - Else => "Unknown (OOV)" (no change)
        """
        if not word:
            return word, "Correct"

        w = word.lower()

        # --- Domain‑specific short‑circuit corrections ---
        #
        # Certain high‑impact real‑word errors are not easily captured by the
        # generic probabilistic model (e.g. "corona versus" should be
        # "corona virus", "covid-18" should be "covid-19").  We define
        # these custom mappings to ensure they are corrected even when the
        # statistical evidence is sparse.  Only apply these corrections
        # when the candidate appears in the vocabulary and the context
        # strongly suggests it.
        spec = self._special_correction(w, prev_word)
        if spec is not None:
            return spec, "Real-word error"

        # Ignore tokens without letters
        if not any(ch.isalpha() for ch in w):
            return word, "Correct"

        original_known = self.is_known(w)

        # If unknown => try to correct as non-word
        if not original_known:
            cands = self._generate_candidates(w)
            if not cands:
                return word, "Unknown (OOV)"
            best = max(cands, key=lambda c: self._score(c, prev_word))
            return best.word, "Non-word error"

        # Known word => default is "Correct"
        # Real-word correction only if NOT protected and evidence is overwhelming.
        if self._protect_realword(w):
            return word, "Correct"

        # Build candidate set for potential real-word correction:
        # we include original (dist=0) + small edit candidates
        cands = [SpellCandidate(word=w, dist=0)]
        cands.extend(self._generate_candidates(w))

        # Score all
        scored = [(self._score(c, prev_word), c) for c in cands]
        scored.sort(key=lambda x: x[0], reverse=True)

        best_score, best_c = scored[0]

        # Find original score
        orig_score = next((s for s, c in scored if c.word == w), -1e9)

        # Do-no-harm: add a strong keep-bonus to the original score
        # (implemented as a margin requirement)
        keep_target = orig_score + self.keep_original_bonus + self.realword_extra_margin

        if best_c.word != w and best_score > keep_target:
            return best_c.word, "Real-word error"

        return word, "Correct"

    def candidates(self, word: str) -> List[str]:
        """
        Return top suggestions for UI. This is for display only.
        """
        if not word:
            return []
        w = word.lower()
        if len(w) <= 2 or not any(ch.isalpha() for ch in w):
            return []

        cands = self._generate_candidates(w)
        # rank without context (prev=None) for UI list
        ranked = sorted(cands, key=lambda c: self._score(c, None), reverse=True)
        return [c.word for c in ranked[:8]]

    # ---- Optional UI helpers (safe even if main.py doesn't use them) ----

    @staticmethod
    def _damerau_levenshtein_distance(s1: str, s2: str) -> int:
        # Exact DL distance (for UI reporting, not for heavy scoring)
        d = {}
        len1, len2 = len(s1), len(s2)
        for i in range(-1, len1 + 1):
            d[(i, -1)] = i + 1
        for j in range(-1, len2 + 1):
            d[(-1, j)] = j + 1

        for i in range(len1):
            for j in range(len2):
                cost = 0 if s1[i] == s2[j] else 1
                d[(i, j)] = min(
                    d[(i - 1, j)] + 1,        # deletion
                    d[(i, j - 1)] + 1,        # insertion
                    d[(i - 1, j - 1)] + cost  # substitution
                )
                if i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                    d[(i, j)] = min(d[(i, j)], d[(i - 2, j - 2)] + cost)  # transposition
        return d[(len1 - 1, len2 - 1)]

    def edit_distance(self, w1: str, w2: str) -> int:
        return self._damerau_levenshtein_distance(w1.lower(), w2.lower())

    def candidates_with_distance(self, word: str) -> Set[Tuple[str, int]]:
        out = set()
        for c in self.candidates(word):
            out.add((c, self.edit_distance(word, c)))
        return out

    def score_candidate(
        self,
        candidate: str,
        prev_word: str,
        precalc_dist: int = 0,
        original_word: str = ""
    ) -> float:
        # Compatibility layer
        cand = SpellCandidate(word=candidate.lower(), dist=int(precalc_dist))
        return self._score(cand, prev_word)

    # ------------------------------------------------------------------
    # Domain‑specific corrections
    # ------------------------------------------------------------------
    def _special_correction(self, w: str, prev: Optional[str]) -> Optional[str]:
        """
        Return a domain‑specific correction for the given word if a known
        confusion pair is detected.  This helps capture common real‑word
        errors that the generic language model may overlook.

        Args:
            w: The lower‑cased token to check.
            prev: The previous token in the sequence (lower‑cased or None).

        Returns:
            The corrected form if a special rule applies, otherwise None.
        """
        if prev:
            prev_l = prev.lower()
        else:
            prev_l = None

        # 1. "versus" -> "virus".
        #
        #   Two heuristics are used:
        #   a) If the previous token is one of the known disease terms then
        #      "versus" is almost certainly intended to be "virus".
        #   b) If the conditional bigram count for the candidate "virus"
        #      is much higher than for "versus", we treat it as a real‑word
        #      error even without the specific preceding term.  This helps
        #      correct phrases like "a versus" which are rare, while
        #      "a virus" is common.
        if w == "versus":
            # a) Preceding disease term
            if prev_l in {"corona", "covid", "covid-19", "sars", "mers", "hiv", "cov"}:
                return "virus"
            # b) Bigram ratio threshold
            if prev_l is not None:
                orig_cnt = self.bigrams.get(prev_l, {}).get("versus", 0)
                virus_cnt = self.bigrams.get(prev_l, {}).get("virus", 0)
                # require at least one occurrence of candidate and a ratio > 20
                if virus_cnt > 0 and virus_cnt >= (orig_cnt + 1) * 20:
                    return "virus"

        # 2. "virtues" -> "virus" (common misspelling)
        if w == "virtues":
            if prev_l in {"corona", "covid", "covid-19", "sars", "mers"}:
                if self.is_known("virus"):
                    return "virus"

        # 3. "covid-18" -> "covid-19" regardless of vocabulary.  The term
        #    covid‑18 is a frequent typographical mistake for covid‑19 and
        #    should always be corrected.
        if w in {"covid-18", "covid18", "covd-18"}:
            return "covid-19"

        # 4. Protect certain domain words from being altered (no correction)
        #    This method does not return a correction here; the protection is
        #    handled in is_known/_protect_realword.  Leaving this for
        #    completeness if future rules require forced no‑changes.
        # Optional external typo map (kept out of code to avoid any hard-coded "demo" lists).
        if self.typo_map:
            mapped = self.typo_map.get(wl)
            if mapped and (not self.is_known(wl)) and self.is_known(mapped):
                return _match_case(word, mapped)

        return None