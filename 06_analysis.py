"""
===============================================================================
AISA - AI Semantic Analyzer
06_analysis.py - Sentiment analysis + AI Buzz Index computation
===============================================================================

Two responsibilities:

  1. SENTIMENT ANALYSIS
     - Primary:  FinBERT (ProsusAI/finbert) - finance-domain sentiment
     - Fallback:  VADER (lexicon-based, no model needed)
     - Hard fail if NEITHER is available
     - Batch processing: all references encoded in one call
     - Special handling for AI governance context (negative → neutral)

  2. AI BUZZ INDEX (ABI)
     Measures frequency and intensity of AI mentions in corporate reports.
     NOT a measure of real-world AI adoption — a proxy for how prominently
     a company discusses AI in its public disclosures.

     Seven weighted sub-dimensions (from AnalyzerConfig.weights):
         volume          - mention density per page
         depth           - semantic specificity of AI language
         breadth         - taxonomy category coverage
         tone            - sentiment polarity
         specificity     - concrete vs. superficial mentions
         forward_looking - future-oriented language
         salience        - investment / deployment signals

     Per company-year → AIBuzzIndex dataclass
     Industry aggregation → IndustryBuzzIndex

VisualizationGenerator is NOT in this module (moved to viz/ - future scope).

CHANGELOG:
    v1.0.0 (2026-02) - AISA initial release
    v1.0.1 (2026-02) - Renamed AI Adoption Index → AI Buzz Index
        Sub-dimensions renamed: intensity→volume, semantic→depth,
        diversity→breadth, sentiment→tone, maturity→specificity,
        future→forward_looking, commitment→salience

Author: TeRa0
License: MIT
===============================================================================
"""

from __future__ import annotations

import importlib
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from version import AISA_VERSION, BUZZ_DIMENSION_LABELS

_m1 = importlib.import_module("01_models")
AIReference     = _m1.AIReference
AnalyzerConfig  = _m1.AnalyzerConfig
DocumentResult  = _m1.DocumentResult
AIBuzzIndex     = _m1.AIBuzzIndex
logger          = _m1.logger


# ============================================================================
# SENTIMENT BACKEND - FinBERT or VADER (hard fail if neither)
# ============================================================================

_FINBERT_AVAILABLE = False
_VADER_AVAILABLE   = False

# --- Try FinBERT first ---
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )
    import torch
    import torch.nn.functional as F

    _FINBERT_TOKENIZER = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    _FINBERT_MODEL     = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert"
    )
    _FINBERT_MODEL.eval()
    _FINBERT_LABELS = ["positive", "negative", "neutral"]
    _FINBERT_AVAILABLE = True
    logger.info("FinBERT loaded: ProsusAI/finbert")
except Exception as _fb_exc:
    logger.warning(f"FinBERT not available: {_fb_exc}")

# --- Try VADER fallback ---
if not _FINBERT_AVAILABLE:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VADER_CLS
        _VADER = _VADER_CLS()
        _VADER_AVAILABLE = True
        logger.info("VADER sentiment loaded as fallback")
    except ImportError:
        pass

if not _FINBERT_AVAILABLE and not _VADER_AVAILABLE:
    raise RuntimeError(
        "Neither FinBERT (transformers + torch) nor VADER (vaderSentiment) "
        "is available. Install at least one:\n"
        "  pip install transformers torch\n"
        "  pip install vaderSentiment"
    )

_SENTIMENT_BACKEND = "finbert" if _FINBERT_AVAILABLE else "vader"
logger.info(f"Sentiment backend: {_SENTIMENT_BACKEND}")


# ============================================================================
# GOVERNANCE CONTEXT PATTERNS
# References about AI ethics/policy are often phrased negatively but
# signal commitment, not risk. Reclassified → neutral.
# ============================================================================

_GOVERNANCE_RE = re.compile(
    r"\b(?:"
    r"responsible AI|ethical AI|AI ethics|AI governance|AI policy|"
    r"bias (?:detection|mitigation)|algorithmic fairness|"
    r"AI safety|guardrails|hallucination|alignment|red teaming|"
    r"explainab(?:le|ility)|interpretab(?:le|ility)|"
    r"trustworthy AI|transparent AI|accountable AI|"
    r"risk framework|compliance framework"
    r")\b",
    re.IGNORECASE,
)

# Forward-looking signals (for forward_looking sub-dimension)
_FORWARD_LOOKING_RE = re.compile(
    r"\b(?:"
    r"will|plan(?:ned|ning)?|intend|expect|target|forecast|roadmap|"
    r"next year|future|upcoming|by \d{4}|going forward|"
    r"initiative|pilot|proof of concept|POC|prototype"
    r")\b",
    re.IGNORECASE,
)

# Salience signals (for salience sub-dimension)
_SALIENCE_RE = re.compile(
    r"\b(?:"
    r"invest(?:ed|ing|ment)|acqui(?:red|ring|sition)|partner(?:ed|ship)|"
    r"launch(?:ed|ing)?|deploy(?:ed|ing|ment)|implement(?:ed|ing|ation)|"
    r"billion|million|\$[\d,.]+|center of excellence|AI team|"
    r"dedicated|committed|strategic"
    r")\b",
    re.IGNORECASE,
)


# ============================================================================
# SENTIMENT COMPUTATION
# ============================================================================

@dataclass
class SentimentResult:
    """Sentiment output for a single text fragment."""
    label:      str    # "positive" | "negative" | "neutral"
    score:      float  # Probability of the winning label (0..1)
    confidence: str    # "finbert" | "vader" | "governance_adjusted"


def _sentiment_finbert_batch(texts: List[str]) -> List[SentimentResult]:
    """
    Run FinBERT on a batch of texts.

    Truncates at 512 tokens. Returns one SentimentResult per text.
    """
    results: List[SentimentResult] = []
    batch_size = 32

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = _FINBERT_TOKENIZER(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = _FINBERT_MODEL(**enc).logits
            probs  = F.softmax(logits, dim=-1).cpu().numpy()

        for row in probs:
            best_idx = int(np.argmax(row))
            results.append(SentimentResult(
                label=_FINBERT_LABELS[best_idx],
                score=round(float(row[best_idx]), 4),
                confidence="finbert",
            ))

    return results


def _sentiment_vader_batch(texts: List[str]) -> List[SentimentResult]:
    """
    Run VADER on a batch of texts.

    Converts compound score to label + probability proxy.
    """
    results: List[SentimentResult] = []
    for text in texts:
        scores   = _VADER.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            label = "positive"
            prob  = round((compound + 1) / 2, 4)
        elif compound <= -0.05:
            label = "negative"
            prob  = round((1 - compound) / 2, 4)
        else:
            label = "neutral"
            prob  = round(1 - abs(compound), 4)
        results.append(SentimentResult(label=label, score=prob, confidence="vader"))

    return results


def _apply_governance_adjustment(
    ref: AIReference,
    result: SentimentResult,
) -> SentimentResult:
    """
    Reclassify negative sentiment as neutral when governance context is present.

    AI governance language (bias detection, responsible AI, etc.) is
    often phrased negatively but signals commitment, not risk.
    """
    if result.label == "negative" and _GOVERNANCE_RE.search(
        f"{ref.text} {ref.context}"
    ):
        return SentimentResult(
            label="neutral",
            score=result.score,
            confidence="governance_adjusted",
        )
    return result


def analyze_sentiment_batch(
    references: List[AIReference],
    config: Optional[AnalyzerConfig] = None,
) -> List[AIReference]:
    """
    Run sentiment analysis on a list of AIReference objects (in place).

    Updates ref.sentiment, ref.sentiment_score, ref.sentiment_confidence
    for each reference. Uses whichever backend is available (FinBERT > VADER).

    Args:
        references: List of AIReference with sentiment == "pending".
        config:     AnalyzerConfig (reserved for future flags).

    Returns:
        The same list with sentiment fields populated.
    """
    if not references:
        return references

    pending = [r for r in references if r.sentiment == "pending"]
    if not pending:
        return references

    texts = [f"{r.text} {r.context}"[:512] for r in pending]

    t0 = time.perf_counter()
    if _FINBERT_AVAILABLE:
        raw_results = _sentiment_finbert_batch(texts)
    else:
        raw_results = _sentiment_vader_batch(texts)
    elapsed = time.perf_counter() - t0
    logger.debug(
        f"Sentiment ({_SENTIMENT_BACKEND}): {len(pending)} refs in {elapsed*1000:.0f}ms"
    )

    for ref, result in zip(pending, raw_results):
        result = _apply_governance_adjustment(ref, result)
        ref.sentiment            = result.label
        ref.sentiment_score      = result.score
        ref.sentiment_confidence = result.confidence

    return references


def analyze_sentiment_document(doc: DocumentResult) -> DocumentResult:
    """
    Convenience wrapper: analyze sentiment for all references in a document.

    Args:
        doc: DocumentResult whose references have sentiment == "pending".

    Returns:
        The same DocumentResult with sentiment populated.
    """
    analyze_sentiment_batch(doc.references)
    return doc


# ============================================================================
# AI BUZZ INDEX - SUB-DIMENSION CALCULATORS
# ============================================================================

def _calc_volume(refs: List[AIReference], total_pages: int) -> float:
    """
    Volume: normalized AI mention density per page.

    Formula: tanh(refs_per_page) — ensures (0,1) range with diminishing returns.
    1 mention/page ≈ 0.76; 3/page ≈ 0.995.
    """
    if total_pages == 0:
        return 0.0
    refs_per_page = len(refs) / total_pages
    return round(float(np.tanh(refs_per_page)), 4)


def _calc_depth(refs: List[AIReference]) -> float:
    """
    Depth: average semantic similarity score across all references.

    Higher scores indicate more specific, technical AI language
    (closer to the anchor embedding space in 05_detect.py).
    """
    if not refs:
        return 0.0
    scores = [r.semantic_score for r in refs if r.semantic_score > 0]
    if not scores:
        return 0.0
    return round(float(np.mean(scores)), 4)


def _calc_breadth(refs: List[AIReference]) -> float:
    """
    Breadth: taxonomy coverage measured via Shannon entropy.

    Normalized by log2(16) — max for 8 application + 8 technology categories.
    For N-dimensional taxonomies the normalization base grows proportionally.
    High breadth = AI mentioned across many different areas.
    """
    import json as _json

    cat_counts: Dict[str, int] = {}
    for r in refs:
        # Backward-compat axes
        if r.category_a:
            cat_counts[r.category_a] = cat_counts.get(r.category_a, 0) + 1
        if r.category_b:
            cat_counts[r.category_b] = cat_counts.get(r.category_b, 0) + 1
        # Extra dimensions (e.g. 3rd axis G for Digitalization taxonomy)
        dj = getattr(r, "dimensions_json", "")
        if dj:
            try:
                dims = _json.loads(dj)
                for code_conf in dims.values():
                    if isinstance(code_conf, (list, tuple)) and code_conf:
                        code = code_conf[0]
                        if code and code not in (r.category_a, r.category_b):
                            cat_counts[code] = cat_counts.get(code, 0) + 1
            except Exception:
                pass

    MAX_CATEGORIES = max(16, len(cat_counts))

    if not cat_counts:
        return 0.0

    total = sum(cat_counts.values())
    entropy = 0.0
    for count in cat_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    max_entropy = math.log2(MAX_CATEGORIES)
    return round(min(entropy / max_entropy, 1.0), 4) if max_entropy > 0 else 0.0


def _calc_tone(refs: List[AIReference]) -> float:
    """
    Tone: weighted sentiment polarity of AI mentions.

    positive → 1.0, neutral → 0.5, negative → 0.0
    Weighted by model confidence (sentiment_score).
    Defaults to 0.5 (neutral) when no scored references exist.
    """
    if not refs:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0

    for r in refs:
        if r.sentiment == "pending":
            continue
        weight = r.sentiment_score if r.sentiment_score > 0 else 0.5
        if r.sentiment == "positive":
            val = 1.0
        elif r.sentiment == "negative":
            val = 0.0
        else:
            val = 0.5
        weighted_sum  += val * weight
        total_weight  += weight

    if total_weight == 0:
        return 0.5
    return round(weighted_sum / total_weight, 4)


def _calc_specificity(refs: List[AIReference]) -> float:
    """
    Specificity: proportion of concrete vs. superficial AI mentions.

    'strong' references count double — they indicate verifiable deployments
    or specific technical details, not just passing mentions.
    """
    if not refs:
        return 0.0
    strong  = sum(1 for r in refs if r.reference_strength == "strong")
    medium  = sum(1 for r in refs if r.reference_strength == "medium")
    weighted = (strong * 2 + medium) / (len(refs) * 2)
    return round(min(weighted, 1.0), 4)


def _calc_forward_looking(refs: List[AIReference]) -> float:
    """
    Forward-looking: proportion of references with future-oriented language.

    High forward_looking + low specificity = planning/aspiration stage.
    High forward_looking + high specificity = active expansion stage.
    """
    if not refs:
        return 0.0
    count = sum(
        1 for r in refs
        if _FORWARD_LOOKING_RE.search(f"{r.text} {r.context}")
    )
    return round(count / len(refs), 4)


def _calc_salience(refs: List[AIReference]) -> float:
    """
    Salience: proportion of references showing resource commitment signals.

    Captures investment amounts, deployment announcements, partnerships,
    and dedicated team references.
    """
    if not refs:
        return 0.0
    count = sum(
        1 for r in refs
        if _SALIENCE_RE.search(f"{r.text} {r.context}")
    )
    return round(count / len(refs), 4)


# ============================================================================
# AI BUZZ INDEX - COMPOSITE CALCULATOR
# ============================================================================

def calculate_buzz_index(
    refs:           List[AIReference],
    company:        str,
    year:           int,
    position:       int,
    industry:       str,
    sector:         str,
    country:        str,
    total_pages:    int,
    config:         AnalyzerConfig,
) -> AIBuzzIndex:
    """
    Compute the AI Buzz Index for one company-year.

    Combines seven sub-dimension scores into a single weighted composite.
    Weights are read from config.weights (must sum to 1.0).

    The resulting score reflects how prominently and specifically a company
    discusses AI in its annual report — not necessarily actual AI adoption.

    Args:
        refs:           Deduplicated AIReference list for this company-year.
        company:        Company name.
        year:           Report year.
        position:       Fortune 500 rank.
        industry:       Industry classification.
        sector:         Sector classification.
        country:        Country.
        total_pages:    Total pages in the document.
        config:         AnalyzerConfig with weight definitions.

    Returns:
        AIBuzzIndex dataclass.
    """
    w = config.weights

    vol = _calc_volume(refs, total_pages)
    dep = _calc_depth(refs)
    bre = _calc_breadth(refs)
    ton = _calc_tone(refs)
    spe = _calc_specificity(refs)
    fwd = _calc_forward_looking(refs)
    sal = _calc_salience(refs)

    composite = (
        vol * w.get("volume",           0.15) +
        dep * w.get("depth",            0.20) +
        bre * w.get("breadth",          0.15) +
        ton * w.get("tone",             0.15) +
        spe * w.get("specificity",      0.15) +
        fwd * w.get("forward_looking",  0.10) +
        sal * w.get("salience",         0.10)
    )

    unique_cats: set = set()
    for r in refs:
        if r.category_a:
            unique_cats.add(r.category_a)
        if r.category_b:
            unique_cats.add(r.category_b)
        dj = getattr(r, "dimensions_json", "")
        if dj:
            try:
                import json as _json
                for code_conf in _json.loads(dj).values():
                    if isinstance(code_conf, (list, tuple)) and code_conf and code_conf[0]:
                        unique_cats.add(code_conf[0])
            except Exception:
                pass

    return AIBuzzIndex(
        company=company,
        year=year,
        position=position,
        industry=industry,
        sector=sector,
        country=country,
        volume_index=vol,
        depth_index=dep,
        breadth_index=bre,
        tone_index=ton,
        specificity_index=spe,
        forward_looking_index=fwd,
        salience_index=sal,
        ai_buzz_index=round(composite, 4),
        total_refs=len(refs),
        total_pages=total_pages,
        categories_used=len(unique_cats),
    )


# ============================================================================
# RANKING UTILITY
# ============================================================================

def rank_buzz_indices(
    indices: List[AIBuzzIndex],
) -> List[AIBuzzIndex]:
    """
    Assign rank_in_year to each index within the same year.

    Modifies each AIBuzzIndex.rank_in_year in place.
    Rank 1 = highest ai_buzz_index within the year.

    Args:
        indices: List of AIBuzzIndex (may span multiple years).

    Returns:
        Same list with rank_in_year populated.
    """
    years = set(i.year for i in indices)
    for year in years:
        year_group = sorted(
            [i for i in indices if i.year == year],
            key=lambda x: x.ai_buzz_index,
            reverse=True,
        )
        for rank, idx_obj in enumerate(year_group, start=1):
            idx_obj.rank_in_year = rank

    return indices


# ============================================================================
# INDUSTRY AGGREGATION
# ============================================================================

@dataclass
class IndustryBuzzIndex:
    """Aggregated AI Buzz Index for an industry-year pair."""
    industry:                   str
    year:                       int
    avg_volume_index:           float = 0.0
    avg_depth_index:            float = 0.0
    avg_breadth_index:          float = 0.0
    avg_tone_index:             float = 0.0
    avg_specificity_index:      float = 0.0
    avg_forward_looking_index:  float = 0.0
    avg_salience_index:         float = 0.0
    ai_buzz_index_industry:     float = 0.0
    rank_among_industries:      int   = 0
    num_companies:              int   = 0
    total_refs:                 int   = 0
    min_index:                  float = 0.0
    max_index:                  float = 0.0
    std_deviation:              float = 0.0
    companies_list:             str   = ""


def aggregate_by_industry(
    indices: List[AIBuzzIndex],
) -> List[IndustryBuzzIndex]:
    """
    Compute industry-level aggregates from company-year buzz indices.

    Args:
        indices: List of AIBuzzIndex objects (with rank_in_year set).

    Returns:
        List of IndustryBuzzIndex, one per industry-year pair,
        sorted by (industry, year) with rank_among_industries set.
    """
    from collections import defaultdict

    groups: Dict[Tuple, List[AIBuzzIndex]] = defaultdict(list)
    for idx in indices:
        groups[(idx.industry, idx.year)].append(idx)

    result: List[IndustryBuzzIndex] = []
    for (industry, year), group in sorted(groups.items()):
        scores = [g.ai_buzz_index for g in group]
        arr    = np.array(scores)

        ind = IndustryBuzzIndex(
            industry=industry,
            year=year,
            avg_volume_index=round(float(np.mean([g.volume_index for g in group])), 4),
            avg_depth_index=round(float(np.mean([g.depth_index for g in group])), 4),
            avg_breadth_index=round(float(np.mean([g.breadth_index for g in group])), 4),
            avg_tone_index=round(float(np.mean([g.tone_index for g in group])), 4),
            avg_specificity_index=round(float(np.mean([g.specificity_index for g in group])), 4),
            avg_forward_looking_index=round(float(np.mean([g.forward_looking_index for g in group])), 4),
            avg_salience_index=round(float(np.mean([g.salience_index for g in group])), 4),
            ai_buzz_index_industry=round(float(np.mean(arr)), 4),
            num_companies=len(group),
            total_refs=sum(g.total_refs for g in group),
            min_index=round(float(arr.min()), 4),
            max_index=round(float(arr.max()), 4),
            std_deviation=round(float(arr.std()), 4),
            companies_list=", ".join(sorted(g.company for g in group)),
        )
        result.append(ind)

    # Rank industries within each year
    years = set(i.year for i in result)
    for year in years:
        yr_group = sorted(
            [i for i in result if i.year == year],
            key=lambda x: x.ai_buzz_index_industry,
            reverse=True,
        )
        for rank, ind in enumerate(yr_group, start=1):
            ind.rank_among_industries = rank

    return result


# ============================================================================
# SMOKE TEST
# ============================================================================

if __name__ == "__main__":
    from version import get_version_string
    print(get_version_string())
    print(f"  Sentiment backend: {_SENTIMENT_BACKEND}")
    print()

    cfg = AnalyzerConfig()

    # Verify weight keys match sub-dimensions
    expected_keys = {"volume", "depth", "breadth", "tone", "specificity", "forward_looking", "salience"}
    assert set(cfg.weights.keys()) == expected_keys, (
        f"Weight keys mismatch: {cfg.weights.keys()}"
    )
    print(f"  Weight keys OK: {list(cfg.weights.keys())}")

    # --- Build synthetic references ---
    def _make_ref(text, context, sent="pending", sem=0.75, strength="strong",
                  cat_a="A2_Operational_Excellence", cat_b="B4_GenAI_LLMs"):
        return AIReference(
            company="TestCorp", year=2024, position=1,
            industry="Technology", sector="Software", country="USA",
            doc_type="Annual Report",
            text=text, context=context,
            page=1, category=f"{cat_a}|{cat_b}",
            detection_method="pattern_hard",
            sentiment=sent, sentiment_score=0.0,
            semantic_score=sem, source="TestCorp_2024.pdf",
            category_a=cat_a, confidence_a=0.8,
            category_b=cat_b, confidence_b=0.75,
            reference_strength=strength, confidence_score=0.8,
        )

    refs = [
        _make_ref(
            "We deployed ChatGPT across customer service, reducing costs by 30%.",
            "The AI deployment initiative was completed in Q2.",
        ),
        _make_ref(
            "Our responsible AI governance framework ensures bias detection.",
            "We invested in ethical AI oversight.",
            cat_a="A7_Governance_Ethics", cat_b="B8_General_AI",
            strength="medium",
        ),
        _make_ref(
            "We plan to expand our ML infrastructure to 5 additional regions by 2025.",
            "This strategic AI investment supports our digital transformation roadmap.",
            cat_a="A6_Strategy_Investment", cat_b="B7_Infrastructure_Platforms",
            strength="medium",
        ),
        _make_ref(
            "SageMaker pipelines process 100M events daily with 99.9% uptime.",
            "MLOps maturity is central to our AI strategy.",
            cat_a="A2_Operational_Excellence", cat_b="B7_Infrastructure_Platforms",
            strength="strong", sem=0.82,
        ),
    ]

    # --- Sentiment ---
    print("--- Sentiment analysis ---")
    analyze_sentiment_batch(refs, cfg)
    for r in refs:
        print(
            f"  [{r.sentiment:<8}] score={r.sentiment_score:.3f} "
            f"conf={r.sentiment_confidence:<22} | {r.text[:60]}..."
        )
    assert all(r.sentiment != "pending" for r in refs)

    gov_ref = refs[1]
    assert gov_ref.sentiment in ("neutral", "positive"), (
        f"Governance ref should be neutral or positive, got: {gov_ref.sentiment}"
    )
    print(f"  Governance adjustment: OK (A7 ref = {gov_ref.sentiment})")

    # --- Sub-dimensions ---
    print()
    print("--- AI Buzz Index sub-dimensions ---")
    total_pages = 150

    dims = {
        "volume":           _calc_volume(refs, total_pages),
        "depth":            _calc_depth(refs),
        "breadth":          _calc_breadth(refs),
        "tone":             _calc_tone(refs),
        "specificity":      _calc_specificity(refs),
        "forward_looking":  _calc_forward_looking(refs),
        "salience":         _calc_salience(refs),
    }
    for name, val in dims.items():
        label = BUZZ_DIMENSION_LABELS.get(name, name)
        print(f"  {label:<28} {val}")
        assert 0.0 <= val <= 1.0, f"{name} out of range: {val}"

    # --- Composite ---
    print()
    print("--- Composite AI Buzz Index ---")
    idx = calculate_buzz_index(
        refs=refs, company="TestCorp", year=2024, position=1,
        industry="Technology", sector="Software", country="USA",
        total_pages=total_pages, config=cfg,
    )
    print(f"  ABI score:        {idx.ai_buzz_index}")
    print(f"  Categories used:  {idx.categories_used}")
    print(f"  Total refs:       {idx.total_refs}")
    assert 0.0 <= idx.ai_buzz_index <= 1.0
    assert idx.categories_used > 0

    # --- Ranking ---
    print()
    print("--- Ranking ---")
    idx2 = calculate_buzz_index(
        refs=refs[:2], company="OtherCorp", year=2024, position=5,
        industry="Technology", sector="Software", country="USA",
        total_pages=100, config=cfg,
    )
    ranked = rank_buzz_indices([idx, idx2])
    for r in ranked:
        print(f"  {r.company:<12} rank={r.rank_in_year}  ABI={r.ai_buzz_index}")
    assert ranked[0].rank_in_year != ranked[1].rank_in_year

    # --- Industry aggregation ---
    print()
    print("--- Industry aggregation ---")
    industries = aggregate_by_industry([idx, idx2])
    assert len(industries) == 1
    ind = industries[0]
    print(f"  Industry: {ind.industry} {ind.year}")
    print(f"  Companies: {ind.num_companies}  avg_ABI: {ind.ai_buzz_index_industry}")
    print(f"  min={ind.min_index}  max={ind.max_index}  std={ind.std_deviation}")
    assert ind.num_companies == 2
    assert ind.rank_among_industries == 1

    print()
    print("  06_analysis.py all checks passed.")
