"""
===============================================================================
AISA - AI Semantic Analyzer
00_text_repair.py - PDF text cleaning and space repair
===============================================================================

Handles two classes of PDF extraction problems:

1. MISSING SPACES  (space_ratio < 0.08)
   Some PDF renderers (Asian-language filings, encrypted PDFs, certain
   annual-report generators) strip inter-word spaces:
       "WeseektoinvestefficientlyinseveralareasoftechnologyAI"
   Fix: wordsegment (English unigram/bigram model) reconstructs word
   boundaries. Known AI product names are protected before segmentation
   so they are not broken (ChatGPT -> "chat gpt" etc.).

2. CAMELCASE CONCATENATION  (space_ratio >= 0.08 but camelCase artifacts)
   PDF ligature encoding sometimes produces:
       "artificialIntelligence", "machineLearningSolutions"
   Fix: regex-based CamelCase splitting with a protected-word list.

Usage in 10_pipeline.py:
    _m00 = importlib.import_module("00_text_repair")
    repair_page_texts = _m00.repair_page_texts

    page_texts, status = extract_pdf_text(doc_meta.source)
    page_texts = repair_page_texts(page_texts)

Module number 00: runs BEFORE any other AISA module.
Has ZERO imports from AISA -- safe inside ProcessPool workers.

CHANGELOG:
    v1.0.0 (2026-02) - initial release

Author: TeRa0
License: MIT
===============================================================================
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger("AISA")

# ============================================================================
# WORDSEGMENT - optional dependency, loaded lazily once per process
# ============================================================================

_ws_loaded: bool = False
_ws_available: bool = False


def _ensure_wordsegment() -> bool:
    """Load wordsegment corpus once per process. Returns True if available."""
    global _ws_loaded, _ws_available
    if _ws_loaded:
        return _ws_available
    _ws_loaded = True
    try:
        import wordsegment
        wordsegment.load()
        _ws_available = True
        logger.info("00_text_repair: wordsegment loaded OK")
    except ImportError:
        _ws_available = False
        logger.warning(
            "00_text_repair: wordsegment not installed -- "
            "space repair uses CamelCase heuristic only. "
            "pip install wordsegment"
        )
    return _ws_available


# ============================================================================
# PROTECTED TERMS
# Known AI/tech compound words that must NOT be broken by any repair step.
# Sorted longest-first so the regex matches greedily.
# ============================================================================

_PROTECTED: List[str] = [
    # Specific products (longest first for regex priority)
    "Azure Cognitive Services", "Azure OpenAI Service", "Azure Machine Learning",
    "Google Cloud AI", "Stable Diffusion", "Hugging Face", "Document AI",
    "Azure OpenAI", "Azure ML", "Vertex AI", "Bing Chat",
    "ChatGPT", "AutoGPT", "LangChain", "LangGraph", "LangSmith",
    "TensorFlow", "PyTorch", "SageMaker",
    "LightGBM", "XGBoost", "DataRobot",
    "OpenAI", "GenAI", "DeepMind", "GitHub", "LinkedIn", "GitLab",
    "MLOps", "LLMOps", "DevOps", "DataOps", "FinOps", "AIOps",
    "AutoML", "AutoGen", "CrewAI", "Watsonx",
    "Copilot", "Bedrock", "Rekognition", "Comprehend",
    "Watson", "Llama", "Mistral", "Mixtral", "Gemini", "Codex",
    "Perplexity", "Midjourney", "Cohere",
    "Databricks", "Snowflake", "MLflow",
    "DALL-E", "DALLE", "spaCy", "YOLO", "BERT", "NLTK",
    "GPT-4", "GPT-3", "GPT4", "GPT3",
    # Common abbreviations wordsegment would split letter-by-letter
    "AI", "ML", "NLP", "OCR", "RPA", "IoT", "ERP", "CRM",
    "ESG", "CEO", "CFO", "COO", "CTO", "IPO",
    "APIs", "API", "SaaS", "PaaS", "IaaS",
]

_PROTECT_RE = re.compile(
    r'(?<![A-Za-z])(' +
    '|'.join(re.escape(t) for t in sorted(_PROTECTED, key=len, reverse=True)) +
    r')(?![A-Za-z])',
    re.IGNORECASE,
)

# ============================================================================
# CAMELCASE SPLITTING PATTERNS (used when wordsegment unavailable)
# ============================================================================

_CAMEL_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'([a-z])([A-Z][a-z])'),  r'\1 \2'),
    (re.compile(r'([a-z])([A-Z]{2,})'),   r'\1 \2'),
    (re.compile(r'([A-Z]{2,})([a-z])'),   r'\1 \2'),
    (re.compile(r'(\.)([A-Z])'),           r'\1 \2'),
    (re.compile(r'(,)([a-zA-Z])'),         r'\1 \2'),
]


# ============================================================================
# DETECTION
# ============================================================================

def is_space_deficient(text: str, threshold: float = 0.08) -> bool:
    """
    Return True if text has suspiciously few spaces.
    Normal English prose: ~15-20% spaces. Below 8% = words concatenated.
    """
    if len(text) < 20:   # too short to judge reliably
        return False
    return (text.count(' ') / len(text)) < threshold


def has_camelcase_artifacts(text: str) -> bool:
    """
    Return True if text has suspicious CamelCase that is likely a PDF
    extraction artifact. Requires >= 2 non-protected transitions.
    """
    candidates = re.findall(r'[a-z]{3,}[A-Z][a-z]{3,}', text)
    if len(candidates) < 2:
        return False
    non_protected = [c for c in candidates if not _PROTECT_RE.search(c)]
    return len(non_protected) >= 2


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _protect_terms(text: str) -> Tuple[str, Dict[str, str]]:
    """Replace protected terms with short placeholders. Returns (text, map)."""
    protected: Dict[str, str] = {}
    counter = [0]

    def replacer(m: re.Match) -> str:
        key = f"xp{counter[0]:03d}x"
        protected[key] = m.group(0)
        counter[0] += 1
        return key

    return _PROTECT_RE.sub(replacer, text), protected


def _restore_terms(text: str, protected: Dict[str, str]) -> str:
    """Restore original terms from placeholders."""
    for key, val in protected.items():
        text = text.replace(key, val)
        text = text.replace(key.lower(), val)
    return text


def _repair_with_wordsegment(text: str) -> str:
    """
    Reconstruct word spaces using wordsegment's English model.
    Processes line by line. Protected terms survive segmentation.
    """
    try:
        from wordsegment import segment

        lines = text.split('\n')
        repaired_lines: List[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped or not is_space_deficient(stripped, threshold=0.05):
                repaired_lines.append(line)
                continue

            protected_line, prot_map = _protect_terms(stripped)
            try:
                segmented = ' '.join(segment(protected_line.lower()))
            except Exception:
                segmented = protected_line

            restored = _restore_terms(segmented, prot_map)
            repaired_lines.append(restored)

        return '\n'.join(repaired_lines)

    except Exception as e:
        logger.warning(f"00_text_repair: wordsegment failed -- {e}")
        return text


def _repair_with_camelcase(text: str) -> str:
    """
    Heuristic CamelCase splitting with protected-term preservation.
    """
    text2, prot_map = _protect_terms(text)
    for pat, repl in _CAMEL_PATTERNS:
        text2 = pat.sub(repl, text2)
    return _restore_terms(text2, prot_map)


# ============================================================================
# PUBLIC API
# ============================================================================

def repair_page_text(text: str) -> str:
    """
    Repair a single page's extracted PDF text.

    Decision tree:
        is_space_deficient AND wordsegment available   -> wordsegment + camelCase pass
        is_space_deficient AND wordsegment unavailable -> camelCase only
        has_camelcase_artifacts (normal spacing)       -> camelCase only
        clean text                                     -> unchanged (fast path)

    Returns:
        Repaired text string. Never raises.
    """
    if not text or len(text) < 20:
        return text

    ws_ok     = _ensure_wordsegment()
    space_bad = is_space_deficient(text)
    camel_bad = has_camelcase_artifacts(text)

    if space_bad:
        if ws_ok:
            logger.debug(
                f"00_text_repair: space-deficient "
                f"({text.count(' ')}/{len(text)} spaces) -- wordsegment"
            )
            text = _repair_with_wordsegment(text)
        else:
            logger.debug("00_text_repair: space-deficient -- CamelCase heuristic")
        # Always apply CamelCase pass after space repair (catches residual artifacts)
        text = _repair_with_camelcase(text)

    elif camel_bad:
        logger.debug("00_text_repair: CamelCase artifacts -- splitting")
        text = _repair_with_camelcase(text)

    # Final normalisation
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def repair_page_texts(page_texts: List[str]) -> List[str]:
    """
    Repair a list of per-page texts extracted from a PDF.
    Returns a new list of the same length.
    """
    return [repair_page_text(p) for p in page_texts]


def detect_language(text: str) -> str:
    """
    Detect the dominant language of a text block.

    Strategy (fast, no heavy dependencies required):
        Step 1: Count CJK characters. If ratio > 0.15, classify as zh/ja/ko.
                - Hiragana/Katakana presence → 'ja'
                - Hangul presence → 'ko'
                - Otherwise (CJK Unified Ideographs dominant) → 'zh'
        Step 2: If CJK ratio is 0.05–0.15 (mixed/ambiguous) and langdetect
                is installed, use it as tiebreaker.
        Step 3: Fallback → 'en'

    Args:
        text: A text string (single page or concatenated pages).

    Returns:
        Language code string: 'zh', 'ja', 'ko', 'en', or 'other'.
        Never raises.
    """
    if not text or len(text) < 20:
        return "en"

    total = len(text)

    # ── Unicode range counters ──────────────────────────────────────────
    cjk_count       = 0   # CJK Unified Ideographs (primarily Chinese)
    hiragana_count  = 0   # Japanese Hiragana
    katakana_count  = 0   # Japanese Katakana
    hangul_count    = 0   # Korean Hangul

    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF:     # CJK Unified Ideographs (core)
            cjk_count += 1
        elif 0x3400 <= cp <= 0x4DBF:   # CJK Extension A
            cjk_count += 1
        elif 0x20000 <= cp <= 0x2A6DF: # CJK Extension B
            cjk_count += 1
        elif 0x3040 <= cp <= 0x309F:   # Hiragana
            hiragana_count += 1
        elif 0x30A0 <= cp <= 0x30FF:   # Katakana
            katakana_count += 1
        elif 0xAC00 <= cp <= 0xD7AF:   # Hangul syllables
            hangul_count += 1

    cjk_ratio      = cjk_count      / total
    hiragana_ratio = hiragana_count / total
    katakana_ratio = katakana_count / total
    hangul_ratio   = hangul_count   / total

    # ── Strong CJK signal (> 15%) ────────────────────────────────────────
    if cjk_ratio > 0.15 or (cjk_ratio + hiragana_ratio + katakana_ratio + hangul_ratio) > 0.15:
        if hiragana_ratio > 0.01 or katakana_ratio > 0.02:
            return "ja"
        if hangul_ratio > 0.05:
            return "ko"
        return "zh"

    # ── Weak CJK signal (5–15%): use langdetect if available ─────────────
    if (cjk_ratio + hiragana_ratio + katakana_ratio + hangul_ratio) > 0.05:
        try:
            from langdetect import detect as _ld_detect
            lang = _ld_detect(text[:2000])  # sample first 2000 chars
            if lang.startswith("zh"):
                return "zh"
            if lang == "ja":
                return "ja"
            if lang == "ko":
                return "ko"
        except Exception:
            pass
        return "zh"  # fallback for weak CJK signal without langdetect

    # ── No CJK — try langdetect for non-CJK non-English ─────────────────
    try:
        from langdetect import detect as _ld_detect
        lang = _ld_detect(text[:2000])
        if lang == "en":
            return "en"
        if lang.startswith("zh"):
            return "zh"
        if lang == "ja":
            return "ja"
        if lang == "ko":
            return "ko"
        if lang in ("en", "en-US", "en-GB"):
            return "en"
        return "other"
    except Exception:
        pass

    return "en"


def detect_document_language(page_texts: List[str], sample_pages: int = 5) -> str:
    """
    Detect the dominant language for an entire document.

    Samples up to `sample_pages` pages (evenly distributed) and returns
    the most common language detected.

    Args:
        page_texts:   List of per-page text strings.
        sample_pages: Maximum number of pages to sample.

    Returns:
        Language code: 'zh', 'ja', 'ko', 'en', or 'other'.
    """
    if not page_texts:
        return "en"

    n = len(page_texts)
    if n <= sample_pages:
        indices = list(range(n))
    else:
        step = n // sample_pages
        indices = [i * step for i in range(sample_pages)]

    counts: Dict[str, int] = {}
    for idx in indices:
        lang = detect_language(page_texts[idx])
        counts[lang] = counts.get(lang, 0) + 1

    return max(counts, key=lambda k: counts[k])


def text_repair_stats(original: str, repaired: str) -> Dict:
    """Diagnostic: compare original vs repaired text statistics."""
    def stats(t: str) -> Dict:
        chars = len(t)
        return {
            "chars":       chars,
            "words":       len(t.split()),
            "space_ratio": round(t.count(' ') / max(chars, 1), 3),
        }
    return {
        "original": stats(original),
        "repaired": stats(repaired),
        "changed":  original != repaired,
    }


# ============================================================================
# SMOKE TEST
# ============================================================================

if __name__ == "__main__":
    print("AISA 00_text_repair.py smoke test")
    print("=" * 60)

    ws_ok = _ensure_wordsegment()
    print(f"  wordsegment available: {ws_ok}")
    print()

    cases = [
        (
            "No-space Amazon-style text",
            "WeseektoinvestefficientlyinseveralareasoftechnologyandinfrastructureincludingAI",
            True,
        ),
        (
            "No-space with AI product names",
            "ThecompanyusedChatGPTandOpenAIsolutionsforoperationalexcellence",
            True,
        ),
        (
            "CamelCase PDF artifact",
            "The company uses artificialIntelligence and machineLearning dataAnalysis tools.",
            True,
        ),
        (
            "Normal text -- fast path (unchanged)",
            "We seek to invest efficiently in several areas of technology and AI infrastructure.",
            False,
        ),
        (
            "Protected terms in normal text (unchanged)",
            "We deployed ChatGPT and OpenAI solutions via LangChain with MLOps pipelines.",
            False,
        ),
    ]

    all_ok = True
    for desc, text, expect_changed in cases:
        repaired = repair_page_text(text)
        changed  = (repaired != text)
        ok       = (changed == expect_changed)
        status   = "  [OK]" if ok else "  [FAIL]"
        print(f"{status} {desc}")
        if changed:
            print(f"         IN : {text[:80]}")
            print(f"         OUT: {repaired[:80]}")
        if not ok:
            all_ok = False
            print(f"         EXPECTED changed={expect_changed}, got changed={changed}")
    print()

    # Protected terms must survive all repair paths
    proto_text = "We use ChatGPT, OpenAI APIs, LangChain, MLOps and TensorFlow."
    proto_result = repair_page_text(proto_text)
    for term in ["ChatGPT", "OpenAI", "LangChain", "MLOps", "TensorFlow"]:
        assert term in proto_result, f"Protected term '{term}' broken in: {proto_result}"
    print("  Protected terms intact in normal text [OK]")

    # repair_page_texts list wrapper
    pages = [
        "WeseektoinvestinAItechnologiesandinnovation",   # len > 30, no spaces
        "We already have spaces here and nothing should change.",
    ]
    fixed = repair_page_texts(pages)
    assert len(fixed) == 2
    assert fixed[0] != pages[0], "Space-deficient page should be repaired"
    assert fixed[1] == pages[1], "Normal page must be unchanged"
    print("  repair_page_texts list wrapper [OK]")

    # Stats helper
    s = text_repair_stats("WeseektoinvestinAI", "we seek to invest in AI")
    assert s["changed"] is True
    print("  text_repair_stats [OK]")

    assert all_ok, "Some tests failed -- check output above"

    # detect_language smoke tests
    print()
    print("  detect_language tests:")

    zh_text = "人工智能和机器学习技术在金融行业中得到广泛应用，大模型推动了生成式AI的发展。" * 5
    assert detect_language(zh_text) == "zh", f"Expected zh, got {detect_language(zh_text)}"
    print("    Chinese text → 'zh' [OK]")

    en_text = "We seek to invest efficiently in AI technologies and machine learning solutions."
    assert detect_language(en_text) == "en", f"Expected en, got {detect_language(en_text)}"
    print("    English text → 'en' [OK]")

    ja_text = "人工知能と機械学習の技術は、ビジネスの効率化に貢献しています。" * 5
    lang = detect_language(ja_text)
    assert lang == "ja", f"Expected ja, got {lang}"
    print("    Japanese text → 'ja' [OK]")

    # detect_document_language
    pages_en = ["We use AI and machine learning." for _ in range(10)]
    assert detect_document_language(pages_en) == "en"
    print("    detect_document_language English [OK]")

    pages_zh = [zh_text for _ in range(10)]
    assert detect_document_language(pages_zh) == "zh"
    print("    detect_document_language Chinese [OK]")

    print()
    print("  00_text_repair.py all checks passed.")
