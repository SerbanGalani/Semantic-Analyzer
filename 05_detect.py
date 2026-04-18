"""
===============================================================================
AISA - AI Semantic Analyzer
05_detect.py - AI reference detection engine
===============================================================================

Core detection pipeline for identifying AI references in corporate document
text. Implements a two-stage approach:

    Stage 1 (CPU, parallelizable):
        - Text segmentation into candidate fragments (sentences / windows)
        - Regex pattern matching (compiled at module level)
        - Keyword pre-filtering (discard obvious non-matches early)
        - False positive check via TaxonomyProvider

    Stage 2 (GPU/CPU, single worker recommended):
        - Batch semantic encoding via SentenceTransformer
        - Cosine similarity scoring against AI anchor embeddings
        - Confidence scoring and reference-strength classification

Design constraints (from AISA_CONTEXT.md):
    - Regex compiled ONCE at module level, never per call or per instantiation
    - Batch encoding: collect ALL fragments per document, encode ONCE
    - Pre-filter before semantic: keyword check first
    - Stage 1 / Stage 2 separated so Stage 1 can run in ProcessPool
    - Hard fail if SentenceTransformer is unavailable - NO fallback

Context extraction:
    - Sentence-based: config.context_sentences_before (default 2) sentences
      before the match + config.context_sentences_after (default 2) after.
    - AI term highlighted with >>>term<<< markers in the context field.
    - In Excel export (07_export.py) markers render as BOLD + RED rich text.

CHANGELOG:
    v1.1.2 (2026-04) - Digitalization Stage-1 support + ZH taxonomy map
        - Added taxonomy aliases for Digitalization_Relational_v2, v2_2 and v2_2_ZH
        - Added dedicated Stage 1 trigger regexes for digitalization (EN + ZH)
        - Expanded term-highlighting priority list for digitalization detections
    v1.1.1 (2026-03) - Embedding cache integration
        - stage2_semantic_score accepts optional db (DatabaseManager)
        - Cache hit: skip model.encode(), load from embedding_cache table
        - Cache miss: encode normally, save to cache for next run
        - Zero breaking changes — db=None disables cache (original behavior)
    v1.0.1 (2026-02) - Sentence-based context extraction (2+2), >>><<< markers
    v1.0.0 (2026-02) - AISA initial release

Author: TeRa0
License: MIT
===============================================================================
"""

from __future__ import annotations

import hashlib
import importlib
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from version import (
    AISA_VERSION,
    SEMANTIC_MODEL_NAME,
    SEMANTIC_THRESHOLD,
    SEMANTIC_THRESHOLD_RELAXED,
    SEMANTIC_THRESHOLD_STRICT,
)

_m1 = importlib.import_module("01_models")
AIReference     = _m1.AIReference
AnalyzerConfig  = _m1.AnalyzerConfig
DocumentResult  = _m1.DocumentResult
logger          = _m1.logger

_m3 = importlib.import_module("03_taxonomy_base")
TaxonomyProvider    = _m3.TaxonomyProvider
CompiledPatternCache = _m3.CompiledPatternCache

# ── Taxonomy — loaded dynamically via set_taxonomy() ─────────────────────────
# Default: built-in AI taxonomy (backward compatible).
# Call set_taxonomy(provider) from 10_pipeline.py before the first document.
_TAXONOMY_MODULE_MAP: Dict[str, str] = {
    "AI_Disclosure":                  "04_taxonomy_builtin",
    "Digitalization_Eco":             "04_taxonomy_digitalization",
    "Digitalization_Relational_v2":   "04_taxonomy_digitalization",
    "Digitalization_Relational_v2_2": "04_taxonomy_digitalization_relational_v2_2_0",
    "Digitalization_Relational_v2_2_ZH": "04_taxonomy_digitalization_zh",
    # Register new taxonomies here: "MyTaxonomy": "04_taxonomy_mytaxonomy"
}

TAXONOMY:       Optional[TaxonomyProvider]      = None
PATTERN_CACHE:  Optional[CompiledPatternCache]  = None


def set_taxonomy(provider: "TaxonomyProvider") -> None:
    """
    Set the active taxonomy for all subsequent detections in this process.

    Must be called once before the first detect_stage1() / detect_stage2() call.
    Calling it again replaces the active taxonomy (allows sequential multi-run).

    If the provider exposes get_anchor_phrases() → Dict[str, List[str]], those
    phrases replace the default AI-specific anchor embeddings for Stage 2 scoring.
    Otherwise the module-level _ANCHOR_TEXTS list is used (backward compat).

    Args:
        provider: Any TaxonomyProvider implementation.
    """
    global TAXONOMY, PATTERN_CACHE, _ANCHOR_EMBEDDINGS
    TAXONOMY = provider
    PATTERN_CACHE = CompiledPatternCache()
    PATTERN_CACHE.compile_from_provider(provider)

    # Dynamic anchor embeddings: use taxonomy-specific phrases when available
    if callable(getattr(provider, "get_anchor_phrases", None)):
        anchor_phrases_map: dict = provider.get_anchor_phrases()
        flat_anchors = [p for phrases in anchor_phrases_map.values() for p in phrases]
        if flat_anchors:
            _ANCHOR_EMBEDDINGS = _SEMANTIC_MODEL.encode(
                flat_anchors,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            logger.info(
                f"Anchor embeddings refreshed from taxonomy: "
                f"{len(flat_anchors)} phrases "
                f"({len(anchor_phrases_map)} categories)"
            )
        else:
            logger.warning("get_anchor_phrases() returned empty — keeping current anchors")
    # else: keep existing _ANCHOR_EMBEDDINGS (AI anchors or previously set)

    logger.info(
        f"Taxonomy set: {type(provider).__name__} "
        f"(v{provider.get_version()}, "
        f"{len(provider.get_all_categories())} categories)"
    )


def load_taxonomy_by_name(taxonomy_name: str) -> "TaxonomyProvider":
    """
    Load and return a TaxonomyProvider by registered name.

    Args:
        taxonomy_name: Key from _TAXONOMY_MODULE_MAP (e.g. 'AI_Disclosure').

    Returns:
        The module-level TAXONOMY singleton from the corresponding module.

    Raises:
        ValueError: If taxonomy_name is not registered.
    """
    module_name = _TAXONOMY_MODULE_MAP.get(taxonomy_name)
    if not module_name:
        registered = list(_TAXONOMY_MODULE_MAP.keys())
        raise ValueError(
            f"Unknown taxonomy '{taxonomy_name}'. "
            f"Registered: {registered}. "
            f"Add it to _TAXONOMY_MODULE_MAP in 05_detect.py."
        )
    m = importlib.import_module(module_name)
    return m.TAXONOMY


def load_taxonomy_from_excel(excel_path: str) -> "TaxonomyProvider":
    """
    Load and return a TaxonomyProvider from an Excel file.

    No registration needed — just point to the file.
    See taxonomy_excel_loader.py for the expected sheet format.

    Args:
        excel_path: Path to .xlsx file (absolute or relative to cwd).

    Returns:
        ExcelTaxonomyProvider instance ready for set_taxonomy().

    Example:
        provider = load_taxonomy_from_excel("my_taxonomies/ESG_v1.xlsx")
        set_taxonomy(provider)
    """
    loader = importlib.import_module("taxonomy_excel_loader")
    return loader.ExcelTaxonomyProvider(excel_path)


# Load default taxonomy at import time so existing code that reads
# module-level TAXONOMY / PATTERN_CACHE without calling set_taxonomy() still works.
set_taxonomy(importlib.import_module("04_taxonomy_builtin").TAXONOMY)


# ============================================================================
# SEMANTIC MODEL - loaded once at module level
# ============================================================================

# SEMANTIC MODEL - loaded once at module level
# ============================================================================
# Dual-model strategy (AISA v1.1.0):
#   English documents   → all-MiniLM-L6-v2     (fast, highest EN accuracy)
#   CJK documents       → paraphrase-multilingual-MiniLM-L12-v2  (lazy-loaded)
# ============================================================================

from version import SEMANTIC_MODEL_MULTILINGUAL  # noqa: E402

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _SEMANTIC_MODEL = SentenceTransformer(SEMANTIC_MODEL_NAME)
    SEMANTIC_AVAILABLE = True
    logger.info(f"SentenceTransformer loaded: {SEMANTIC_MODEL_NAME}")
except ImportError:
    raise RuntimeError(
        f"SentenceTransformer is required but not installed. "
        f"Run: pip install sentence-transformers"
    )
except Exception as exc:
    raise RuntimeError(
        f"Failed to load SentenceTransformer model '{SEMANTIC_MODEL_NAME}': {exc}"
    )

# Multilingual model — loaded lazily when first CJK document is encountered
_SEMANTIC_MODEL_MULTI: Optional[object] = None
_MULTI_MODEL_LOADED: bool = False


def _get_semantic_model(language: str = "en"):
    """
    Return the appropriate SentenceTransformer for the given language.

    English (and any non-CJK) → _SEMANTIC_MODEL (all-MiniLM-L6-v2)
    Chinese / Japanese / Korean → _SEMANTIC_MODEL_MULTI (multilingual)

    The multilingual model is loaded lazily on first call for a CJK language.
    """
    global _SEMANTIC_MODEL_MULTI, _MULTI_MODEL_LOADED

    if language not in ("zh", "ja", "ko"):
        return _SEMANTIC_MODEL

    if not _MULTI_MODEL_LOADED:
        _MULTI_MODEL_LOADED = True
        try:
            logger.info(
                f"Loading multilingual model for language='{language}': "
                f"{SEMANTIC_MODEL_MULTILINGUAL}"
            )
            _SEMANTIC_MODEL_MULTI = SentenceTransformer(SEMANTIC_MODEL_MULTILINGUAL)
            logger.info(f"Multilingual model loaded: {SEMANTIC_MODEL_MULTILINGUAL}")
        except Exception as exc:
            logger.warning(
                f"Failed to load multilingual model '{SEMANTIC_MODEL_MULTILINGUAL}': {exc}. "
                f"Falling back to English model for language='{language}'."
            )
            _SEMANTIC_MODEL_MULTI = None

    return _SEMANTIC_MODEL_MULTI if _SEMANTIC_MODEL_MULTI is not None else _SEMANTIC_MODEL


def release_multilingual_model() -> None:
    """
    Unload the multilingual SentenceTransformer from memory.
    Call this after finishing a batch of CJK documents when switching
    back to an EN-only corpus segment, or at end of pipeline run.
    Safe to call multiple times — no-op if model is not loaded.
    """
    global _SEMANTIC_MODEL_MULTI, _MULTI_MODEL_LOADED
    if _SEMANTIC_MODEL_MULTI is not None:
        del _SEMANTIC_MODEL_MULTI
        _SEMANTIC_MODEL_MULTI = None
        _MULTI_MODEL_LOADED = False
        import gc
        gc.collect()
        logger.info("Multilingual model unloaded — memory released")


# ============================================================================
# ANCHOR EMBEDDINGS FOR SEMANTIC MATCHING
# Encoded once at module load from carefully chosen representative phrases.
# ============================================================================

_ANCHOR_TEXTS = [
    "artificial intelligence implementation in business",
    "machine learning model deployed for operations",
    "deep learning neural network training",
    "AI-powered product or service feature",
    "natural language processing text analysis",
    "generative AI large language model GPT",
    "computer vision image recognition system",
    "robotic process automation intelligent automation",
    "AI governance ethics responsible AI policy",
    "AI talent workforce upskilling data scientist",
    "predictive analytics forecasting machine learning",
    "AI investment strategy transformation initiative",
    "AI risk compliance fraud detection cybersecurity",
    "foundation model fine-tuning RLHF prompt engineering",
    "AI infrastructure MLOps platform deployment",
    "autonomous vehicle agentic AI multi-agent system",
]

_ANCHOR_EMBEDDINGS = _SEMANTIC_MODEL.encode(
    _ANCHOR_TEXTS,
    convert_to_tensor=True,
    show_progress_bar=False,
    normalize_embeddings=True,
)
logger.info(f"Anchor embeddings computed: {len(_ANCHOR_TEXTS)} anchors")


# ============================================================================
# MODULE-LEVEL COMPILED REGEX (Stage 1)
# Compiled once; reused across all documents and all calls.
# ============================================================================

# Hard trigger patterns - high-confidence AI signals (AI taxonomy)
_HARD_TRIGGER_RE = re.compile(
    r"\b(?:"
    r"artificial intelligence|machine learning|deep learning|"
    r"neural network|natural language processing|"
    r"generative AI|GenAI|large language model|LLM|foundation model|"
    r"ChatGPT|GPT-[34]|GPT-4o|Copilot|Gemini|Claude AI|"
    r"computer vision|image recognition|"
    r"robotic process automation|RPA|"
    r"MLOps|SageMaker|Vertex AI|Azure ML|Watsonx|"
    r"reinforcement learning|transfer learning|"
    r"transformer model|BERT|XGBoost|random forest|"
    r"prompt engineering|fine[\s-]?tuning|RLHF|RAG|"
    r"autonomous vehicle|self[\s-]?driving|agentic AI|AI agent|"
    r"responsible AI|ethical AI|AI governance|"
    r"AI[\s-](?:powered|driven|enabled|first)|AI\s+(?:system|platform|model|solution)"
    r")\b",
    re.IGNORECASE,
)

# Soft trigger patterns - need semantic validation (AI taxonomy)
_SOFT_TRIGGER_RE = re.compile(
    r"\b(?:"
    r"intelligent|algorithm|automation|embedding|"
    r"model|prediction|analytics|robot(?:ic)?s?|"
    r"cognitive|smart|digital transform|"
    r"data[\s-]driven|data science|"
    r"transformer|multimodal|"
    r"NLP|ML\b|DL\b|"
    r"Claude|Gemini|Llama|Mistral"
    r")\b",
    re.IGNORECASE,
)

# Hard trigger patterns - digitalization taxonomy (EN + ZH)
_DIGITAL_HARD_TRIGGER_RE = re.compile(
    r"(?:"
    r"partner onboarding portal|supplier onboarding portal|vendor registration portal|"
    r"partner portal|supplier portal|dealer portal|distributor portal|"
    r"electronic data interchange|\bEDI\b|API integration|API gateway|iPaaS|middleware|"
    r"control tower|shared data space|data sovereignty|sovereign data exchange|"
    r"GAIA-X|Catena-X|digital product passport|battery passport|product carbon footprint|"
    r"blockchain traceability|chain of custody|provenance tracking|"
    r"industrial IoT|\bIIoT\b|digital twin|connected asset|remote monitoring|"
    r"federated identity|identity federation|zero trust|secure data exchange|"
    r"API marketplace|ecosystem orchestration|complementor|multi-sided platform|"
    r"digital servitization|outcome-based contract|cross-company data sharing|interoperability framework|"
    r"数字化|数字化转型|伙伴门户|合作伙伴门户|供应商门户|经销商门户|分销商门户|"
    r"电子数据交换|数据共享|数据空间|共享数据空间|数据主权|主权数据交换|"
    r"互操作|互联互通|控制塔|数字产品护照|电池护照|产品碳足迹|"
    r"区块链追溯|全程追溯|来源追踪|工业物联网|数字孪生|远程监控|"
    r"联邦身份|身份联邦|零信任|安全数据交换|API市场|生态系统编排|补充者|"
    r"平台治理|数字服务化|跨企业数据共享|互操作框架"
    r")",
    re.IGNORECASE,
)

# Soft trigger patterns - digitalization taxonomy (EN + ZH)
_DIGITAL_SOFT_TRIGGER_RE = re.compile(
    r"(?:"
    r"platform|portal|ecosystem|partner|supplier|dealer|distributor|channel|"
    r"workflow|automation|integration|interoperability|shared|exchange|connect|connected|"
    r"cloud platform|data platform|data exchange|traceability|compliance platform|"
    r"visibility|transparency|governance|knowledge sharing|co-innovation|servitization|"
    r"marketplace|dashboard|network|interface|"
    r"平台|门户|生态|合作伙伴|供应商|经销商|分销商|渠道|"
    r"工作流|自动化|集成|整合|互操作|共享|交换|连接|协同|"
    r"云平台|数据平台|数据交换|追溯|合规平台|可视化|透明度|治理|知识共享|"
    r"共同创新|服务化|市场平台|仪表板|网络|接口"
    r")",
    re.IGNORECASE,
)

# Sentence boundary splitter - compiled once
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])"
)

# Boilerplate patterns to strip from context (page numbers, URLs, etc.)
_BOILERPLATE_RE = re.compile(
    r"(?:^|\n)\s*\d+\s*(?:\n|$)"          # standalone page numbers
    r"|(?:^|\n)\s*(?:Page|Pagina)\s+\d+"  # "Page N"
    r"|(?:^|\n)\s*©\s*\d{4}"              # copyright line
    r"|(?:^|\n)\s*All\s+rights\s+reserved"
    r"|(?:www\.|https?://)\S+",            # URLs
    re.IGNORECASE,
)

# Priority list for highlighting semantic detections (AI + digitalization EN/ZH)
_AI_TERM_PRIORITY_RE = [
    re.compile(p, re.IGNORECASE) for p in [
        # AI-first high-priority terms
        r"\bgenerative AI\b", r"\bGenAI\b", r"\blarge language model(?:s)?\b",
        r"\bLLM(?:s)?\b", r"\bartificial intelligence\b", r"\bmachine learning\b",
        r"\bdeep learning\b", r"\bneural network(?:s)?\b", r"\bGPT\b",
        r"\bChatGPT\b", r"\bCopilot\b", r"\bpredictive analytics\b",
        r"\bnatural language processing\b", r"\bcomputer vision\b",
        r"\bautonomous\b", r"\brobot(?:ic)?(?:s)?\b", r"\bRPA\b",
        r"\bRAG\b", r"\bembedding(?:s)?\b", r"\bvector\s+database\b",
        r"\b(?:AI|ML)\b(?![\w-])", r"\bchatbot\b", r"\bautomation\b",
        # Digitalization terms (EN)
        r"\bpartner onboarding portal\b", r"\bsupplier portal\b", r"\bpartner portal\b",
        r"\bcontrol tower\b", r"\bshared data space\b", r"\bdata sovereignty\b",
        r"\bsovereign data exchange\b", r"\bGAIA-X\b", r"\bCatena-X\b",
        r"\bdigital product passport\b", r"\bbattery passport\b",
        r"\bblockchain traceability\b", r"\bindustrial IoT\b", r"\bdigital twin\b",
        r"\bfederated identity\b", r"\bzero trust\b", r"\bAPI marketplace\b",
        r"\becosystem orchestration\b", r"\bdigital servitization\b",
        # Digitalization terms (ZH)
        r"数字化转型", r"数字化", r"合作伙伴门户", r"供应商门户", r"控制塔",
        r"数据共享", r"数据空间", r"数据主权", r"主权数据交换", r"互操作",
        r"数字产品护照", r"电池护照", r"区块链追溯", r"工业物联网", r"数字孪生",
        r"联邦身份", r"零信任", r"生态系统编排", r"平台治理", r"知识共享",
    ]
]

# Deduplication hash fields
_REF_HASH_FIELDS = ("company", "year", "doc_type", "page", "text")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CandidateFragment:
    """
    A text fragment that passed Stage 1 filtering and is queued for
    Stage 2 semantic encoding.
    """
    text:             str
    context:          str        # sentence-based context with >>>markers<<<
    ai_term:          str        # the specific AI term that triggered detection
    page:             int
    char_offset:      int
    has_hard_trigger: bool
    pattern_matches:  List[Tuple[str, str]] = field(default_factory=list)
    # List of (pattern_string, category_code) from regex matches


@dataclass
class DetectionCandidate:
    """
    A fragment that has been semantically scored and passed all filters.
    Ready to be converted to AIReference.

    dimensions: maps dimension_name → (category_code, confidence)
    Replaces the old category_a/b fields — supports any number of dimensions.
    Backward-compat properties category_a/b/confidence_a/b read first/second dim.
    """
    fragment:           CandidateFragment
    semantic_score:     float
    dimensions:         Dict[str, Tuple[str, float]]   # dim_name → (code, conf)
    detection_method:   str   # "pattern_hard" | "pattern_soft" | "semantic"
    reference_strength: str
    confidence_score:   float
    confidence_reasons: str
    embedding:          Optional[object] = None  # tensor from Stage 2, used for cosine dedupe

    # ── Backward-compat properties ────────────────────────────────────────────
    @property
    def category_a(self) -> str:
        v = self._nth(0)
        return v[0] if v else ""

    @property
    def confidence_a(self) -> float:
        v = self._nth(0)
        return v[1] if v else 0.0

    @property
    def category_b(self) -> str:
        v = self._nth(1)
        return v[0] if v else ""

    @property
    def confidence_b(self) -> float:
        v = self._nth(1)
        return v[1] if v else 0.0

    def _nth(self, n: int) -> Optional[Tuple[str, float]]:
        vals = list(self.dimensions.values())
        return vals[n] if len(vals) > n else None


# ============================================================================
# CONTEXT EXTRACTION - SENTENCE-BASED WITH >>>MARKERS<<<
# ============================================================================

_HIGHLIGHT_START = ">>>"
_HIGHLIGHT_END   = "<<<"


def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using the sentence boundary regex.
    Filters out empty/whitespace-only fragments.
    """
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in parts if s.strip()]


def _clean_boilerplate(text: str) -> str:
    """Remove boilerplate (page numbers, URLs, copyright lines) from context."""
    return _BOILERPLATE_RE.sub(" ", text).strip()


def extract_context_sentences(
    full_text: str,
    match_start: int,
    match_end: int,
    matched_term: str,
    sentences_before: int = 2,
    sentences_after: int = 2,
    max_context_length: int = 2000,
) -> str:
    """
    Extract N sentences before and after the sentence containing the match.
    Highlights the matched AI term with >>>term<<< markers.

    Strategy:
        1. Take a large window (±2000 chars) around the match for sentence detection.
        2. Split into sentences.
        3. Find the sentence containing the match.
        4. Keep sentences_before sentences before + target + sentences_after after.
        5. Re-insert >>>term<<< marker around the matched term in the target sentence.
        6. Strip boilerplate and truncate to max_context_length.

    Args:
        full_text:          Complete document text.
        match_start:        Start char index of the match in full_text.
        match_end:          End char index of the match in full_text.
        matched_term:       The exact text that was matched (for highlighting).
        sentences_before:   Number of sentences to include before target (default 2).
        sentences_after:    Number of sentences to include after target (default 2).
        max_context_length: Maximum output length in characters.

    Returns:
        Context string with >>>matched_term<<< highlighting applied.
    """
    # Step 1: extract a generous window for sentence detection
    window_start = max(0, match_start - 2000)
    window_end   = min(len(full_text), match_end + 2000)
    window_text  = full_text[window_start:window_end]

    # relative position of match within the window
    rel_start = match_start - window_start

    # Step 2: split window into sentences
    sentences = _split_into_sentences(window_text)
    if not sentences:
        # Fallback: return char-window with marker
        ctx = window_text[:max_context_length].strip()
        return _insert_marker(ctx, matched_term)

    # Step 3: find which sentence contains rel_start
    target_idx = 0
    cursor = 0
    for idx, sent in enumerate(sentences):
        # find this sentence in window_text starting from cursor
        sent_pos = window_text.find(sent, cursor)
        if sent_pos == -1:
            cursor += len(sent)
            continue
        sent_end = sent_pos + len(sent)
        if sent_pos <= rel_start < sent_end:
            target_idx = idx
            break
        cursor = sent_pos + 1

    # Step 4: slice sentences around target
    start_idx = max(0, target_idx - sentences_before)
    end_idx   = min(len(sentences), target_idx + sentences_after + 1)
    selected  = sentences[start_idx:end_idx]

    context = " ".join(selected)

    # Step 5: insert >>>marker<<<
    context = _insert_marker(context, matched_term)

    # Step 6: clean boilerplate and truncate
    context = _clean_boilerplate(context)
    context = re.sub(r"\s{2,}", " ", context).strip()

    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."

    return context


def _insert_marker(context: str, term: str) -> str:
    """
    Insert >>>term<<< around the first occurrence of term in context.
    Case-insensitive match; preserves original capitalisation.
    If term is already marked (contains >>>), returns context unchanged.
    """
    if not term or _HIGHLIGHT_START in context:
        return context

    pat = re.compile(re.escape(term), re.IGNORECASE)
    m = pat.search(context)
    if m:
        return (
            context[:m.start()]
            + _HIGHLIGHT_START + m.group() + _HIGHLIGHT_END
            + context[m.end():]
        )
    return context


def _find_and_mark_ai_term(paragraph: str) -> Tuple[str, str]:
    """
    For semantic detections: find the most prominent AI term in a paragraph,
    mark it with >>><<<, and return (ai_term, marked_paragraph).
    Falls back to first 50 chars if no term found.
    """
    for pat in _AI_TERM_PRIORITY_RE:
        m = pat.search(paragraph)
        if m:
            ai_term = m.group()
            marked = (
                paragraph[:m.start()]
                + _HIGHLIGHT_START + ai_term + _HIGHLIGHT_END
                + paragraph[m.end():]
            )
            return ai_term, marked
    return paragraph[:50], paragraph


# ============================================================================
# STAGE 1: TEXT SEGMENTATION + PRE-FILTER
# ============================================================================

def extract_text_windows(
    text: str,
    window_chars: int = 500,
    overlap_chars: int = 100,
) -> List[Tuple[str, int]]:
    """
    Split document text into overlapping windows for candidate extraction.

    Tries sentence-based splitting first; falls back to character windows.
    Returns list of (window_text, char_offset).
    """
    if not text or len(text) < 20:
        return []

    sentences = _split_into_sentences(text)
    if len(sentences) <= 1:
        # Fallback: character windows
        windows = []
        for start in range(0, len(text), window_chars - overlap_chars):
            chunk = text[start : start + window_chars]
            windows.append((chunk, start))
        return windows

    # Group sentences into windows of approximately window_chars length
    windows: List[Tuple[str, int]] = []
    current_parts: List[str] = []
    current_len = 0
    current_offset = 0
    char_pos = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > window_chars and current_parts:
            window_text = " ".join(current_parts)
            windows.append((window_text, current_offset))
            # Overlap: keep last sentence for next window
            overlap = current_parts[-1:]
            current_parts = overlap
            current_len = sum(len(p) for p in overlap)
            current_offset = char_pos - current_len
        current_parts.append(sent)
        current_len += sent_len
        char_pos += sent_len + 1  # +1 for separator

    if current_parts:
        windows.append((" ".join(current_parts), current_offset))

    return windows


def _is_digitalization_taxonomy(taxonomy: TaxonomyProvider) -> bool:
    """Best-effort detection for D/T/G digitalization taxonomies."""
    try:
        dims = taxonomy.get_dimensions()
        category_codes: List[str] = []
        for cats in dims.values():
            category_codes.extend(cats.keys())
        return any(code.startswith(("D", "T", "G")) for code in category_codes)
    except Exception:
        return False


def _get_stage1_trigger_regexes(taxonomy: TaxonomyProvider):
    """Return taxonomy-appropriate hard/soft Stage 1 triggers."""
    if _is_digitalization_taxonomy(taxonomy):
        return _DIGITAL_HARD_TRIGGER_RE, _DIGITAL_SOFT_TRIGGER_RE
    return _HARD_TRIGGER_RE, _SOFT_TRIGGER_RE




def _make_gate_window(full_text: str, match_start: int, match_end: int, radius: int) -> str:
    """Return a symmetric text window around the matched trigger for gate checks."""
    start = max(0, match_start - radius)
    end = min(len(full_text), match_end + radius)
    return full_text[start:end]

def stage1_extract_candidates(
    text: str,
    page_texts: List[str],
    config: AnalyzerConfig,
    taxonomy: TaxonomyProvider = TAXONOMY,
) -> List[CandidateFragment]:
    """
    Stage 1: Extract and pre-filter candidate reference fragments.

    Stable variant:
        - keeps the original working trigger flow
        - adds real gate windows (±500 / ±400)
        - supports the G soft-gate memory (prior D/T hit in document)
        - allows a controlled semantic fallback for digitalization taxonomies
    """
    candidates: List[CandidateFragment] = []
    seen_hashes: set = set()
    doc_has_dt_hit = False

    detection_patterns = PATTERN_CACHE.get_detection_patterns()
    hard_trigger_re, soft_trigger_re = _get_stage1_trigger_regexes(taxonomy)

    sentences_before = getattr(config, "context_sentences_before", 2)
    sentences_after  = getattr(config, "context_sentences_after", 2)
    max_ctx_len      = getattr(config, "max_context_length", 2000)
    is_digital_tax   = _is_digitalization_taxonomy(taxonomy)

    page_start_offsets: List[int] = []
    cumulative = 0
    for pt in page_texts:
        page_start_offsets.append(cumulative)
        cumulative += len(pt) + 1

    for page_idx, page_text in enumerate(page_texts):
        if not page_text or len(page_text) < config.min_text_length:
            continue

        page_num = page_idx + 1
        page_global_offset = page_start_offsets[page_idx]

        windows = extract_text_windows(page_text, window_chars=500, overlap_chars=100)

        for window_text, char_offset_in_page in windows:
            if len(window_text.strip()) < 30:
                continue

            hard_match = hard_trigger_re.search(window_text)
            soft_match = None
            if not hard_match:
                soft_match = soft_trigger_re.search(window_text)
                if not soft_match:
                    continue

            trigger_match = hard_match or soft_match
            matched_term = trigger_match.group(0)

            global_window_start = page_global_offset + char_offset_in_page
            trigger_global_start = global_window_start + trigger_match.start()
            trigger_global_end = global_window_start + trigger_match.end()

            fp_result = taxonomy.check_false_positive(matched_term, window_text)
            if fp_result.is_fp:
                logger.debug(
                    f"FP filtered (page {page_num}): '{matched_term}' [{fp_result.category}]"
                )
                continue

            gate_window_t = _make_gate_window(text, trigger_global_start, trigger_global_end, 500)
            gate_window_g = _make_gate_window(text, trigger_global_start, trigger_global_end, 400)

            has_actor = bool(callable(getattr(taxonomy, "has_external_actor", None)) and taxonomy.has_external_actor(gate_window_t))
            has_verb = bool(callable(getattr(taxonomy, "has_relational_verb", None)) and taxonomy.has_relational_verb(gate_window_t))
            has_marker = bool(callable(getattr(taxonomy, "has_digital_marker", None)) and taxonomy.has_digital_marker(gate_window_g))
            soft_relational_pass = is_digital_tax and sum([1 if has_actor else 0, 1 if has_verb else 0, 1 if has_marker else 0]) >= 2

            saw_dt_pass = False
            has_passing_hit = False
            if callable(getattr(taxonomy, "get_gate_type", None)):
                for compiled_pat, code in detection_patterns:
                    if not compiled_pat.search(window_text):
                        continue
                    gate = taxonomy.get_gate_type(code)
                    passed = False
                    if gate == "none":
                        passed = True
                    elif gate == "cooccur":
                        passed = has_actor
                    elif gate == "cooccur_verb":
                        passed = has_actor and has_verb
                    elif gate == "soft":
                        passed = has_marker or doc_has_dt_hit
                    if passed:
                        has_passing_hit = True
                        code_root = (code or "").split("_")[0]
                        if code_root.startswith(("D", "T")):
                            saw_dt_pass = True
                if not has_passing_hit and not soft_relational_pass:
                    logger.debug(
                        f"Gate filtered (page {page_num}): '{matched_term}' — no ungated pattern passed"
                    )
                    continue
            if saw_dt_pass:
                doc_has_dt_hit = True

            context = extract_context_sentences(
                full_text=text,
                match_start=trigger_global_start,
                match_end=trigger_global_end,
                matched_term=matched_term,
                sentences_before=sentences_before,
                sentences_after=sentences_after,
                max_context_length=max_ctx_len,
            )

            pattern_matches: List[Tuple[str, str]] = []
            _has_gate_fn = callable(getattr(taxonomy, "get_gate_type", None))
            for compiled_pat, code in detection_patterns:
                if not compiled_pat.search(window_text):
                    continue
                if _has_gate_fn:
                    gate = taxonomy.get_gate_type(code)
                    if gate == "cooccur" and not has_actor:
                        continue
                    if gate == "cooccur_verb" and not (has_actor and has_verb):
                        continue
                    if gate == "soft" and not (has_marker or doc_has_dt_hit):
                        continue
                pattern_matches.append((compiled_pat.pattern, code))

            frag_hash = hashlib.md5((window_text[:200]).encode("utf-8", errors="ignore")).hexdigest()
            if frag_hash in seen_hashes:
                continue
            seen_hashes.add(frag_hash)

            candidates.append(CandidateFragment(
                text=window_text,
                context=context,
                ai_term=matched_term,
                page=page_num,
                char_offset=global_window_start,
                has_hard_trigger=bool(hard_match),
                pattern_matches=pattern_matches,
            ))

    logger.debug(f"Stage 1: {len(candidates)} candidates extracted")
    return candidates


# ============================================================================
# STAGE 2: BATCH SEMANTIC SCORING
# ============================================================================

def _compute_max_semantic_score(embedding) -> float:
    """
    Compute cosine similarity against all anchor embeddings.
    Returns the maximum similarity score.
    """
    import torch
    sims = st_util.cos_sim(embedding, _ANCHOR_EMBEDDINGS)
    return float(sims.max().item())


def _classify_reference_strength(
    semantic_score: float,
    has_hard_trigger: bool,
    pattern_count: int,
) -> Tuple[str, float, str]:
    """
    Classify reference strength based on available signals.

    Returns:
        (reference_strength, confidence_score, confidence_reasons)
    """
    reasons: List[str] = []
    score = 0.0

    if has_hard_trigger:
        score += 0.5
        reasons.append("hard_trigger")

    if pattern_count >= 2:
        score += 0.25
        reasons.append(f"pattern_matches={pattern_count}")
    elif pattern_count == 1:
        score += 0.15
        reasons.append("pattern_match=1")

    if semantic_score >= SEMANTIC_THRESHOLD_STRICT:
        score += 0.25
        reasons.append(f"semantic_strict={semantic_score:.3f}")
    elif semantic_score >= SEMANTIC_THRESHOLD:
        score += 0.15
        reasons.append(f"semantic={semantic_score:.3f}")

    score = min(score, 1.0)

    if score >= 0.7:
        strength = "strong"
    elif score >= 0.45:
        strength = "medium"
    elif score > 0.0:
        strength = "mention_only"
    else:
        strength = "unknown"

    return strength, round(score, 3), "|".join(reasons)


def _get_detection_method(
    has_hard_trigger: bool,
    pattern_count: int,
    semantic_score: float,
    threshold: float,
) -> str:
    """Determine detection method label for audit trail."""
    if has_hard_trigger and pattern_count > 0:
        return "pattern_hard"
    if has_hard_trigger:
        return "pattern_hard_kw"
    if pattern_count > 0:
        return "pattern_soft"
    if semantic_score >= threshold:
        return "semantic"
    return "unknown"


def stage2_semantic_score(
    candidates: List[CandidateFragment],
    config: AnalyzerConfig,
    taxonomy: TaxonomyProvider = TAXONOMY,
    language: str = "en",
    db=None,  # Optional[DatabaseManager] — passed for embedding cache
) -> List[DetectionCandidate]:
    """
    Stage 2: Batch-encode all candidate fragments and score semantically.

    Key design:
        - ALL fragments encoded in a single model.encode() call (batch)
        - Hard-trigger fragments use relaxed threshold
        - Soft-trigger-only fragments use strict threshold
        - Classification (A/B taxonomy) runs on scored candidates only
        - language: selects EN or multilingual model (AISA v1.1.0)
        - db: if provided, uses embedding_cache table to skip re-encoding
          fragments already seen in previous runs (AISA v1.1.1)

    Args:
        candidates: Output from stage1_extract_candidates().
        config:     AnalyzerConfig for threshold settings.
        taxonomy:   TaxonomyProvider for classification.
        language:   Document language code ('en', 'zh', 'ja', 'ko').
        db:         DatabaseManager instance for embedding cache (optional).
                    If None, cache is disabled — original behavior preserved.

    Returns:
        List of DetectionCandidate objects (passed all filters).
    """
    if not candidates:
        return []

    model       = _get_semantic_model(language)
    model_name  = model.get_sentence_embedding_dimension and getattr(
        model, "_model_card_vars", {}
    ).get("model_name", SEMANTIC_MODEL_NAME if language == "en" else SEMANTIC_MODEL_MULTILINGUAL)

    # ── Embedding cache lookup (v1.1.1) ──────────────────────────────────────
    # Split candidates into cache hits and misses.
    # Cache hits: load numpy array from DB, convert to tensor.
    # Cache misses: collect texts for batch encoding.
    import torch

    embeddings_map: Dict[int, object] = {}  # index → tensor
    texts_to_encode: List[str]  = []
    indices_to_encode: List[int] = []

    cache_hits = 0
    if db is not None:
        for i, c in enumerate(candidates):
            cached = db.get_embedding(c.text, model_name)
            if cached is not None:
                embeddings_map[i] = torch.tensor(cached)
                cache_hits += 1
            else:
                texts_to_encode.append(c.text)
                indices_to_encode.append(i)
    else:
        texts_to_encode   = [c.text for c in candidates]
        indices_to_encode = list(range(len(candidates)))

    # ── Batch encode cache misses ─────────────────────────────────────────────
    if texts_to_encode:
        t0 = time.perf_counter()
        new_embeddings = model.encode(
            texts_to_encode,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=64,
        )
        encode_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            f"Stage 2: encoded {len(texts_to_encode)} fragments "
            f"({cache_hits} cache hits) in {encode_ms:.0f}ms"
        )

        # Store newly computed embeddings in cache and map
        for local_idx, (global_idx, text) in enumerate(
            zip(indices_to_encode, texts_to_encode)
        ):
            emb_tensor = new_embeddings[local_idx]
            embeddings_map[global_idx] = emb_tensor
            # Save to cache if db available
            if db is not None:
                emb_np = emb_tensor.cpu().numpy().astype("float32")
                db.save_embedding(text, model_name, emb_np)
    else:
        logger.debug(
            f"Stage 2: all {cache_hits} fragments served from embedding cache"
        )

    results: List[DetectionCandidate] = []

    for i, candidate in enumerate(candidates):
        embedding = embeddings_map[i]
        semantic_score = _compute_max_semantic_score(embedding)

        # Hard floor: reject regardless of trigger strength.
        # Score < semantic_floor indicates no meaningful similarity to AI corpora
        # — almost always a false positive (mirrored PDF text, boilerplate, financial tables).
        semantic_floor = getattr(config, "semantic_floor", 0.20)
        if semantic_score < semantic_floor:
            logger.debug(
                f"Stage 2 reject (floor): score={semantic_score:.3f} < {semantic_floor} "
                f"| '{candidate.text[:60]}'"
            )
            continue

        # Choose threshold based on trigger strength
        threshold = (
            config.semantic_threshold_relaxed
            if candidate.has_hard_trigger
            else config.semantic_threshold_strict
        )

        # Accept if: semantic passes OR has hard trigger with pattern evidence
        pattern_count   = len(candidate.pattern_matches)
        passes_semantic = semantic_score >= threshold
        passes_pattern  = candidate.has_hard_trigger and pattern_count >= 1

        if not (passes_semantic or passes_pattern):
            continue

        # Taxonomy classification
        clf = taxonomy.classify(candidate.text, candidate.context)

        # Reference strength
        strength, conf_score, conf_reasons = _classify_reference_strength(
            semantic_score=semantic_score,
            has_hard_trigger=candidate.has_hard_trigger,
            pattern_count=pattern_count,
        )

        detection_method = _get_detection_method(
            has_hard_trigger=candidate.has_hard_trigger,
            pattern_count=pattern_count,
            semantic_score=semantic_score,
            threshold=threshold,
        )

        results.append(DetectionCandidate(
            fragment=candidate,
            semantic_score=round(semantic_score, 4),
            dimensions=clf.dimensions,
            detection_method=detection_method,
            reference_strength=strength,
            confidence_score=conf_score,
            confidence_reasons=conf_reasons,
            embedding=embedding,  # stored for cosine dedupe in pipeline
        ))

    logger.debug(
        f"Stage 2: {len(results)}/{len(candidates)} fragments accepted"
    )
    return results


# ============================================================================
# REFERENCE BUILDER
# Converts DetectionCandidate → AIReference (the storable model object)
# ============================================================================

def build_ai_reference(
    candidate:  DetectionCandidate,
    company:    str,
    year:       int,
    position:   int,
    industry:   str,
    sector:     str,
    country:    str,
    doc_type:   str,
    source:     str,
    page_count: int = 0,
    taxonomy:   TaxonomyProvider = TAXONOMY,
) -> AIReference:
    """
    Build an AIReference from a DetectionCandidate + document metadata.

    The context field already contains >>>term<<< markers from Stage 1.
    These markers are rendered as BOLD+RED rich text by 07_export.py.

    Args:
        candidate:  Output from stage2_semantic_score().
        company:    Company name.
        year:       Report year.
        position:   Fortune 500 rank.
        industry:   Industry classification.
        sector:     Sector classification.
        country:    Country of registration.
        doc_type:   Document type (Annual Report / Sustainability / ...).
        source:     Source PDF path.
        page_count: Total pages in the document.
        taxonomy:   TaxonomyProvider (for code validation).

    Returns:
        AIReference dataclass instance.
    """
    import json as _json

    # Build combined category string from all dimensions
    dim_codes = [code for code, _ in candidate.dimensions.values() if code]
    combined  = "|".join(dim_codes) if dim_codes else "UNCLASSIFIED"

    # Backward-compat: first two dimensions → category_a/b
    cat_a = candidate.category_a or ""
    cat_b = candidate.category_b or ""

    # Full multi-dim payload as JSON (all dimensions, not just first two)
    dimensions_json = _json.dumps(
        {dim: [code, round(conf, 4)] for dim, (code, conf) in candidate.dimensions.items()},
        ensure_ascii=False,
    )

    # Use the specific AI term detected (not the full window text)
    ai_term = candidate.fragment.ai_term or candidate.fragment.text[:100]

    # Truncate text/context to DB-safe lengths
    text    = ai_term[:1000].strip()
    context = candidate.fragment.context[:2000].strip()

    return AIReference(
        company=company,
        year=year,
        position=position,
        industry=industry,
        sector=sector,
        country=country,
        doc_type=doc_type,
        text=text,
        context=context,
        page=candidate.fragment.page,
        page_count=page_count,
        category=combined,
        dimensions_json=dimensions_json,
        category_a=cat_a,
        confidence_a=candidate.confidence_a,
        category_b=cat_b,
        confidence_b=candidate.confidence_b,
        detection_method=candidate.detection_method,
        semantic_score=candidate.semantic_score,
        reference_strength=candidate.reference_strength,
        confidence_score=candidate.confidence_score,
        confidence_reasons=candidate.confidence_reasons,
        source=source,
        embedding=candidate.embedding,  # passed for cosine dedupe; not persisted to DB
        # Sentiment filled in by 06_analysis.py
        sentiment="pending",
        sentiment_score=0.0,
        sentiment_confidence="standard",
    )


# ============================================================================
# DOCUMENT-LEVEL ORCHESTRATOR
# Used by 10_pipeline.py (wraps Stage 1 + Stage 2 for a single document)
# ============================================================================

def detect_references_in_document(
    page_texts:     List[str],
    company:        str,
    year:           int,
    position:       int,
    industry:       str,
    sector:         str,
    country:        str,
    doc_type:       str,
    source:         str,
    page_count:     int,
    config:         AnalyzerConfig,
    taxonomy:       TaxonomyProvider = TAXONOMY,
    language:       str = "en",
    db=None,        # Optional[DatabaseManager] — for embedding cache (v1.1.1)
) -> DocumentResult:
    """
    Full detection pipeline for a single document.

    Runs Stage 1 (candidate extraction with sentence-based context) and
    Stage 2 (semantic scoring) sequentially.

    Args:
        page_texts:  List of page text strings (index = page - 1).
        company:     Company name.
        year:        Report year.
        position:    Fortune 500 rank.
        industry:    Industry.
        sector:      Sector.
        country:     Country.
        doc_type:    Document type string.
        source:      Source PDF path.
        page_count:  Total pages.
        config:      AnalyzerConfig (uses context_sentences_before/after).
        taxonomy:    TaxonomyProvider.
        language:    Document language code ('en', 'zh', 'ja', 'ko'). Selects
                     the appropriate semantic model (AISA v1.1.0).
        db:          DatabaseManager instance for embedding cache (v1.1.1).
                     If None, cache is disabled — original behavior preserved.

    Returns:
        DocumentResult with all detected AIReference objects.
        Context fields contain >>>term<<< markers ready for Excel rich text.
    """
    t_start = time.perf_counter()

    full_text   = "\n".join(page_texts)
    text_length = len(full_text)

    result = DocumentResult(
        company=company,
        year=year,
        position=position,
        industry=industry,
        sector=sector,
        country=country,
        doc_type=doc_type,
        source=source,
        total_pages=page_count,
        text_length=text_length,
    )

    if not full_text.strip():
        result.text_status = "empty"
        logger.warning(f"Empty text for {company} {year} ({source})")
        return result

    # Stage 1
    candidates = stage1_extract_candidates(
        text=full_text,
        page_texts=page_texts,
        config=config,
        taxonomy=taxonomy,
    )

    if not candidates:
        result.processing_time = time.perf_counter() - t_start
        return result

    # Stage 2
    scored = stage2_semantic_score(
        candidates=candidates,
        config=config,
        taxonomy=taxonomy,
        language=language,
        db=db,
    )

    # Build AIReference objects
    for det in scored:
        ref = build_ai_reference(
            candidate=det,
            company=company,
            year=year,
            position=position,
            industry=industry,
            sector=sector,
            country=country,
            doc_type=doc_type,
            source=source,
            page_count=page_count,
            taxonomy=taxonomy,
        )
        ref.language = language  # store document language on each reference (v1.1.0)
        result.add_reference(ref)

    result.processing_time = time.perf_counter() - t_start
    logger.info(
        f"{company} {year}: {result.total_refs} refs "
        f"({result.pattern_refs} pattern / {result.semantic_refs} semantic) "
        f"in {result.processing_time:.1f}s"
    )
    return result


# ============================================================================
# REF HASH UTILITY
# Used by 02_db.py for deduplication across runs
# ============================================================================

def compute_ref_hash(ref: AIReference) -> str:
    """
    Compute a stable MD5 hash for an AIReference for cross-run deduplication.

    Uses company + year + doc_type + page + first 200 chars of text.
    """
    raw = "|".join([
        ref.company,
        str(ref.year),
        ref.doc_type,
        str(ref.page),
        ref.text[:200],
    ])
    return hashlib.md5(raw.encode("utf-8", errors="ignore")).hexdigest()


# ============================================================================
# SMOKE TEST
# ============================================================================

if __name__ == "__main__":
    from version import get_version_string
    print(get_version_string())
    print()

    cfg = AnalyzerConfig()
    print(f"  Config: sentences_before={cfg.context_sentences_before}, "
          f"sentences_after={cfg.context_sentences_after}")
    print()

    # --- Context extraction unit test ---
    print("--- Context extraction (sentence-based) ---")
    sample_text = (
        "The company reported strong financial results in Q3. "
        "Revenue grew by 15% year over year. "
        "We deployed ChatGPT across our customer service operations in 2024. "
        "This AI initiative improved response times by 40%. "
        "Further investments are planned for next year. "
        "Our CFO commented on the strong outlook."
    )

    # Simulate finding 'ChatGPT' in the text
    m = re.search(r"ChatGPT", sample_text)
    assert m, "ChatGPT should be in sample text"
    ctx = extract_context_sentences(
        full_text=sample_text,
        match_start=m.start(),
        match_end=m.end(),
        matched_term="ChatGPT",
        sentences_before=2,
        sentences_after=2,
    )
    print(f"  Context: {ctx}")
    assert ">>>ChatGPT<<<" in ctx, f"Expected >>>ChatGPT<<< in context, got: {ctx}"
    print("  >>>marker<<< OK")
    print()

    # --- Stage 1 test ---
    sample_pages = [
        # Page 1 - clear AI content
        (
            "We deployed ChatGPT across our customer service operations in 2024, "
            "leveraging generative AI to improve response times by 40%. "
            "Our MLOps platform on Azure ML processes over 50 million requests daily. "
            "The fraud detection model uses XGBoost with real-time inference."
        ),
        # Page 2 - FP content (should be filtered)
        (
            "We are deeply embedding ESG values across all our operations. "
            "Our power transformers at the Budapest substation were upgraded. "
            "Multimodal transport logistics hub opened in Rotterdam. "
            "Jean-Claude Dupont was appointed Chairman of the Board."
        ),
        # Page 3 - mixed
        (
            "The company invested $50 million in AI infrastructure, including "
            "GPU clusters for training large language models. "
            "Responsible AI governance framework adopted in Q3. "
            "Traditional welding robots at our assembly plant were upgraded."
        ),
    ]

    print("--- Stage 1: Candidate extraction ---")
    full_text = "\n".join(sample_pages)
    candidates = stage1_extract_candidates(
        text=full_text,
        page_texts=sample_pages,
        config=cfg,
        taxonomy=TAXONOMY,
    )
    print(f"  Candidates extracted: {len(candidates)}")
    for c in candidates:
        trigger_label = "HARD" if c.has_hard_trigger else "soft"
        has_marker    = ">>>" in c.context
        print(
            f"  [{trigger_label}] page={c.page} term='{c.ai_term}' "
            f"marker={'✓' if has_marker else '✗'} | "
            f"ctx={c.context[:80]}..."
        )

    assert len(candidates) > 0, "Should have extracted candidates"

    # Verify markers are present
    for c in candidates:
        assert ">>>" in c.context, (
            f"Missing >>>marker<<< in candidate context for term '{c.ai_term}'"
        )
    print("  All candidates have >>>markers<<< ✓")

    # Verify FP page 2 items were filtered
    page2_candidates = [c for c in candidates if c.page == 2]
    assert len(page2_candidates) == 0, (
        f"Page 2 (FP content) should produce 0 candidates, got {len(page2_candidates)}"
    )
    print(f"  FP filtering: page 2 correctly produced 0 candidates ✓")
    print()

    print("--- Stage 2: Semantic scoring ---")
    scored = stage2_semantic_score(candidates, cfg, TAXONOMY)
    print(f"  Accepted: {len(scored)}/{len(candidates)}")
    for s in scored:
        print(
            f"  [sem={s.semantic_score:.3f}] [{s.detection_method}] "
            f"A={s.category_a} B={s.category_b} | "
            f"term='{s.fragment.ai_term}'"
        )

    assert len(scored) > 0, "Stage 2 should accept at least one fragment"
    print()

    print("--- Full pipeline: detect_references_in_document ---")
    doc_result = detect_references_in_document(
        page_texts=sample_pages,
        company="TestCorp",
        year=2024,
        position=1,
        industry="Technology",
        sector="Software",
        country="USA",
        doc_type="Annual Report",
        source="test_annual_report.pdf",
        page_count=3,
        config=cfg,
        taxonomy=TAXONOMY,
    )
    print(f"  Total refs: {doc_result.total_refs}")
    print(f"  Pattern refs: {doc_result.pattern_refs}")
    print(f"  Semantic refs: {doc_result.semantic_refs}")
    print(f"  Processing time: {doc_result.processing_time:.2f}s")
    assert doc_result.total_refs > 0

    # Verify context markers and ai_term in stored refs
    for ref in doc_result.references:
        assert ">>>" in ref.context, (
            f"AIReference missing >>>marker<<< in context for text='{ref.text}'"
        )
        assert ref.text, "AIReference text should not be empty"
    print(f"  All {doc_result.total_refs} AIReference objects have >>>markers<<< ✓")
    print()

    print("--- Ref hash utility ---")
    ref = doc_result.references[0]
    h = compute_ref_hash(ref)
    assert len(h) == 32, "MD5 hash should be 32 chars"
    print(f"  Hash: {h}")
    print(f"  AIReference: text='{ref.text}' A={ref.category_a} "
          f"B={ref.category_b} method={ref.detection_method}")
    print()
    print("  05_detect.py all checks passed.")
