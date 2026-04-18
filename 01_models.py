"""
===============================================================================
AISA - AI Semantic Analyzer
01_models.py - Core data models, configuration, and logging
===============================================================================

Provides all foundational dataclasses, configuration management, and logging
setup. No database, no taxonomy, no external ML dependencies - this module
must be importable with only stdlib + pandas + numpy.

COMPONENTS:
    1. Logging setup
    2. AnalyzerConfig       - Central configuration dataclass
    3. AIReference          - Single detected AI reference
    4. DocumentResult       - Result of processing one PDF
    5. ProcessingStats      - Aggregate stats for a full run
    6. AIBuzzIndex          - Per company-year AI Buzz Index

CHANGELOG:
    v1.0.0 (2026) - AISA rewrite from ai_analyzer_v6_3_module1
        - Removed all DB code (moved to 02_db.py)
        - Removed taxonomy imports (moved to 03_taxonomy_base.py)
        - Removed interactive config wizard (moved to 11_cli.py)
        - Removed Plotly imports
        - All version strings imported from version.py
        - Full English docstrings
    v1.0.1 (2026) - AIAdoptionIndex renamed to AIBuzzIndex
        - Sub-dimensions renamed: intensity→volume, semantic→depth,
          diversity→breadth, sentiment→tone, maturity→specificity,
          future→forward_looking, commitment→salience
        - AISA_DESCRIPTION updated to reflect buzz measurement
    v1.0.2 (2026) - DocumentResult extended with per-document text stats
        - Added word_count, sentence_count, paragraph_count defaults
        - Backward compatible with existing constructors

Author: TeRa0
License: MIT
===============================================================================
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from version import (
    AISA_VERSION,
    SEMANTIC_THRESHOLD,
    SEMANTIC_THRESHOLD_RELAXED,
    SEMANTIC_THRESHOLD_STRICT,
    get_version_string,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(
    log_file: str = "aisa.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure and return the AISA root logger.

    Creates both a file handler and a console handler. Safe to call multiple
    times - clears existing handlers before re-configuring.

    Args:
        log_file: Path to the log file.
        level:    Logging level (default INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger("AISA")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


# Module-level logger - all other modules import this
logger = setup_logging()


# ============================================================================
# CONSTANTS
# ============================================================================

CONFIG_FILE     = "aisa_config.json"
DOCUMENT_TYPES  = ["Annual Report", "Sustainability", "Proxy", "Quarterly Report"]

# Optional dependency flags - set by importers
OCR_AVAILABLE                   = False   # updated by 05_detect.py
FINBERT_AVAILABLE               = False   # updated by 06_analysis.py
PRODUCT_EXTRACTION_AVAILABLE    = False   # updated by 04_taxonomy_builtin.py
ADOPTION_MEMORY_AVAILABLE       = False   # updated by 08_memory.py


# ============================================================================
# ANALYZER CONFIG
# ============================================================================

@dataclass
class AnalyzerConfig:
    """
    Central configuration for an AISA analysis run.

    All processing parameters live here. Instantiate via:
        - AnalyzerConfig()                   for defaults
        - AnalyzerConfig.from_dict(d)        from a dict / saved JSON
        - AnalyzerConfig.from_json(path)     from a JSON file

    The CLI (11_cli.py) is responsible for interactive configuration and
    calls from_dict() with the collected values.
    """

    # --- Taxonomy ---
    taxonomy_name:  str           = "AI_Disclosure"
    # Registered name of the active taxonomy. Controls:
    #   - which 04_taxonomy_*.py module is loaded by 05_detect.py
    #   - output_folder default:  Results_{taxonomy_name}/
    #   - database_name default:  aisa_{taxonomy_name}.db
    taxonomy_excel: Optional[str] = None
    # Path to an Excel taxonomy file (e.g. "taxonomies/ESG_v1.xlsx").
    # When set, overrides taxonomy_name — no Python module needed.
    # taxonomy_name is still used for output_folder / database_name naming.

    # --- Paths ---
    input_folder:   str           = "Fortune500_PDFs"
    output_folder:  str           = ""          # auto-derived if left empty
    database_name:  str           = ""          # auto-derived if left empty
    fortune500_csv: Optional[str] = None

    # --- Semantic thresholds ---
    semantic_threshold:         float = SEMANTIC_THRESHOLD
    semantic_threshold_strict:  float = SEMANTIC_THRESHOLD_STRICT
    semantic_threshold_relaxed: float = SEMANTIC_THRESHOLD_RELAXED
    deduplication_threshold:    float = 0.85

    # --- Context extraction ---
    context_chars:              int = 150
    context_sentences_before:   int = 2
    context_sentences_after:    int = 2
    max_context_length:         int = 2000
    min_text_length:            int = 100

    # --- Processing ---
    max_workers:    int           = 2
    semantic_floor: float         = 0.20
    # Hard floor below which fragments are rejected regardless of trigger
    # Already used in stage2_semantic_score — making it explicit in config
    # allows users to raise it (e.g. 0.25) to reduce memory pressure from
    # processing large numbers of low-quality candidates
    batch_size:     int           = 5
    top_n:          Optional[int] = None

    # --- Filters ---
    filter_traditional_robotics: bool = True
    filter_traditional_rpa:      bool = True

    # --- AI Buzz Index weights (must sum to 1.0) ---
    weights: Dict[str, float] = field(default_factory=lambda: {
        "volume":           0.15,   # mention density per page
        "depth":            0.20,   # semantic specificity of language
        "breadth":          0.15,   # taxonomy category coverage
        "tone":             0.15,   # sentiment polarity
        "specificity":      0.15,   # concrete vs. superficial mentions
        "forward_looking":  0.10,   # future-oriented language
        "salience":         0.10,   # investment / deployment signals
    })

    # --- Year range ---
    start_year: int = 2020
    end_year:   int = 2025

    def __post_init__(self):
        # Auto-derive paths from taxonomy_name when not explicitly set
        if not self.output_folder:
            self.output_folder = f"Results_{self.taxonomy_name}"
        if not self.database_name:
            self.database_name = f"aisa_{self.taxonomy_name}.db"

        total = sum(self.weights.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.6f}. "
                f"Adjust weights: {self.weights}"
            )
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        logger.info(
            f"AnalyzerConfig initialized | "
            f"taxonomy={self.taxonomy_name} | "
            f"db={self.database_name} | output={self.output_folder} | "
            f"semantic={self.semantic_threshold} / strict={self.semantic_threshold_strict} | "
            f"workers={self.max_workers} | "
            f"robotics_filter={self.filter_traditional_robotics}"
        )

    @classmethod
    def from_dict(cls, d: Dict) -> "AnalyzerConfig":
        """Build config from a plain dict (e.g. loaded from JSON or CLI)."""
        return cls(
            taxonomy_name               = d.get("taxonomy_name",                "AI_Disclosure"),
            taxonomy_excel              = d.get("taxonomy_excel"),
            input_folder                = d.get("input_folder",                 "Fortune500_PDFs"),
            output_folder               = d.get("output_folder",                ""),   # "" → auto-derive
            database_name               = d.get("database_name",                ""),   # "" → auto-derive
            fortune500_csv              = d.get("fortune500_csv"),
            top_n                       = d.get("top_n"),
            max_workers                 = d.get("max_workers",                  2),
            semantic_floor              = d.get("semantic_floor",               0.20),
            start_year                  = d.get("start_year",                   2020),
            end_year                    = d.get("end_year",                     2025),
            semantic_threshold          = d.get("semantic_threshold",           SEMANTIC_THRESHOLD),
            semantic_threshold_strict   = d.get("semantic_threshold_strict",    SEMANTIC_THRESHOLD_STRICT),
            semantic_threshold_relaxed  = d.get("semantic_threshold_relaxed",   SEMANTIC_THRESHOLD_RELAXED),
            deduplication_threshold     = d.get("deduplication_threshold",      0.85),
            context_chars               = d.get("context_chars",                150),
            context_sentences_before    = d.get("context_sentences_before",     2),
            context_sentences_after     = d.get("context_sentences_after",      2),
            max_context_length          = d.get("max_context_length",           2000),
            min_text_length             = d.get("min_text_length",              100),
            batch_size                  = d.get("batch_size",                   5),
            filter_traditional_robotics = d.get("filter_traditional_robotics",  True),
            filter_traditional_rpa      = d.get("filter_traditional_rpa",       True),
            weights                     = d.get("weights", {
                "volume":           0.15,
                "depth":            0.20,
                "breadth":          0.15,
                "tone":             0.15,
                "specificity":      0.15,
                "forward_looking":  0.10,
                "salience":         0.10,
            }),
        )

    @classmethod
    def from_json(cls, path: str) -> "AnalyzerConfig":
        """Load config from a JSON file saved by save_json()."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save_json(self, path: str = CONFIG_FILE):
        """Persist config to JSON for reuse across runs."""
        data = {
            "taxonomy_name":    self.taxonomy_name,
            "taxonomy_excel":   self.taxonomy_excel,
            "input_folder":     self.input_folder,
            "output_folder":    self.output_folder,
            "database_name":    self.database_name,
            "fortune500_csv":   self.fortune500_csv,
            "top_n":            self.top_n,
            "max_workers":      self.max_workers,
            "start_year":       self.start_year,
            "end_year":         self.end_year,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Config saved to {path}")

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# AI REFERENCE
# ============================================================================

@dataclass
class AIReference:
    """
    A single AI reference detected in a corporate document.

    Dual taxonomy (v1.0):
        category_a  - Application dimension  (A1-A8): WHAT FOR?
        category_b  - Technology dimension   (B1-B8): WHAT TECHNOLOGY?

    Adoption Memory fields (v1.0):
        product_name    - Specific product (ChatGPT, Copilot, SageMaker, ...)
        product_vendor  - Vendor (OpenAI, Microsoft, AWS, ...)
        granularity_level - SPECIFIC | VENDOR_ONLY | CATEGORY_ONLY | INTERNAL
        event_type      - NEW_ADOPTION | CONFIRMED | DISCONTINUED | REPLACED
    """

    # --- Required fields (no defaults) ---
    company:            str
    year:               int
    position:           int
    industry:           str
    sector:             str
    country:            str
    doc_type:           str
    text:               str
    context:            str
    page:               int
    category:           str
    detection_method:   str
    sentiment:          str
    sentiment_score:    float
    semantic_score:     float
    source:             str
    language:           str   = "en"

    # --- Classification fields ---
    robotics_type:          str   = "not_robotics"   # ai_robotics | traditional_robotics | not_robotics
    rpa_type:               str   = "not_rpa"        # ai_rpa | traditional_rpa | not_rpa
    sentiment_confidence:   str   = "standard"       # standard | governance_adjusted | confirmed_negative

    category_confidence:    float = 1.0

    # --- Confidence / strength (FP reduction) ---
    reference_strength:     str   = "unknown"   # strong | medium | mention_only | unknown
    confidence_score:       float = 0.0         # 0..1
    confidence_reasons:     str   = ""          # pipe-separated audit trail

    # --- Taxonomy dimensions ---
    # dimensions_json stores all N dimensions as JSON for multi-dim taxonomies.
    # category_a/b are kept for backward compat (DB columns, export, old code).
    dimensions_json: str  = ""    # JSON: {"Application": ["A1_...", 0.9], "Technology": [...]}
    category_a:     str   = ""    # First dimension  category code (backward compat)
    confidence_a:   float = 0.0
    category_b:     str   = ""    # Second dimension category code (backward compat)
    confidence_b:   float = 0.0
    page_count:     int   = 0     # Total pages in source document

    # --- Adoption Memory ---
    product_name:               Optional[str] = None
    product_vendor:             Optional[str] = None
    granularity_level:          str           = "CATEGORY_ONLY"
    event_type:                 str           = "NEW_ADOPTION"
    adoption_memory_processed:  bool          = False

    # --- Dedupe support ---
    # Embedding tensor from Stage 2, used for cosine dedupe within a document.
    # Not persisted to DB (None after loading from DB).
    embedding:                  Optional[object] = field(default=None, repr=False)

    # --- Metadata ---
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    def __post_init__(self):
        # Build legacy combined category if not set
        if not self.category and self.category_a and self.category_b:
            self.category = f"{self.category_a}|{self.category_b}"

    def to_dict(self) -> Dict:
        return asdict(self)

    def get_product_display(self) -> str:
        """Human-readable product/vendor string."""
        if self.product_name and self.product_vendor:
            return f"{self.product_name} ({self.product_vendor})"
        if self.product_name:
            return self.product_name
        if self.product_vendor:
            return f"[{self.product_vendor}]"
        return "[Generic]"

    def get_combined_category(self) -> str:
        """Returns 'A3|B4' style combined category."""
        a = self.category_a or "?"
        b = self.category_b or "?"
        return f"{a}|{b}"


# ============================================================================
# DOCUMENT RESULT
# ============================================================================

@dataclass
class DocumentResult:
    """
    Aggregated result for a single processed PDF document.

    Collects all AIReference objects found in the document along with
    metadata and processing statistics.
    """

    company:    str
    year:       int
    position:   int
    industry:   str
    sector:     str
    country:    str
    doc_type:   str
    source:     str
    total_pages:    int
    text_length:    int

    # Per-document text statistics (used by pipeline progress/UI).
    # Defaults keep backward compatibility with older constructors.
    word_count:      int = 0
    sentence_count:  int = 0
    paragraph_count: int = 0

    references:             List[AIReference] = field(default_factory=list)
    total_refs:             int  = 0
    pattern_refs:           int  = 0
    semantic_refs:          int  = 0
    categories_count:       Dict[str, int] = field(default_factory=dict)
    sentiment_distribution: Dict[str, int] = field(default_factory=dict)
    processing_time:        float = 0.0
    text_status:            str   = "valid"
    # text_status values:
    #   valid | corrupted_ocr_success | corrupted_ocr_failed | empty

    def add_reference(self, ref: AIReference):
        """Add a reference and update all counters."""
        self.references.append(ref)
        self.total_refs += 1

        if ref.detection_method.startswith("pattern"):
            self.pattern_refs += 1
        elif ref.detection_method.startswith("semantic"):
            self.semantic_refs += 1

        self.categories_count[ref.category] = (
            self.categories_count.get(ref.category, 0) + 1
        )
        self.sentiment_distribution[ref.sentiment] = (
            self.sentiment_distribution.get(ref.sentiment, 0) + 1
        )


# ============================================================================
# PROCESSING STATS
# ============================================================================

@dataclass
class ProcessingStats:
    """
    Aggregate statistics for a complete AISA processing run.

    Populated incrementally during pipeline execution.
    Call calculate_derived_metrics() before exporting.
    """

    # Documents
    total_documents:        int = 0
    successful_documents:   int = 0
    failed_documents:       int = 0

    # Pages
    total_pages:        int = 0
    valid_pages:        int = 0
    corrupted_pages:    int = 0
    ocr_pages:          int = 0

    # Text
    total_text_chars:   int = 0
    total_text_mb:      float = 0.0
    total_words:        int = 0
    total_sentences:    int = 0

    # References
    total_references_raw:           int = 0
    total_references_deduplicated:  int = 0

    # Timing
    total_processing_time:  float = 0.0
    start_time:             Optional[datetime] = None
    end_time:               Optional[datetime] = None

    # Derived (calculated)
    avg_pages_per_doc:          float = 0.0
    avg_references_per_page:    float = 0.0
    avg_chars_per_page:         int   = 0
    avg_words_per_page:         int   = 0
    avg_words_per_doc:          int   = 0
    refs_per_1000_words:        float = 0.0

    def calculate_derived_metrics(self):
        """Compute all ratio metrics. Call before printing or exporting."""
        d = self.successful_documents
        p = self.total_pages
        w = self.total_words

        if d > 0:
            self.avg_pages_per_doc  = self.total_pages / d
            self.avg_words_per_doc  = int(w / d)
        if p > 0:
            self.avg_references_per_page = self.total_references_deduplicated / p
            self.avg_chars_per_page      = int(self.total_text_chars / p)
            self.avg_words_per_page      = int(w / p)
        if w > 0:
            self.refs_per_1000_words = (
                self.total_references_deduplicated / w
            ) * 1000

        self.total_text_mb = self.total_text_chars / (1024 * 1024)

    def to_dict(self) -> Dict:
        """Export as nested dict for JSON / Excel."""
        self.calculate_derived_metrics()
        sr = self.successful_documents
        td = max(1, self.total_documents)
        rd = self.total_references_deduplicated
        rr = max(1, self.total_references_raw)

        return {
            "Documents": {
                "Total Processed":  self.total_documents,
                "Successful":       self.successful_documents,
                "Failed":           self.failed_documents,
                "Success Rate %":   round(100 * sr / td, 1),
            },
            "Pages": {
                "Total Pages Read":     self.total_pages,
                "Valid Pages":          self.valid_pages,
                "Corrupted (OCR)":      self.ocr_pages,
                "Avg Pages/Document":   round(self.avg_pages_per_doc, 1),
            },
            "Text Extracted": {
                "Total Characters":     f"{self.total_text_chars:,}",
                "Total Size (MB)":      round(self.total_text_mb, 1),
                "Total Words":          f"{self.total_words:,}",
                "Total Sentences":      f"{self.total_sentences:,}",
                "Avg Chars/Page":       self.avg_chars_per_page,
                "Avg Words/Page":       self.avg_words_per_page,
                "Avg Words/Document":   self.avg_words_per_doc,
            },
            "AI References": {
                "Raw References":           self.total_references_raw,
                "After Deduplication":      self.total_references_deduplicated,
                "Deduplication Rate %":     round(100 * (1 - rd / rr), 1),
                "Avg References/Page":      round(self.avg_references_per_page, 3),
                "Refs per 1000 Words":      round(self.refs_per_1000_words, 3),
            },
            "Performance": {
                "Total Processing Time":    self._fmt(self.total_processing_time),
                "Avg Time/Document":        self._fmt(
                    self.total_processing_time / max(1, sr)
                ),
                "Start Time": (
                    self.start_time.strftime("%Y-%m-%d %H:%M:%S")
                    if self.start_time else "N/A"
                ),
                "End Time": (
                    self.end_time.strftime("%Y-%m-%d %H:%M:%S")
                    if self.end_time else "N/A"
                ),
            },
        }

    def print_summary(self):
        """Print a formatted summary to stdout."""
        self.calculate_derived_metrics()
        sr = self.successful_documents
        td = max(1, self.total_documents)
        rd = self.total_references_deduplicated
        rr = max(1, self.total_references_raw)

        print("\n" + "=" * 70)
        print(f"  AISA v{AISA_VERSION} - PROCESSING SUMMARY")
        print("=" * 70)

        print(f"\n  DOCUMENTS:")
        print(f"    Total processed:        {self.total_documents}")
        print(f"    Successful:             {self.successful_documents}")
        print(f"    Failed:                 {self.failed_documents}")
        print(f"    Success rate:           {100 * sr / td:.1f}%")

        print(f"\n  PAGES:")
        print(f"    Total pages read:       {self.total_pages:,}")
        print(f"    Valid pages:            {self.valid_pages:,}")
        print(f"    OCR pages:              {self.ocr_pages:,}")
        print(f"    Avg pages/document:     {self.avg_pages_per_doc:.1f}")

        print(f"\n  TEXT EXTRACTED:")
        print(f"    Total characters:       {self.total_text_chars:,}")
        print(f"    Total size:             {self.total_text_mb:.1f} MB")
        print(f"    Total words:            {self.total_words:,}")
        print(f"    Total sentences:        {self.total_sentences:,}")
        print(f"    Avg chars/page:         {self.avg_chars_per_page:,}")
        print(f"    Avg words/page:         {self.avg_words_per_page:,}")
        print(f"    Avg words/document:     {self.avg_words_per_doc:,}")

        print(f"\n  AI REFERENCES:")
        print(f"    Raw detected:           {self.total_references_raw:,}")
        print(f"    After deduplication:    {self.total_references_deduplicated:,}")
        print(f"    Deduplication rate:     {100 * (1 - rd / rr):.1f}%")
        print(f"    Avg refs/page:          {self.avg_references_per_page:.3f}")
        print(f"    Refs per 1000 words:    {self.refs_per_1000_words:.3f}")

        print(f"\n  PERFORMANCE:")
        print(f"    Total time:             {self._fmt(self.total_processing_time)}")
        print(f"    Avg time/document:      {self._fmt(self.total_processing_time / max(1, sr))}")
        if self.start_time:
            print(f"    Start:                  {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if self.end_time:
            print(f"    End:                    {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")

        print("=" * 70 + "\n")

    @staticmethod
    def _fmt(seconds: float) -> str:
        """Format seconds to human-readable duration string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}h {m}m {s}s"


# ============================================================================
# AI BUZZ INDEX
# ============================================================================

@dataclass
class AIBuzzIndex:
    """
    Composite AI Buzz Index for a single company-year pair.

    Measures the frequency and intensity of AI mentions in corporate reports.
    NOT a measure of real-world AI adoption - rather a proxy for how much
    a company talks about AI in its public disclosures.

    Seven weighted sub-dimensions produce a final ai_buzz_index score.
    Weights are defined in AnalyzerConfig.weights and must sum to 1.0.

    Sub-dimensions:
        volume          - mention density per page (tanh-normalized)
        depth           - semantic specificity of AI language used
        breadth         - taxonomy category coverage (Shannon entropy)
        tone            - sentiment polarity of AI mentions
        specificity     - concrete deployments vs. superficial mentions
        forward_looking - future-oriented AI language
        salience        - investment / deployment commitment signals
    """

    company:    str
    year:       int
    position:   int
    industry:   str
    sector:     str
    country:    str

    # Sub-dimension scores (0..1 each)
    volume_index:           float = 0.0
    depth_index:            float = 0.0
    breadth_index:          float = 0.0
    tone_index:             float = 0.0
    specificity_index:      float = 0.0
    forward_looking_index:  float = 0.0
    salience_index:         float = 0.0

    # Final composite score (0..1)
    ai_buzz_index:          float = 0.0

    # Supporting stats
    total_refs:             int = 0
    total_pages:            int = 0
    categories_used:        int = 0
    rank_in_year:           int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    def sub_dimensions(self) -> Dict[str, float]:
        """Return sub-dimension scores as a labelled dict."""
        return {
            "volume":           self.volume_index,
            "depth":            self.depth_index,
            "breadth":          self.breadth_index,
            "tone":             self.tone_index,
            "specificity":      self.specificity_index,
            "forward_looking":  self.forward_looking_index,
            "salience":         self.salience_index,
        }


# ============================================================================
# SMOKE TEST
# ============================================================================

if __name__ == "__main__":
    print(get_version_string())
    print()

    # Test AnalyzerConfig
    cfg = AnalyzerConfig(output_folder="test_output")
    print(f"  AnalyzerConfig OK | output: {cfg.output_folder}")
    print(f"  Weights keys: {list(cfg.weights.keys())}")
    assert "volume" in cfg.weights
    assert "salience" in cfg.weights

    # Test AIReference
    ref = AIReference(
        company="TestCorp", year=2023, position=1,
        industry="Technology", sector="Software",
        country="USA", doc_type="Annual Report",
        text="We use machine learning", context="...",
        page=12, category="B1_Traditional_ML",
        detection_method="pattern_hard",
        sentiment="positive", sentiment_score=0.8,
        semantic_score=0.0, source="TestCorp_2023.pdf",
        category_a="A1_Product_Innovation",
        category_b="B1_Traditional_ML",
    )
    print(f"  AIReference OK    | product: {ref.get_product_display()}")
    print(f"  Combined category : {ref.get_combined_category()}")

    # Test DocumentResult
    doc = DocumentResult(
        company="TestCorp", year=2023, position=1,
        industry="Technology", sector="Software",
        country="USA", doc_type="Annual Report",
        source="TestCorp_2023.pdf",
        total_pages=120, text_length=250000,
    )
    doc.add_reference(ref)
    print(f"  DocumentResult OK | refs: {doc.total_refs}")

    # Test ProcessingStats
    stats = ProcessingStats(
        total_documents=1, successful_documents=1,
        total_pages=120, total_words=50000,
        total_references_raw=10, total_references_deduplicated=7,
        total_processing_time=45.3,
        start_time=datetime.now(),
    )
    stats.calculate_derived_metrics()
    print(f"  ProcessingStats OK | refs/1000w: {stats.refs_per_1000_words:.3f}")

    # Test AIBuzzIndex
    idx = AIBuzzIndex(
        company="TestCorp", year=2023, position=1,
        industry="Technology", sector="Software", country="USA",
        volume_index=0.65,
        depth_index=0.72,
        breadth_index=0.48,
        tone_index=0.80,
        specificity_index=0.55,
        forward_looking_index=0.40,
        salience_index=0.60,
        ai_buzz_index=0.634,
        total_refs=7,
    )
    print(f"  AIBuzzIndex OK    | score: {idx.ai_buzz_index}")
    dims = idx.sub_dimensions()
    assert len(dims) == 7
    print(f"  Sub-dimensions    : {list(dims.keys())}")

    print()
    print("  All models OK.")

    # Cleanup test folder
    import shutil
    if Path("test_output").exists():
        shutil.rmtree("test_output")
