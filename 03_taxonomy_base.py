"""
===============================================================================
AISA - AI Semantic Analyzer
03_taxonomy_base.py - Taxonomy protocol / interface
===============================================================================

Defines TaxonomyProvider: the interface that ANY taxonomy source must implement.
Detection code (05_detect.py) depends only on this protocol - it does not care
whether the taxonomy comes from the builtin module, an Excel file, or an API.

Current implementations:
    04_taxonomy_builtin.py  - Built-in taxonomy (keywords + patterns hardcoded)

Planned implementations (future):
    taxonomy/loader.py      - Load from Excel / CSV / XML

CHANGELOG:
    v1.0.0 (2026) - AISA initial release

Author: TeRa0
License: MIT
===============================================================================
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================================
# SUPPORTING TYPES
# ============================================================================

@dataclass
class CategoryInfo:
    """
    Metadata for a single taxonomy category.

    Used by the detector for classification and confidence scoring.
    """
    code:           str             # e.g. "A1_Product_Innovation"
    name:           str             # e.g. "Product & Service Innovation"
    description:    str             # Short description for export/reporting
    dimension:      str             # "A" (Application) or "B" (Technology)
    keywords:       List[str]       = field(default_factory=list)
    patterns:       List[str]       = field(default_factory=list)
    keyword_tiers:  Dict[str, int]  = field(default_factory=dict)
    # Tier 1 = high confidence, Tier 2 = medium, Tier 3 = low


@dataclass
class FPResult:
    """Result of a false positive check."""
    is_fp:      bool
    category:   Optional[str]   = None   # Which FP category triggered
    pattern:    Optional[str]   = None   # Which pattern matched


@dataclass
class ClassificationResult:
    """
    N-dimensional taxonomy classification result for a single text fragment.

    dimensions: maps dimension_name → (category_code, confidence)
    combined:   pipe-separated codes for all matched dimensions (export/legacy)

    Supports any number of dimensions (1, 2, 3, ...):
        1D — {"Topic": ("T1_ESG", 0.9)}
        2D — {"Application": ("A1_...", 0.9), "Technology": ("B4_...", 0.85)}
        3D — {"Application": ..., "Technology": ..., "Geography": ...}

    Backward-compat properties category_a/b and confidence_a/b read from the
    first and second dimension respectively, so existing code keeps working.
    """
    dimensions: Dict[str, Tuple[str, float]] = field(default_factory=dict)
    combined:   str = ""

    def __post_init__(self):
        if not self.combined and self.dimensions:
            parts = [code for code, _ in self.dimensions.values() if code]
            self.combined = "|".join(parts)

    # ── Backward-compat properties (for code expecting dual A/B) ─────────────
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
# TAXONOMY PROVIDER PROTOCOL
# ============================================================================

class TaxonomyProvider(ABC):
    """
    Abstract base class for all taxonomy sources.

    Any class that provides taxonomy data to the AISA detection pipeline
    must subclass this and implement all abstract methods.

    Design principle: detect.py imports TaxonomyProvider and calls these
    methods. It never imports from a specific taxonomy implementation.
    This allows swapping builtin ↔ Excel ↔ API without touching detect.py.
    """

    @abstractmethod
    def get_version(self) -> str:
        """Return the taxonomy version string (e.g. '1.0.0')."""
        ...

    @abstractmethod
    def get_dimensions(self) -> Dict[str, Dict[str, "CategoryInfo"]]:
        """
        Return ALL dimensions and their categories.

        The primary API for taxonomy content — implement this in every
        TaxonomyProvider subclass. Any number of dimensions is supported:

            1D (topic-only):
                {"Topic": {"T1_ESG": CategoryInfo(...), ...}}

            2D (classic AISA dual):
                {
                  "Application": {"A1_...": CategoryInfo(...), ...},
                  "Technology":  {"B1_...": CategoryInfo(...), ...},
                }

            3D:
                {
                  "Application": {...},
                  "Technology":  {...},
                  "Geography":   {...},
                }

        There is no limit on number of categories per dimension.

        Returns:
            OrderedDict (or regular dict) of dimension_name → {code: CategoryInfo}.
            Dimension order matters: the first dimension maps to category_a
            (backward compat), the second to category_b.
        """
        ...

    # ── Backward-compat helpers (concrete — delegate to get_dimensions()) ─────

    def get_applications(self) -> Dict[str, "CategoryInfo"]:
        """First dimension — backward compat alias. Delegates to get_dimensions()."""
        dims = self.get_dimensions()
        if not dims:
            return {}
        return next(iter(dims.values()))

    def get_technologies(self) -> Dict[str, "CategoryInfo"]:
        """Second dimension — backward compat alias. Delegates to get_dimensions()."""
        dims = self.get_dimensions()
        values = list(dims.values())
        return values[1] if len(values) > 1 else {}

    @abstractmethod
    def get_fp_patterns(self) -> Dict[str, List[str]]:
        """
        Return false positive filter patterns grouped by category name.

        Returns:
            Dict mapping fp_category_name → list of regex pattern strings.
            Example: {"embedding_non_ai": [r"embedding\\s+ESG", ...]}
        """
        ...

    @abstractmethod
    def check_false_positive(self, text: str, context: str = "") -> FPResult:
        """
        Check whether text+context is a false positive.

        Args:
            text:    The matched text fragment.
            context: Surrounding context (±N chars).

        Returns:
            FPResult with is_fp=True if this should be filtered out.
        """
        ...

    @abstractmethod
    def classify(self, text: str, context: str = "") -> ClassificationResult:
        """
        Classify text into dual taxonomy (Application + Technology).

        Args:
            text:    The matched text fragment.
            context: Surrounding context for better classification.

        Returns:
            ClassificationResult with best-match category codes and confidence.
        """
        ...

    # ------------------------------------------------------------------
    # Concrete helpers (available to all implementations)
    # ------------------------------------------------------------------

    def get_all_categories(self) -> Dict[str, "CategoryInfo"]:
        """Return combined dict of all categories across ALL dimensions."""
        cats = {}
        for dim_cats in self.get_dimensions().values():
            cats.update(dim_cats)
        return cats

    def get_category_info(self, code: str) -> Optional[CategoryInfo]:
        """Look up a category by its code. Returns None if not found."""
        return self.get_all_categories().get(code)

    def get_all_keywords(self) -> List[str]:
        """Flat list of all keywords across all categories."""
        keywords = []
        for cat in self.get_all_categories().values():
            keywords.extend(cat.keywords)
        return list(set(keywords))

    def get_all_patterns(self) -> List[str]:
        """Flat list of all detection patterns across all categories."""
        patterns = []
        for cat in self.get_all_categories().values():
            patterns.extend(cat.patterns)
        return list(set(patterns))

    def validate_code(self, code: str) -> bool:
        """Return True if code is a known category code."""
        return code in self.get_all_categories()

    def get_keyword_tier(self, code: str, keyword: str) -> int:
        """
        Return confidence tier for a keyword in a category.

        Tier 1 = high confidence (requires less additional context)
        Tier 2 = medium confidence
        Tier 3 = low confidence (requires strong additional context)
        Returns 2 (medium) as default if not explicitly tiered.
        """
        cat = self.get_category_info(code)
        if cat and keyword in cat.keyword_tiers:
            return cat.keyword_tiers[keyword]
        return 2  # default: medium


# ============================================================================
# COMPILED PATTERN CACHE
# ============================================================================

class CompiledPatternCache:
    """
    Module-level cache for compiled regex patterns.

    Patterns are compiled once at import time (or first use) and reused
    across all calls. This replaces per-instance compilation in the old code.

    Usage:
        cache = CompiledPatternCache()
        cache.compile_from_provider(taxonomy_provider)
        patterns = cache.get_detection_patterns()
        fp_patterns = cache.get_fp_patterns()
    """

    def __init__(self):
        self._detection: List[Tuple[re.Pattern, str]] = []
        # List of (compiled_pattern, category_code)

        self._fp: Dict[str, List[re.Pattern]] = {}
        # Dict of fp_category → list of compiled patterns

        self._compiled = False

    def compile_from_provider(self, provider: TaxonomyProvider):
        """
        Compile all patterns from a TaxonomyProvider instance.

        Safe to call multiple times - only compiles once.
        """
        if self._compiled:
            return

        # Detection patterns
        for code, cat in provider.get_all_categories().items():
            for pattern_str in cat.patterns:
                try:
                    compiled = re.compile(pattern_str, re.IGNORECASE)
                    self._detection.append((compiled, code))
                except re.error as e:
                    import logging
                    logging.getLogger("AISA").warning(
                        f"Invalid detection pattern in {code}: {pattern_str!r} → {e}"
                    )

        # FP patterns
        for fp_cat, pattern_list in provider.get_fp_patterns().items():
            self._fp[fp_cat] = []
            for pattern_str in pattern_list:
                try:
                    self._fp[fp_cat].append(
                        re.compile(pattern_str, re.IGNORECASE)
                    )
                except re.error as e:
                    import logging
                    logging.getLogger("AISA").warning(
                        f"Invalid FP pattern in {fp_cat}: {pattern_str!r} → {e}"
                    )

        self._compiled = True

    def get_detection_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """Return list of (compiled_pattern, category_code) tuples."""
        return self._detection

    def get_fp_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Return dict of fp_category → compiled patterns."""
        return self._fp

    def is_compiled(self) -> bool:
        return self._compiled


# ============================================================================
# SMOKE TEST
# ============================================================================

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from version import get_version_string

    print(get_version_string())
    print()

    # Verify the abstract interface can be inspected
    abstract_methods = TaxonomyProvider.__abstractmethods__
    print(f"  TaxonomyProvider abstract methods: {sorted(abstract_methods)}")
    assert "get_version" in abstract_methods
    assert "get_dimensions" in abstract_methods
    assert "get_fp_patterns" in abstract_methods
    assert "check_false_positive" in abstract_methods
    assert "classify" in abstract_methods
    print("  Protocol definition OK")

    # Verify dataclasses
    cat = CategoryInfo(
        code="A1_Product_Innovation",
        name="Product & Service Innovation",
        description="AI-enhanced products",
        dimension="A",
        keywords=["AI-powered", "ChatGPT"],
        keyword_tiers={"AI-powered": 1, "ChatGPT": 1},
    )
    print(f"  CategoryInfo OK: {cat.code} / tier ChatGPT={cat.keyword_tiers['ChatGPT']}")

    fp = FPResult(is_fp=True, category="embedding_non_ai", pattern=r"embedding\s+ESG")
    print(f"  FPResult OK: is_fp={fp.is_fp}, category={fp.category}")

    clf = ClassificationResult(
        dimensions={
            "Application": ("A1_Product_Innovation", 0.9),
            "Technology":  ("B4_GenAI_LLMs", 0.85),
        }
    )
    print(f"  ClassificationResult OK: combined={clf.combined}")

    # Verify cache
    cache = CompiledPatternCache()
    assert not cache.is_compiled()
    print("  CompiledPatternCache OK (empty)")

    print()
    print("  03_taxonomy_base.py all checks passed.")
