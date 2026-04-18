"""
===============================================================================
AISA - AI Semantic Analyzer
taxonomy_excel_loader.py - Load a TaxonomyProvider from an Excel file
===============================================================================

Allows researchers to define taxonomies in Excel without touching Python code.

EXCEL FORMAT
============

Sheet "Categories" (required):
    dimension   | code                  | name                    | description
    ------------|----------------------|-------------------------|-------------
    Application | D1_Supply_Chain      | Supply Chain Integration| ...
    Technology  | T1_ERP_Core_Systems  | ERP & Core Systems      | ...

    - dimension: any string label — becomes a key in get_dimensions()
    - code:      unique identifier, used as dict key everywhere
    - name:      human-readable label
    - description: shown in Taxonomy_Meta Excel export (can be empty)

Sheet "Keywords" (required):
    code                 | keyword          | tier
    ---------------------|-----------------|------
    D1_Supply_Chain      | supplier portal | 1
    D1_Supply_Chain      | EDI integration | 1
    D1_Supply_Chain      | ERP integration | 3

    - tier: 1 (high confidence), 2 (medium, default), 3 (low / needs context)
    - If tier column is missing, all keywords default to tier 2

Sheet "Patterns" (optional):
    code                 | pattern
    ---------------------|------------------------------------------
    D1_Supply_Chain      | \\bsupplier\\s+portal\\b
    D1_Supply_Chain      | \\bEDI\\b.*\\b(integration|platform)\\b

    - Raw regex strings; compiled at load time with re.IGNORECASE
    - Rows with invalid regex are skipped with a warning

Sheet "FP_Patterns" (optional):
    fp_category          | pattern
    ---------------------|------------------------------------------
    erp_non_digital      | \\bERP\\b.*\\b(vulnerability|CVE|patch)\\b

    - fp_category: any string label grouping related FP patterns
    - Rows with invalid regex are skipped with a warning

Sheet "Taxonomy_Meta" (optional, first row used):
    taxonomy_name        | display_name | version | description
    ---------------------|-------------|---------|------------------------------
    Digitalization_Eco   | Digitaliz.. | 1.0.0   | ...

USAGE
=====

    from taxonomy_excel_loader import ExcelTaxonomyProvider
    provider = ExcelTaxonomyProvider("my_taxonomy.xlsx")
    # Use directly:
    provider.classify("SAP S/4HANA migration", context)
    # Or register with AISA pipeline:
    from 05_detect import set_taxonomy
    set_taxonomy(provider)

CHANGELOG:
    v1.0.0 (2026-04) - Initial implementation

Author: TeRa0
License: MIT
===============================================================================
"""

from __future__ import annotations

import importlib
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_m3 = importlib.import_module("03_taxonomy_base")
TaxonomyProvider     = _m3.TaxonomyProvider
CategoryInfo         = _m3.CategoryInfo
ClassificationResult = _m3.ClassificationResult
FPResult             = _m3.FPResult
CompiledPatternCache = _m3.CompiledPatternCache

logger = logging.getLogger("AISA")


# ============================================================================
# LOADER HELPERS
# ============================================================================

def _read_sheet(xl: pd.ExcelFile, sheet: str) -> Optional[pd.DataFrame]:
    """Read a sheet from an ExcelFile, return None if sheet doesn't exist."""
    if sheet not in xl.sheet_names:
        return None
    df = xl.parse(sheet)
    # Normalize column names: lowercase + strip
    df.columns = [str(c).strip().lower() for c in df.columns]
    # Drop completely empty rows
    df = df.dropna(how="all")
    return df


def _require_cols(df: pd.DataFrame, sheet: str, cols: List[str]):
    """Raise ValueError if any required column is missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Sheet '{sheet}' is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )


# ============================================================================
# EXCEL TAXONOMY PROVIDER
# ============================================================================

class ExcelTaxonomyProvider(TaxonomyProvider):
    """
    TaxonomyProvider backed by an Excel file.

    Loads once at construction; immutable after that.
    Thread-safe for reads (no mutable state after __init__).

    Args:
        path: Path to the .xlsx file (absolute or relative to cwd).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If required sheets / columns are missing.
    """

    def __init__(self, path: str):
        self._path    = Path(path).resolve()
        self._version = "1.0.0"
        self._name    = self._path.stem   # default: filename without extension

        if not self._path.exists():
            raise FileNotFoundError(
                f"Taxonomy Excel file not found: {self._path}\n"
                f"Expected sheets: Categories, Keywords (required); "
                f"Patterns, FP_Patterns, Taxonomy_Meta (optional)."
            )

        self._dimensions: Dict[str, Dict[str, CategoryInfo]] = {}
        self._fp_patterns: Dict[str, List[str]] = {}
        self._compiled_cache: Optional[CompiledPatternCache] = None

        self._load(self._path)
        logger.info(
            f"ExcelTaxonomyProvider loaded: {self._path.name} | "
            f"v{self._version} | "
            f"{len(self._dimensions)} dimensions | "
            f"{sum(len(c) for c in self._dimensions.values())} categories"
        )

    # ── TaxonomyProvider interface ────────────────────────────────────────────

    def get_version(self) -> str:
        return self._version

    def get_dimensions(self) -> Dict[str, Dict[str, CategoryInfo]]:
        return self._dimensions

    def get_fp_patterns(self) -> Dict[str, List[str]]:
        return self._fp_patterns

    def check_false_positive(self, text: str, context: str = "") -> FPResult:
        combined = f"{text} {context}"
        cache = self._get_compiled_cache()
        for fp_cat, patterns in cache.get_fp_patterns().items():
            for pattern in patterns:
                if pattern.search(combined):
                    return FPResult(
                        is_fp=True,
                        category=fp_cat,
                        pattern=pattern.pattern,
                    )
        return FPResult(is_fp=False)

    def classify(self, text: str, context: str = "") -> ClassificationResult:
        """
        Classify text across ALL dimensions.

        For each dimension, finds the category with the highest keyword/pattern
        score. Returns ClassificationResult with all matched dimensions.
        """
        combined = f"{text} {context}"
        cache    = self._get_compiled_cache()

        dims: Dict[str, Tuple[str, float]] = {}

        # Group compiled detection patterns by dimension
        dim_patterns: Dict[str, List[Tuple[re.Pattern, str]]] = {}
        for compiled_pat, code in cache.get_detection_patterns():
            # Reverse-lookup: code → dimension
            dim = self._code_to_dim.get(code)
            if dim is None:
                continue
            dim_patterns.setdefault(dim, []).append((compiled_pat, code))

        for dim_name, cats in self._dimensions.items():
            cat_code, conf = self._match_dimension(
                combined,
                dim_patterns.get(dim_name, []),
                cats,
            )
            dims[dim_name] = (cat_code, conf)

        return ClassificationResult(dimensions=dims)

    # ── Internal loading logic ────────────────────────────────────────────────

    def _load(self, path: Path):
        xl = pd.ExcelFile(path, engine="openpyxl")

        # --- Optional: Taxonomy_Meta ---
        meta_df = _read_sheet(xl, "Taxonomy_Meta")
        if meta_df is not None and len(meta_df) > 0:
            row = meta_df.iloc[0]
            if "version" in meta_df.columns and pd.notna(row.get("version")):
                self._version = str(row["version"]).strip()
            if "taxonomy_name" in meta_df.columns and pd.notna(row.get("taxonomy_name")):
                self._name = str(row["taxonomy_name"]).strip()

        # --- Required: Categories ---
        cat_df = _read_sheet(xl, "Categories")
        if cat_df is None:
            raise ValueError(
                f"Sheet 'Categories' not found in {path.name}. "
                f"Available sheets: {xl.sheet_names}"
            )
        _require_cols(cat_df, "Categories", ["dimension", "code", "name"])

        categories: Dict[str, CategoryInfo] = {}
        for _, row in cat_df.iterrows():
            code = str(row["code"]).strip()
            dim  = str(row["dimension"]).strip()
            cat  = CategoryInfo(
                code        = code,
                name        = str(row["name"]).strip(),
                description = str(row.get("description", "")).strip()
                              if pd.notna(row.get("description")) else "",
                dimension   = dim,
            )
            categories[code] = cat
            self._dimensions.setdefault(dim, {})[code] = cat

        # Reverse lookup: code → dimension name (used in classify())
        self._code_to_dim: Dict[str, str] = {}
        for dim_name, cats in self._dimensions.items():
            for code in cats:
                self._code_to_dim[code] = dim_name

        # --- Required: Keywords ---
        kw_df = _read_sheet(xl, "Keywords")
        if kw_df is None:
            raise ValueError(
                f"Sheet 'Keywords' not found in {path.name}. "
                f"At minimum, Categories + Keywords sheets are required."
            )
        _require_cols(kw_df, "Keywords", ["code", "keyword"])

        has_tier = "tier" in kw_df.columns
        for _, row in kw_df.iterrows():
            code = str(row["code"]).strip()
            kw   = str(row["keyword"]).strip()
            if not kw or kw.lower() == "nan":
                continue
            if code not in categories:
                logger.warning(
                    f"Keywords sheet: code '{code}' not found in Categories — skipped."
                )
                continue
            tier = 2  # default medium
            if has_tier and pd.notna(row.get("tier")):
                try:
                    tier = int(row["tier"])
                except (ValueError, TypeError):
                    tier = 2
            cat = categories[code]
            if kw not in cat.keywords:
                cat.keywords.append(kw)
            cat.keyword_tiers[kw] = tier

        # --- Optional: Patterns ---
        pat_df = _read_sheet(xl, "Patterns")
        if pat_df is not None:
            _require_cols(pat_df, "Patterns", ["code", "pattern"])
            for _, row in pat_df.iterrows():
                code    = str(row["code"]).strip()
                pattern = str(row["pattern"]).strip()
                if not pattern or pattern.lower() == "nan":
                    continue
                if code not in categories:
                    logger.warning(
                        f"Patterns sheet: code '{code}' not found in Categories — skipped."
                    )
                    continue
                # Validate regex before adding
                try:
                    re.compile(pattern, re.IGNORECASE)
                    categories[code].patterns.append(pattern)
                except re.error as e:
                    logger.warning(
                        f"Patterns sheet: invalid regex for '{code}': "
                        f"{pattern!r} → {e} — skipped."
                    )

        # --- Optional: FP_Patterns ---
        fp_df = _read_sheet(xl, "FP_Patterns")
        if fp_df is not None:
            _require_cols(fp_df, "FP_Patterns", ["fp_category", "pattern"])
            for _, row in fp_df.iterrows():
                fp_cat  = str(row["fp_category"]).strip()
                pattern = str(row["pattern"]).strip()
                if not pattern or pattern.lower() == "nan":
                    continue
                try:
                    re.compile(pattern, re.IGNORECASE)
                    self._fp_patterns.setdefault(fp_cat, []).append(pattern)
                except re.error as e:
                    logger.warning(
                        f"FP_Patterns sheet: invalid regex in '{fp_cat}': "
                        f"{pattern!r} → {e} — skipped."
                    )

    # ── Classification helpers ────────────────────────────────────────────────

    def _get_compiled_cache(self) -> CompiledPatternCache:
        if self._compiled_cache is None:
            self._compiled_cache = CompiledPatternCache()
            self._compiled_cache.compile_from_provider(self)
        return self._compiled_cache

    def _match_dimension(
        self,
        text: str,
        compiled_patterns: List[Tuple[re.Pattern, str]],
        categories: Dict[str, CategoryInfo],
    ) -> Tuple[str, float]:
        """
        Find best-matching category in one dimension.

        Strategy (same as BuiltinTaxonomy._match_dimension):
            1. Regex patterns  → adds 0.6/0.7/0.8 confidence per tier
            2. Keyword match   → adds 0.4/0.6/0.8 per tier
            Best total score wins.

        Returns (category_code, confidence) or ("", 0.0) if no match.
        """
        scores: Dict[str, float] = {}
        text_lower = text.lower()

        # Pattern matches
        for compiled, code in compiled_patterns:
            if compiled.search(text):
                cat = categories.get(code)
                if cat is None:
                    continue
                scores[code] = scores.get(code, 0.0) + 0.7

        # Keyword matches
        for code, cat in categories.items():
            for kw in cat.keywords:
                if kw.lower() in text_lower:
                    tier = cat.keyword_tiers.get(kw, 2)
                    bonus = {1: 0.8, 2: 0.6, 3: 0.4}.get(tier, 0.6)
                    scores[code] = scores.get(code, 0.0) + bonus

        if not scores:
            return ("", 0.0)

        best_code = max(scores, key=lambda c: scores[c])
        best_conf = min(scores[best_code], 1.0)
        return (best_code, round(best_conf, 3))


# ============================================================================
# TEMPLATE GENERATOR
# ============================================================================

def create_template(output_path: str = "taxonomy_template.xlsx"):
    """
    Generate a blank Excel template with correct sheet names and column headers.

    Run this once to get a starting file, then fill in your categories.

    Args:
        output_path: Where to save the template.

    Example:
        python taxonomy_excel_loader.py
        # Creates taxonomy_template.xlsx in the current directory
    """
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

        pd.DataFrame([{
            "taxonomy_name": "My_Taxonomy",
            "display_name":  "My Custom Taxonomy",
            "version":       "1.0.0",
            "description":   "Describe what this taxonomy analyzes",
        }]).to_excel(writer, sheet_name="Taxonomy_Meta", index=False)

        pd.DataFrame([
            {"dimension": "Application", "code": "A1_Example",  "name": "Example Application Category",  "description": "What companies DO with the topic"},
            {"dimension": "Application", "code": "A2_Example2", "name": "Second Application Category",    "description": "Another application type"},
            {"dimension": "Technology",  "code": "T1_Example",  "name": "Example Technology Category",   "description": "HOW companies implement it"},
        ]).to_excel(writer, sheet_name="Categories", index=False)

        pd.DataFrame([
            {"code": "A1_Example",  "keyword": "example keyword one",  "tier": 1},
            {"code": "A1_Example",  "keyword": "example keyword two",  "tier": 2},
            {"code": "A1_Example",  "keyword": "generic keyword",       "tier": 3},
            {"code": "T1_Example",  "keyword": "technology keyword",    "tier": 1},
        ]).to_excel(writer, sheet_name="Keywords", index=False)

        pd.DataFrame([
            {"code": "A1_Example", "pattern": r"\bexample\s+keyword\s+one\b"},
            {"code": "T1_Example", "pattern": r"\btechnology\s+(?:platform|system|solution)\b"},
        ]).to_excel(writer, sheet_name="Patterns", index=False)

        pd.DataFrame([
            {"fp_category": "example_fp", "pattern": r"\bexample\b.*\b(irrelevant|unrelated)\b"},
        ]).to_excel(writer, sheet_name="FP_Patterns", index=False)

    print(f"Template created: {output_path}")
    print("Fill in Categories + Keywords sheets, then load with:")
    print("  from taxonomy_excel_loader import ExcelTaxonomyProvider")
    print(f"  provider = ExcelTaxonomyProvider('{output_path}')")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # No args: generate template
        create_template()
    elif sys.argv[1] == "validate":
        # Validate an existing Excel taxonomy file
        if len(sys.argv) < 3:
            print("Usage: python taxonomy_excel_loader.py validate <path.xlsx>")
            sys.exit(1)
        path = sys.argv[2]
        try:
            p = ExcelTaxonomyProvider(path)
            dims = p.get_dimensions()
            print(f"OK: {Path(path).name} | v{p.get_version()}")
            for dim_name, cats in dims.items():
                total_kw  = sum(len(c.keywords) for c in cats.values())
                total_pat = sum(len(c.patterns) for c in cats.values())
                print(f"  [{dim_name}] {len(cats)} categories | {total_kw} keywords | {total_pat} patterns")
            fp = p.get_fp_patterns()
            if fp:
                print(f"  [FP] {sum(len(v) for v in fp.values())} false positive patterns in {len(fp)} groups")
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    else:
        print("Usage:")
        print("  python taxonomy_excel_loader.py                    # generate template")
        print("  python taxonomy_excel_loader.py validate file.xlsx # validate file")
