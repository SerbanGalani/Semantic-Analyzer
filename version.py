"""
===============================================================================
AISA - AI Semantic Analyzer
version.py - Centralized version constants
===============================================================================

Single source of truth for all version numbers used across AISA modules.
Import from here; never hardcode versions inside modules.

CHANGELOG:
    v1.1.0 (2026-03) - Multilingual support: dual semantic model + Chinese keywords
                       Added SEMANTIC_MODEL_MULTILINGUAL, SUPPORTED_LANGUAGES,
                       TRANSLATION_ENABLED. Schema bumped to v4.
    v1.0.0 (2026-02) - AISA initial release

Author: TeRa0
License: MIT
===============================================================================
"""

# ── Core versions ────────────────────────────────────────────────────────────
AISA_VERSION        = "2.0.0"
TAXONOMY_VERSION    = "1.2.1"   # version of the built-in AI_Disclosure taxonomy
SCHEMA_VERSION      = 9          # DB schema; bump when adding migrations

# ── Taxonomy registry ─────────────────────────────────────────────────────────
# Maps taxonomy_name → human-readable label.
# When you create a new 04_taxonomy_*.py, add an entry here AND in
# _TAXONOMY_MODULE_MAP inside 05_detect.py.
TAXONOMY_REGISTRY: dict = {
    "AI_Disclosure":      "AI Disclosure (Fortune Global 500)",
    "Digitalization_Eco": "Digitalization & Business Ecosystems",
    # "ESG_Reporting":    "ESG & Sustainability Reporting",   ← example future entry
}
DEFAULT_TAXONOMY = "AI_Disclosure"

# ── Semantic models ──────────────────────────────────────────────────────────
SEMANTIC_MODEL_NAME         = "all-MiniLM-L6-v2"                     # English
SEMANTIC_MODEL_MULTILINGUAL = "paraphrase-multilingual-MiniLM-L12-v2" # CJK + more

# ── Thresholds ───────────────────────────────────────────────────────────────
SEMANTIC_THRESHOLD          = 0.35   # default accept threshold
SEMANTIC_THRESHOLD_RELAXED  = 0.28   # used when strong keyword already matched
SEMANTIC_THRESHOLD_STRICT   = 0.50   # used for low-tier keywords

# ── Multilingual / translation ───────────────────────────────────────────────
SUPPORTED_LANGUAGES = ["en", "zh", "ja", "ko"]   # languages with dedicated support
TRANSLATION_ENABLED = True                        # set False to skip 15_translate.py

# ── Sub-module versions ───────────────────────────────────────────────────────
MEMORY_VERSION      = "1.1.0"   # 08_memory.py
TPDI_VERSION        = "1.1.0"   # 09_tpdi.py

# ── AI Buzz Index dimension labels (used in 06_analysis.py reports) ──────────
BUZZ_DIMENSION_LABELS: dict = {
    "volume":           "Volume",
    "depth":            "Depth",
    "breadth":          "Breadth",
    "tone":             "Tone",
    "specificity":      "Specificity",
    "forward_looking":  "Forward Looking",
    "salience":         "Salience",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_version_string() -> str:
    return (
        f"AISA v{AISA_VERSION} | "
        f"Taxonomy v{TAXONOMY_VERSION} | "
        f"Schema v{SCHEMA_VERSION}"
    )


if __name__ == "__main__":
    print(get_version_string())
    print(f"  Semantic model (EN):    {SEMANTIC_MODEL_NAME}")
    print(f"  Semantic model (multi): {SEMANTIC_MODEL_MULTILINGUAL}")
    print(f"  Supported languages:    {SUPPORTED_LANGUAGES}")
    print(f"  Translation enabled:    {TRANSLATION_ENABLED}")
    print(f"  Schema version:         {SCHEMA_VERSION}")
