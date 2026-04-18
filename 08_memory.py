"""
===============================================================================
AISA - AI Semantic Analyzer
08_memory.py - Adoption Memory / AITI (AI Technology Intelligence)
===============================================================================

Builds and maintains a longitudinal portfolio of AI technology adoptions
extracted from annual reports. Tracks which AI applications and technologies
each Fortune 500 company has adopted, when, and how their portfolio evolved.

Core concept:
    Each AIReference that survived detection + deduplication + sentiment
    represents a data point. This module aggregates those points into:

    Portfolio A  - Application layer: WHAT the company uses AI FOR
                   (one record per company × category_a)
    Portfolio B  - Technology layer: WHAT AI TECH the company uses
                   (one record per company × category_b × product)
    Events       - Timeline: NEW_ADOPTION, CONFIRMED, DISCONTINUED, REPLACED
    Links A↔B    - Cross-dimension relationships per company

Important distinction:
    Adoption Memory tracks WHAT COMPANIES CLAIM in reports.
    It is NOT validated external evidence of actual deployment.
    This aligns with the AI Buzz Index philosophy.

Processing flow (called from 10_pipeline.py or 11_cli.py):
    1. process_document_memory(refs, db)      → updates portfolios from one doc
    2. finalize_year_memory(year, db)         → detect DISCONTINUED after full year
    3. get_company_portfolio(company, db)     → query portfolio snapshot

CHANGELOG:
    v1.0.0 (2026-02) - AISA initial release

Author: TeRa0
License: MIT
===============================================================================
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from version import MEMORY_VERSION

_m1 = importlib.import_module("01_models")
AIReference     = _m1.AIReference
AnalyzerConfig  = _m1.AnalyzerConfig
logger          = _m1.logger

_m2 = importlib.import_module("02_db")
DatabaseManager = _m2.DatabaseManager


# ============================================================================
# CONSTANTS
# ============================================================================

# event_type values (canonical)
EVENT_NEW_ADOPTION  = "NEW_ADOPTION"
EVENT_CONFIRMED     = "CONFIRMED"
EVENT_DISCONTINUED  = "DISCONTINUED"
EVENT_REPLACED      = "REPLACED"

# granularity_level values (canonical)
GRANULARITY_SPECIFIC    = "SPECIFIC"      # Named product + vendor (ChatGPT / OpenAI)
GRANULARITY_VENDOR_ONLY = "VENDOR_ONLY"   # Vendor only (Microsoft)
GRANULARITY_CATEGORY    = "CATEGORY_ONLY" # Category only (B4_GenAI_LLMs)
GRANULARITY_INTERNAL    = "INTERNAL"      # Internal/custom tool, no vendor

# How many years of silence before marking DISCONTINUED
DISCONTINUITY_THRESHOLD_YEARS = 2


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PortfolioEntryA:
    """
    A single Application-dimension portfolio entry for a company.
    Tracks one category_a across all years it was mentioned.
    """
    company:                str
    category_a:             str
    first_seen_year:        int
    last_confirmed_year:    int
    discontinued_year:      Optional[int]   = None
    status:                 str             = "ACTIVE"   # ACTIVE | DISCONTINUED
    linked_categories_b:    List[str]       = field(default_factory=list)


@dataclass
class PortfolioEntryB:
    """
    A single Technology-dimension portfolio entry for a company.
    Tracks one category_b × product × vendor combination across years.
    """
    company:                str
    category_b:             str
    product_name:           Optional[str]
    product_vendor:         Optional[str]
    granularity_level:      str
    first_seen_year:        int
    last_confirmed_year:    int
    discontinued_year:      Optional[int]   = None
    status:                 str             = "ACTIVE"
    replaced_by_id:         Optional[int]   = None
    linked_categories_a:    List[str]       = field(default_factory=list)


@dataclass
class AdoptionEvent:
    """A single adoption timeline event for a company."""
    company:        str
    year:           int
    dimension:      str         # "A" or "B"
    category:       str
    product_name:   Optional[str]
    product_vendor: Optional[str]
    event_type:     str         # NEW_ADOPTION | CONFIRMED | DISCONTINUED | REPLACED
    reference_id:   Optional[int] = None
    context:        str           = ""


@dataclass
class CompanyPortfolio:
    """
    Full adoption portfolio snapshot for a company.

    Returned by get_company_portfolio() for export or analysis.
    """
    company:            str
    snapshot_year:      int         # Year this snapshot was taken
    portfolio_a:        List[Dict]  = field(default_factory=list)
    portfolio_b:        List[Dict]  = field(default_factory=list)
    events:             List[Dict]  = field(default_factory=list)
    links:              List[Dict]  = field(default_factory=list)
    summary:            Dict        = field(default_factory=dict)

    def __post_init__(self):
        if not self.summary:
            self.summary = {
                "total_applications":   len(self.portfolio_a),
                "total_technologies":   len(self.portfolio_b),
                "active_applications":  sum(1 for e in self.portfolio_a if e.get("status") == "ACTIVE"),
                "active_technologies":  sum(1 for e in self.portfolio_b if e.get("status") == "ACTIVE"),
                "specific_products":    sum(1 for e in self.portfolio_b if e.get("granularity_level") == GRANULARITY_SPECIFIC),
            }


# ============================================================================
# REFERENCE → MEMORY PROCESSOR
# ============================================================================

def _determine_event_type(
    existing_first_year: Optional[int],
    current_year: int,
) -> str:
    """
    Determine event type for a portfolio entry based on history.

    NEW_ADOPTION if this is the first time the category/product appears.
    CONFIRMED if it was already present in previous years.
    """
    if existing_first_year is None:
        return EVENT_NEW_ADOPTION
    if existing_first_year < current_year:
        return EVENT_CONFIRMED
    return EVENT_NEW_ADOPTION


def _normalize_product(ref: AIReference) -> Tuple[Optional[str], Optional[str], str]:
    """
    Extract and normalize product name, vendor, and granularity from a reference.

    Returns: (product_name, product_vendor, granularity_level)
    """
    name    = ref.product_name.strip() if ref.product_name else None
    vendor  = ref.product_vendor.strip() if ref.product_vendor else None

    if name and vendor:
        return name, vendor, GRANULARITY_SPECIFIC
    if vendor and not name:
        return None, vendor, GRANULARITY_VENDOR_ONLY
    if name and not vendor:
        return name, None, GRANULARITY_SPECIFIC
    return None, None, GRANULARITY_CATEGORY


def process_reference_memory(
    ref: AIReference,
    db: DatabaseManager,
    ref_db_id: Optional[int] = None,
) -> bool:
    """
    Process a single AIReference into the adoption memory tables.

    Updates:
        - adoption_portfolio_a  (upsert)
        - adoption_portfolio_b  (upsert)
        - adoption_events       (insert)
        - adoption_links        (upsert)

    Args:
        ref:        AIReference with category_a, category_b populated.
        db:         Open DatabaseManager.
        ref_db_id:  ID of the reference in ai_references_raw (for event linkage).

    Returns:
        True if any memory record was created or updated.
    """
    if not ref.category_a and not ref.category_b:
        return False

    year    = ref.year
    company = ref.company
    updated = False

    product_name, product_vendor, granularity = _normalize_product(ref)

    # ----------------------------------------------------------------
    # Portfolio A (Application dimension)
    # ----------------------------------------------------------------
    if ref.category_a:
        existing = db.conn.execute(
            """
            SELECT id, first_seen_year, status, linked_categories_b
            FROM adoption_portfolio_a
            WHERE company = ? AND category_a = ?
            """,
            (company, ref.category_a),
        ).fetchone()

        event_type = _determine_event_type(
            existing["first_seen_year"] if existing else None,
            year,
        )

        if existing:
            # Update last_confirmed_year and linked B categories
            linked_b = json.loads(existing["linked_categories_b"] or "[]")
            if ref.category_b and ref.category_b not in linked_b:
                linked_b.append(ref.category_b)

            db.conn.execute(
                """
                UPDATE adoption_portfolio_a
                SET last_confirmed_year = MAX(last_confirmed_year, ?),
                    status              = 'ACTIVE',
                    discontinued_year   = NULL,
                    linked_categories_b = ?,
                    updated_at          = CURRENT_TIMESTAMP
                WHERE company = ? AND category_a = ?
                """,
                (year, json.dumps(linked_b), company, ref.category_a),
            )
        else:
            linked_b = [ref.category_b] if ref.category_b else []
            db.conn.execute(
                """
                INSERT INTO adoption_portfolio_a
                    (company, category_a, first_seen_year, last_confirmed_year,
                     status, linked_categories_b)
                VALUES (?, ?, ?, ?, 'ACTIVE', ?)
                """,
                (company, ref.category_a, year, year, json.dumps(linked_b)),
            )

        # Event
        db.conn.execute(
            """
            INSERT INTO adoption_events
                (company, year, dimension, category,
                 product_name, product_vendor, event_type, reference_id, context)
            VALUES (?, ?, 'A', ?, ?, ?, ?, ?, ?)
            """,
            (
                company, year, ref.category_a,
                product_name, product_vendor,
                event_type, ref_db_id,
                ref.context[:500] if ref.context else "",
            ),
        )
        updated = True

    # ----------------------------------------------------------------
    # Portfolio B (Technology dimension)
    # ----------------------------------------------------------------
    if ref.category_b:
        existing_b = db.conn.execute(
            """
            SELECT id, first_seen_year, status, linked_categories_a
            FROM adoption_portfolio_b
            WHERE company = ? AND category_b = ?
              AND COALESCE(product_name, '') = COALESCE(?, '')
              AND COALESCE(product_vendor, '') = COALESCE(?, '')
            """,
            (company, ref.category_b, product_name, product_vendor),
        ).fetchone()

        event_type_b = _determine_event_type(
            existing_b["first_seen_year"] if existing_b else None,
            year,
        )

        if existing_b:
            linked_a = json.loads(existing_b["linked_categories_a"] or "[]")
            if ref.category_a and ref.category_a not in linked_a:
                linked_a.append(ref.category_a)

            db.conn.execute(
                """
                UPDATE adoption_portfolio_b
                SET last_confirmed_year = MAX(last_confirmed_year, ?),
                    status              = 'ACTIVE',
                    discontinued_year   = NULL,
                    granularity_level   = ?,
                    linked_categories_a = ?,
                    updated_at          = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (year, granularity, json.dumps(linked_a), existing_b["id"]),
            )
        else:
            linked_a = [ref.category_a] if ref.category_a else []
            db.conn.execute(
                """
                INSERT INTO adoption_portfolio_b
                    (company, category_b, product_name, product_vendor,
                     granularity_level, first_seen_year, last_confirmed_year,
                     status, linked_categories_a)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?)
                """,
                (
                    company, ref.category_b, product_name, product_vendor,
                    granularity, year, year, json.dumps(linked_a),
                ),
            )

        # Event B
        db.conn.execute(
            """
            INSERT INTO adoption_events
                (company, year, dimension, category,
                 product_name, product_vendor, event_type, reference_id, context)
            VALUES (?, ?, 'B', ?, ?, ?, ?, ?, ?)
            """,
            (
                company, year, ref.category_b,
                product_name, product_vendor,
                event_type_b, ref_db_id,
                ref.context[:500] if ref.context else "",
            ),
        )
        updated = True

    # ----------------------------------------------------------------
    # Links A↔B
    # ----------------------------------------------------------------
    if ref.category_a and ref.category_b:
        existing_link = db.conn.execute(
            """
            SELECT id, years, reference_count
            FROM adoption_links
            WHERE company = ? AND category_a = ? AND category_b = ?
              AND COALESCE(product_name, '') = COALESCE(?, '')
            """,
            (company, ref.category_a, ref.category_b, product_name),
        ).fetchone()

        if existing_link:
            years = json.loads(existing_link["years"] or "[]")
            if year not in years:
                years.append(year)
            db.conn.execute(
                """
                UPDATE adoption_links
                SET years           = ?,
                    reference_count = reference_count + 1,
                    updated_at      = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (json.dumps(sorted(years)), existing_link["id"]),
            )
        else:
            db.conn.execute(
                """
                INSERT INTO adoption_links
                    (company, category_a, category_b, product_name, years, reference_count)
                VALUES (?, ?, ?, ?, ?, 1)
                """,
                (company, ref.category_a, ref.category_b, product_name, json.dumps([year])),
            )

    # Mark reference as memory-processed
    if ref_db_id:
        db.conn.execute(
            """
            UPDATE ai_references_raw
            SET adoption_memory_processed = 1
            WHERE id = ?
            """,
            (ref_db_id,),
        )

    return updated


def process_document_memory(
    refs: List[AIReference],
    db: DatabaseManager,
    ref_ids: Optional[List[int]] = None,
) -> Dict[str, int]:
    """
    Process all references from one document into adoption memory.

    Deduplicates within the document: if the same category_a + category_b
    + product combination appears multiple times in one document, only the
    first occurrence triggers a NEW_ADOPTION event; subsequent ones are
    merged silently.

    Args:
        refs:     List of AIReference objects from one document.
        db:       Open DatabaseManager.
        ref_ids:  Optional list of DB IDs matching refs (same order).
                  If None, reference_id in events will be NULL.

    Returns:
        Dict with counts: {"processed": N, "new_adoptions": N, "confirmed": N}
    """
    if not refs:
        return {"processed": 0, "new_adoptions": 0, "confirmed": 0}

    seen_combos: Set[Tuple] = set()
    stats = {"processed": 0, "new_adoptions": 0, "confirmed": 0}

    for i, ref in enumerate(refs):
        ref_db_id = ref_ids[i] if ref_ids and i < len(ref_ids) else None

        # De-duplicate within the document
        combo = (
            ref.company, ref.year, ref.category_a, ref.category_b,
            ref.product_name or "", ref.product_vendor or "",
        )
        if combo in seen_combos:
            continue
        seen_combos.add(combo)

        was_updated = process_reference_memory(ref, db, ref_db_id=ref_db_id)
        if was_updated:
            stats["processed"] += 1

    db.conn.commit()

    # Count new vs confirmed from events just inserted (approximate)
    if refs:
        new_count = db.conn.execute(
            """
            SELECT COUNT(*) FROM adoption_events
            WHERE company = ? AND year = ? AND event_type = ?
            """,
            (refs[0].company, refs[0].year, EVENT_NEW_ADOPTION),
        ).fetchone()[0]
        conf_count = db.conn.execute(
            """
            SELECT COUNT(*) FROM adoption_events
            WHERE company = ? AND year = ? AND event_type = ?
            """,
            (refs[0].company, refs[0].year, EVENT_CONFIRMED),
        ).fetchone()[0]
        stats["new_adoptions"] = new_count
        stats["confirmed"]     = conf_count

    logger.info(
        f"Memory: {refs[0].company if refs else '?'} {refs[0].year if refs else '?'} → "
        f"{stats['processed']} processed, "
        f"{stats['new_adoptions']} new, "
        f"{stats['confirmed']} confirmed"
    )
    return stats


# ============================================================================
# YEAR FINALIZATION - DISCONTINUED DETECTION
# ============================================================================

def finalize_year_memory(
    year: int,
    db: DatabaseManager,
    threshold_years: int = DISCONTINUITY_THRESHOLD_YEARS,
) -> Dict[str, int]:
    """
    After all documents for a year are processed, detect DISCONTINUED entries.

    Logic: if a portfolio entry was ACTIVE but its last_confirmed_year is
    more than threshold_years behind the current year, mark it DISCONTINUED
    and insert a DISCONTINUED event.

    This should be called ONCE per year, AFTER all documents for that year
    have been processed through process_document_memory().

    Args:
        year:            The year just completed.
        db:              Open DatabaseManager.
        threshold_years: Years of silence before marking DISCONTINUED.

    Returns:
        Dict with counts: {"discontinued_a": N, "discontinued_b": N}
    """
    cutoff = year - threshold_years
    stats  = {"discontinued_a": 0, "discontinued_b": 0}

    # --- Portfolio A ---
    stale_a = db.conn.execute(
        """
        SELECT id, company, category_a
        FROM adoption_portfolio_a
        WHERE status = 'ACTIVE'
          AND last_confirmed_year <= ?
        """,
        (cutoff,),
    ).fetchall()

    for row in stale_a:
        db.conn.execute(
            """
            UPDATE adoption_portfolio_a
            SET status           = 'DISCONTINUED',
                discontinued_year = ?,
                updated_at        = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (year, row["id"]),
        )
        db.conn.execute(
            """
            INSERT INTO adoption_events
                (company, year, dimension, category,
                 product_name, product_vendor, event_type, reference_id, context)
            VALUES (?, ?, 'A', ?, NULL, NULL, ?, NULL, ?)
            """,
            (
                row["company"], year, row["category_a"],
                EVENT_DISCONTINUED,
                f"No mention in {threshold_years} years (last: {row['company']})",
            ),
        )
        stats["discontinued_a"] += 1

    # --- Portfolio B ---
    stale_b = db.conn.execute(
        """
        SELECT id, company, category_b, product_name, product_vendor
        FROM adoption_portfolio_b
        WHERE status = 'ACTIVE'
          AND last_confirmed_year <= ?
        """,
        (cutoff,),
    ).fetchall()

    for row in stale_b:
        db.conn.execute(
            """
            UPDATE adoption_portfolio_b
            SET status            = 'DISCONTINUED',
                discontinued_year = ?,
                updated_at        = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (year, row["id"]),
        )
        db.conn.execute(
            """
            INSERT INTO adoption_events
                (company, year, dimension, category,
                 product_name, product_vendor, event_type, reference_id, context)
            VALUES (?, ?, 'B', ?, ?, ?, ?, NULL, ?)
            """,
            (
                row["company"], year, row["category_b"],
                row["product_name"], row["product_vendor"],
                EVENT_DISCONTINUED,
                f"No mention in {threshold_years} years",
            ),
        )
        stats["discontinued_b"] += 1

    db.conn.commit()

    if stats["discontinued_a"] or stats["discontinued_b"]:
        logger.info(
            f"Memory finalize {year}: "
            f"{stats['discontinued_a']} A-discontinued, "
            f"{stats['discontinued_b']} B-discontinued"
        )

    return stats


# ============================================================================
# PORTFOLIO QUERY
# ============================================================================

def get_company_portfolio(
    company: str,
    db: DatabaseManager,
    snapshot_year: Optional[int] = None,
) -> CompanyPortfolio:
    """
    Retrieve the full adoption portfolio for a company.

    Args:
        company:       Company name (exact match).
        db:            Open DatabaseManager.
        snapshot_year: If set, return the portfolio state AS OF this year
                       (entries first seen after snapshot_year are excluded).
                       If None, returns the latest state.

    Returns:
        CompanyPortfolio dataclass.
    """
    year_filter = snapshot_year or 9999

    # Portfolio A
    rows_a = db.conn.execute(
        """
        SELECT * FROM adoption_portfolio_a
        WHERE company = ? AND first_seen_year <= ?
        ORDER BY status DESC, first_seen_year
        """,
        (company, year_filter),
    ).fetchall()

    portfolio_a = []
    for r in rows_a:
        entry = dict(r)
        entry["linked_categories_b"] = json.loads(entry.get("linked_categories_b") or "[]")
        portfolio_a.append(entry)

    # Portfolio B
    rows_b = db.conn.execute(
        """
        SELECT * FROM adoption_portfolio_b
        WHERE company = ? AND first_seen_year <= ?
        ORDER BY status DESC, granularity_level, first_seen_year
        """,
        (company, year_filter),
    ).fetchall()

    portfolio_b = []
    for r in rows_b:
        entry = dict(r)
        entry["linked_categories_a"] = json.loads(entry.get("linked_categories_a") or "[]")
        portfolio_b.append(entry)

    # Events
    year_condition = f"AND year <= {year_filter}" if snapshot_year else ""
    rows_e = db.conn.execute(
        f"""
        SELECT * FROM adoption_events
        WHERE company = ? {year_condition}
        ORDER BY year, dimension, event_type
        """,
        (company,),
    ).fetchall()
    events = [dict(r) for r in rows_e]

    # Links
    rows_l = db.conn.execute(
        """
        SELECT * FROM adoption_links
        WHERE company = ?
        ORDER BY category_a, category_b
        """,
        (company,),
    ).fetchall()

    links = []
    for r in rows_l:
        entry = dict(r)
        entry["years"] = json.loads(entry.get("years") or "[]")
        links.append(entry)

    portfolio = CompanyPortfolio(
        company=company,
        snapshot_year=snapshot_year or datetime.now().year,
        portfolio_a=portfolio_a,
        portfolio_b=portfolio_b,
        events=events,
        links=links,
    )
    # Recompute summary now that we have data
    portfolio.summary = {
        "total_applications":   len(portfolio_a),
        "total_technologies":   len(portfolio_b),
        "active_applications":  sum(1 for e in portfolio_a if e.get("status") == "ACTIVE"),
        "active_technologies":  sum(1 for e in portfolio_b if e.get("status") == "ACTIVE"),
        "specific_products":    sum(
            1 for e in portfolio_b
            if e.get("granularity_level") == GRANULARITY_SPECIFIC
        ),
        "years_tracked":        sorted({e["year"] for e in events}) if events else [],
    }

    return portfolio


# ============================================================================
# BULK QUERY - cross-company analysis
# ============================================================================

def get_technology_adoption_timeline(
    category_b: str,
    db: DatabaseManager,
    start_year: int = 2020,
    end_year: int = 2025,
) -> List[Dict]:
    """
    Get the adoption timeline for a specific technology across all companies.

    Useful for diffusion curve analysis (→ 09_tpdi.py).

    Args:
        category_b:  Technology category code (e.g. "B4_GenAI_LLMs").
        db:          Open DatabaseManager.
        start_year:  Start of analysis window.
        end_year:    End of analysis window.

    Returns:
        List of dicts: {year, company, event_type, product_name, product_vendor}
        Sorted by year.
    """
    rows = db.conn.execute(
        """
        SELECT e.year, e.company, e.event_type, e.product_name, e.product_vendor,
               p.granularity_level, p.industry
        FROM adoption_events e
        LEFT JOIN adoption_portfolio_b p
            ON e.company = p.company AND e.category = p.category_b
        WHERE e.dimension = 'B'
          AND e.category  = ?
          AND e.year BETWEEN ? AND ?
        ORDER BY e.year, e.company
        """,
        (category_b, start_year, end_year),
    ).fetchall()

    return [dict(r) for r in rows]


def get_product_mentions(
    product_name: str,
    db: DatabaseManager,
    vendor: Optional[str] = None,
) -> List[Dict]:
    """
    Find all companies that mention a specific product, with year range.

    Args:
        product_name:  Product name (case-insensitive partial match).
        db:            Open DatabaseManager.
        vendor:        Optional vendor filter (exact match).

    Returns:
        List of dicts with company, first/last year, total mentions.
    """
    vendor_clause = "AND LOWER(product_vendor) = LOWER(?)" if vendor else ""
    params = [f"%{product_name}%"]
    if vendor:
        params.append(vendor)

    rows = db.conn.execute(
        f"""
        SELECT company, product_vendor,
               MIN(first_seen_year) AS first_seen,
               MAX(last_confirmed_year) AS last_seen,
               status
        FROM adoption_portfolio_b
        WHERE LOWER(product_name) LIKE LOWER(?)
          {vendor_clause}
        GROUP BY company, product_vendor, status
        ORDER BY first_seen, company
        """,
        params,
    ).fetchall()

    return [dict(r) for r in rows]


# ============================================================================
# UNPROCESSED REFERENCES BATCH PROCESSOR
# ============================================================================

def process_unprocessed_references(
    db: DatabaseManager,
    batch_size: int = 500,
) -> Dict[str, int]:
    """
    Process all AIReference records not yet run through adoption memory.

    Used by the CLI `memory` subcommand to (re-)run memory processing
    on references already in the DB that haven't been memory-processed.

    Args:
        db:         Open DatabaseManager.
        batch_size: Number of references to process per DB batch.

    Returns:
        Dict with counts: {"total": N, "processed": N, "batches": N}
    """
    total_count = db.conn.execute(
        "SELECT COUNT(*) FROM ai_references_raw WHERE adoption_memory_processed = 0"
    ).fetchone()[0]

    if total_count == 0:
        logger.info("Memory: no unprocessed references found.")
        return {"total": 0, "processed": 0, "batches": 0}

    logger.info(f"Memory: processing {total_count} unprocessed references...")

    processed   = 0
    batches     = 0
    offset      = 0

    while True:
        rows = db.conn.execute(
            """
            SELECT id, company, year, position, industry, sector, country,
                   doc_type, text, context, page, category,
                   category_a, confidence_a, category_b, confidence_b,
                   detection_method, sentiment, sentiment_score,
                   semantic_score, source,
                   reference_strength, confidence_score, confidence_reasons,
                   product_name, product_vendor, granularity_level, event_type,
                   robotics_type, rpa_type, sentiment_confidence, category_confidence,
                   page_count
            FROM ai_references_raw
            WHERE adoption_memory_processed = 0
            ORDER BY company, year
            LIMIT ? OFFSET ?
            """,
            (batch_size, offset),
        ).fetchall()

        if not rows:
            break

        for row in rows:
            d = dict(row)
            ref = AIReference(
                company=d["company"], year=d["year"],
                position=d.get("position") or 0,
                industry=d.get("industry") or "",
                sector=d.get("sector") or "",
                country=d.get("country") or "",
                doc_type=d.get("doc_type") or "",
                text=d.get("text") or "",
                context=d.get("context") or "",
                page=d.get("page") or 0,
                category=d.get("category") or "",
                detection_method=d.get("detection_method") or "",
                sentiment=d.get("sentiment") or "neutral",
                sentiment_score=d.get("sentiment_score") or 0.0,
                semantic_score=d.get("semantic_score") or 0.0,
                source=d.get("source") or "",
                category_a=d.get("category_a") or "",
                confidence_a=d.get("confidence_a") or 0.0,
                category_b=d.get("category_b") or "",
                confidence_b=d.get("confidence_b") or 0.0,
                product_name=d.get("product_name"),
                product_vendor=d.get("product_vendor"),
                granularity_level=d.get("granularity_level") or GRANULARITY_CATEGORY,
                event_type=d.get("event_type") or EVENT_NEW_ADOPTION,
                reference_strength=d.get("reference_strength") or "unknown",
                confidence_score=d.get("confidence_score") or 0.0,
                confidence_reasons=d.get("confidence_reasons") or "",
            )

            # Enrich with product/vendor if not already stored
            if not ref.product_name and not ref.product_vendor and ref.category_b:
                try:
                    import importlib as _il
                    _m14 = _il.import_module("14_ai_products_v1")
                    pname, pvendor, _ = _m14.extract_product_info(
                        text=ref.text,
                        category_b=ref.category_b or None,
                        context=ref.context or None,
                        report_year=ref.year,
                    )
                    if pname:
                        ref.product_name = pname
                    if pvendor:
                        ref.product_vendor = pvendor
                except Exception:
                    pass  # 14_ai_products_v1 unavailable — skip enrichment

            was_updated = process_reference_memory(ref, db, ref_db_id=d["id"])
            if was_updated:
                processed += 1

        db.conn.commit()
        offset  += batch_size
        batches += 1
        logger.info(f"Memory batch {batches}: processed {min(offset, total_count)}/{total_count}")

    logger.info(f"Memory complete: {processed}/{total_count} references processed in {batches} batches.")
    return {"total": total_count, "processed": processed, "batches": batches}



# ============================================================================
# AITI — AI Technology Integration Index (pair-based scoring)
# ============================================================================

def calculate_aiti(
    company: str,
    db: "DatabaseManager",
    maturity_weight: float = 0.1,
    maturity_cap: float = 2.0,
) -> Dict[int, Dict]:
    """
    Calculeaza AITI (pair-based) pentru o companie.

    Formula: AITI(t) = AITI(t-1) + pair_points + maturity_bonus

    Scoring rules:
      - Prima aparitie a oricarui A[i] in orice pereche = +2 puncte (use case nou)
      - B[k] nou pentru A[i] existent = +1 punct (tehnologie noua pentru acel use case)
      - Pereche (A[i], B[k]) repetata = 0 puncte
      - Tech standalone fara App link (A = _NO_APP_) = +1 punct
      - App standalone fara Tech link (B = _NO_TECH_) = +2 daca A[i] e nou, altfel 0
      - Maturity bonus = min(maturity_cap, continued_pairs x maturity_weight)

    Schema note:
      - adoption_portfolio_a/b au first_seen_year + last_confirmed_year (NU years_active)
      - years_active e derivat din adoption_events (dimension A sau B, per year)
      - adoption_links.years e JSON list cu anii in care perechea A x B a fost activa

    Returns:
        Dict[year] -> {aiti, pair_points, maturity, new_pairs, continued_pairs, total_pairs}
    """
    # 1. Incarca toate linkurile A x B pentru companie (years e JSON list)
    rows = db.conn.execute(
        "SELECT category_a, category_b, product_name, years"
        " FROM adoption_links WHERE company = ?",
        (company,)
    ).fetchall()

    year_pairs: Dict[int, set] = defaultdict(set)
    for row in rows:
        years_list = json.loads(row["years"] or "[]")
        a = row["category_a"] or "_NO_APP_"
        b = row["category_b"] or "_NO_TECH_"
        for y in years_list:
            year_pairs[y].add((a, b))

    # 2. Standalone A (use case fara tehnologie pereche) — din adoption_events
    # FIX: schema nu are years_active; derivam din adoption_events dimension='A'
    rows_a = db.conn.execute(
        "SELECT DISTINCT category, year FROM adoption_events"
        " WHERE company = ? AND dimension = 'A'",
        (company,)
    ).fetchall()
    for row in rows_a:
        a = row["category"]
        y = int(row["year"])
        # Adauga standalone doar daca nu exista deja o pereche cu acest A in anul y
        if not any(pair[0] == a for pair in year_pairs[y]):
            year_pairs[y].add((a, "_NO_TECH_"))

    # 3. Standalone B (tehnologie fara use case pereche) — din adoption_events
    rows_b = db.conn.execute(
        "SELECT DISTINCT category, year FROM adoption_events"
        " WHERE company = ? AND dimension = 'B'",
        (company,)
    ).fetchall()
    for row in rows_b:
        b = row["category"]
        y = int(row["year"])
        if not any(pair[1] == b for pair in year_pairs[y]):
            year_pairs[y].add(("_NO_APP_", b))

    # 4. Calcul pair-based scoring
    aiti_score = 0.0
    all_seen_pairs: set = set()
    all_seen_apps: set = set()
    results: Dict[int, Dict] = {}

    for year in sorted(year_pairs.keys()):
        curr_pairs = year_pairs[year]
        new_pairs = curr_pairs - all_seen_pairs
        continued_pairs = curr_pairs & all_seen_pairs

        pair_points = 0.0
        for (a, b) in new_pairs:
            if a != "_NO_APP_":
                if a not in all_seen_apps:
                    pair_points += 2  # use case nou
                    all_seen_apps.add(a)
                if b != "_NO_TECH_":
                    pair_points += 1  # tehnologie noua pentru use case
            else:  # a == "_NO_APP_" → standalone tech
                pair_points += 1

        maturity = min(maturity_cap, len(continued_pairs) * maturity_weight)
        aiti_score += pair_points + maturity

        results[year] = {
            "aiti":             round(aiti_score, 2),
            "pair_points":      pair_points,
            "maturity":         round(maturity, 2),
            "new_pairs":        len(new_pairs),
            "continued_pairs":  len(continued_pairs),
            "total_pairs":      len(curr_pairs),
        }

        all_seen_pairs.update(new_pairs)

    return results


# ============================================================================
# SMOKE TEST
# ============================================================================

if __name__ == "__main__":
    import tempfile
    from version import get_version_string

    print(get_version_string())
    print(f"  Memory version: {MEMORY_VERSION}")
    print()

    def _make_ref(cat_a, cat_b, product=None, vendor=None, year=2022, strength="strong"):
        return AIReference(
            company="TestCorp", year=year, position=1,
            industry="Technology", sector="Software", country="USA",
            doc_type="Annual Report",
            text=f"We use {cat_b} for {cat_a}.",
            context="AI implementation context.",
            page=10, category=f"{cat_a}|{cat_b}",
            detection_method="pattern_hard",
            sentiment="positive", sentiment_score=0.85,
            semantic_score=0.78, source=f"TestCorp_{year}.pdf",
            category_a=cat_a, confidence_a=0.85,
            category_b=cat_b, confidence_b=0.80,
            product_name=product, product_vendor=vendor,
            granularity_level=GRANULARITY_SPECIFIC if (product and vendor) else GRANULARITY_CATEGORY,
            event_type=EVENT_NEW_ADOPTION,
            reference_strength=strength, confidence_score=0.82,
        )

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test_memory.db")

        # Need migrations folder for DatabaseManager
        mig_dir = os.path.join(tmp, "migrations")
        os.makedirs(mig_dir)

        # Copy 001_init.sql if available, otherwise create minimal schema
        init_sql_candidates = [
            os.path.join(os.path.dirname(__file__), "migrations", "001_init.sql"),
            "/mnt/user-data/uploads/001_init.sql",
        ]
        init_sql = None
        for candidate in init_sql_candidates:
            if os.path.exists(candidate):
                init_sql = open(candidate).read()
                break

        if init_sql is None:
            # Minimal schema for smoke test
            init_sql = """
                CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, description TEXT);
                CREATE TABLE IF NOT EXISTS ai_references_raw (id INTEGER PRIMARY KEY AUTOINCREMENT, company TEXT, year INTEGER, position INTEGER, industry TEXT, sector TEXT, country TEXT, doc_type TEXT, page INTEGER, text TEXT, context TEXT, detection_method TEXT, category TEXT, category_a TEXT DEFAULT '', confidence_a REAL DEFAULT 0.0, category_b TEXT DEFAULT '', confidence_b REAL DEFAULT 0.0, sentiment TEXT, sentiment_score REAL, sentiment_confidence TEXT DEFAULT 'standard', semantic_score REAL, category_confidence REAL DEFAULT 1.0, reference_strength TEXT DEFAULT 'unknown', confidence_score REAL DEFAULT 0.0, confidence_reasons TEXT DEFAULT '', robotics_type TEXT DEFAULT 'not_robotics', rpa_type TEXT DEFAULT 'not_rpa', product_name TEXT, product_vendor TEXT, granularity_level TEXT DEFAULT 'CATEGORY_ONLY', event_type TEXT DEFAULT 'NEW_ADOPTION', adoption_memory_processed INTEGER DEFAULT 0, ref_hash TEXT, occurrence_count INTEGER DEFAULT 1, page_count INTEGER DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, source TEXT, UNIQUE(company, year, doc_type, page, text));
                CREATE TABLE IF NOT EXISTS adoption_portfolio_a (id INTEGER PRIMARY KEY AUTOINCREMENT, company TEXT NOT NULL, category_a TEXT NOT NULL, first_seen_year INTEGER NOT NULL, last_confirmed_year INTEGER NOT NULL, discontinued_year INTEGER, status TEXT DEFAULT 'ACTIVE', linked_categories_b TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, UNIQUE(company, category_a));
                CREATE TABLE IF NOT EXISTS adoption_portfolio_b (id INTEGER PRIMARY KEY AUTOINCREMENT, company TEXT NOT NULL, category_b TEXT NOT NULL, product_name TEXT, product_vendor TEXT, granularity_level TEXT DEFAULT 'CATEGORY_ONLY', first_seen_year INTEGER NOT NULL, last_confirmed_year INTEGER NOT NULL, discontinued_year INTEGER, status TEXT DEFAULT 'ACTIVE', replaced_by_id INTEGER, linked_categories_a TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, UNIQUE(company, category_b, product_name, product_vendor));
                CREATE TABLE IF NOT EXISTS adoption_events (id INTEGER PRIMARY KEY AUTOINCREMENT, company TEXT NOT NULL, year INTEGER NOT NULL, dimension TEXT NOT NULL, category TEXT NOT NULL, product_name TEXT, product_vendor TEXT, event_type TEXT NOT NULL, reference_id INTEGER, context TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS adoption_links (id INTEGER PRIMARY KEY AUTOINCREMENT, company TEXT NOT NULL, category_a TEXT NOT NULL, category_b TEXT NOT NULL, product_name TEXT, years TEXT, reference_count INTEGER DEFAULT 1, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, UNIQUE(company, category_a, category_b, product_name));
                INSERT OR IGNORE INTO schema_version(version, description) VALUES (1, 'smoke test schema');
                INSERT OR IGNORE INTO schema_version(version, description) VALUES (2, 'smoke test schema v2');
            """

        with open(os.path.join(mig_dir, "001_init.sql"), "w") as f:
            f.write(init_sql)

        with DatabaseManager(db_path) as db:
            db.apply_migrations()

            # --- Year 2022: first appearance ---
            print("--- Year 2022: Initial adoption ---")
            refs_2022 = [
                _make_ref("A3_Customer_Experience", "B4_GenAI_LLMs",
                          product="ChatGPT", vendor="OpenAI", year=2022),
                _make_ref("A2_Operational_Excellence", "B7_Infrastructure_Platforms",
                          product="SageMaker", vendor="AWS", year=2022),
                _make_ref("A7_Governance_Ethics", "B8_General_AI", year=2022),
            ]
            stats = process_document_memory(refs_2022, db)
            print(f"  Processed: {stats['processed']}")
            print(f"  New adoptions: {stats['new_adoptions']}")

            port = get_company_portfolio("TestCorp", db, snapshot_year=2022)
            print(f"  Portfolio A entries: {len(port.portfolio_a)}")
            print(f"  Portfolio B entries: {len(port.portfolio_b)}")
            print(f"  Events: {len(port.events)}")
            print(f"  Links: {len(port.links)}")
            assert len(port.portfolio_a) == 3
            assert len(port.portfolio_b) == 3
            assert all(e["event_type"] == EVENT_NEW_ADOPTION for e in port.events)

            # --- Year 2023: same categories → CONFIRMED ---
            print()
            print("--- Year 2023: Confirmation ---")
            refs_2023 = [
                _make_ref("A3_Customer_Experience", "B4_GenAI_LLMs",
                          product="ChatGPT", vendor="OpenAI", year=2023),
            ]
            stats2 = process_document_memory(refs_2023, db)
            print(f"  New adoptions (should be 0): {stats2['new_adoptions']}")
            print(f"  Confirmed: {stats2['confirmed']}")
            assert stats2["new_adoptions"] == 0 or stats2["confirmed"] >= 1

            # --- Year 2025: finalize → DISCONTINUED for A7/B8 (last seen 2022) ---
            print()
            print("--- Year 2025: Finalize (discontinued detection) ---")
            disc = finalize_year_memory(2025, db, threshold_years=2)
            print(f"  Discontinued A: {disc['discontinued_a']}")
            print(f"  Discontinued B: {disc['discontinued_b']}")

            # A7 and B8 last seen in 2022, threshold=2, so 2022 <= 2025-2 = 2023 → discontinued
            assert disc["discontinued_a"] >= 1  # A7_Governance_Ethics
            assert disc["discontinued_b"] >= 1  # B8_General_AI

            # --- Full portfolio after all years ---
            print()
            print("--- Final portfolio ---")
            final = get_company_portfolio("TestCorp", db)
            print(f"  Active applications:  {final.summary['active_applications']}")
            print(f"  Active technologies:  {final.summary['active_technologies']}")
            print(f"  Specific products:    {final.summary['specific_products']}")
            print(f"  Years tracked:        {final.summary['years_tracked']}")

            # --- Technology timeline ---
            print()
            print("--- Technology timeline (B4_GenAI_LLMs) ---")
            timeline = get_technology_adoption_timeline("B4_GenAI_LLMs", db, 2020, 2025)
            for t in timeline:
                print(f"  {t['year']} | {t['company']} | {t['event_type']} | {t.get('product_name', '-')}")
            assert len(timeline) >= 2  # 2022 NEW + 2023 CONFIRMED

            # --- Product mentions ---
            print()
            print("--- Product mentions (ChatGPT) ---")
            mentions = get_product_mentions("ChatGPT", db)
            for m in mentions:
                print(f"  {m['company']} | {m['first_seen']}–{m['last_seen']} | {m['status']}")
            assert len(mentions) >= 1

            print()
            print("  08_memory.py all checks passed.")
