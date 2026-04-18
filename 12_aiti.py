"""
12_aiti.py — AI Adoption Trajectory Index (AITI) Calculator
============================================================
AISA v1.0 | Schema v3

Calculates AITI for each company × year, starting exclusively from
adoption memory tables (adoption_events, adoption_portfolio_a/b,
adoption_links) — not from raw mentions.

Formula (Breadth):
    AITI_breadth(t) = ΔA(t) + ΔB(t) + ΔP(t) + M(t) + R(t)

    ΔA(t)  = +w_new_a × NEW_A  + w_disc_a × DISC_A
    ΔB(t)  = +w_new_b × NEW_B  + w_disc_b × DISC_B
    ΔP(t)  = Σ w_granularity × NEW_P  (capped per year)
    M(t)   = maturity bonus: B categories confirmed ≥ X years
    R(t)   = retirement penalty: DISCONTINUED explicit + inferred

Formula (Depth):
    AITI_depth(t) = min(1, log2(1 + total_refs_dedup(t)) / divisor)

Tabele scrise:
    aiti_company_year          — scorul principal
    aiti_event_contributions   — audit event-level
    aiti_company_state         — snapshot state per an

Tabele citite (read-only):
    adoption_events, adoption_portfolio_a/b, ai_references_deduplicated

Autor: AISA project
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ── Lazy imports (same pattern as rest of AISA) ───────────────────────────────
def _load(mod: str):
    return importlib.import_module(mod)

logger = logging.getLogger("AISA")

# ── Version constants (loaded lazily to avoid circular imports) ───────────────
_VER: Optional[Dict] = None

def _versions() -> Dict[str, str]:
    global _VER
    if _VER is None:
        v = _load("version")
        _VER = {
            "aisa":           getattr(v, "AISA_VERSION",     ""),
            "taxonomy":       getattr(v, "TAXONOMY_VERSION", ""),
            "memory":         getattr(v, "MEMORY_VERSION",   ""),
            "tpdi":           getattr(v, "TPDI_VERSION",     ""),
            "semantic_model": getattr(v, "SEMANTIC_MODEL_NAME", ""),
        }
    return _VER


# ============================================================================
# PARAMETERS
# ============================================================================

CALC_VERSION = "aiti_v1.0"

@dataclass
class AITIParams:
    """
    All tunable parameters for AITI calculation.
    Mirrors the aiti_parameters table — one instance per run.
    """
    # ΔA
    w_new_a:                     float = 2.0
    w_disc_a:                    float = -2.0

    # ΔB
    w_new_b:                     float = 1.0
    w_disc_b:                    float = -1.0

    # ΔP — weights per granularity
    w_new_p_specific:            float = 0.5
    w_new_p_internal:            float = 0.75
    w_new_p_vendor_only:         float = 0.25
    cap_delta_p_year:            float = 2.0

    # Maturity
    maturity_x_years:            int   = 5
    maturity_cap_per_b:          float = 2.0

    # Retirement / discontinuity
    discontinuity_threshold_years: int = 2
    inferred_retirement_penalty:   float = -0.25

    # Depth
    depth_cap_year:              float = 1.0
    depth_log_base:              float = 2.0
    depth_divisor:               float = 4.0

    notes: str = ""

    def config_hash(self) -> str:
        """Deterministic hash of all numeric parameters (for reproducibility)."""
        d = {k: v for k, v in self.__dict__.items() if k != "notes"}
        raw = json.dumps(d, sort_keys=True)
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    def granularity_weight(self, granularity: str) -> float:
        """Map granularity_level string → ΔP weight."""
        g = (granularity or "").upper()
        if g == "SPECIFIC":
            return self.w_new_p_specific
        if g == "INTERNAL":
            return self.w_new_p_internal
        if g == "VENDOR_ONLY":
            return self.w_new_p_vendor_only
        # CATEGORY_ONLY or unknown → 0 (no product signal)
        return 0.0


# ============================================================================
# INTERMEDIATE DATA STRUCTURES
# ============================================================================

@dataclass
class EventContrib:
    """One scored event to be written to aiti_event_contributions."""
    company:          str
    year:             int
    adoption_event_id: Optional[int]
    reference_id:     Optional[int]
    dimension:        str           # 'A' | 'B' | 'P'
    category:         str
    category_a:       str
    category_b:       str
    product_name:     str
    product_vendor:   str
    granularity_level: str
    event_type:       str
    evidence_type:    str           # EXPLICIT | INFERRED
    points:           float
    points_component: str           # delta_a | delta_b | delta_p | maturity | retirement
    reason:           str
    context_snippet:  str


@dataclass
class AITIResult:
    """Computed AITI scores + breakdown for one company × year."""
    company:              str
    year:                 int

    aiti_breadth:         float = 0.0
    aiti_depth:           float = 0.0

    delta_a:              float = 0.0
    delta_b:              float = 0.0
    delta_p:              float = 0.0
    maturity_m:           float = 0.0
    retirement_r:         float = 0.0
    retirement_explicit:  float = 0.0
    retirement_inferred:  float = 0.0

    active_a_count:       int   = 0
    active_b_count:       int   = 0
    active_p_count:       int   = 0
    confirmed_b_count:    int   = 0
    confirmed_years_mean: float = 0.0

    contribs: List[EventContrib] = field(default_factory=list)

    # State snapshot (written to aiti_company_state)
    active_a_json:           str = "[]"
    active_b_json:           str = "[]"
    active_p_json:           str = "[]"
    confirmed_streak_b_json: str = "{}"
    confirmed_streak_p_json: str = "{}"


# ============================================================================
# CORE CALCULATION
# ============================================================================

def _fetch_events(company: str, year: int, db) -> List[Dict]:
    """Load adoption_events for company × year."""
    rows = db.conn.execute(
        """
        SELECT id, dimension, category, product_name, product_vendor,
               event_type, reference_id, context
        FROM   adoption_events
        WHERE  company = ? AND year = ?
        ORDER  BY id
        """,
        (company, year),
    ).fetchall()
    return [dict(r) for r in rows]


def _fetch_portfolio_a(company: str, db) -> List[Dict]:
    """All portfolio_a entries for a company (any status)."""
    rows = db.conn.execute(
        """
        SELECT category_a, first_seen_year, last_confirmed_year,
               status, discontinued_year, linked_categories_b
        FROM   adoption_portfolio_a
        WHERE  company = ?
        """,
        (company,),
    ).fetchall()
    return [dict(r) for r in rows]


def _fetch_portfolio_b(company: str, db) -> List[Dict]:
    """All portfolio_b entries for a company (any status)."""
    rows = db.conn.execute(
        """
        SELECT category_b, product_name, product_vendor, granularity_level,
               first_seen_year, last_confirmed_year, status, discontinued_year
        FROM   adoption_portfolio_b
        WHERE  company = ?
        """,
        (company,),
    ).fetchall()
    return [dict(r) for r in rows]


def _count_dedup_refs(company: str, year: int, db) -> int:
    """Count deduplicated references for Depth metric."""
    row = db.conn.execute(
        """
        SELECT COUNT(*)
        FROM   ai_references_deduplicated
        WHERE  company = ? AND year = ?
        """,
        (company, year),
    ).fetchone()
    return row[0] if row else 0


def calculate_aiti_for_company_year(
    company: str,
    year: int,
    db,
    params: AITIParams,
) -> AITIResult:
    """
    Calculate AITI (breadth + depth) for one company × year.

    Reads:
        adoption_events (for year)
        adoption_portfolio_a/b (full history, for maturity streaks)
        ai_references_deduplicated (count, for depth)

    Returns AITIResult with full breakdown and event contributions.
    """
    result = AITIResult(company=company, year=year)
    events = _fetch_events(company, year, db)

    # ── Seen sets for de-duplication within the same year ────────────────────
    seen_new_a:    Set[str]            = set()
    seen_disc_a:   Set[str]            = set()
    seen_new_b:    Set[Tuple]          = set()
    seen_disc_b:   Set[Tuple]          = set()
    seen_new_p:    Set[Tuple]          = set()

    delta_a = 0.0
    delta_b = 0.0
    delta_p_raw = 0.0

    # ── ΔA — Application dimension ───────────────────────────────────────────
    for ev in events:
        if ev["dimension"] != "A":
            continue

        cat_a = ev["category"] or ""
        etype = ev["event_type"]

        if etype == "NEW_ADOPTION" and cat_a not in seen_new_a:
            seen_new_a.add(cat_a)
            pts = params.w_new_a
            delta_a += pts
            result.contribs.append(EventContrib(
                company=company, year=year,
                adoption_event_id=ev["id"], reference_id=ev.get("reference_id"),
                dimension="A", category=cat_a,
                category_a=cat_a, category_b="",
                product_name="", product_vendor="", granularity_level="",
                event_type=etype, evidence_type="EXPLICIT",
                points=pts, points_component="delta_a",
                reason=f"New application category adopted: {cat_a}",
                context_snippet=(ev.get("context") or "")[:400],
            ))

        elif etype == "DISCONTINUED" and cat_a not in seen_disc_a:
            seen_disc_a.add(cat_a)
            pts = params.w_disc_a
            delta_a += pts
            result.contribs.append(EventContrib(
                company=company, year=year,
                adoption_event_id=ev["id"], reference_id=ev.get("reference_id"),
                dimension="A", category=cat_a,
                category_a=cat_a, category_b="",
                product_name="", product_vendor="", granularity_level="",
                event_type=etype, evidence_type="EXPLICIT",
                points=pts, points_component="delta_a",
                reason=f"Application category discontinued: {cat_a}",
                context_snippet=(ev.get("context") or "")[:400],
            ))

    result.delta_a = delta_a

    # ── ΔB — Technology category dimension ───────────────────────────────────
    for ev in events:
        if ev["dimension"] != "B":
            continue

        cat_b = ev["category"] or ""
        etype = ev["event_type"]
        key   = (cat_b,)

        if etype == "NEW_ADOPTION" and key not in seen_new_b:
            seen_new_b.add(key)
            pts = params.w_new_b
            delta_b += pts
            result.contribs.append(EventContrib(
                company=company, year=year,
                adoption_event_id=ev["id"], reference_id=ev.get("reference_id"),
                dimension="B", category=cat_b,
                category_a="", category_b=cat_b,
                product_name=ev.get("product_name") or "",
                product_vendor=ev.get("product_vendor") or "",
                granularity_level="",
                event_type=etype, evidence_type="EXPLICIT",
                points=pts, points_component="delta_b",
                reason=f"New technology category adopted: {cat_b}",
                context_snippet=(ev.get("context") or "")[:400],
            ))

        elif etype == "DISCONTINUED" and key not in seen_disc_b:
            seen_disc_b.add(key)
            pts = params.w_disc_b
            delta_b += pts
            result.contribs.append(EventContrib(
                company=company, year=year,
                adoption_event_id=ev["id"], reference_id=ev.get("reference_id"),
                dimension="B", category=cat_b,
                category_a="", category_b=cat_b,
                product_name=ev.get("product_name") or "",
                product_vendor=ev.get("product_vendor") or "",
                granularity_level="",
                event_type=etype, evidence_type="EXPLICIT",
                points=pts, points_component="delta_b",
                reason=f"Technology category discontinued: {cat_b}",
                context_snippet=(ev.get("context") or "")[:400],
            ))

    result.delta_b = delta_b

    # ── ΔP — Product / granularity dimension ─────────────────────────────────
    for ev in events:
        if ev["dimension"] != "B":
            continue
        if ev["event_type"] != "NEW_ADOPTION":
            continue

        pname  = ev.get("product_name")  or ""
        pvend  = ev.get("product_vendor") or ""
        cat_b  = ev["category"] or ""

        if not pname and not pvend:
            continue  # CATEGORY_ONLY — no product signal

        # Fetch granularity from portfolio_b (most specific match)
        gran_row = db.conn.execute(
            """
            SELECT granularity_level
            FROM   adoption_portfolio_b
            WHERE  company = ?
              AND  category_b = ?
              AND  COALESCE(product_name,  '') = ?
              AND  COALESCE(product_vendor,'') = ?
            LIMIT 1
            """,
            (company, cat_b, pname, pvend),
        ).fetchone()
        granularity = gran_row[0] if gran_row else "CATEGORY_ONLY"
        w = params.granularity_weight(granularity)
        if w == 0.0:
            continue

        key = (cat_b, pname, pvend)
        if key in seen_new_p:
            continue
        seen_new_p.add(key)

        delta_p_raw += w
        result.contribs.append(EventContrib(
            company=company, year=year,
            adoption_event_id=ev["id"], reference_id=ev.get("reference_id"),
            dimension="P", category=f"{cat_b}::{pname or pvend}",
            category_a="", category_b=cat_b,
            product_name=pname, product_vendor=pvend,
            granularity_level=granularity,
            event_type=ev["event_type"], evidence_type="EXPLICIT",
            points=w, points_component="delta_p",
            reason=f"Product adoption ({granularity}): {pname or pvend}",
            context_snippet=(ev.get("context") or "")[:400],
        ))

    delta_p = min(delta_p_raw, params.cap_delta_p_year)
    result.delta_p = delta_p

    # ── M(t) — Maturity bonus ─────────────────────────────────────────────────
    portfolio_b = _fetch_portfolio_b(company, db)
    maturity_total = 0.0
    confirmed_years_list: List[int] = []
    confirmed_b_count = 0
    active_b_keys: List[str] = []
    streak_b: Dict[str, int] = {}

    for pb in portfolio_b:
        cat_b  = pb["category_b"] or ""
        status = pb["status"] or "ACTIVE"
        fy     = pb["first_seen_year"]
        lcy    = pb["last_confirmed_year"]

        if status != "ACTIVE":
            continue

        active_b_keys.append(cat_b)

        # Years confirmed up to and including current year
        confirmed_years = max(0, min(year, lcy) - fy + 1) if fy else 0
        streak_b[cat_b] = confirmed_years

        if confirmed_years >= params.maturity_x_years:
            bonus = min(confirmed_years - params.maturity_x_years + 1,
                        params.maturity_cap_per_b)
            maturity_total += bonus
            confirmed_b_count += 1
            confirmed_years_list.append(confirmed_years)

            result.contribs.append(EventContrib(
                company=company, year=year,
                adoption_event_id=None, reference_id=None,
                dimension="B", category=cat_b,
                category_a="", category_b=cat_b,
                product_name="", product_vendor="", granularity_level="",
                event_type="CONFIRMED", evidence_type="INFERRED",
                points=bonus, points_component="maturity",
                reason=(
                    f"Maturity: {cat_b} active for {confirmed_years} yrs "
                    f"(≥ {params.maturity_x_years}), bonus={bonus:.2f}"
                ),
                context_snippet="",
            ))

    result.maturity_m          = maturity_total
    result.confirmed_b_count   = confirmed_b_count
    result.confirmed_years_mean = (
        sum(confirmed_years_list) / len(confirmed_years_list)
        if confirmed_years_list else 0.0
    )
    result.active_b_count = len(active_b_keys)

    # ── R(t) — Retirement penalty ─────────────────────────────────────────────
    retirement_explicit = 0.0
    retirement_inferred = 0.0

    # Explicit: DISCONTINUED events in this year
    explicit_disc_b: Set[str] = set()
    for ev in events:
        if ev["dimension"] == "B" and ev["event_type"] == "DISCONTINUED":
            cat_b = ev["category"] or ""
            if cat_b not in explicit_disc_b:
                explicit_disc_b.add(cat_b)
                pts = params.w_disc_b   # already counted in delta_b, but not R
                # R is separate from delta_b — don't double count delta_b
                # R tracks explicit portfolio-level exits (same sign as disc_b)
                # We set R_explicit from disc_b events (already in delta_b)
                # so just track the count here for the breakdown field
                retirement_explicit += abs(params.w_disc_b)

    # Inferred: portfolio_b entries ACTIVE but not confirmed in threshold window
    # (already marked DISCONTINUED by finalize_year_memory — check if disc_year == year)
    inferred_disc = db.conn.execute(
        """
        SELECT category_b, product_name, product_vendor
        FROM   adoption_portfolio_b
        WHERE  company         = ?
          AND  discontinued_year = ?
          AND  status          = 'DISCONTINUED'
          AND  last_confirmed_year < ?
        """,
        (company, year, year - params.discontinuity_threshold_years),
    ).fetchall()

    for row in inferred_disc:
        cat_b = row[0] or ""
        if cat_b in explicit_disc_b:
            continue   # already counted as explicit
        pts = params.inferred_retirement_penalty
        retirement_inferred += abs(pts)
        result.contribs.append(EventContrib(
            company=company, year=year,
            adoption_event_id=None, reference_id=None,
            dimension="B", category=cat_b,
            category_a="", category_b=cat_b,
            product_name=row[1] or "", product_vendor=row[2] or "",
            granularity_level="",
            event_type="DISCONTINUED", evidence_type="INFERRED",
            points=pts, points_component="retirement",
            reason=(
                f"Inferred discontinuation: {cat_b} not seen for "
                f"{params.discontinuity_threshold_years}+ years"
            ),
            context_snippet="",
        ))

    # R(t) total is negative (penalty) but stored as components for audit
    result.retirement_explicit = -retirement_explicit
    result.retirement_inferred = retirement_inferred * params.inferred_retirement_penalty
    result.retirement_r        = result.retirement_explicit + result.retirement_inferred

    # ── Active counts (portfolio A) ───────────────────────────────────────────
    portfolio_a = _fetch_portfolio_a(company, db)
    active_a = [p for p in portfolio_a if p["status"] == "ACTIVE"]
    result.active_a_count = len(active_a)

    # Active products (portfolio_b with product name)
    active_p = [
        p for p in portfolio_b
        if p["status"] == "ACTIVE" and (p["product_name"] or p["product_vendor"])
    ]
    result.active_p_count = len(active_p)

    # ── State snapshots (for aiti_company_state) ──────────────────────────────
    result.active_a_json = json.dumps(
        [p["category_a"] for p in active_a], ensure_ascii=False
    )
    result.active_b_json = json.dumps(active_b_keys, ensure_ascii=False)
    result.active_p_json = json.dumps(
        [
            {
                "cat_b": p["category_b"],
                "name": p.get("product_name") or "",
                "vendor": p.get("product_vendor") or "",
                "gran": p.get("granularity_level") or "",
            }
            for p in active_p
        ],
        ensure_ascii=False,
    )
    result.confirmed_streak_b_json = json.dumps(streak_b, ensure_ascii=False)

    # ── AITI Breadth ──────────────────────────────────────────────────────────
    result.aiti_breadth = (
        result.delta_a
        + result.delta_b
        + result.delta_p
        + result.maturity_m
        + result.retirement_r
    )

    # ── AITI Depth ────────────────────────────────────────────────────────────
    total_dedup = _count_dedup_refs(company, year, db)
    raw_depth = (
        math.log(1 + total_dedup, params.depth_log_base) / params.depth_divisor
    )
    result.aiti_depth = min(params.depth_cap_year, raw_depth)

    logger.debug(
        f"AITI {company} {year}: "
        f"breadth={result.aiti_breadth:.3f} "
        f"(ΔA={result.delta_a:.2f} ΔB={result.delta_b:.2f} "
        f"ΔP={result.delta_p:.2f} M={result.maturity_m:.2f} "
        f"R={result.retirement_r:.2f}) "
        f"depth={result.aiti_depth:.3f}"
    )

    return result


# ============================================================================
# DB WRITE
# ============================================================================

def _upsert_aiti_company_year(result: AITIResult, params: AITIParams, db) -> None:
    """Write / update aiti_company_year row."""
    ver = _versions()
    db.conn.execute(
        """
        INSERT INTO aiti_company_year (
            company, year,
            aiti_breadth, aiti_depth,
            delta_a, delta_b, delta_p, maturity_m, retirement_r,
            retirement_explicit, retirement_inferred,
            active_a_count, active_b_count, active_p_count,
            confirmed_b_count, confirmed_years_mean,
            calc_version, aisa_version, taxonomy_version,
            memory_version, tpdi_version, semantic_model, config_hash,
            updated_at
        ) VALUES (
            ?,?, ?,?, ?,?,?,?,?, ?,?, ?,?,?,?,?, ?,?,?,?,?,?,?, CURRENT_TIMESTAMP
        )
        ON CONFLICT(company, year) DO UPDATE SET
            aiti_breadth         = excluded.aiti_breadth,
            aiti_depth           = excluded.aiti_depth,
            delta_a              = excluded.delta_a,
            delta_b              = excluded.delta_b,
            delta_p              = excluded.delta_p,
            maturity_m           = excluded.maturity_m,
            retirement_r         = excluded.retirement_r,
            retirement_explicit  = excluded.retirement_explicit,
            retirement_inferred  = excluded.retirement_inferred,
            active_a_count       = excluded.active_a_count,
            active_b_count       = excluded.active_b_count,
            active_p_count       = excluded.active_p_count,
            confirmed_b_count    = excluded.confirmed_b_count,
            confirmed_years_mean = excluded.confirmed_years_mean,
            calc_version         = excluded.calc_version,
            aisa_version         = excluded.aisa_version,
            taxonomy_version     = excluded.taxonomy_version,
            memory_version       = excluded.memory_version,
            tpdi_version         = excluded.tpdi_version,
            semantic_model       = excluded.semantic_model,
            config_hash          = excluded.config_hash,
            updated_at           = CURRENT_TIMESTAMP
        """,
        (
            result.company, result.year,
            result.aiti_breadth, result.aiti_depth,
            result.delta_a, result.delta_b, result.delta_p,
            result.maturity_m, result.retirement_r,
            result.retirement_explicit, result.retirement_inferred,
            result.active_a_count, result.active_b_count, result.active_p_count,
            result.confirmed_b_count, result.confirmed_years_mean,
            CALC_VERSION,
            ver["aisa"], ver["taxonomy"], ver["memory"],
            ver["tpdi"], ver["semantic_model"],
            params.config_hash(),
        ),
    )


def _insert_event_contributions(result: AITIResult, db) -> None:
    """Bulk-insert event contributions (replace for this company × year)."""
    # Clear previous run for this company × year × calc_version
    db.conn.execute(
        "DELETE FROM aiti_event_contributions WHERE company=? AND year=? AND calc_version=?",
        (result.company, result.year, CALC_VERSION),
    )
    for c in result.contribs:
        db.conn.execute(
            """
            INSERT INTO aiti_event_contributions (
                company, year,
                adoption_event_id, reference_id,
                dimension, category, category_a, category_b,
                product_name, product_vendor, granularity_level,
                event_type, evidence_type,
                points, points_component, reason, context_snippet,
                calc_version
            ) VALUES (?,?, ?,?, ?,?,?,?, ?,?,?, ?,?, ?,?,?,?, ?)
            """,
            (
                c.company, c.year,
                c.adoption_event_id, c.reference_id,
                c.dimension, c.category, c.category_a, c.category_b,
                c.product_name, c.product_vendor, c.granularity_level,
                c.event_type, c.evidence_type,
                c.points, c.points_component, c.reason, c.context_snippet,
                CALC_VERSION,
            ),
        )


def _upsert_company_state(result: AITIResult, db) -> None:
    """Write snapshot state for aiti_company_state."""
    db.conn.execute(
        """
        INSERT INTO aiti_company_state (
            company, year,
            active_a_json, active_b_json, active_p_json,
            confirmed_streak_b_json, confirmed_streak_p_json,
            calc_version
        ) VALUES (?,?, ?,?,?, ?,?, ?)
        ON CONFLICT(company, year, calc_version) DO UPDATE SET
            active_a_json            = excluded.active_a_json,
            active_b_json            = excluded.active_b_json,
            active_p_json            = excluded.active_p_json,
            confirmed_streak_b_json  = excluded.confirmed_streak_b_json,
            confirmed_streak_p_json  = excluded.confirmed_streak_p_json
        """,
        (
            result.company, result.year,
            result.active_a_json, result.active_b_json, result.active_p_json,
            result.confirmed_streak_b_json, result.confirmed_streak_p_json,
            CALC_VERSION,
        ),
    )


def ensure_default_params(db) -> None:
    """Insert default aiti_v1.0 parameters if not already present."""
    existing = db.conn.execute(
        "SELECT 1 FROM aiti_parameters WHERE calc_version = ?",
        (CALC_VERSION,),
    ).fetchone()
    if existing:
        return
    p = AITIParams()
    db.conn.execute(
        """
        INSERT OR IGNORE INTO aiti_parameters (
            calc_version, schema_version,
            w_new_a, w_disc_a, w_new_b, w_disc_b,
            w_new_p_specific, w_new_p_internal, w_new_p_vendor_only, cap_delta_p_year,
            maturity_x_years, maturity_cap_per_b,
            discontinuity_threshold_years, inferred_retirement_penalty,
            depth_cap_year, depth_log_base, depth_divisor, notes
        ) VALUES (?,3, ?,?,?,?, ?,?,?,?, ?,?, ?,?, ?,?,?,?)
        """,
        (
            CALC_VERSION,
            p.w_new_a, p.w_disc_a, p.w_new_b, p.w_disc_b,
            p.w_new_p_specific, p.w_new_p_internal, p.w_new_p_vendor_only, p.cap_delta_p_year,
            p.maturity_x_years, p.maturity_cap_per_b,
            p.discontinuity_threshold_years, p.inferred_retirement_penalty,
            p.depth_cap_year, p.depth_log_base, p.depth_divisor,
            "Default parameters — AISA v1.0 initial release",
        ),
    )
    db.conn.commit()
    logger.info(f"Inserted default AITI parameters for {CALC_VERSION}")


# ============================================================================
# PUBLIC API
# ============================================================================

def calculate_aiti(
    company: str,
    year: int,
    db,
    params: Optional[AITIParams] = None,
    write_contributions: bool = True,
    write_state: bool = True,
) -> AITIResult:
    """
    Calculate and persist AITI for one company × year.

    Args:
        company:              Company name (must match adoption_events).
        year:                 Year to calculate.
        db:                   Open DatabaseManager.
        params:               AITIParams; uses defaults if None.
        write_contributions:  If True, write audit rows to aiti_event_contributions.
        write_state:          If True, write snapshot to aiti_company_state.

    Returns:
        AITIResult with all breakdown fields populated.
    """
    if params is None:
        params = AITIParams()

    result = calculate_aiti_for_company_year(company, year, db, params)

    _upsert_aiti_company_year(result, params, db)
    if write_contributions:
        _insert_event_contributions(result, db)
    if write_state:
        _upsert_company_state(result, db)

    db.conn.commit()
    return result


def calculate_aiti_all(
    db,
    years: Optional[List[int]] = None,
    companies: Optional[List[str]] = None,
    params: Optional[AITIParams] = None,
    write_contributions: bool = True,
    write_state: bool = True,
) -> Dict[str, int]:
    """
    Calculate AITI for all companies × years (or a filtered subset).

    Fetches distinct (company, year) pairs from adoption_events.

    Args:
        db:        Open DatabaseManager.
        years:     If given, only calculate for these years.
        companies: If given, only calculate for these companies.
        params:    AITIParams; uses defaults if None.
        write_contributions: Passed to calculate_aiti().
        write_state:         Passed to calculate_aiti().

    Returns:
        {"calculated": N, "errors": N}
    """
    if params is None:
        params = AITIParams()

    ensure_default_params(db)

    # Build dynamic WHERE clause
    where_parts = []
    bind: List = []

    if years:
        placeholders = ",".join("?" * len(years))
        where_parts.append(f"year IN ({placeholders})")
        bind.extend(years)
    if companies:
        placeholders = ",".join("?" * len(companies))
        where_parts.append(f"company IN ({placeholders})")
        bind.extend(companies)

    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    pairs = db.conn.execute(
        f"""
        SELECT DISTINCT company, year
        FROM   adoption_events
        {where_sql}
        ORDER  BY year, company
        """,
        bind,
    ).fetchall()

    stats = {"calculated": 0, "errors": 0}

    for company, year in pairs:
        try:
            calculate_aiti(
                company, year, db, params,
                write_contributions=write_contributions,
                write_state=write_state,
            )
            stats["calculated"] += 1
        except Exception as exc:
            logger.error(f"AITI error {company} {year}: {exc}")
            stats["errors"] += 1

    logger.info(
        f"AITI batch complete: {stats['calculated']} OK, {stats['errors']} errors"
    )
    return stats


def get_aiti_scores(
    db,
    company: Optional[str] = None,
    year: Optional[int] = None,
) -> List[Dict]:
    """
    Query aiti_company_year with optional filters.

    Returns list of dicts with all columns.
    """
    where_parts = []
    bind: List = []
    if company:
        where_parts.append("company = ?")
        bind.append(company)
    if year:
        where_parts.append("year = ?")
        bind.append(year)

    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    rows = db.conn.execute(
        f"""
        SELECT * FROM aiti_company_year
        {where_sql}
        ORDER BY year, aiti_breadth DESC
        """,
        bind,
    ).fetchall()
    return [dict(r) for r in rows]


def get_aiti_contributions(
    db,
    company: str,
    year: int,
) -> List[Dict]:
    """
    Query event contributions for one company × year (audit / explain).
    Sorted by |points| DESC.
    """
    rows = db.conn.execute(
        """
        SELECT * FROM aiti_event_contributions
        WHERE  company = ? AND year = ? AND calc_version = ?
        ORDER  BY ABS(points) DESC, dimension, category
        """,
        (company, year, CALC_VERSION),
    ).fetchall()
    return [dict(r) for r in rows]


# ============================================================================
# SELF-TEST
# ============================================================================

def _selftest():
    """
    Minimal smoke test — creates an in-memory SQLite DB,
    applies migrations 001-003, inserts synthetic adoption events,
    runs calculate_aiti_all(), and checks the results.
    """
    import sqlite3, os

    print("\n" + "="*60)
    print("12_aiti.py — self-test")
    print("="*60)

    # ── Stub version module for standalone test ──────────────────────────────
    import sys, types
    stub = types.ModuleType("version")
    stub.AISA_VERSION       = "1.0.0-test"
    stub.TAXONOMY_VERSION   = "1.0.0-test"
    stub.MEMORY_VERSION     = "1.0.0-test"
    stub.TPDI_VERSION       = "1.0.0-test"
    stub.SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"
    sys.modules["version"] = stub
    global _VER; _VER = None   # reset cached versions

    # ── Setup in-memory DB ────────────────────────────────────────────────────
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # Apply migrations (001→003) from local files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    migrations_dir = os.path.join(script_dir, "migrations")
    for fname in sorted(os.listdir(migrations_dir)):
        if fname.endswith(".sql"):
            sql = open(os.path.join(migrations_dir, fname), encoding="utf-8").read()
            conn.executescript(sql)
    conn.commit()

    # Minimal DatabaseManager stub
    class _DB:
        def __init__(self, c):
            self.conn = c

    db = _DB(conn)

    # ── Synthetic data ────────────────────────────────────────────────────────
    # Company "TestCo": adopts A1, B4, B2 (with product) in 2021
    #                   confirms B4 in 2022, 2023, 2024, 2025
    #                   B2 discontinued in 2023

    company = "TestCo"
    events = [
        # 2021
        (company, 2021, "A", "A1_Customer_Experience", None, None, "NEW_ADOPTION", None, "AI for CX"),
        (company, 2021, "B", "B4_GenAI_LLMs",          "GPT-3", "OpenAI", "NEW_ADOPTION", None, "GPT-3 pilot"),
        (company, 2021, "B", "B2_Deep_Learning",        None, None,      "NEW_ADOPTION", None, "DL models"),
        # 2022
        (company, 2022, "B", "B4_GenAI_LLMs",          "GPT-4", "OpenAI", "CONFIRMED",   None, "GPT-4 prod"),
        # 2023
        (company, 2023, "B", "B4_GenAI_LLMs",          "GPT-4", "OpenAI", "CONFIRMED",   None, "GPT-4 scale"),
        (company, 2023, "B", "B2_Deep_Learning",        None, None,       "DISCONTINUED", None, "DL sunset"),
        # 2024
        (company, 2024, "B", "B4_GenAI_LLMs",          "GPT-4o","OpenAI", "CONFIRMED",   None, "GPT-4o"),
        # 2025
        (company, 2025, "B", "B4_GenAI_LLMs",          "GPT-4o","OpenAI", "CONFIRMED",   None, "GPT-4o prod"),
        (company, 2025, "A", "A2_Operational_Efficiency", None, None,     "NEW_ADOPTION", None, "Ops AI"),
    ]
    conn.executemany(
        """
        INSERT INTO adoption_events
            (company, year, dimension, category, product_name, product_vendor,
             event_type, reference_id, context)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        events,
    )

    # Portfolio A
    conn.execute("""
        INSERT INTO adoption_portfolio_a
            (company, category_a, first_seen_year, last_confirmed_year, status)
        VALUES ('TestCo','A1_Customer_Experience',2021,2025,'ACTIVE')
    """)
    conn.execute("""
        INSERT INTO adoption_portfolio_a
            (company, category_a, first_seen_year, last_confirmed_year, status)
        VALUES ('TestCo','A2_Operational_Efficiency',2025,2025,'ACTIVE')
    """)

    # Portfolio B
    conn.execute("""
        INSERT INTO adoption_portfolio_b
            (company, category_b, product_name, product_vendor,
             granularity_level, first_seen_year, last_confirmed_year, status)
        VALUES ('TestCo','B4_GenAI_LLMs','GPT-4o','OpenAI',
                'SPECIFIC', 2021, 2025, 'ACTIVE')
    """)
    conn.execute("""
        INSERT INTO adoption_portfolio_b
            (company, category_b, product_name, product_vendor,
             granularity_level, first_seen_year, last_confirmed_year,
             status, discontinued_year)
        VALUES ('TestCo','B2_Deep_Learning', NULL, NULL,
                'CATEGORY_ONLY', 2021, 2021,
                'DISCONTINUED', 2023)
    """)

    # Dedup refs for depth
    for yr in range(2021, 2026):
        for i in range(5 * (yr - 2019)):
            conn.execute(
                "INSERT INTO ai_references_deduplicated "
                "(company,year,text,context) VALUES (?,?,?,?)",
                (company, yr, f"ref_{yr}_{i}", f"ctx_{yr}_{i}"),
            )
    conn.commit()

    # ── Run calculate_aiti_all ────────────────────────────────────────────────
    params = AITIParams()
    stats = calculate_aiti_all(db, params=params)
    print(f"\n  batch stats: {stats}")

    scores = get_aiti_scores(db, company=company)
    print(f"\n  Scores for {company}:")
    for s in scores:
        print(
            f"    {s['year']}: breadth={s['aiti_breadth']:+.3f}  "
            f"depth={s['aiti_depth']:.3f}  "
            f"(ΔA={s['delta_a']:+.1f} ΔB={s['delta_b']:+.1f} "
            f"ΔP={s['delta_p']:+.2f} M={s['maturity_m']:+.2f} "
            f"R={s['retirement_r']:+.2f})"
        )

    contribs_2025 = get_aiti_contributions(db, company, 2025)
    print(f"\n  Event contributions {company} 2025 ({len(contribs_2025)} items):")
    for c in contribs_2025:
        print(f"    [{c['dimension']}] {c['category'][:35]:<35} "
              f"{c['points_component']:<12} {c['points']:+.2f}  {c['reason'][:60]}")

    # Assertions
    assert stats["calculated"] == 5, f"Expected 5 years, got {stats['calculated']}"
    assert stats["errors"] == 0

    y2021 = next(s for s in scores if s["year"] == 2021)
    assert y2021["delta_a"] == params.w_new_a, \
        f"ΔA 2021 expected {params.w_new_a}, got {y2021['delta_a']}"
    assert y2021["delta_b"] == 2 * params.w_new_b, \
        f"ΔB 2021 expected {2*params.w_new_b}, got {y2021['delta_b']}"

    y2023 = next(s for s in scores if s["year"] == 2023)
    assert y2023["retirement_r"] != 0.0, "Expected retirement penalty in 2023"

    y2025 = next(s for s in scores if s["year"] == 2025)
    assert y2025["maturity_m"] > 0, "Expected maturity bonus in 2025 (B4 active 5 years)"

    print("\n  ✓ Toate assertiunile OK")
    print("="*60 + "\n")


if __name__ == "__main__":
    _selftest()
