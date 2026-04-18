-- =============================================================================
-- AISA - AI Semantic Analyzer
-- Migration 003: AITI tables (Schema v2 -> v3)
-- =============================================================================
-- Adds 4 new tables for the AI Adoption Trajectory Index (AITI).
-- Does NOT modify any existing table.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. aiti_company_year  — scorul AITI principal per companie per an
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS aiti_company_year (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,

    company                 TEXT    NOT NULL,
    year                    INTEGER NOT NULL,

    -- Core outputs
    aiti_breadth            REAL    NOT NULL DEFAULT 0.0,
    aiti_depth              REAL    NOT NULL DEFAULT 0.0,

    -- Component breakdown (breadth)
    delta_a                 REAL    NOT NULL DEFAULT 0.0,
    delta_b                 REAL    NOT NULL DEFAULT 0.0,
    delta_p                 REAL    NOT NULL DEFAULT 0.0,
    maturity_m              REAL    NOT NULL DEFAULT 0.0,
    retirement_r            REAL    NOT NULL DEFAULT 0.0,
    retirement_explicit     REAL    NOT NULL DEFAULT 0.0,
    retirement_inferred     REAL    NOT NULL DEFAULT 0.0,

    -- Coverage / debug counts
    active_a_count          INTEGER NOT NULL DEFAULT 0,
    active_b_count          INTEGER NOT NULL DEFAULT 0,
    active_p_count          INTEGER NOT NULL DEFAULT 0,
    confirmed_b_count       INTEGER NOT NULL DEFAULT 0,
    confirmed_years_mean    REAL    NOT NULL DEFAULT 0.0,

    -- Reproducibility metadata
    calc_version            TEXT    NOT NULL DEFAULT 'aiti_v1.0',
    aisa_version            TEXT    NOT NULL DEFAULT '',
    taxonomy_version        TEXT    NOT NULL DEFAULT '',
    memory_version          TEXT    NOT NULL DEFAULT '',
    tpdi_version            TEXT    NOT NULL DEFAULT '',
    semantic_model          TEXT    NOT NULL DEFAULT '',
    config_hash             TEXT    NOT NULL DEFAULT '',

    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(company, year)
);

CREATE INDEX IF NOT EXISTS idx_aiti_cy_company
    ON aiti_company_year(company);
CREATE INDEX IF NOT EXISTS idx_aiti_cy_year
    ON aiti_company_year(year);

-- ---------------------------------------------------------------------------
-- 2. aiti_parameters  — parametri de calcul (sensitivity / replicare)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS aiti_parameters (
    id                              INTEGER PRIMARY KEY AUTOINCREMENT,

    calc_version                    TEXT    NOT NULL,
    schema_version                  INTEGER NOT NULL DEFAULT 3,

    -- ΔA weights
    w_new_a                         REAL    NOT NULL DEFAULT  2.0,
    w_disc_a                        REAL    NOT NULL DEFAULT -2.0,

    -- ΔB weights
    w_new_b                         REAL    NOT NULL DEFAULT  1.0,
    w_disc_b                        REAL    NOT NULL DEFAULT -1.0,

    -- ΔP weights
    w_new_p_specific                REAL    NOT NULL DEFAULT  0.5,
    w_new_p_internal                REAL    NOT NULL DEFAULT  0.75,
    w_new_p_vendor_only             REAL    NOT NULL DEFAULT  0.25,
    cap_delta_p_year                REAL    NOT NULL DEFAULT  2.0,

    -- Maturity
    maturity_x_years                INTEGER NOT NULL DEFAULT  5,
    maturity_cap_per_b              REAL    NOT NULL DEFAULT  2.0,

    -- Discontinuity / retirement
    discontinuity_threshold_years   INTEGER NOT NULL DEFAULT  2,
    inferred_retirement_penalty     REAL    NOT NULL DEFAULT -0.25,

    -- Depth metric
    depth_cap_year                  REAL    NOT NULL DEFAULT  1.0,
    depth_log_base                  REAL    NOT NULL DEFAULT  2.0,
    depth_divisor                   REAL    NOT NULL DEFAULT  4.0,

    notes                           TEXT    NOT NULL DEFAULT '',
    created_at                      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(calc_version)
);

-- Default parameter set for aiti_v1.0
INSERT OR IGNORE INTO aiti_parameters (
    calc_version, schema_version,
    w_new_a, w_disc_a,
    w_new_b, w_disc_b,
    w_new_p_specific, w_new_p_internal, w_new_p_vendor_only, cap_delta_p_year,
    maturity_x_years, maturity_cap_per_b,
    discontinuity_threshold_years, inferred_retirement_penalty,
    depth_cap_year, depth_log_base, depth_divisor,
    notes
) VALUES (
    'aiti_v1.0', 3,
     2.0, -2.0,
     1.0, -1.0,
     0.5,  0.75, 0.25, 2.0,
     5,    2.0,
     2,   -0.25,
     1.0,  2.0,  4.0,
    'Default parameters — AISA v1.0 initial release'
);

-- ---------------------------------------------------------------------------
-- 3. aiti_event_contributions  — audit: ce a contribuit la fiecare scor
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS aiti_event_contributions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,

    company                 TEXT    NOT NULL,
    year                    INTEGER NOT NULL,

    adoption_event_id       INTEGER,            -- logical FK → adoption_events.id
    reference_id            INTEGER,            -- logical FK → ai_references_raw.id

    dimension               TEXT    NOT NULL,   -- 'A' | 'B' | 'P'
    category                TEXT    NOT NULL,
    category_a              TEXT    NOT NULL DEFAULT '',
    category_b              TEXT    NOT NULL DEFAULT '',
    product_name            TEXT    NOT NULL DEFAULT '',
    product_vendor          TEXT    NOT NULL DEFAULT '',
    granularity_level       TEXT    NOT NULL DEFAULT '',

    event_type              TEXT    NOT NULL,   -- NEW_ADOPTION|CONFIRMED|DISCONTINUED|REPLACED
    evidence_type           TEXT    NOT NULL DEFAULT 'EXPLICIT', -- EXPLICIT|INFERRED

    points                  REAL    NOT NULL DEFAULT 0.0,
    points_component        TEXT    NOT NULL,   -- delta_a|delta_b|delta_p|maturity|retirement
    reason                  TEXT    NOT NULL DEFAULT '',
    context_snippet         TEXT    NOT NULL DEFAULT '',

    calc_version            TEXT    NOT NULL DEFAULT 'aiti_v1.0',
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_aiti_ec_company_year
    ON aiti_event_contributions(company, year);
CREATE INDEX IF NOT EXISTS idx_aiti_ec_event_id
    ON aiti_event_contributions(adoption_event_id);

-- ---------------------------------------------------------------------------
-- 4. aiti_company_state  — snapshot state-machine per companie per an
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS aiti_company_state (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,

    company                 TEXT    NOT NULL,
    year                    INTEGER NOT NULL,

    active_a_json           TEXT    NOT NULL DEFAULT '[]',
    active_b_json           TEXT    NOT NULL DEFAULT '[]',
    active_p_json           TEXT    NOT NULL DEFAULT '[]',

    confirmed_streak_b_json TEXT    NOT NULL DEFAULT '{}',
    confirmed_streak_p_json TEXT    NOT NULL DEFAULT '{}',

    calc_version            TEXT    NOT NULL DEFAULT 'aiti_v1.0',
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(company, year, calc_version)
);

-- ---------------------------------------------------------------------------
-- Record migration
-- ---------------------------------------------------------------------------
INSERT OR IGNORE INTO schema_version (version, description)
VALUES (3, 'AITI tables: aiti_company_year, aiti_parameters, aiti_event_contributions, aiti_company_state');
