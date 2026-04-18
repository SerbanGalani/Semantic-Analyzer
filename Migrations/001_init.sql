-- =============================================================================
-- AISA - AI Semantic Analyzer
-- Migration 001: Initial schema (Schema v1)
-- =============================================================================

CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

-- ---------------------------------------------------------------------------
-- PROCESSED DOCUMENTS
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS processed_documents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT    NOT NULL UNIQUE,
    company         TEXT    NOT NULL,
    year            INTEGER NOT NULL,
    position        INTEGER DEFAULT 0,
    industry        TEXT    DEFAULT '',
    doc_type        TEXT    DEFAULT '',
    total_pages     INTEGER DEFAULT 0,
    text_length     INTEGER DEFAULT 0,
    refs_found      INTEGER DEFAULT 0,
    file_hash       TEXT,
    text_status     TEXT    DEFAULT 'valid',
    process_count   INTEGER DEFAULT 1,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_procdoc_company ON processed_documents(company);
CREATE INDEX IF NOT EXISTS idx_procdoc_year    ON processed_documents(year);

-- ---------------------------------------------------------------------------
-- AI REFERENCES RAW
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ai_references_raw (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    company                     TEXT,
    year                        INTEGER,
    position                    INTEGER,
    industry                    TEXT,
    sector                      TEXT,
    country                     TEXT,
    doc_type                    TEXT,
    page                        INTEGER,
    source                      TEXT,
    page_count                  INTEGER DEFAULT 0,
    text                        TEXT,
    context                     TEXT,
    detection_method            TEXT,
    category                    TEXT,
    category_a                  TEXT    DEFAULT '',
    confidence_a                REAL    DEFAULT 0.0,
    category_b                  TEXT    DEFAULT '',
    confidence_b                REAL    DEFAULT 0.0,
    sentiment                   TEXT,
    sentiment_score             REAL,
    sentiment_confidence        TEXT    DEFAULT 'standard',
    semantic_score              REAL,
    category_confidence         REAL    DEFAULT 1.0,
    reference_strength          TEXT    DEFAULT 'unknown',
    confidence_score            REAL    DEFAULT 0.0,
    confidence_reasons          TEXT    DEFAULT '',
    robotics_type               TEXT    DEFAULT 'not_robotics',
    rpa_type                    TEXT    DEFAULT 'not_rpa',
    product_name                TEXT,
    product_vendor              TEXT,
    granularity_level           TEXT    DEFAULT 'CATEGORY_ONLY',
    event_type                  TEXT    DEFAULT 'NEW_ADOPTION',
    adoption_memory_processed   INTEGER DEFAULT 0,
    ref_hash                    TEXT,
    occurrence_count            INTEGER DEFAULT 1,
    created_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_raw_company ON ai_references_raw(company);
CREATE INDEX IF NOT EXISTS idx_raw_year    ON ai_references_raw(year);
CREATE INDEX IF NOT EXISTS idx_raw_hash    ON ai_references_raw(ref_hash);
CREATE INDEX IF NOT EXISTS idx_raw_compact ON ai_references_raw(company, year, doc_type, page, text);
CREATE INDEX IF NOT EXISTS idx_raw_cat_a   ON ai_references_raw(category_a);
CREATE INDEX IF NOT EXISTS idx_raw_cat_b   ON ai_references_raw(category_b);

-- ---------------------------------------------------------------------------
-- AI REFERENCES DEDUPLICATED
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ai_references_deduplicated (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    company                 TEXT,
    year                    INTEGER,
    position                INTEGER,
    industry                TEXT,
    sector                  TEXT,
    country                 TEXT,
    text                    TEXT,
    context                 TEXT,
    category                TEXT,
    category_a              TEXT    DEFAULT '',
    confidence_a            REAL    DEFAULT 0.0,
    category_b              TEXT    DEFAULT '',
    confidence_b            REAL    DEFAULT 0.0,
    sources                 TEXT,
    pages                   TEXT,
    doc_count               INTEGER DEFAULT 1,
    total_occurrences       INTEGER DEFAULT 1,
    avg_sentiment_score     REAL,
    avg_semantic_score      REAL,
    max_confidence_score    REAL,
    original_refs           TEXT,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company, year, context)
);
CREATE INDEX IF NOT EXISTS idx_dedup_company ON ai_references_deduplicated(company);
CREATE INDEX IF NOT EXISTS idx_dedup_year    ON ai_references_deduplicated(year);

-- ---------------------------------------------------------------------------
-- ADOPTION INDEX  (per company per year)
-- Column named ai_adoption_index in v1 — renamed to ai_buzz_index in migration 002
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS adoption_index (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    company                 TEXT    NOT NULL,
    year                    INTEGER NOT NULL,
    position                INTEGER DEFAULT 0,
    industry                TEXT    DEFAULT '',
    sector                  TEXT    DEFAULT '',
    country                 TEXT    DEFAULT '',
    volume_index            REAL    DEFAULT 0.0,
    depth_index             REAL    DEFAULT 0.0,
    breadth_index           REAL    DEFAULT 0.0,
    tone_index              REAL    DEFAULT 0.0,
    specificity_index       REAL    DEFAULT 0.0,
    forward_looking_index   REAL    DEFAULT 0.0,
    salience_index          REAL    DEFAULT 0.0,
    ai_adoption_index       REAL    DEFAULT 0.0,
    rank_in_year            INTEGER DEFAULT 0,
    total_refs              INTEGER DEFAULT 0,
    total_pages             INTEGER DEFAULT 0,
    categories_used         INTEGER DEFAULT 0,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company, year)
);
CREATE INDEX IF NOT EXISTS idx_adopt_company ON adoption_index(company);
CREATE INDEX IF NOT EXISTS idx_adopt_year    ON adoption_index(year);

-- ---------------------------------------------------------------------------
-- INDUSTRY BUZZ INDEX  (per industry per year)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS adoption_index_industry (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    industry                    TEXT    NOT NULL,
    year                        INTEGER NOT NULL,
    avg_volume_index            REAL    DEFAULT 0.0,
    avg_depth_index             REAL    DEFAULT 0.0,
    avg_breadth_index           REAL    DEFAULT 0.0,
    avg_tone_index              REAL    DEFAULT 0.0,
    avg_specificity_index       REAL    DEFAULT 0.0,
    avg_forward_looking_index   REAL    DEFAULT 0.0,
    avg_salience_index          REAL    DEFAULT 0.0,
    ai_buzz_index_industry      REAL    DEFAULT 0.0,
    rank_among_industries       INTEGER DEFAULT 0,
    num_companies               INTEGER DEFAULT 0,
    total_refs                  INTEGER DEFAULT 0,
    min_index                   REAL    DEFAULT 0.0,
    max_index                   REAL    DEFAULT 0.0,
    std_deviation               REAL    DEFAULT 0.0,
    companies_list              TEXT    DEFAULT '',
    created_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(industry, year)
);

-- ---------------------------------------------------------------------------
-- ADOPTION MEMORY
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS adoption_portfolio_a (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    company                 TEXT    NOT NULL,
    category_a              TEXT    NOT NULL,
    first_seen_year         INTEGER NOT NULL,
    last_confirmed_year     INTEGER NOT NULL,
    discontinued_year       INTEGER,
    status                  TEXT    DEFAULT 'ACTIVE',
    linked_categories_b     TEXT,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company, category_a)
);

CREATE TABLE IF NOT EXISTS adoption_portfolio_b (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    company                 TEXT    NOT NULL,
    category_b              TEXT    NOT NULL,
    product_name            TEXT,
    product_vendor          TEXT,
    granularity_level       TEXT    DEFAULT 'CATEGORY_ONLY',
    first_seen_year         INTEGER NOT NULL,
    last_confirmed_year     INTEGER NOT NULL,
    discontinued_year       INTEGER,
    status                  TEXT    DEFAULT 'ACTIVE',
    replaced_by_id          INTEGER,
    linked_categories_a     TEXT,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company, category_b, product_name, product_vendor)
);

CREATE TABLE IF NOT EXISTS adoption_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    company         TEXT    NOT NULL,
    year            INTEGER NOT NULL,
    dimension       TEXT    NOT NULL,
    category        TEXT    NOT NULL,
    product_name    TEXT,
    product_vendor  TEXT,
    event_type      TEXT    NOT NULL,
    reference_id    INTEGER,
    context         TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS adoption_links (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    company             TEXT    NOT NULL,
    category_a          TEXT    NOT NULL,
    category_b          TEXT    NOT NULL,
    product_name        TEXT,
    years               TEXT,
    reference_count     INTEGER DEFAULT 1,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company, category_a, category_b, product_name)
);

-- ---------------------------------------------------------------------------
-- Record migration
-- ---------------------------------------------------------------------------
INSERT OR IGNORE INTO schema_version (version, description)
VALUES (1, 'Initial schema: all core tables');
