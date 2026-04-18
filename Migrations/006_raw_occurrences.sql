-- ============================================================================
-- Migration 006: Preserve all raw occurrences (remove UNIQUE constraint)
-- ============================================================================
-- Problem fixed:
--   ai_references_raw was created with UNIQUE(company, year, doc_type, page, text).
--   That compressed multiple real occurrences into one row, causing:
--     - missing raw rows in export
--     - context from later detections to be lost
--     - occurrence_count to rise while raw Excel looked incomplete
--
-- Solution:
--   Rebuild ai_references_raw WITHOUT the UNIQUE constraint.
--   Keep a normal lookup index on (company, year, doc_type, page, text)
--   for performance, but allow multiple raw rows.
-- ============================================================================

CREATE TABLE IF NOT EXISTS ai_references_raw_new (
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
    created_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    language                    TEXT    DEFAULT 'en',
    text_translated             TEXT,
    translation_source          TEXT,
    dimensions_json             TEXT    DEFAULT ''
);

INSERT INTO ai_references_raw_new (
    id, company, year, position, industry, sector, country,
    doc_type, page, source, page_count, text, context, detection_method,
    category, category_a, confidence_a, category_b, confidence_b,
    sentiment, sentiment_score, sentiment_confidence, semantic_score, category_confidence,
    reference_strength, confidence_score, confidence_reasons, robotics_type, rpa_type,
    product_name, product_vendor, granularity_level, event_type, adoption_memory_processed,
    ref_hash, occurrence_count, created_at, language, text_translated, translation_source, dimensions_json
)
SELECT
    id, company, year, position, industry, sector, country,
    doc_type, page, source, page_count, text, context, detection_method,
    category, category_a, confidence_a, category_b, confidence_b,
    sentiment, sentiment_score, sentiment_confidence, semantic_score, category_confidence,
    reference_strength, confidence_score, confidence_reasons, robotics_type, rpa_type,
    product_name, product_vendor, granularity_level, event_type, adoption_memory_processed,
    ref_hash, occurrence_count, created_at, language, text_translated, translation_source, dimensions_json
FROM ai_references_raw;

DROP TABLE ai_references_raw;
ALTER TABLE ai_references_raw_new RENAME TO ai_references_raw;

CREATE INDEX IF NOT EXISTS idx_raw_company ON ai_references_raw(company);
CREATE INDEX IF NOT EXISTS idx_raw_year    ON ai_references_raw(year);
CREATE INDEX IF NOT EXISTS idx_raw_hash    ON ai_references_raw(ref_hash);
CREATE INDEX IF NOT EXISTS idx_raw_compact ON ai_references_raw(company, year, doc_type, page, text);
CREATE INDEX IF NOT EXISTS idx_raw_cat_a   ON ai_references_raw(category_a);
CREATE INDEX IF NOT EXISTS idx_raw_cat_b   ON ai_references_raw(category_b);
CREATE INDEX IF NOT EXISTS idx_ai_refs_raw_language ON ai_references_raw(language);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (6, '006_raw_occurrences: rebuild ai_references_raw without UNIQUE(company, year, doc_type, page, text)');
