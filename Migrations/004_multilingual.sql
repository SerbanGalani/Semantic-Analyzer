-- ============================================================================
-- AISA Migration 004 — Multilingual support (Schema v4)
-- Adds language detection and translation fields to ai_references tables.
-- Adds translation_log audit table.
-- ============================================================================

-- Add language + translation fields to ai_references_raw
ALTER TABLE ai_references_raw ADD COLUMN language TEXT DEFAULT 'en';
ALTER TABLE ai_references_raw ADD COLUMN text_translated TEXT;
ALTER TABLE ai_references_raw ADD COLUMN translation_source TEXT;

-- Add language + translation fields to ai_references_deduplicated
ALTER TABLE ai_references_deduplicated ADD COLUMN language TEXT DEFAULT 'en';
ALTER TABLE ai_references_deduplicated ADD COLUMN text_translated TEXT;
ALTER TABLE ai_references_deduplicated ADD COLUMN translation_source TEXT;

-- Add language field to processed_documents for document-level tracking
ALTER TABLE processed_documents ADD COLUMN language TEXT DEFAULT 'en';

-- Translation audit log
CREATE TABLE IF NOT EXISTS translation_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ref_id              INTEGER,
    company             TEXT,
    year                INTEGER,
    source_language     TEXT,
    original_text       TEXT,
    translated_text     TEXT,
    translation_engine  TEXT,      -- 'deepl' / 'google' / 'helsinki' / 'none'
    translated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status              TEXT,      -- 'ok' / 'error' / 'skipped'
    error_message       TEXT
);

CREATE INDEX IF NOT EXISTS idx_translation_log_company_year
    ON translation_log (company, year);

CREATE INDEX IF NOT EXISTS idx_ai_refs_raw_language
    ON ai_references_raw (language);

CREATE INDEX IF NOT EXISTS idx_ai_refs_dedup_language
    ON ai_references_deduplicated (language);

-- Record this migration
INSERT OR IGNORE INTO schema_version (version, description)
VALUES (4, 'Multilingual support: language detection + translation fields');
