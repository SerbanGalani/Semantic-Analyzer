-- Migration 009: add missing columns to ai_references_deduplicated
--
-- insert_deduplicated_reference() in 02_db.py inserts doc_type, language,
-- and dimensions_json, but these columns were never added to the deduplicated
-- table (they existed only on ai_references_raw). The missing columns caused
-- every INSERT to raise an OperationalError that was silently swallowed in
-- _populate_dedup_table(), leaving written=0 for all dedupe runs.
--
-- The migration system handles "duplicate column name" gracefully, so this
-- migration is safe to apply against databases that already have these columns.

ALTER TABLE ai_references_deduplicated ADD COLUMN doc_type        TEXT DEFAULT '';
ALTER TABLE ai_references_deduplicated ADD COLUMN language        TEXT DEFAULT 'en';
ALTER TABLE ai_references_deduplicated ADD COLUMN dimensions_json TEXT DEFAULT '';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (9, 'add doc_type, language, dimensions_json to ai_references_deduplicated');
