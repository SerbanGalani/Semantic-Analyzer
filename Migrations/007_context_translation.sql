-- ============================================================================
-- Migration 007: Add context translation fields (Schema v7)
-- ============================================================================

ALTER TABLE ai_references_raw ADD COLUMN context_translated TEXT;
ALTER TABLE ai_references_deduplicated ADD COLUMN context_translated TEXT;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (7, '007_context_translation: add context_translated to raw and deduplicated reference tables');
