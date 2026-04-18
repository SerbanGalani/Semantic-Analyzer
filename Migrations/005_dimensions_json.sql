-- ============================================================================
-- Migration 005: Add dimensions_json column for N-dimensional taxonomy support
-- ============================================================================
-- Adds dimensions_json TEXT to both reference tables.
-- Stores the full multi-dimensional classification as a JSON blob, e.g.:
--   {"Application": ["D1_Partner_Onboarding_Access", 0.87],
--    "Technology":  ["T3_Platform_Integration", 0.75],
--    "Governance":  ["G2_Data_Sharing_Agreement", 0.60]}
-- This allows any taxonomy with more than 2 dimensions (e.g. tri-axial D/T/G)
-- to persist all classification results without changing the existing
-- category_a / category_b backward-compat columns.
-- ============================================================================

ALTER TABLE ai_references_raw
    ADD COLUMN dimensions_json TEXT DEFAULT '';

ALTER TABLE ai_references_deduplicated
    ADD COLUMN dimensions_json TEXT DEFAULT '';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (5, '005_dimensions_json: N-dimensional taxonomy JSON blob on both reference tables');
