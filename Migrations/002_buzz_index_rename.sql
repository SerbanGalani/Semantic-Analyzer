-- =============================================================================
-- AISA - AI Semantic Analyzer
-- Migration 002: Rename ai_adoption_index -> ai_buzz_index  (Schema v1 -> v2)
-- =============================================================================
-- SQLite ALTER TABLE ... RENAME COLUMN requires SQLite 3.25+.
-- We use the safe recreate-copy-swap pattern.
-- =============================================================================

-- Step 1: Create new table with the correct column name
CREATE TABLE IF NOT EXISTS adoption_index_new (
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
    ai_buzz_index           REAL    DEFAULT 0.0,
    rank_in_year            INTEGER DEFAULT 0,
    total_refs              INTEGER DEFAULT 0,
    total_pages             INTEGER DEFAULT 0,
    categories_used         INTEGER DEFAULT 0,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company, year)
);

-- Step 2: Copy data from old table, mapping ai_adoption_index -> ai_buzz_index
INSERT OR IGNORE INTO adoption_index_new
    (id, company, year, position, industry, sector, country,
     volume_index, depth_index, breadth_index, tone_index,
     specificity_index, forward_looking_index, salience_index,
     ai_buzz_index,
     rank_in_year, total_refs, total_pages, categories_used, created_at)
SELECT
    id, company, year, position, industry, sector, country,
    volume_index, depth_index, breadth_index, tone_index,
    specificity_index, forward_looking_index, salience_index,
    ai_adoption_index,
    rank_in_year, total_refs, total_pages, categories_used, created_at
FROM adoption_index;

-- Step 3: Drop old table and rename new one
DROP TABLE adoption_index;
ALTER TABLE adoption_index_new RENAME TO adoption_index;

-- Step 4: Recreate indexes
CREATE INDEX IF NOT EXISTS idx_adopt_company ON adoption_index(company);
CREATE INDEX IF NOT EXISTS idx_adopt_year    ON adoption_index(year);

-- Step 5: Record migration
INSERT OR IGNORE INTO schema_version (version, description)
VALUES (2, 'Rename ai_adoption_index to ai_buzz_index in adoption_index table');
