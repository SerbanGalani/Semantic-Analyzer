"""
===============================================================================
AISA - AI Semantic Analyzer
02_db.py - Unified Database Manager with migration system
===============================================================================

Single DatabaseManager class that consolidates:
    - Core table creation (previously in module1.DatabaseManager)
    - Document tracking (previously in menu.ExtendedDatabaseManager)
    - Reference deduplication tracking
    - Migration system (replaces try/except ALTER TABLE pattern)

Migration system:
    - Reads SQL files from migrations/ folder
    - Applies only pending migrations (tracks via schema_version table)
    - Never re-applies a migration
    - Hard-fails on missing migration files (no silent fallbacks)

Performance settings (SQLite):
    - WAL journal mode for concurrent reads
    - NORMAL synchronous (safe + faster than FULL)
    - Batch commits (commit every N inserts, not per insert)

CHANGELOG:
    v1.0.1 (2026-04) - processed_documents optional metrics support
        - mark_document_processed now persists word_count / sentence_count /
          paragraph_count when the columns exist
        - safe runtime fallback for older DB schemas without these columns
        - get_processing_stats aggregates optional text metrics when available
    v1.0.0 (2026) - AISA rewrite
        - Merged DatabaseManager + ExtendedDatabaseManager into one class
        - Replaced try/except ALTER TABLE with SQL migration files
        - Added WAL + batch commit support
        - Full English docstrings

Author: TeRa0
License: MIT
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import sys as _sys, os as _os, importlib as _il
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from version import SCHEMA_VERSION

_m1 = _il.import_module('01_models')
AIBuzzIndex     = _m1.AIBuzzIndex
AIReference     = _m1.AIReference
DocumentResult  = _m1.DocumentResult
logger          = _m1.logger

# Folder containing numbered SQL migration files
MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# How many inserts before an automatic commit (pipeline performance)
BATCH_COMMIT_SIZE = 100


# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """
    Unified SQLite database manager for AISA.

    Lifecycle:
        db = DatabaseManager("aisa_results.db")
        db.apply_migrations()       # run once at startup
        ...use db...
        db.close()

    Or use as context manager:
        with DatabaseManager("aisa_results.db") as db:
            db.apply_migrations()
            ...
    """

    def __init__(self, db_path: str, batch_commit_size: int = BATCH_COMMIT_SIZE):
        self.db_path = db_path
        self.batch_commit_size = batch_commit_size
        self._insert_counter = 0
        self.conn: Optional[sqlite3.Connection] = None
        self._connect()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self):
        """Open connection and apply performance PRAGMAs."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row

            # Performance settings
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            self.conn.execute("PRAGMA foreign_keys=ON;")
            self.conn.execute("PRAGMA cache_size=-32000;")   # ~32 MB cache

            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def close(self):
        """Commit any pending inserts and close the connection."""
        if self.conn:
            self.conn.commit()
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ------------------------------------------------------------------
    # Migration system
    # ------------------------------------------------------------------

    def apply_migrations(self):
        """
        Apply all pending SQL migrations in order.

        Reads files from migrations/ named NNN_description.sql.
        Only applies migrations with version > current schema_version.
        Raises RuntimeError if migrations folder or files are missing.
        """
        # Guard: migrations folder must exist alongside this file
        if not MIGRATIONS_DIR.exists():
            raise RuntimeError(
                f"Migrations directory not found: {MIGRATIONS_DIR}\n"
                "Ensure the 'migrations/' folder is present next to 02_db.py "
                "and contains 001_init.sql, 002_buzz_index_rename.sql, 004_multilingual.sql, 007_context_translation.sql."
            )

        expected = [
            "001_init.sql", "002_buzz_index_rename.sql", "003_aiti_tables.sql",
            "004_multilingual.sql", "005_dimensions_json.sql", "006_raw_occurrences.sql",
            "007_context_translation.sql", "008_embedding_cache.sql",
            "009_dedup_missing_cols.sql",
        ]
        missing  = [f for f in expected if not (MIGRATIONS_DIR / f).exists()]
        if missing:
            raise RuntimeError(
                f"Required migration files missing from {MIGRATIONS_DIR}: "
                f"{missing}\nCannot initialize database."
            )
        # Ensure schema_version table exists (bootstrap)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version     INTEGER PRIMARY KEY,
                applied_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
        """)
        self.conn.commit()

        current = self._get_current_schema_version()

        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        max_file_version = 0
        for mf in migration_files:
            try:
                max_file_version = max(max_file_version, int(mf.stem.split("_")[0]))
            except (ValueError, IndexError):
                continue

        target_version = max(SCHEMA_VERSION, max_file_version)
        logger.info(f"Current schema version: {current} | Target: {target_version}")

        if current >= target_version:
            logger.info("Schema is up to date.")
            return

        if not migration_files:
            raise RuntimeError(
                f"No migration files found in {MIGRATIONS_DIR}. "
                "Cannot initialize database."
            )
        if not migration_files:
            raise RuntimeError(
                f"No migration files found in {MIGRATIONS_DIR}. "
                "Cannot initialize database."
            )

        for mf in migration_files:
            # Extract version number from filename (e.g. 001_init.sql → 1)
            try:
                file_version = int(mf.stem.split("_")[0])
            except (ValueError, IndexError):
                logger.warning(f"Skipping unrecognised migration file: {mf.name}")
                continue

            if file_version <= current:
                continue  # Already applied

            logger.info(f"Applying migration {mf.name} ...")
            sql = mf.read_text(encoding="utf-8")

            try:
                # Execute each statement individually so that pre-existing
                # columns (ALTER TABLE ADD COLUMN on an already-migrated DB)
                # are silently skipped instead of aborting the whole migration.
                statements = [s.strip() for s in sql.split(";") if s.strip()]
                for stmt in statements:
                    try:
                        self.conn.execute(stmt)
                    except sqlite3.OperationalError as stmt_err:
                        if "duplicate column name" in str(stmt_err).lower():
                            logger.warning(
                                f"  {mf.name}: column already exists, skipping: "
                                f"{stmt[:80].replace(chr(10), ' ')}"
                            )
                        else:
                            raise
                self.conn.commit()
                logger.info(f"Migration {mf.name} applied successfully.")
            except sqlite3.Error as e:
                raise RuntimeError(
                    f"Migration {mf.name} failed: {e}\n"
                    "Fix the SQL file and restart."
                ) from e

        final = self._get_current_schema_version()
        logger.info(f"Schema updated to version {final}.")

    def _get_current_schema_version(self) -> int:
        """Return the highest applied migration version, or 0 if none."""
        try:
            row = self.conn.execute(
                "SELECT MAX(version) FROM schema_version"
            ).fetchone()
            return row[0] if row and row[0] is not None else 0
        except sqlite3.Error:
            return 0

    def _get_table_columns(self, table_name: str) -> Set[str]:
        """Return the current SQLite column names for a table (empty set if missing)."""
        try:
            rows = self.conn.execute(
                f"PRAGMA table_info({table_name})"
            ).fetchall()
            return {r[1] for r in rows}
        except sqlite3.Error:
            return set()

    def _first_existing_column(self, table_name: str, *candidates: str) -> Optional[str]:
        """Return the first candidate column that exists in the given table."""
        cols = self._get_table_columns(table_name)
        for col in candidates:
            if col in cols:
                return col
        return None

    # ------------------------------------------------------------------
    # Document tracking
    # ------------------------------------------------------------------

    def is_document_processed(self, source: str) -> bool:
        """Return True if the document (by source path/name) was already processed."""
        row = self.conn.execute(
            "SELECT id FROM processed_documents WHERE source = ?", (source,)
        ).fetchone()
        return row is not None

    def get_processed_documents(self) -> Set[str]:
        """Return the set of all processed document source paths."""
        rows = self.conn.execute(
            "SELECT source FROM processed_documents"
        ).fetchall()
        return {r[0] for r in rows}

    def mark_document_processed(
        self,
        doc_result: DocumentResult,
        refs_found: int,
        file_hash: Optional[str] = None,
        text_status: str = "valid",
    ):
        """
        Insert or update the processed_documents record for a document.

        Persists optional per-document text metrics when the current DB schema
        provides matching columns. Older schemas remain fully supported.

        Args:
            doc_result:   Completed DocumentResult from the pipeline.
            refs_found:   Number of references detected.
            file_hash:    Optional SHA-256 of the source file.
            text_status:  One of: valid | corrupted_ocr_success |
                          corrupted_ocr_failed | empty | error
        """
        cols = self._get_table_columns("processed_documents")
        existing = self.conn.execute(
            "SELECT id, process_count FROM processed_documents WHERE source = ?",
            (doc_result.source,),
        ).fetchone()

        total_pages_col    = self._first_existing_column("processed_documents", "total_pages", "pages")
        text_length_col    = self._first_existing_column("processed_documents", "text_length")
        word_count_col     = self._first_existing_column("processed_documents", "word_count")
        sentence_count_col = self._first_existing_column("processed_documents", "sentence_count")
        paragraph_count_col = self._first_existing_column("processed_documents", "paragraph_count")
        updated_at_col     = self._first_existing_column("processed_documents", "updated_at", "last_processed_at")

        if existing:
            set_parts = []
            params: List[object] = []

            if updated_at_col:
                set_parts.append(f"{updated_at_col} = CURRENT_TIMESTAMP")
            if "process_count" in cols:
                set_parts.append("process_count = process_count + 1")
            if "refs_found" in cols:
                set_parts.append("refs_found = ?")
                params.append(refs_found)
            if "file_hash" in cols:
                set_parts.append("file_hash = COALESCE(?, file_hash)")
                params.append(file_hash)
            if "text_status" in cols:
                set_parts.append("text_status = ?")
                params.append(text_status)
            if total_pages_col:
                set_parts.append(f"{total_pages_col} = ?")
                params.append(doc_result.total_pages)
            if text_length_col:
                set_parts.append(f"{text_length_col} = ?")
                params.append(doc_result.text_length)
            if word_count_col:
                set_parts.append(f"{word_count_col} = ?")
                params.append(getattr(doc_result, "word_count", 0))
            if sentence_count_col:
                set_parts.append(f"{sentence_count_col} = ?")
                params.append(getattr(doc_result, "sentence_count", 0))
            if paragraph_count_col:
                set_parts.append(f"{paragraph_count_col} = ?")
                params.append(getattr(doc_result, "paragraph_count", 0))

            if set_parts:
                params.append(doc_result.source)
                self.conn.execute(
                    f"UPDATE processed_documents SET {', '.join(set_parts)} WHERE source = ?",
                    tuple(params),
                )
            logger.debug(
                f"Updated processed_documents: {doc_result.source} "
                f"(run #{existing['process_count'] + 1}, status={text_status})"
            )
        else:
            insert_cols: List[str] = [
                "source", "company", "year", "position", "industry", "doc_type",
            ]
            insert_vals: List[object] = [
                doc_result.source, doc_result.company, doc_result.year,
                doc_result.position, doc_result.industry, doc_result.doc_type,
            ]

            if total_pages_col:
                insert_cols.append(total_pages_col)
                insert_vals.append(doc_result.total_pages)
            if text_length_col:
                insert_cols.append(text_length_col)
                insert_vals.append(doc_result.text_length)
            if "refs_found" in cols:
                insert_cols.append("refs_found")
                insert_vals.append(refs_found)
            if "file_hash" in cols:
                insert_cols.append("file_hash")
                insert_vals.append(file_hash)
            if "text_status" in cols:
                insert_cols.append("text_status")
                insert_vals.append(text_status)
            if word_count_col:
                insert_cols.append(word_count_col)
                insert_vals.append(getattr(doc_result, "word_count", 0))
            if sentence_count_col:
                insert_cols.append(sentence_count_col)
                insert_vals.append(getattr(doc_result, "sentence_count", 0))
            if paragraph_count_col:
                insert_cols.append(paragraph_count_col)
                insert_vals.append(getattr(doc_result, "paragraph_count", 0))

            placeholders = ", ".join(["?"] * len(insert_cols))
            sql = (
                f"INSERT INTO processed_documents ({', '.join(insert_cols)}) "
                f"VALUES ({placeholders})"
            )
            self.conn.execute(sql, tuple(insert_vals))
            logger.debug(
                f"New document processed: {doc_result.source} (status={text_status})"
            )

        self.conn.commit()

    def get_processing_stats(self) -> Dict:
        """Return aggregate statistics about processed documents."""
        stats: Dict = {}
        cols = self._get_table_columns("processed_documents")

        stats["total_documents_processed"] = self.conn.execute(
            "SELECT COUNT(*) FROM processed_documents"
        ).fetchone()[0]

        if "process_count" in cols:
            stats["documents_reprocessed"] = self.conn.execute(
                "SELECT COUNT(*) FROM processed_documents WHERE process_count > 1"
            ).fetchone()[0]
            stats["total_processing_runs"] = (
                self.conn.execute(
                    "SELECT SUM(process_count) FROM processed_documents"
                ).fetchone()[0] or 0
            )
        else:
            stats["documents_reprocessed"] = 0
            stats["total_processing_runs"] = stats["total_documents_processed"]

        row = self.conn.execute(
            "SELECT COUNT(*), SUM(COALESCE(occurrence_count, 1)) FROM ai_references_raw"
        ).fetchone()
        stats["unique_references"] = row[0] or 0
        stats["total_occurrences"] = row[1] or 0

        if "refs_found" in cols:
            stats["documents_without_refs"] = self.conn.execute(
                "SELECT COUNT(*) FROM processed_documents WHERE refs_found = 0"
            ).fetchone()[0]
        else:
            stats["documents_without_refs"] = 0

        if "text_status" in cols:
            rows = self.conn.execute(
                "SELECT text_status, COUNT(*) FROM processed_documents GROUP BY text_status"
            ).fetchall()
            stats["by_text_status"] = {(r[0] or "unknown"): r[1] for r in rows}
            stats["documents_with_text_issues"] = self.conn.execute(
                """
                SELECT COUNT(*) FROM processed_documents
                WHERE text_status IN ('corrupted_ocr_failed', 'empty', 'error')
                """
            ).fetchone()[0]
        else:
            stats["by_text_status"] = {}
            stats["documents_with_text_issues"] = 0

        total_pages_col = self._first_existing_column("processed_documents", "total_pages", "pages")
        if total_pages_col:
            stats["total_pages"] = (
                self.conn.execute(
                    f"SELECT SUM(COALESCE({total_pages_col}, 0)) FROM processed_documents"
                ).fetchone()[0] or 0
            )
        else:
            stats["total_pages"] = 0

        if "text_length" in cols:
            stats["total_text_chars"] = (
                self.conn.execute(
                    "SELECT SUM(COALESCE(text_length, 0)) FROM processed_documents"
                ).fetchone()[0] or 0
            )
        else:
            stats["total_text_chars"] = 0

        if "word_count" in cols:
            stats["total_words"] = (
                self.conn.execute(
                    "SELECT SUM(COALESCE(word_count, 0)) FROM processed_documents"
                ).fetchone()[0] or 0
            )
        else:
            stats["total_words"] = 0

        if "sentence_count" in cols:
            stats["total_sentences"] = (
                self.conn.execute(
                    "SELECT SUM(COALESCE(sentence_count, 0)) FROM processed_documents"
                ).fetchone()[0] or 0
            )
        else:
            stats["total_sentences"] = 0

        if "paragraph_count" in cols:
            stats["total_paragraphs"] = (
                self.conn.execute(
                    "SELECT SUM(COALESCE(paragraph_count, 0)) FROM processed_documents"
                ).fetchone()[0] or 0
            )
        else:
            stats["total_paragraphs"] = 0

        return stats

    # ------------------------------------------------------------------
    # Reference insertion
    # ------------------------------------------------------------------

    def insert_reference(self, ref: AIReference) -> Tuple[bool, bool]:
        """
        Insert a new reference or update an existing one.

        Deduplication key: (company, year, doc_type, page, text).
        If the record exists, increments occurrence_count and updates
        scores/confidence if the new values are better.

        Returns:
            (is_new, was_updated): tuple of booleans.
        """
        ref_hash = self._hash_reference(ref)

        existing = self.conn.execute(
            """
            SELECT id, occurrence_count FROM ai_references_raw
            WHERE company = ? AND year = ? AND doc_type = ? AND page = ? AND text = ?
            """,
            (ref.company, ref.year, ref.doc_type, ref.page, ref.text),
        ).fetchone()

        if existing:
            ref_id, current_count = existing["id"], existing["occurrence_count"] or 1
            self.conn.execute(
                """
                UPDATE ai_references_raw SET
                    occurrence_count  = ?,
                    ref_hash          = COALESCE(ref_hash, ?),
                    sentiment_score   = ?,
                    semantic_score    = ?,
                    reference_strength = COALESCE(?, reference_strength),
                    confidence_score  = CASE
                        WHEN ? > COALESCE(confidence_score, 0) THEN ?
                        ELSE COALESCE(confidence_score, 0) END,
                    confidence_reasons = CASE
                        WHEN ? != '' THEN ? ELSE COALESCE(confidence_reasons, '') END,
                    category_a = CASE
                        WHEN COALESCE(category_a,'') = '' THEN ? ELSE category_a END,
                    confidence_a = CASE
                        WHEN ? > COALESCE(confidence_a, 0) THEN ? ELSE COALESCE(confidence_a, 0) END,
                    category_b = CASE
                        WHEN COALESCE(category_b,'') = '' THEN ? ELSE category_b END,
                    confidence_b = CASE
                        WHEN ? > COALESCE(confidence_b, 0) THEN ? ELSE COALESCE(confidence_b, 0) END,
                    dimensions_json = CASE
                        WHEN ? != '' THEN ? ELSE COALESCE(dimensions_json, '') END
                WHERE id = ?
                """,
                (
                    current_count + 1,
                    ref_hash,
                    ref.sentiment_score,
                    ref.semantic_score,
                    ref.reference_strength,
                    ref.confidence_score, ref.confidence_score,
                    ref.confidence_reasons, ref.confidence_reasons,
                    ref.category_a,
                    ref.confidence_a, ref.confidence_a,
                    ref.category_b,
                    ref.confidence_b, ref.confidence_b,
                    getattr(ref, "dimensions_json", ""), getattr(ref, "dimensions_json", ""),
                    ref_id,
                ),
            )
            self._auto_commit()
            return (False, True)

        # New reference
        try:
            self.conn.execute(
                """
                INSERT INTO ai_references_raw (
                    company, year, position, industry, sector, country,
                    doc_type, page, text, context,
                    category, category_a, confidence_a, category_b, confidence_b,
                    sentiment, sentiment_score, sentiment_confidence,
                    semantic_score, category_confidence,
                    reference_strength, confidence_score, confidence_reasons,
                    robotics_type, rpa_type,
                    product_name, product_vendor, granularity_level,
                    event_type, adoption_memory_processed,
                    ref_hash, occurrence_count, page_count, source,
                    language, text_translated, translation_source,
                    dimensions_json
                ) VALUES (
                    ?,?,?,?,?,?,  ?,?,?,?,
                    ?,?,?,?,?,   ?,?,?,
                    ?,?,         ?,?,?,
                    ?,?,         ?,?,?,
                    ?,?,         ?,1,?,?,
                    ?,?,?,       ?
                )
                """,
                (
                    ref.company, ref.year, ref.position,
                    ref.industry, ref.sector, ref.country,
                    ref.doc_type, ref.page, ref.text, ref.context,
                    ref.category, ref.category_a, ref.confidence_a,
                    ref.category_b, ref.confidence_b,
                    ref.sentiment, ref.sentiment_score, ref.sentiment_confidence,
                    ref.semantic_score, ref.category_confidence,
                    ref.reference_strength, ref.confidence_score, ref.confidence_reasons,
                    ref.robotics_type, ref.rpa_type,
                    ref.product_name, ref.product_vendor, ref.granularity_level,
                    ref.event_type, int(ref.adoption_memory_processed),
                    ref_hash, ref.page_count, ref.source,
                    # EN: text_translated = original text, source = "original"
                    # non-EN: NULL → filled later by 15_translate.py
                    getattr(ref, "language", "en"),
                    ref.text if getattr(ref, "language", "en") == "en" else None,
                    "original" if getattr(ref, "language", "en") == "en" else None,
                    getattr(ref, "dimensions_json", ""),
                ),
            )
            self._auto_commit()
            return (True, False)

        except sqlite3.IntegrityError:
            # Race condition - just increment
            self.conn.execute(
                """
                UPDATE ai_references_raw
                SET occurrence_count = COALESCE(occurrence_count, 1) + 1,
                    ref_hash = COALESCE(ref_hash, ?)
                WHERE company=? AND year=? AND doc_type=? AND page=? AND text=?
                """,
                (ref_hash, ref.company, ref.year, ref.doc_type, ref.page, ref.text),
            )
            self._auto_commit()
            return (False, True)

    def insert_deduplicated_reference(self, dedup: Dict):
        """
        Upsert a deduplicated reference cluster.

        Args:
            dedup: Dict produced by SemanticDeduplicator._create_deduplicated_ref()
        """
        self.conn.execute(
            """
            INSERT INTO ai_references_deduplicated (
                company, year, position, industry, sector, country,
                doc_type, language,
                text, context, category, category_a, confidence_a,
                category_b, confidence_b,
                sources, pages, doc_count, total_occurrences,
                avg_sentiment_score, avg_semantic_score,
                max_confidence_score, original_refs, dimensions_json
            ) VALUES (?,?,?,?,?,?,?,?,  ?,?,?,?,?,  ?,?,  ?,?,?,?,  ?,?,  ?,?,?)
            ON CONFLICT(company, year, context) DO UPDATE SET
                total_occurrences    = excluded.total_occurrences,
                avg_sentiment_score  = excluded.avg_sentiment_score,
                avg_semantic_score   = excluded.avg_semantic_score,
                max_confidence_score = excluded.max_confidence_score,
                sources              = excluded.sources,
                pages                = excluded.pages,
                dimensions_json      = CASE
                    WHEN excluded.dimensions_json != '' THEN excluded.dimensions_json
                    ELSE COALESCE(dimensions_json, '') END
            """,
            (
                dedup.get("company"), dedup.get("year"), dedup.get("position"),
                dedup.get("industry"), dedup.get("sector"), dedup.get("country"),
                dedup.get("doc_type", ""), dedup.get("language", "en"),
                dedup.get("text"), dedup.get("context"),
                dedup.get("category"), dedup.get("category_a"), dedup.get("confidence_a"),
                dedup.get("category_b"), dedup.get("confidence_b"),
                dedup.get("sources"), dedup.get("pages"),
                dedup.get("doc_count"), dedup.get("total_occurrences"),
                dedup.get("avg_sentiment_score"), dedup.get("avg_semantic_score"),
                dedup.get("max_confidence_score"),
                json.dumps(dedup.get("original_refs", [])),
                dedup.get("dimensions_json", ""),
            ),
        )
        self._auto_commit()

    def insert_buzz_index(self, idx: AIBuzzIndex):
        """Upsert an AIBuzzIndex record into adoption_index table."""
        self.conn.execute(
            """
            INSERT INTO adoption_index (
                company, year, position, industry, sector, country,
                volume_index, depth_index, breadth_index,
                tone_index, specificity_index, forward_looking_index, salience_index,
                ai_buzz_index, rank_in_year,
                total_refs, total_pages, categories_used
            ) VALUES (?,?,?,?,?,?,  ?,?,?,?,?,?,?,  ?,?,  ?,?,?)
            ON CONFLICT(company, year) DO UPDATE SET
                volume_index          = excluded.volume_index,
                depth_index           = excluded.depth_index,
                breadth_index         = excluded.breadth_index,
                tone_index            = excluded.tone_index,
                specificity_index     = excluded.specificity_index,
                forward_looking_index = excluded.forward_looking_index,
                salience_index        = excluded.salience_index,
                ai_buzz_index         = excluded.ai_buzz_index,
                rank_in_year          = excluded.rank_in_year,
                total_refs            = excluded.total_refs,
                total_pages           = excluded.total_pages,
                categories_used       = excluded.categories_used
            """,
            (
                idx.company, idx.year, idx.position,
                idx.industry, idx.sector, idx.country,
                idx.volume_index, idx.depth_index, idx.breadth_index,
                idx.tone_index, idx.specificity_index, idx.forward_looking_index,
                idx.salience_index, idx.ai_buzz_index, idx.rank_in_year,
                idx.total_refs, idx.total_pages, idx.categories_used,
            ),
        )
        self._auto_commit()

    def update_rankings(self, year: int):
        """
        Recalculate rank_in_year for all companies in a given year,
        ordered by ai_buzz_index DESC.
        """
        rows = self.conn.execute(
            """
            SELECT id FROM adoption_index
            WHERE year = ?
            ORDER BY ai_buzz_index DESC
            """,
            (year,),
        ).fetchall()

        for rank, row in enumerate(rows, start=1):
            self.conn.execute(
                "UPDATE adoption_index SET rank_in_year = ? WHERE id = ?",
                (rank, row["id"]),
            )
        self.conn.commit()
        logger.info(f"Rankings updated for year {year} ({len(rows)} companies).")

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_references(
        self,
        company: Optional[str] = None,
        year: Optional[int] = None,
        deduplicated: bool = False,
    ) -> List[Dict]:
        """
        Fetch raw or deduplicated references, optionally filtered.

        Args:
            company:      Filter by company name (exact match).
            year:         Filter by year.
            deduplicated: If True, query ai_references_deduplicated.

        Returns:
            List of row dicts.
        """
        table = "ai_references_deduplicated" if deduplicated else "ai_references_raw"
        conditions, params = [], []

        if company:
            conditions.append("company = ?")
            params.append(company)
        if year:
            conditions.append("year = ?")
            params.append(year)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self.conn.execute(
            f"SELECT * FROM {table} {where}", params
        ).fetchall()

        return [dict(r) for r in rows]

    def get_buzz_indices(
        self,
        year: Optional[int] = None,
        company: Optional[str] = None,
    ) -> List[Dict]:
        """Fetch AI Buzz Index records, optionally filtered."""
        conditions, params = [], []
        if year:
            conditions.append("year = ?")
            params.append(year)
        if company:
            conditions.append("company = ?")
            params.append(company)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self.conn.execute(
            f"SELECT * FROM adoption_index {where} ORDER BY year, rank_in_year",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_industry_buzz(
        self,
        year: Optional[int] = None,
    ) -> List[Dict]:
        """Fetch industry-level buzz index records, optionally filtered by year."""
        conditions, params = [], []
        if year:
            conditions.append("year = ?")
            params.append(year)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self.conn.execute(
            f"""
            SELECT * FROM adoption_index_industry
            {where}
            ORDER BY year, rank_among_industries
            """,
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Embedding cache (Stage 2 acceleration — migration 008)
    # ------------------------------------------------------------------

    def get_embedding(self, text: str, model_name: str):
        """
        Retrieve a cached embedding vector for (text, model_name).

        Returns:
            numpy float32 array if found, None otherwise.
        """
        import numpy as np
        text_hash = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
        try:
            row = self.conn.execute(
                "SELECT embedding FROM embedding_cache WHERE text_hash = ? AND model_name = ?",
                (text_hash, model_name),
            ).fetchone()
            if row is None:
                return None
            return np.frombuffer(row[0], dtype="float32")
        except Exception:
            return None

    def save_embedding(self, text: str, model_name: str, embedding) -> None:
        """
        Persist an embedding vector for (text, model_name).

        Args:
            text:       The candidate fragment text used as cache key.
            model_name: Name of the SentenceTransformer model.
            embedding:  numpy float32 array (shape: [dim]).
        """
        text_hash = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO embedding_cache (text_hash, model_name, embedding) "
                "VALUES (?, ?, ?)",
                (text_hash, model_name, embedding.tobytes()),
            )
            self._auto_commit()
        except Exception:
            pass  # Cache write failure is non-fatal — next run will re-encode

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hash_reference(self, ref: AIReference) -> str:
        """
        Generate a stable SHA-256 hash for a reference.

        Hash inputs: company | year | doc_type | page | text[:200]
        Returns first 32 hex characters.
        """
        normalized = " ".join(ref.text.split())[:200]
        unique_str = f"{ref.company}|{ref.year}|{ref.doc_type}|{ref.page}|{normalized}"
        return hashlib.sha256(unique_str.encode("utf-8")).hexdigest()[:32]

    def _auto_commit(self):
        """Commit every BATCH_COMMIT_SIZE inserts instead of per-insert."""
        self._insert_counter += 1
        if self._insert_counter >= self.batch_commit_size:
            self.conn.commit()
            self._insert_counter = 0

    def commit(self):
        """Force an immediate commit (call at end of each document batch)."""
        self.conn.commit()
        self._insert_counter = 0


# ============================================================================
# SMOKE TEST
# ============================================================================

if __name__ == "__main__":
    import tempfile, os
    from version import get_version_string
    AnalyzerConfig = _m1.AnalyzerConfig

    print(get_version_string())
    print()

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test_aisa.db")

        with DatabaseManager(db_path) as db:
            # Apply migrations
            db.apply_migrations()
            print("  apply_migrations()     OK")

            # Check schema version
            v = db._get_current_schema_version()
            print(f"  schema_version         {v}")
            assert v == SCHEMA_VERSION, f"Expected {SCHEMA_VERSION}, got {v}"

            # Insert a reference
            ref = AIReference(
                company="TestCorp", year=2023, position=1,
                industry="Technology", sector="Software", country="USA",
                doc_type="Annual Report",
                text="We use machine learning extensively.",
                context="Our platform leverages machine learning.",
                page=42, category="B1_Traditional_ML",
                detection_method="pattern|hard",
                sentiment="positive", sentiment_score=0.82,
                semantic_score=0.0, source="TestCorp_2023.pdf",
                category_a="A1_Product_Innovation",
                category_b="B1_Traditional_ML",
                confidence_a=0.9, confidence_b=0.85,
                reference_strength="strong", confidence_score=0.88,
            )
            is_new, _ = db.insert_reference(ref)
            assert is_new, "First insert should be new"
            print("  insert_reference()     OK (new)")

            # Insert duplicate - should update
            _, was_updated = db.insert_reference(ref)
            assert was_updated, "Second insert should update"
            print("  insert_reference()     OK (update)")

            # Document tracking
            doc = DocumentResult(
                company="TestCorp", year=2023, position=1,
                industry="Technology", sector="Software", country="USA",
                doc_type="Annual Report", source="TestCorp_2023.pdf",
                total_pages=120, text_length=250000,
                word_count=42000, sentence_count=1800, paragraph_count=350,
            )
            db.mark_document_processed(doc, refs_found=1, text_status="valid")
            assert db.is_document_processed("TestCorp_2023.pdf")
            print("  mark_document_processed() OK")

            # Stats
            stats = db.get_processing_stats()
            assert stats["total_documents_processed"] == 1
            assert stats["unique_references"] == 1
            print(
                "  get_processing_stats() OK | "
                f"docs={stats['total_documents_processed']} "
                f"words={stats.get('total_words', 0)} "
                f"sent={stats.get('total_sentences', 0)}"
            )

            # Buzz index
            idx = AIBuzzIndex(
                company="TestCorp", year=2023, position=1,
                industry="Technology", sector="Software", country="USA",
                ai_buzz_index=0.634, total_refs=1,
                volume_index=0.65, depth_index=0.72,
                breadth_index=0.48, tone_index=0.80,
                specificity_index=0.55, forward_looking_index=0.40,
                salience_index=0.60,
            )
            db.insert_buzz_index(idx)
            db.update_rankings(2023)
            indices = db.get_buzz_indices(year=2023)
            assert len(indices) == 1
            assert indices[0]["rank_in_year"] == 1
            print(f"  insert_buzz_index()    OK | score={indices[0]['ai_buzz_index']}")

            db.commit()

    print()
    print("  All DatabaseManager tests passed.")
