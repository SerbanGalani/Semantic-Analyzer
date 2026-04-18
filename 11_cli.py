"""
===============================================================================
AISA - AI Semantic Analyzer
11_cli.py - Command-line interface
===============================================================================

Entry point for all AISA operations. Uses argparse subcommands.

Subcommands:
    ingest      Run the full ingestion pipeline (PDF → DB)
    dedupe      Re-run semantic deduplication on existing DB references
    sentiment   Re-run sentiment analysis on pending references
    index       Recompute AI Buzz Index for all or specific years
    memory      Process/re-process adoption memory from DB references
    tpdi        Compute Technology-Product Diffusion Index report
    export      Export results to Excel / JSON / CSV
    status      Show processing status and database summary
    config      Show or save the current configuration

Usage examples:
    python 11_cli.py ingest --input Fortune500_PDFs/ --workers 8
    python 11_cli.py ingest --input Fortune500_PDFs/ --config aisa_config.json
    python 11_cli.py export --format excel --output Results_AISA/
    python 11_cli.py export --format all --year 2023
    python 11_cli.py status
    python 11_cli.py tpdi --output Results_AISA/tpdi_report.xlsx
    python 11_cli.py memory --reprocess-all
    python 11_cli.py config --save aisa_config.json

CHANGELOG:
    v1.0.0 (2026-02) - AISA initial release (argparse, no interactive menu)

Author: TeRa0
License: MIT
===============================================================================
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from version import AISA_VERSION, SCHEMA_VERSION, get_version_string


# ============================================================================
# LAZY MODULE LOADER
# Heavy modules (SentenceTransformer, FinBERT) are only imported when the
# subcommand that needs them is actually invoked.
# ============================================================================

def _load_core():
    """Load models module (always lightweight)."""
    _m1 = importlib.import_module("01_models")
    return _m1

def _load_db():
    _m2 = importlib.import_module("02_db")
    return _m2

def _load_pipeline():
    _m10 = importlib.import_module("10_pipeline")
    return _m10

def _load_analysis():
    _m6 = importlib.import_module("06_analysis")
    return _m6

def _load_memory():
    _m8 = importlib.import_module("08_memory")
    return _m8

def _load_tpdi():
    _m9 = importlib.import_module("09_tpdi")
    return _m9

def _load_export():
    _m7 = importlib.import_module("07_export")
    return _m7


def _load_taxonomy_for_export(config):
    """
    Load taxonomy provider without importing 05_detect.py.

    Export does not need the semantic model, so this helper imports the
    taxonomy module directly to avoid SentenceTransformer / Hugging Face
    network access during export.
    """
    if getattr(config, "taxonomy_excel", None):
        try:
            loader = importlib.import_module("taxonomy_excel_loader")
            return loader.ExcelTaxonomyProvider(config.taxonomy_excel)
        except Exception as exc:
            raise RuntimeError(f"Excel taxonomy load failed: {exc}") from exc

    module_map = {
        "AI_Disclosure": "04_taxonomy_builtin",
        "Digitalization_Eco": "04_taxonomy_digitalization",
        "Digitalization_Relational_v2": "04_taxonomy_digitalization",
        "Digitalization_Relational_v2_2": "04_taxonomy_digitalization_relational_v2_2_0",
        "Digitalization_Relational_v2_2_ZH": "04_taxonomy_digitalization_zh",
    }

    module_name = module_map.get(getattr(config, "taxonomy_name", None))
    if not module_name:
        raise RuntimeError(
            f"Unknown taxonomy for export: {getattr(config, 'taxonomy_name', None)}"
        )

    try:
        mod = importlib.import_module(module_name)
        return mod.TAXONOMY
    except Exception as exc:
        raise RuntimeError(f"Taxonomy module load failed ({module_name}): {exc}") from exc


# ============================================================================
# CONFIG RESOLUTION
# Priority: CLI args > config file > defaults
# ============================================================================

def _resolve_config(args: argparse.Namespace):
    """
    Build AnalyzerConfig from CLI args and optional JSON config file.

    Priority (highest to lowest):
        1. Explicit CLI arguments (e.g. --workers 8)
        2. Values from --config JSON file
        3. AnalyzerConfig defaults

    Returns:
        (AnalyzerConfig, db_path: str)
    """
    m1 = _load_core()
    AnalyzerConfig = m1.AnalyzerConfig

    # Load base config from file if provided
    base: dict = {}
    config_path = getattr(args, "config", None)
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            base = json.load(f)
        print(f"  Config loaded from: {config_path}")

    # Override with explicit CLI args
    if getattr(args, "input", None):
        base["input_folder"] = args.input
    if getattr(args, "output", None):
        base["output_folder"] = args.output
    if getattr(args, "db", None):
        base["database_name"] = args.db
    if getattr(args, "csv", None):
        base["fortune500_csv"] = args.csv
    if getattr(args, "taxonomy_name", None):
        base["taxonomy_name"] = args.taxonomy_name
    if getattr(args, "taxonomy_excel", None):
        base["taxonomy_excel"] = args.taxonomy_excel
    if getattr(args, "workers", None):
        base["max_workers"] = args.workers
    if getattr(args, "start_year", None):
        base["start_year"] = args.start_year
    if getattr(args, "end_year", None):
        base["end_year"] = args.end_year
    if getattr(args, "top_n", None):
        base["top_n"] = args.top_n

    config = AnalyzerConfig.from_dict(base)

    db_path = os.path.join(config.output_folder, config.database_name)
    return config, db_path


def _open_db(db_path: str, apply_migrations: bool = True):
    """Open DatabaseManager, optionally running pending migrations."""
    m2 = _load_db()
    DatabaseManager = m2.DatabaseManager
    db = DatabaseManager(db_path)
    if apply_migrations:
        db.apply_migrations()
    return db


# ============================================================================
# PROGRESS CALLBACK (printed to stdout)
# ============================================================================

def _make_progress_callback(total: int):
    """Return a progress callback that prints a simple progress line."""
    start = time.time()

    def callback(current: int, total_: int, company: str, year: int):
        elapsed  = time.time() - start
        avg_sec  = elapsed / current if current > 0 else 0
        remaining = avg_sec * (total_ - current)
        pct = 100 * current / total_ if total_ > 0 else 0
        eta = _fmt_seconds(remaining)
        print(
            f"\r  [{current:>4}/{total_:>4}] {pct:5.1f}%  "
            f"{company[:28]:<28} {year}  "
            f"avg={avg_sec:.1f}s  ETA={eta}   ",
            end="",
            flush=True,
        )
        if current == total_:
            print()

    return callback


def _fmt_seconds(s: float) -> str:
    if s < 60:
        return f"{int(s)}s"
    if s < 3600:
        return f"{int(s//60)}m{int(s%60):02d}s"
    return f"{int(s//3600)}h{int((s%3600)//60):02d}m"


# ============================================================================
# SUBCOMMAND HANDLERS
# ============================================================================

def cmd_ingest(args: argparse.Namespace) -> int:
    """
    Run the full ingestion pipeline: PDF → detect → sentiment → buzz → memory.
    """
    print(f"\n{get_version_string()}")
    print("=" * 60)
    print("  INGEST")
    print("=" * 60)

    config, db_path = _resolve_config(args)

    print(f"  Input folder : {config.input_folder}")
    print(f"  Output folder: {config.output_folder}")
    print(f"  Database     : {db_path}")
    print(f"  Years        : {config.start_year}–{config.end_year}")
    print(f"  Workers      : {config.max_workers}")
    if config.top_n:
        print(f"  Top N        : {config.top_n}")
    print()

    m10 = _load_pipeline()
    run_pipeline = m10.run_pipeline

    with _open_db(db_path) as db:
        stats = run_pipeline(
            config=config,
            db=db,
            progress_callback=_make_progress_callback(0),
        )

    # Auto-export if --export flag given
    if getattr(args, "export_after", False):
        _do_export(db_path, config.output_folder, formats=["excel", "json"])

    return 0


def cmd_dedupe(args: argparse.Namespace) -> int:
    """
    Re-run cross-document deduplication on all references in the DB.

    Groups references by (company, year), applies Jaccard clustering
    (threshold 0.7) across documents, and writes clusters to
    ai_references_deduplicated. Does NOT delete from ai_references_raw.

    Clusters preserve all source documents (doc_type + pages) so that
    the origin of each reference remains traceable.
    """
    print(f"\n{get_version_string()}")
    print("=" * 60)
    print("  DEDUPE")
    print("=" * 60)

    _, db_path = _resolve_config(args)
    year_filter = getattr(args, "year", None)
    m10 = _load_pipeline()

    with _open_db(db_path) as db:
        where_clause = "WHERE year = ?" if year_filter else ""
        params       = [year_filter] if year_filter else []
        count = db.conn.execute(
            f"SELECT COUNT(*) FROM ai_references_raw {where_clause}", params
        ).fetchone()[0]

        print(f"  References in DB: {count}")
        if count == 0:
            print("  Nothing to deduplicate.")
            return 0

        result = m10.run_dedup_for_db(db, year_filter=year_filter)

        total_clusters = sum(result.values())
        print(f"  Deduplication complete (cross-document, per company per year):")
        for year, clusters in sorted(result.items()):
            print(f"    {year}: {clusters} unique clusters → ai_references_deduplicated")
        print(f"  Total clusters written: {total_clusters}")

    return 0


def cmd_sentiment(args: argparse.Namespace) -> int:
    """
    Re-run sentiment analysis on references with sentiment == 'pending'.
    """
    print(f"\n{get_version_string()}")
    print("=" * 60)
    print("  SENTIMENT")
    print("=" * 60)

    config, db_path = _resolve_config(args)
    year_filter = getattr(args, "year", None)
    batch_size  = getattr(args, "batch_size", 256) or 256

    m6  = _load_analysis()
    analyze_sentiment_batch = m6.analyze_sentiment_batch
    m1  = _load_core()
    AIReference = m1.AIReference

    with _open_db(db_path) as db:
        conditions = ["sentiment = 'pending'"]
        params: list = []
        if year_filter:
            conditions.append("year = ?")
            params.append(year_filter)

        count = db.conn.execute(
            f"SELECT COUNT(*) FROM ai_references_raw WHERE {' AND '.join(conditions)}",
            params,
        ).fetchone()[0]

        print(f"  Pending sentiment analysis: {count} references")
        if count == 0:
            print("  Nothing to do.")
            return 0

        processed = 0
        offset = 0
        while True:
            rows = db.conn.execute(
                f"""
                SELECT id, text, context, category_a
                FROM ai_references_raw
                WHERE {' AND '.join(conditions)}
                ORDER BY id
                LIMIT ? OFFSET ?
                """,
                params + [batch_size, offset],
            ).fetchall()

            if not rows:
                break

            # Build minimal AIReference stubs for sentiment
            refs = []
            for r in rows:
                ref = AIReference(
                    company="", year=0, position=0,
                    industry="", sector="", country="",
                    doc_type="", text=r["text"] or "",
                    context=r["context"] or "",
                    page=0, category=r.get("category_a") or "",
                    detection_method="", sentiment="pending",
                    sentiment_score=0.0, semantic_score=0.0,
                    source="", category_a=r.get("category_a") or "",
                )
                refs.append((r["id"], ref))

            ref_objs = [r for _, r in refs]
            analyze_sentiment_batch(ref_objs, config)

            # Write back to DB
            for (ref_id, ref_obj) in refs:
                db.conn.execute(
                    """
                    UPDATE ai_references_raw
                    SET sentiment            = ?,
                        sentiment_score      = ?,
                        sentiment_confidence = ?
                    WHERE id = ?
                    """,
                    (ref_obj.sentiment, ref_obj.sentiment_score,
                     ref_obj.sentiment_confidence, ref_id),
                )
            db.conn.commit()

            processed += len(rows)
            offset    += batch_size
            print(f"\r  Processed: {processed}/{count}", end="", flush=True)

        print()
        print(f"  Done. {processed} references updated.")

    return 0


def cmd_index(args: argparse.Namespace) -> int:
    """
    Recompute AI Buzz Index for all companies (or a specific year).
    """
    print(f"\n{get_version_string()}")
    print("=" * 60)
    print("  INDEX (AI Buzz Index recompute)")
    print("=" * 60)

    config, db_path = _resolve_config(args)
    year_filter = getattr(args, "year", None)

    m6  = _load_analysis()
    m1  = _load_core()
    AIReference = m1.AIReference
    calculate_buzz_index    = m6.calculate_buzz_index
    rank_buzz_indices       = m6.rank_buzz_indices
    aggregate_by_industry   = m6.aggregate_by_industry

    m10 = _load_pipeline()
    _upsert_industry_buzz = m10._upsert_industry_buzz
    IndustryBuzzIndex = m6.IndustryBuzzIndex

    with _open_db(db_path) as db:
        # Get all distinct company-year pairs
        conditions = []
        params: list = []
        if year_filter:
            conditions.append("year = ?")
            params.append(year_filter)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        pairs = db.conn.execute(
            f"""
            SELECT company, year, position, industry, sector, country,
                   COUNT(*) as ref_count,
                   MAX(page_count) as total_pages
            FROM ai_references_raw
            {where}
            GROUP BY company, year
            ORDER BY company, year
            """,
            params,
        ).fetchall()

        print(f"  Company-year pairs to reindex: {len(pairs)}")

        buzz_list = []
        for row in pairs:
            refs_rows = db.conn.execute(
                """
                SELECT text, context, category_a, category_b,
                       sentiment, sentiment_score, semantic_score,
                       reference_strength, confidence_score,
                       confidence_reasons, product_name, product_vendor
                FROM ai_references_raw
                WHERE company = ? AND year = ?
                """,
                (row["company"], row["year"]),
            ).fetchall()

            refs = []
            for r in refs_rows:
                ref = AIReference(
                    company=row["company"], year=row["year"],
                    position=row["position"] or 0,
                    industry=row["industry"] or "",
                    sector=row["sector"] or "",
                    country=row["country"] or "",
                    doc_type="Annual Report",
                    text=r["text"] or "", context=r["context"] or "",
                    page=0, category="",
                    detection_method="",
                    sentiment=r["sentiment"] or "neutral",
                    sentiment_score=r["sentiment_score"] or 0.0,
                    semantic_score=r["semantic_score"] or 0.0,
                    source="",
                    category_a=r["category_a"] or "",
                    category_b=r["category_b"] or "",
                    reference_strength=r["reference_strength"] or "unknown",
                    confidence_score=r["confidence_score"] or 0.0,
                    confidence_reasons=r["confidence_reasons"] or "",
                )
                refs.append(ref)

            buzz = calculate_buzz_index(
                refs=refs,
                company=row["company"], year=row["year"],
                position=row["position"] or 0,
                industry=row["industry"] or "",
                sector=row["sector"] or "",
                country=row["country"] or "",
                total_pages=row["total_pages"] or 1,
                config=config,
            )
            db.insert_buzz_index(buzz)
            buzz_list.append(buzz)

        # Rank + industry aggregation
        if buzz_list:
            rank_buzz_indices(buzz_list)
            for buzz in buzz_list:
                db.insert_buzz_index(buzz)   # re-upsert with rank

            industry_indices = aggregate_by_industry(buzz_list)
            for ind in industry_indices:
                _upsert_industry_buzz(ind, db)

            # Update DB rankings per year
            years = {b.year for b in buzz_list}
            for y in years:
                db.update_rankings(y)

        db.commit()
        print(f"  Done. {len(buzz_list)} buzz index records updated.")

    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    """
    Process/re-process adoption memory from references in DB.
    """
    print(f"\n{get_version_string()}")
    print("=" * 60)
    print("  MEMORY (Adoption Memory)")
    print("=" * 60)

    config, db_path = _resolve_config(args)
    reprocess_all = getattr(args, "reprocess_all", False)
    year_filter   = getattr(args, "year", None)

    m8 = _load_memory()
    process_unprocessed = m8.process_unprocessed_references
    finalize_year       = m8.finalize_year_memory

    with _open_db(db_path) as db:
        if reprocess_all:
            # Reset memory-processed flag
            db.conn.execute(
                "UPDATE ai_references_raw SET adoption_memory_processed = 0"
                + (" WHERE year = ?" if year_filter else ""),
                ([year_filter] if year_filter else []),
            )
            db.conn.commit()
            print("  Reset adoption_memory_processed flag.")

        stats = process_unprocessed(db, batch_size=500)
        print(f"  Total refs      : {stats['total']}")
        print(f"  Processed       : {stats['processed']}")
        print(f"  Batches         : {stats['batches']}")

        # Finalize years
        years_to_finalize = []
        if year_filter:
            years_to_finalize = [year_filter]
        else:
            rows = db.conn.execute(
                "SELECT DISTINCT year FROM adoption_events ORDER BY year"
            ).fetchall()
            years_to_finalize = [r["year"] for r in rows]

        for year in years_to_finalize:
            disc = finalize_year(year, db)
            if disc["discontinued_a"] or disc["discontinued_b"]:
                print(
                    f"  Year {year}: {disc['discontinued_a']} app discontinued, "
                    f"{disc['discontinued_b']} tech discontinued"
                )

    return 0


def cmd_tpdi(args: argparse.Namespace) -> int:
    """
    Compute Technology-Product Diffusion Index and export report.
    """
    print(f"\n{get_version_string()}")
    print("=" * 60)
    print("  TPDI (Technology-Product Diffusion Index)")
    print("=" * 60)

    config, db_path = _resolve_config(args)
    output_path     = getattr(args, "tpdi_output", None)
    min_adopters    = getattr(args, "min_adopters", 3) or 3
    no_products     = getattr(args, "no_products", False)

    m9  = _load_tpdi()
    build_tpdi_report = m9.build_tpdi_report

    with _open_db(db_path) as db:
        report = build_tpdi_report(
            db=db,
            config=config,
            include_products=not no_products,
            min_adopters=min_adopters,
        )

    print(f"\n  Curves computed: {len(report.curves)}")
    if report.analysis_years:
        print(f"  Analysis years: {report.analysis_years[0]}\u2013{report.analysis_years[-1]}")
    else:
        print(f"  Analysis years: N/A")
    print(f"  Corpus size   : {report.total_companies} companies")
    print()

    if not report.curves:
        print("  \u26a0\ufe0f  No curves met the minimum adopter threshold.")
        print(f"     Try running with --min-adopters 1 to include all data.")
        return 0

    # Print top 10 summary
    print("  TOP 10 TECHNOLOGIES BY TPDI:")
    print(f"  {'Rank':<5} {'Label':<35} {'TPDI':>6} {'Penetration':>12} {'Stage':<14}")
    print("  " + "-" * 80)
    for i, curve in enumerate(report.top_n(10), 1):
        print(
            f"  {i:<5} {curve.label():<35} "
            f"{curve.tpdi_score:>6.3f} "
            f"{curve.max_penetration*100:>10.1f}%  "
            f"{curve.lifecycle_stage:<14}"
        )

    # Export if output specified
    if output_path:
        import pandas as pd
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.suffix.lower() == ".xlsx":
            try:
                with pd.ExcelWriter(str(out_path), engine="openpyxl") as writer:
                    pd.DataFrame(report.to_records()).to_excel(
                        writer, sheet_name="TPDI Summary", index=False
                    )
                    pd.DataFrame(report.yearly_to_records()).to_excel(
                        writer, sheet_name="Yearly Diffusion", index=False
                    )
                print(f"\n  Excel export: {out_path}")
            except ImportError:
                print("  openpyxl not installed — skipping Excel export")
        elif out_path.suffix.lower() == ".csv":
            pd.DataFrame(report.to_records()).to_csv(str(out_path), index=False)
            print(f"\n  CSV export: {out_path}")
        elif out_path.suffix.lower() == ".json":
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "meta": {
                        "aisa_version": AISA_VERSION,
                        "analysis_years": report.analysis_years,
                        "total_companies": report.total_companies,
                    },
                    "curves":  report.to_records(),
                    "yearly":  report.yearly_to_records(),
                }, f, indent=2, ensure_ascii=False)
            print(f"\n  JSON export: {out_path}")

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """
    Export results to Excel / JSON / CSV.
    """
    print(f"\n{get_version_string()}")
    print("=" * 60)
    print("  EXPORT")
    print("=" * 60)

    config, db_path = _resolve_config(args)

    fmt        = getattr(args, "format", "all") or "all"
    year_filter = getattr(args, "year", None)
    base_name  = getattr(args, "base_name", "aisa_results") or "aisa_results"

    formats = ["excel", "json", "csv"] if fmt == "all" else [fmt]

    m7 = _load_export()
    export_all = m7.export_all

    taxonomy = None
    try:
        taxonomy = _load_taxonomy_for_export(config)
    except Exception as exc:
        print(f"  Warning: could not load taxonomy for export: {exc}")

    export_type = getattr(args, "export_type", "full") or "full"

    with _open_db(db_path) as db:
        result = export_all(
            db=db,
            output_dir=config.output_folder,
            base_name=base_name,
            year=year_filter,
            formats=formats,
            taxonomy=taxonomy,
            export_type=export_type,
        )

    print()
    if result.get("excel"):
        print(f"  Excel : {result['excel']}")
    if result.get("json"):
        print(f"  JSON  : {result['json']}")
    if result.get("csv"):
        for f in result["csv"]:
            print(f"  CSV   : {f}")

    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """
    Show database status and processing summary.
    """
    print(f"\n{get_version_string()}")
    print("=" * 60)
    print("  STATUS")
    print("=" * 60)

    config, db_path = _resolve_config(args)

    if not os.path.exists(db_path):
        print(f"  Database not found: {db_path}")
        print("  Run 'ingest' first.")
        return 1

    with _open_db(db_path, apply_migrations=False) as db:
        # Schema version
        sv = db._get_current_schema_version()
        print(f"  Database        : {db_path}")
        print(f"  Schema version  : {sv} (target: {SCHEMA_VERSION})")

        # Processing stats
        stats = db.get_processing_stats()
        print(f"\n  Documents processed   : {stats.get('total_documents_processed', 0)}")
        print(f"  Unique references     : {stats.get('unique_references', 0)}")
        print(f"  Total occurrences     : {stats.get('total_occurrences', 0)}")
        print(f"  Docs without refs     : {stats.get('documents_without_refs', 0)}")
        print(f"  Docs with issues      : {stats.get('documents_with_text_issues', 0)}")

        by_status = stats.get("by_text_status", {})
        if by_status:
            print(f"\n  Text status breakdown:")
            for status, count in sorted(by_status.items()):
                print(f"    {status:<30} {count}")

        # Buzz index summary
        buzz_rows = db.conn.execute(
            """
            SELECT year, COUNT(*) as companies,
                   ROUND(AVG(ai_buzz_index), 3) as avg_abi,
                   ROUND(MAX(ai_buzz_index), 3) as max_abi
            FROM adoption_index
            GROUP BY year ORDER BY year
            """
        ).fetchall()

        if buzz_rows:
            print(f"\n  AI Buzz Index by year:")
            print(f"    {'Year':<6} {'Companies':>10} {'Avg ABI':>10} {'Max ABI':>10}")
            for r in buzz_rows:
                print(
                    f"    {r['year']:<6} {r['companies']:>10} "
                    f"{r['avg_abi']:>10} {r['max_abi']:>10}"
                )

        # Memory summary
        mem_a = db.conn.execute(
            "SELECT COUNT(*) FROM adoption_portfolio_a"
        ).fetchone()[0]
        mem_b = db.conn.execute(
            "SELECT COUNT(*) FROM adoption_portfolio_b"
        ).fetchone()[0]
        events = db.conn.execute(
            "SELECT COUNT(*) FROM adoption_events"
        ).fetchone()[0]

        print(f"\n  Adoption Memory:")
        print(f"    Portfolio A entries : {mem_a}")
        print(f"    Portfolio B entries : {mem_b}")
        print(f"    Events              : {events}")

    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """
    Show or save the current configuration.
    """
    print(f"\n{get_version_string()}")
    print("=" * 60)
    print("  CONFIG")
    print("=" * 60)

    config, db_path = _resolve_config(args)
    save_path = getattr(args, "save", None)

    d = config.to_dict()
    # Remove weights from display (too verbose); shown separately
    weights = d.pop("weights", {})

    print()
    for k, v in d.items():
        print(f"  {k:<28} {v}")

    print(f"\n  AI Buzz Index weights:")
    for k, v in weights.items():
        print(f"    {k:<24} {v}")

    print(f"\n  DB path: {db_path}")

    if save_path:
        config.save_json(save_path)
        print(f"\n  Config saved to: {save_path}")

    return 0


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared across multiple subcommands."""
    parser.add_argument(
        "--config", "-c", metavar="PATH",
        help="Path to JSON config file (overrides defaults)",
    )
    parser.add_argument(
        "--db", metavar="NAME",
        help="SQLite database filename (default: aisa_results.db)",
    )
    parser.add_argument(
        "--output", "-o", metavar="DIR",
        help="Output folder for results (default: Results_AISA/)",
    )


def _add_year_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--year", "-y", type=int, metavar="YEAR",
        help="Filter to a specific year",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="aisa",
        description=(
            f"AISA v{AISA_VERSION} - AI Semantic Analyzer\n"
            "NLP-powered analysis of AI mentions in Fortune 500 annual reports."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  aisa ingest --input Fortune500_PDFs/ --workers 8\n"
            "  aisa ingest --config aisa_config.json\n"
            "  aisa export --format excel --year 2023\n"
            "  aisa status\n"
            "  aisa tpdi --output Results_AISA/tpdi_report.xlsx\n"
            "  aisa memory --reprocess-all\n"
            "  aisa config --save aisa_config.json\n"
        ),
    )
    parser.add_argument(
        "--version", action="version",
        version=get_version_string(),
    )

    sub = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
    sub.required = True

    # ----------------------------------------------------------------
    # ingest
    # ----------------------------------------------------------------
    p_ingest = sub.add_parser(
        "ingest",
        help="Run full ingestion pipeline (PDF → DB)",
        description="Process Fortune 500 PDFs through the full AISA pipeline.",
    )
    _add_common_args(p_ingest)
    p_ingest.add_argument(
        "--input", "-i", metavar="DIR",
        help="Folder containing PDF files (default: Fortune500_PDFs/)",
    )
    p_ingest.add_argument(
        "--csv", metavar="PATH",
        help="Fortune 500 CSV with company metadata",
    )
    p_ingest.add_argument(
        "--workers", "-w", type=int, metavar="N",
        help="Number of parallel Stage 1 workers (default: 4)",
    )
    p_ingest.add_argument(
        "--start-year", dest="start_year", type=int, metavar="YEAR",
        help="First year to process (default: 2020)",
    )
    p_ingest.add_argument(
        "--end-year", dest="end_year", type=int, metavar="YEAR",
        help="Last year to process (default: 2025)",
    )
    p_ingest.add_argument(
        "--top-n", dest="top_n", type=int, metavar="N",
        help="Process only first N PDFs (for testing)",
    )
    p_ingest.add_argument(
        "--export-after", dest="export_after", action="store_true",
        help="Auto-export Excel + JSON after ingestion completes",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # ----------------------------------------------------------------
    # dedupe
    # ----------------------------------------------------------------
    p_dedupe = sub.add_parser(
        "dedupe",
        help="Re-run semantic deduplication on DB references",
    )
    _add_common_args(p_dedupe)
    _add_year_arg(p_dedupe)
    p_dedupe.set_defaults(func=cmd_dedupe)

    # ----------------------------------------------------------------
    # sentiment
    # ----------------------------------------------------------------
    p_sent = sub.add_parser(
        "sentiment",
        help="Re-run sentiment analysis on pending references",
    )
    _add_common_args(p_sent)
    _add_year_arg(p_sent)
    p_sent.add_argument(
        "--batch-size", dest="batch_size", type=int, default=256, metavar="N",
        help="References per sentiment batch (default: 256)",
    )
    p_sent.set_defaults(func=cmd_sentiment)

    # ----------------------------------------------------------------
    # index
    # ----------------------------------------------------------------
    p_index = sub.add_parser(
        "index",
        help="Recompute AI Buzz Index for all companies",
    )
    _add_common_args(p_index)
    _add_year_arg(p_index)
    p_index.set_defaults(func=cmd_index)

    # ----------------------------------------------------------------
    # memory
    # ----------------------------------------------------------------
    p_mem = sub.add_parser(
        "memory",
        help="Process/re-process adoption memory from DB references",
    )
    _add_common_args(p_mem)
    _add_year_arg(p_mem)
    p_mem.add_argument(
        "--reprocess-all", dest="reprocess_all", action="store_true",
        help="Reset memory flags and reprocess all references",
    )
    p_mem.set_defaults(func=cmd_memory)

    # ----------------------------------------------------------------
    # tpdi
    # ----------------------------------------------------------------
    p_tpdi = sub.add_parser(
        "tpdi",
        help="Compute Technology-Product Diffusion Index",
    )
    _add_common_args(p_tpdi)
    p_tpdi.add_argument(
        "--tpdi-output", dest="tpdi_output", metavar="PATH",
        help="Output file for TPDI report (.xlsx / .csv / .json)",
    )
    p_tpdi.add_argument(
        "--min-adopters", dest="min_adopters", type=int, default=3, metavar="N",
        help="Minimum companies for product-level curve inclusion (default: 3)",
    )
    p_tpdi.add_argument(
        "--no-products", dest="no_products", action="store_true",
        help="Only compute category-level curves (skip named products)",
    )
    p_tpdi.add_argument(
        "--start-year", dest="start_year", type=int, metavar="YEAR",
    )
    p_tpdi.add_argument(
        "--end-year", dest="end_year", type=int, metavar="YEAR",
    )
    p_tpdi.set_defaults(func=cmd_tpdi)

    # ----------------------------------------------------------------
    # export
    # ----------------------------------------------------------------
    p_export = sub.add_parser(
        "export",
        help="Export results to Excel / JSON / CSV",
    )
    _add_common_args(p_export)
    _add_year_arg(p_export)
    p_export.add_argument(
        "--format", "-f",
        choices=["excel", "json", "csv", "all"],
        default="all",
        help="Export format (default: all)",
    )
    p_export.add_argument(
        "--base-name", dest="base_name", metavar="NAME", default="aisa_results",
        help="Base filename for Excel/JSON exports (default: aisa_results)",
    )
    p_export.set_defaults(func=cmd_export)

    # ----------------------------------------------------------------
    # status
    # ----------------------------------------------------------------
    p_status = sub.add_parser(
        "status",
        help="Show processing status and database summary",
    )
    _add_common_args(p_status)
    p_status.set_defaults(func=cmd_status)

    # ----------------------------------------------------------------
    # config
    # ----------------------------------------------------------------
    p_cfg = sub.add_parser(
        "config",
        help="Show or save the current configuration",
    )
    _add_common_args(p_cfg)
    p_cfg.add_argument(
        "--input", "-i", metavar="DIR",
        help="Input folder for PDFs",
    )
    p_cfg.add_argument(
        "--save", metavar="PATH",
        help="Save config to this JSON file",
    )
    p_cfg.add_argument(
        "--workers", "-w", type=int, metavar="N",
    )
    p_cfg.add_argument(
        "--start-year", dest="start_year", type=int, metavar="YEAR",
    )
    p_cfg.add_argument(
        "--end-year", dest="end_year", type=int, metavar="YEAR",
    )
    p_cfg.set_defaults(func=cmd_config)

    return parser


# ============================================================================
# ENTRY POINT
# ============================================================================

def main(argv=None) -> int:
    """Parse args and dispatch to the appropriate subcommand handler."""
    parser = build_parser()
    args   = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")
        return 130
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        return 2
    except RuntimeError as e:
        print(f"\n  RUNTIME ERROR: {e}")
        return 3
    except Exception as e:
        print(f"\n  UNEXPECTED ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


# ============================================================================
# SMOKE TEST
# ============================================================================

if __name__ == "__main__":
    from version import get_version_string as gvs
    print(gvs())
    print()

    parser = build_parser()

    # Test --version
    try:
        parser.parse_args(["--version"])
    except SystemExit:
        pass

    # Test help texts (should not raise)
    for subcmd in ["ingest", "export", "status", "tpdi", "memory",
                   "sentiment", "dedupe", "index", "config"]:
        try:
            parser.parse_args([subcmd, "--help"])
        except SystemExit:
            pass

    # Test argument parsing
    args = parser.parse_args([
        "ingest",
        "--input", "Fortune500_PDFs/",
        "--workers", "8",
        "--start-year", "2020",
        "--end-year", "2024",
        "--top-n", "10",
    ])
    assert args.subcommand == "ingest"
    assert args.input == "Fortune500_PDFs/"
    assert args.workers == 8
    assert args.start_year == 2020
    assert args.top_n == 10
    print("  ingest args:   OK")

    args2 = parser.parse_args([
        "export",
        "--format", "excel",
        "--year", "2023",
        "--output", "Results/",
    ])
    assert args2.subcommand == "export"
    assert args2.format == "excel"
    assert args2.year == 2023
    print("  export args:   OK")

    args3 = parser.parse_args([
        "tpdi",
        "--tpdi-output", "results/tpdi.xlsx",
        "--min-adopters", "5",
        "--no-products",
    ])
    assert args3.min_adopters == 5
    assert args3.no_products is True
    print("  tpdi args:     OK")

    args4 = parser.parse_args([
        "memory",
        "--reprocess-all",
        "--year", "2022",
    ])
    assert args4.reprocess_all is True
    assert args4.year == 2022
    print("  memory args:   OK")

    args5 = parser.parse_args([
        "config",
        "--input", "PDFs/",
        "--workers", "4",
        "--save", "aisa_config.json",
    ])
    assert args5.save == "aisa_config.json"
    print("  config args:   OK")

    # Test _resolve_config with minimal namespace
    import types
    ns = types.SimpleNamespace(
        config=None, input="Fortune500_PDFs/",
        output="Results_AISA/", db=None, csv=None,
        workers=4, start_year=2020, end_year=2025, top_n=None,
    )
    config, db_path = _resolve_config(ns)
    assert config.input_folder == "Fortune500_PDFs/"
    assert config.max_workers == 4
    assert "Results_AISA" in db_path
    print("  _resolve_config: OK")

    print()
    print("  11_cli.py all checks passed.")
    print()
    print("  Usage: python 11_cli.py --help")
