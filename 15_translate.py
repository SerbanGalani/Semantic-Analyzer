"""
===============================================================================
AISA - AI Semantic Analyzer
15_translate.py - Fragment translation for non-English documents
===============================================================================

Translates AI reference text fragments detected in non-English (primarily
Chinese) corporate annual reports to English for thesis citation and manual
verification.

Translation strategy:
    - NEVER translate entire documents (expensive, slow)
    - Translate ONLY detected AI reference fragments (50–100 per doc × 2-3 sentences)
    - Translations stored in DB: text_translated + translation_source fields
    - Export: original text + English translation side-by-side in Excel

Supported engines (in priority order):
    1. DeepL          (preferred — highest quality for ZH→EN)
                       Requires: pip install deepl; DEEPL_API_KEY env variable
    2. OpenAI / LLM   (high quality fallback; also useful for difficult fragments)
                       Requires: pip install openai; OPENAI_API_KEY env variable
                       Optional: OPENAI_BASE_URL, TRANSLATION_LLM_MODEL
    3. Google Translate (fallback)
                       Requires: pip install google-cloud-translate;
                                 GOOGLE_APPLICATION_CREDENTIALS env variable
    4. Helsinki-NLP    (offline, free — quality lower than commercial)
                       Requires: pip install transformers sentencepiece
                                 Model: Helsinki-NLP/opus-mt-zh-en

Usage:
    # Translate a single fragment
    from 15_translate import translate_fragment
    translated, engine = translate_fragment("人工智能技术", source_lang="zh")

    # Post-process: translate all untranslated non-EN references in DB
    from 15_translate import update_translations_in_db
    stats = update_translations_in_db("aisa_results.db", engine="deepl")

    # CLI usage (standalone)
    python 15_translate.py --db aisa_results.db --engine deepl --batch-size 50

CHANGELOG:
    v1.0.0 (2026-03) - AISA v1.1.0: initial release

Author: TeRa0
License: MIT
===============================================================================
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import html

logger = logging.getLogger("AISA")

# ============================================================================
# CONSTANTS
# ============================================================================

SUPPORTED_SOURCE_LANGS = ["zh", "ja", "ko", "other"]
LLM_DEFAULT_MODEL = os.environ.get("TRANSLATION_LLM_MODEL", "gpt-4.1-mini")

# DeepL language codes (slightly different from ISO 639-1)
_DEEPL_LANG_MAP = {
    "zh": "ZH",
    "ja": "JA",
    "ko": "KO",
    "other": None,  # DeepL auto-detect
}

# Helsinki-NLP model map (source_lang → HuggingFace model ID)
_HELSINKI_MODEL_MAP = {
    "zh": "Helsinki-NLP/opus-mt-zh-en",
    "ja": "Helsinki-NLP/opus-mt-ja-en",
    "ko": "Helsinki-NLP/opus-mt-ko-en",
}

# Rate limit: pause between API calls to avoid throttling
_DEEPL_DELAY_SEC    = 0.05   # 50ms between calls
_GOOGLE_DELAY_SEC   = 0.10   # 100ms between calls

# Character limit: skip fragments that are too long (likely boilerplate)
MAX_FRAGMENT_CHARS  = 1000


# ============================================================================
# ENGINE AVAILABILITY CHECK
# ============================================================================

def check_engine_availability(engine: str) -> Tuple[bool, str]:
    """
    Check whether a translation engine is available (installed + configured).

    Returns:
        (available: bool, message: str)
    """
    if engine == "deepl":
        try:
            import deepl  # noqa: F401
            api_key = os.environ.get("DEEPL_API_KEY", "")
            if not api_key:
                return False, "DEEPL_API_KEY environment variable not set"
            return True, "DeepL available"
        except ImportError:
            return False, "deepl package not installed (pip install deepl)"

    if engine == "llm":
        try:
            from openai import OpenAI  # noqa: F401
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                return False, "OPENAI_API_KEY environment variable not set"
            return True, f"LLM available ({LLM_DEFAULT_MODEL})"
        except ImportError:
            return False, "openai package not installed (pip install openai)"

    if engine == "google":
        try:
            from google.cloud import translate_v2  # noqa: F401
            creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
            if not creds:
                return False, "GOOGLE_APPLICATION_CREDENTIALS not set"
            return True, "Google Translate available"
        except ImportError:
            return False, "google-cloud-translate not installed"

    if engine == "helsinki":
        try:
            from transformers import pipeline  # noqa: F401
            return True, "Helsinki-NLP available (transformers installed)"
        except ImportError:
            return False, "transformers not installed (pip install transformers sentencepiece)"

    return False, f"Unknown engine: {engine}"


def get_best_available_engine() -> Optional[str]:
    """Return the best available translation engine, or None if none available."""
    for engine in ("deepl", "llm", "google", "helsinki"):
        available, _ = check_engine_availability(engine)
        if available:
            return engine
    return None


# ============================================================================
# INTERNAL ENGINE IMPLEMENTATIONS
# ============================================================================

def _translate_deepl(text: str, source_lang: str) -> str:
    import deepl
    api_key = os.environ["DEEPL_API_KEY"]
    translator = deepl.Translator(api_key)
    src = _DEEPL_LANG_MAP.get(source_lang)  # None → auto-detect
    result = translator.translate_text(text, source_lang=src, target_lang="EN-US")
    return result.text


def _translate_google(text: str, source_lang: str) -> str:
    from google.cloud import translate_v2 as google_translate
    client = google_translate.Client()
    result = client.translate(text, source_language=source_lang, target_language="en")
    return html.unescape(result["translatedText"])


def _translate_llm(text: str, source_lang: str) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )
    model = os.environ.get("TRANSLATION_LLM_MODEL", LLM_DEFAULT_MODEL)
    lang_hint = {
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "other": "the source language",
    }.get(source_lang, "the source language")

    prompt = (
        f"Translate the following {lang_hint} corporate disclosure fragment into clear, natural English. "
        "Preserve factual meaning, numbers, company/product names, and reporting tone. "
        "Return only the translation, with no commentary.\n\n"
        f"Text:\n{text}"
    )

    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0,
    )
    return response.output_text.strip()


# Helsinki pipeline cache (loaded once per process per source language)
_helsinki_pipelines: Dict[str, object] = {}


def _translate_helsinki(text: str, source_lang: str) -> str:
    from transformers import pipeline as hf_pipeline
    model_id = _HELSINKI_MODEL_MAP.get(source_lang)
    if not model_id:
        raise ValueError(f"No Helsinki-NLP model for source_lang='{source_lang}'")
    if source_lang not in _helsinki_pipelines:
        logger.info(f"Loading Helsinki-NLP model: {model_id}")
        _helsinki_pipelines[source_lang] = hf_pipeline(
            "translation",
            model=model_id,
            device=-1,  # CPU
        )
    pipe = _helsinki_pipelines[source_lang]
    result = pipe(text[:512])   # HuggingFace translation pipelines have token limits
    return result[0]["translation_text"]


# ============================================================================
# PUBLIC API: translate_fragment
# ============================================================================

def translate_fragment(
    text: str,
    source_lang: str = "zh",
    engine: str = "deepl",
    fallback_chain: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """
    Translate a single text fragment to English.

    Args:
        text:           Source text to translate.
        source_lang:    ISO language code of source text ('zh', 'ja', 'ko').
        engine:         Preferred translation engine ('deepl', 'google', 'helsinki').
        fallback_chain: If given, try these engines in order if preferred fails.
                        Default: ['google', 'helsinki'] for deepl, etc.

    Returns:
        (translated_text: str, engine_used: str)
        On failure, returns (original_text, 'none').
    """
    if not text or not text.strip():
        return text, "none"

    if len(text) > MAX_FRAGMENT_CHARS:
        text = text[:MAX_FRAGMENT_CHARS]

    if fallback_chain is None:
        fallback_chain = [e for e in ("deepl", "llm", "google", "helsinki") if e != engine]

    engines_to_try = [engine] + fallback_chain

    for eng in engines_to_try:
        available, msg = check_engine_availability(eng)
        if not available:
            logger.debug(f"Engine '{eng}' skipped: {msg}")
            continue
        try:
            if eng == "deepl":
                translated = _translate_deepl(text, source_lang)
                time.sleep(_DEEPL_DELAY_SEC)
            elif eng == "llm":
                translated = _translate_llm(text, source_lang)
            elif eng == "google":
                translated = _translate_google(text, source_lang)
                time.sleep(_GOOGLE_DELAY_SEC)
            elif eng == "helsinki":
                translated = _translate_helsinki(text, source_lang)
            else:
                continue
            logger.debug(f"Translated [{source_lang}→EN] via {eng}: {text[:40]!r}")
            return translated, eng
        except Exception as exc:
            logger.warning(f"Translation via '{eng}' failed: {exc}")
            continue

    logger.warning(f"All translation engines failed for text: {text[:60]!r}")
    return text, "none"


# ============================================================================
# PUBLIC API: translate_batch
# ============================================================================

def translate_batch(
    fragments: List[str],
    source_lang: str = "zh",
    engine: str = "deepl",
    batch_size: int = 50,
    progress_callback=None,
) -> List[Tuple[str, str]]:
    """
    Translate a list of fragments to English.

    Args:
        fragments:         List of source text strings.
        source_lang:       Source language code.
        engine:            Translation engine.
        batch_size:        Number of fragments to process per progress report.
        progress_callback: Optional callable(current, total) for progress.

    Returns:
        List of (translated_text, engine_used) tuples, same order as input.
    """
    results: List[Tuple[str, str]] = []
    total = len(fragments)

    for i, frag in enumerate(fragments):
        translated, eng_used = translate_fragment(frag, source_lang=source_lang, engine=engine)
        results.append((translated, eng_used))
        if progress_callback and (i + 1) % batch_size == 0:
            progress_callback(i + 1, total)

    if progress_callback:
        progress_callback(total, total)

    return results


# ============================================================================
# PUBLIC API: update_translations_in_db
# ============================================================================

def update_translations_in_db(
    db_path: str,
    engine: str = "deepl",
    batch_size: int = 50,
    dry_run: bool = False,
    include_deduplicated: bool = True,
) -> Dict:
    """
    Post-processing step: translate all untranslated non-English references.

    Translates both:
        - text -> text_translated
        - context -> context_translated   (when column exists)

    By default also updates ai_references_deduplicated when that table has the
    relevant translation columns.
    """
    db_path = str(db_path)
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    avail, msg = check_engine_availability(engine)
    if not avail:
        best = get_best_available_engine()
        if best is None:
            raise RuntimeError(
                f"Translation engine '{engine}' unavailable ({msg}) and no fallback available. "
                "Install deepl, openai, google-cloud-translate, or transformers."
            )
        logger.warning(f"Engine '{engine}' unavailable: {msg}. Using '{best}' instead.")
        engine = best

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    has_ctx_raw = _column_exists(conn, "ai_references_raw", "context_translated")
    has_ctx_dedup = _column_exists(conn, "ai_references_deduplicated", "context_translated")

    en_backfill = conn.execute(
        """
        UPDATE ai_references_raw
        SET text_translated    = text,
            translation_source = 'original'
        WHERE (language = 'en' OR language IS NULL)
          AND (text_translated IS NULL OR text_translated = '')
        """
    ).rowcount
    if has_ctx_raw:
        conn.execute(
            """
            UPDATE ai_references_raw
            SET context_translated = context
            WHERE (language = 'en' OR language IS NULL)
              AND (context_translated IS NULL OR context_translated = '')
            """
        )

    if include_deduplicated:
        conn.execute(
            """
            UPDATE ai_references_deduplicated
            SET text_translated    = text,
                translation_source = 'original'
            WHERE (language = 'en' OR language IS NULL)
              AND (text_translated IS NULL OR text_translated = '')
            """
        )
        if has_ctx_dedup:
            conn.execute(
                """
                UPDATE ai_references_deduplicated
                SET context_translated = context
                WHERE (language = 'en' OR language IS NULL)
                  AND (context_translated IS NULL OR context_translated = '')
                """
            )
    conn.commit()

    rows = conn.execute(
        """
        SELECT id, company, year, language, text, context
        FROM ai_references_raw
        WHERE language != 'en'
          AND (
                (text_translated IS NULL OR text_translated = '')
                OR (? = 1 AND (context_translated IS NULL OR context_translated = ''))
              )
        ORDER BY company, year
        """,
        (1 if has_ctx_raw else 0,),
    ).fetchall()

    stats = {
        "translated": 0,
        "translated_context": 0,
        "skipped": 0,
        "failed": 0,
        "en_backfilled": en_backfill,
        "engine_used": engine,
        "elapsed_sec": 0.0,
    }

    if not rows:
        logger.info("No untranslated non-English references found.")
        conn.close()
        return stats

    logger.info(f"Found {len(rows)} untranslated non-English references. Engine: {engine}")
    t_start = time.perf_counter()

    for i, row in enumerate(rows):
        ref_id  = row["id"]
        company = row["company"]
        year    = row["year"]
        lang    = row["language"]
        text_value = row["text"] or ""
        context_value = row["context"] or ""

        if (not text_value or len(text_value.strip()) < 3) and (not context_value or len(context_value.strip()) < 3):
            stats["skipped"] += 1
            continue

        (translated_text, text_engine), (translated_context, context_engine) = _translate_row_fields(
            text_value=text_value,
            context_value=context_value,
            source_lang=lang,
            engine=engine,
        )

        if text_value and text_engine == "none" and (not context_value or context_engine == "none"):
            stats["failed"] += 1
            _log_translation(conn, ref_id, company, year, lang, text_value, None, "none", "error", "all engines failed", dry_run)
            continue

        if translated_text and text_engine != "none":
            stats["translated"] += 1
        if translated_context and context_engine != "none":
            stats["translated_context"] += 1

        if not dry_run:
            if has_ctx_raw:
                conn.execute(
                    """
                    UPDATE ai_references_raw
                    SET text_translated = COALESCE(?, text_translated),
                        translation_source = CASE WHEN ? != 'none' THEN ? ELSE translation_source END,
                        context_translated = COALESCE(?, context_translated)
                    WHERE id = ?
                    """,
                    (translated_text, text_engine, text_engine, translated_context, ref_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE ai_references_raw
                    SET text_translated = COALESCE(?, text_translated),
                        translation_source = CASE WHEN ? != 'none' THEN ? ELSE translation_source END
                    WHERE id = ?
                    """,
                    (translated_text, text_engine, text_engine, ref_id),
                )

            _log_translation(conn, ref_id, company, year, lang, text_value, translated_text, text_engine, "ok", None, dry_run)
            if translated_context and context_engine != "none":
                _log_translation(conn, ref_id, company, year, lang, f"[context] {context_value}", translated_context, context_engine, "ok", None, dry_run)

        if (i + 1) % batch_size == 0:
            if not dry_run:
                conn.commit()
            pct = (i + 1) / len(rows) * 100
            logger.info(
                f"  Progress: {i+1}/{len(rows)} ({pct:.0f}%) — "
                f"translated={stats['translated']} context={stats['translated_context']} failed={stats['failed']}"
            )

    if include_deduplicated:
        if has_ctx_dedup:
            conn.execute(
                """
                UPDATE ai_references_deduplicated
                SET context_translated = text_translated
                WHERE context = text
                  AND language != 'en'
                  AND (context_translated IS NULL OR context_translated = '')
                """
            )
        if has_ctx_dedup:
            conn.execute(
                """
                UPDATE ai_references_deduplicated
                SET text_translated = COALESCE((
                        SELECT r.text_translated
                        FROM ai_references_raw r
                        WHERE r.company = ai_references_deduplicated.company
                          AND r.year = ai_references_deduplicated.year
                          AND r.text = ai_references_deduplicated.text
                          AND r.language = ai_references_deduplicated.language
                          AND COALESCE(r.text_translated, '') != ''
                        ORDER BY r.id DESC
                        LIMIT 1
                    ), text_translated),
                    translation_source = COALESCE((
                        SELECT r.translation_source
                        FROM ai_references_raw r
                        WHERE r.company = ai_references_deduplicated.company
                          AND r.year = ai_references_deduplicated.year
                          AND r.text = ai_references_deduplicated.text
                          AND r.language = ai_references_deduplicated.language
                          AND COALESCE(r.text_translated, '') != ''
                        ORDER BY r.id DESC
                        LIMIT 1
                    ), translation_source),
                    context_translated = COALESCE((
                        SELECT r.context_translated
                        FROM ai_references_raw r
                        WHERE r.company = ai_references_deduplicated.company
                          AND r.year = ai_references_deduplicated.year
                          AND r.context = ai_references_deduplicated.context
                          AND r.language = ai_references_deduplicated.language
                          AND COALESCE(r.context_translated, '') != ''
                        ORDER BY r.id DESC
                        LIMIT 1
                    ), context_translated)
                WHERE language != 'en'
                """
            )
        else:
            conn.execute(
                """
                UPDATE ai_references_deduplicated
                SET text_translated = COALESCE((
                        SELECT r.text_translated
                        FROM ai_references_raw r
                        WHERE r.company = ai_references_deduplicated.company
                          AND r.year = ai_references_deduplicated.year
                          AND r.text = ai_references_deduplicated.text
                          AND r.language = ai_references_deduplicated.language
                          AND COALESCE(r.text_translated, '') != ''
                        ORDER BY r.id DESC
                        LIMIT 1
                    ), text_translated),
                    translation_source = COALESCE((
                        SELECT r.translation_source
                        FROM ai_references_raw r
                        WHERE r.company = ai_references_deduplicated.company
                          AND r.year = ai_references_deduplicated.year
                          AND r.text = ai_references_deduplicated.text
                          AND r.language = ai_references_deduplicated.language
                          AND COALESCE(r.text_translated, '') != ''
                        ORDER BY r.id DESC
                        LIMIT 1
                    ), translation_source)
                WHERE language != 'en'
                """
            )

    if not dry_run:
        conn.commit()

    stats["elapsed_sec"] = round(time.perf_counter() - t_start, 2)
    conn.close()

    logger.info(
        f"Translation complete: {stats['translated']} text + {stats['translated_context']} context translated, "
        f"{stats['skipped']} skipped, {stats['failed']} failed in {stats['elapsed_sec']}s"
    )
    return stats

    logger.info(f"Found {len(rows)} untranslated non-English references. Engine: {engine}")
    t_start = time.perf_counter()

    for i, row in enumerate(rows):
        ref_id  = row["id"]
        company = row["company"]
        year    = row["year"]
        lang    = row["language"]
        text    = row["text"]

        if not text or len(text.strip()) < 3:
            stats["skipped"] += 1
            continue

        translated, eng_used = translate_fragment(text, source_lang=lang, engine=engine)

        if eng_used == "none":
            stats["failed"] += 1
            _log_translation(conn, ref_id, company, year, lang, text,
                             None, "none", "error", "all engines failed", dry_run)
            continue

        stats["translated"] += 1

        if not dry_run:
            conn.execute(
                """
                UPDATE ai_references_raw
                SET text_translated    = ?,
                    translation_source = ?
                WHERE id = ?
                """,
                (translated, eng_used, ref_id),
            )
            _log_translation(conn, ref_id, company, year, lang, text,
                             translated, eng_used, "ok", None, dry_run)

        if (i + 1) % batch_size == 0:
            if not dry_run:
                conn.commit()
            pct = (i + 1) / len(rows) * 100
            logger.info(f"  Progress: {i+1}/{len(rows)} ({pct:.0f}%) — "
                        f"translated={stats['translated']} failed={stats['failed']}")

    if not dry_run:
        conn.commit()

    stats["elapsed_sec"] = round(time.perf_counter() - t_start, 2)
    conn.close()

    logger.info(
        f"Translation complete: {stats['translated']} translated, "
        f"{stats['skipped']} skipped, {stats['failed']} failed "
        f"in {stats['elapsed_sec']}s"
    )
    return stats


def _log_translation(
    conn: sqlite3.Connection,
    ref_id: int,
    company: str,
    year: int,
    source_language: str,
    original_text: str,
    translated_text: Optional[str],
    engine: str,
    status: str,
    error_message: Optional[str],
    dry_run: bool,
) -> None:
    """Write an entry to translation_log. Silently skips if dry_run."""
    if dry_run:
        return
    try:
        conn.execute(
            """
            INSERT INTO translation_log
                (ref_id, company, year, source_language, original_text,
                 translated_text, translation_engine, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ref_id, company, year, source_language, original_text,
             translated_text, engine, status, error_message),
        )
    except sqlite3.Error as e:
        logger.debug(f"translation_log insert failed (non-critical): {e}")


# ============================================================================
# TRANSLATION STATS QUERY
# ============================================================================

def get_translation_stats(db_path: str) -> Dict:
    """
    Query DB for translation coverage statistics.

    Returns:
        Dict with counts by language and translation status.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT language,
               COUNT(*) AS total,
               SUM(CASE WHEN text_translated IS NOT NULL AND text_translated != '' THEN 1 ELSE 0 END) AS translated,
               SUM(CASE WHEN COALESCE(context_translated, '') != '' THEN 1 ELSE 0 END) AS context_translated,
               SUM(CASE WHEN translation_source = 'deepl' THEN 1 ELSE 0 END) AS via_deepl,
               SUM(CASE WHEN translation_source = 'llm' THEN 1 ELSE 0 END) AS via_llm,
               SUM(CASE WHEN translation_source = 'google' THEN 1 ELSE 0 END) AS via_google,
               SUM(CASE WHEN translation_source = 'helsinki' THEN 1 ELSE 0 END) AS via_helsinki
        FROM ai_references_raw
        GROUP BY language
        ORDER BY total DESC
        """
    ).fetchall()
    conn.close()

    return {
        row["language"]: {
            "total":       row["total"],
            "translated":  row["translated"],
            "context_translated": row["context_translated"],
            "pending":     row["total"] - row["translated"],
            "via_deepl":   row["via_deepl"],
            "via_llm":     row["via_llm"],
            "via_google":  row["via_google"],
            "via_helsinki": row["via_helsinki"],
        }
        for row in rows
    }


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="AISA 15_translate.py — Translate non-English AI reference fragments"
    )
    parser.add_argument("--db", required=True, help="Path to AISA SQLite database")
    parser.add_argument(
        "--engine",
        default="deepl",
        choices=["deepl", "llm", "google", "helsinki"],
        help="Translation engine (default: deepl)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Commit to DB every N translations (default: 50)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Translate without writing to DB",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show translation coverage statistics and exit",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.stats:
        stats = get_translation_stats(args.db)
        print("\nTranslation coverage by language:")
        for lang, s in stats.items():
            pct = (s["translated"] / s["total"] * 100) if s["total"] > 0 else 0
            print(f"  {lang:6s}: text={s['translated']}/{s['total']} ({pct:.0f}%) context={s['context_translated']} "
                  f"[deepl={s['via_deepl']} llm={s['via_llm']} google={s['via_google']} helsinki={s['via_helsinki']}]")
        return

    result = update_translations_in_db(
        db_path=args.db,
        engine=args.engine,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
    print(f"\nResult: {result}")


# ============================================================================
# SMOKE TEST
# ============================================================================

if __name__ == "__main__":
    import sys

    if "--db" in sys.argv:
        _cli()
    else:
        print("AISA 15_translate.py smoke test")
        print("=" * 60)

        # Engine availability
        for eng in ("deepl", "llm", "google", "helsinki"):
            avail, msg = check_engine_availability(eng)
            status = "available" if avail else "unavailable"
            print(f"  {eng:10s}: {status} — {msg}")

        best = get_best_available_engine()
        print(f"\n  Best available engine: {best}")

        # translate_fragment with no engine (returns original)
        t, e = translate_fragment("人工智能", source_lang="zh",
                                  engine="deepl", fallback_chain=[])
        assert isinstance(t, str)
        assert isinstance(e, str)
        print(f"\n  translate_fragment (no engine available): '{t}' via '{e}' [OK]")

        # translate_batch
        frags = ["人工智能", "机器学习", "自然语言处理"]
        results = translate_batch(frags, source_lang="zh",
                                  engine="deepl")
        assert len(results) == 3
        assert all(isinstance(r[0], str) and isinstance(r[1], str) for r in results)
        print(f"  translate_batch {len(frags)} fragments [OK]")

        # get_translation_stats with nonexistent DB (should raise)
        try:
            get_translation_stats("/nonexistent/path.db")
            print("  ERROR: should have raised")
        except Exception:
            print("  get_translation_stats raises on missing DB [OK]")

        print()
        print("  15_translate.py all checks passed.")
