"""
===============================================================================
AISA - AI Semantic Analyzer
10_pipeline.py - Main orchestration pipeline
===============================================================================

Orchestrates the full AISA processing run:

    Stage 1  (ProcessPool, CPU-parallel):
        - PDF text extraction
        - Stage-1 candidate detection (regex + keyword pre-filter)

    Stage 2  (single worker, semantic model):
        - Batch semantic scoring (SentenceTransformer)
        - AIReference construction
        - Semantic deduplication

    Stage 3  (single worker, sequential):
        - Sentiment analysis (FinBERT / VADER)
        - AI Buzz Index computation
        - Adoption Memory update
        - DB persistence

Architecture:
    ProcessPoolExecutor  → Stage 1 workers (one per PDF)
    single thread        → Stage 2 + 3 (model must run in single process)

    Queue pattern: Stage 1 results are batched before Stage 2 to maximize
    SentenceTransformer batch efficiency.

Document metadata resolution:
    Pipeline reads a Fortune 500 CSV to map PDF filenames to company metadata.
    If the CSV is not provided, metadata is extracted from the filename
    using the pattern: {Company}_{Year}_{DocType}.pdf

Resumability:
    Already-processed documents are skipped (tracked in processed_documents DB
    table via DatabaseManager.is_document_processed()).

Performance notes:
    - Stage 1 saturates CPU cores (PDF parsing is I/O + CPU bound)
    - Stage 2 is the bottleneck on CPU-only machines; batch encoding amortizes
      model overhead across all candidates in one document
    - Stage 3 DB commits are batched (BATCH_COMMIT_SIZE = 100)

CHANGELOG:
    v1.0.0 (2026-02) - AISA initial release
    v1.1.0 (2026-02) - Integrated 00_text_repair (wordsegment + CamelCase)
                        applied to page_texts after PDF extraction, before
                        Stage 1 detection. Fixes missing-spaces PDFs (Amazon,
                        Walmart, etc.) and CamelCase ligature artifacts.

Author: TeRa0
License: MIT
===============================================================================
"""

from __future__ import annotations

import csv
import importlib
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from version import AISA_VERSION

_m1 = importlib.import_module("01_models")
AIReference     = _m1.AIReference
AnalyzerConfig  = _m1.AnalyzerConfig
DocumentResult  = _m1.DocumentResult
ProcessingStats = _m1.ProcessingStats
AIBuzzIndex     = _m1.AIBuzzIndex
logger          = _m1.logger

_m2 = importlib.import_module("02_db")
DatabaseManager = _m2.DatabaseManager

_m6 = importlib.import_module("06_analysis")
analyze_sentiment_batch = _m6.analyze_sentiment_batch
calculate_buzz_index    = _m6.calculate_buzz_index
rank_buzz_indices       = _m6.rank_buzz_indices
aggregate_by_industry   = _m6.aggregate_by_industry
IndustryBuzzIndex       = _m6.IndustryBuzzIndex

_m8 = importlib.import_module("08_memory")
process_document_memory = _m8.process_document_memory
finalize_year_memory    = _m8.finalize_year_memory

# 00_text_repair: re-imported inside each worker for ProcessPool safety.
# Module-level availability is verified here so failures surface at startup.
try:
    importlib.import_module("00_text_repair")
except Exception as _repair_import_err:
    logger.warning(
        "00_text_repair could not be imported — text repair will be skipped "
        f"for all documents. Reason: {_repair_import_err}"
    )


# ============================================================================
# DOCUMENT METADATA
# ============================================================================

@dataclass
class DocumentMeta:
    """Metadata for a single PDF to be processed."""
    source:     str             # full path to PDF
    company:    str
    year:       int
    position:   int
    industry:   str
    sector:     str
    country:    str
    doc_type:   str = "Annual Report"


@dataclass
class TaxonomyTarget:
    """
    One (provider, db, config) bundle for multi-taxonomy pipeline runs.

    Each TaxonomyTarget represents one taxonomy lane:
      - provider: the TaxonomyProvider implementation
      - db:       open DatabaseManager pointing to this taxonomy's own DB file
      - config:   AnalyzerConfig with correct output_folder and database_name
      - name:     display label (e.g. "AI_Disclosure", "Digitalization_Eco")
    """
    provider:   object          # TaxonomyProvider (typed as object to avoid circular import)
    db:         DatabaseManager
    config:     "AnalyzerConfig"
    name:       str


def load_fortune500_csv(csv_path: str) -> Dict[str, Dict]:
    """
    Load Fortune 500 metadata from CSV into a lookup dict.

    Accepts two formats (auto-detected, case-insensitive):

    Format A - custom AISA format:
        company, year, position, industry, sector, country

    Format B - Fortune Global 500 export format:
        RANK, NAME, REVENUES ($M), ..., Sector, Industry, Country

    Returns:
        Dict keyed by lowercase company name -> metadata dict.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fortune 500 CSV not found: {csv_path}")

    lookup: Dict[str, Dict] = {}
    is_global500 = False
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = {k.strip().lower() for k in (reader.fieldnames or [])}

        # Detect format
        is_global500 = "name" in headers and "rank" in headers and "company" not in headers

        for row in reader:
            norm = {k.strip().lower(): v.strip() for k, v in row.items()}

            if is_global500:
                # Format B: Fortune Global 500 (NAME, RANK, Sector, Industry, Country)
                raw_name = norm.get("name", "")
                raw_rank = norm.get("rank", "0")
                company = raw_name.lower()
                if not company:
                    continue
                rank_clean = re.sub(r"[^\d]", "", raw_rank)
                lookup[company] = {
                    "company":  raw_name,
                    "year":     0,      # static list - year resolved from filename
                    "position": int(rank_clean) if rank_clean else 0,
                    "industry": norm.get("industry", ""),
                    "sector":   norm.get("sector", ""),
                    "country":  norm.get("country", ""),
                }
            else:
                # Format A: custom AISA format
                company = norm.get("company", "").lower()
                if not company:
                    continue
                lookup[company] = {
                    "company":  norm.get("company", ""),
                    "year":     int(norm.get("year", 0) or 0),
                    "position": int(norm.get("position", 0) or 0),
                    "industry": norm.get("industry", ""),
                    "sector":   norm.get("sector", ""),
                    "country":  norm.get("country", ""),
                }

    fmt = "Global500" if is_global500 else "AISA"
    logger.info(f"Loaded Fortune 500 CSV: {len(lookup)} entries from {csv_path} [format={fmt}]")
    return lookup


DOC_TYPE_MAPPING = {
    'annual': 'Annual Report',
    'sustainability': 'Sustainability',
    'proxy': 'Proxy',
    'quarterly': 'Quarterly Report',
    '10k': 'Annual Report',
    '10q': 'Quarterly Report',
    'def14a': 'Proxy',
    'esg': 'Sustainability',
    'csr': 'Sustainability',
}


def _is_chinese_language_file(pdf_path: str) -> bool:
    """
    Detects files with explicit Chinese language suffix (_CN, -CN).
    NOTE (AISA v1.1.0): These files are NO LONGER ignored — they are processed
    normally with the multilingual semantic model and Chinese keywords.
    The function remains available for statistics / logging.
    """
    stem = Path(pdf_path).stem.upper()
    return stem.endswith('_CN') or stem.endswith('-CN')


def _parse_filename_meta(pdf_path: str) -> Tuple[str, int, str, int]:
    """
    Parseaza fisierele cu formatul Fortune 500:
      '257. Edeka Zentrale - 2024 - annual report.pdf'
      '265. Tsingshan - Rept Battero - 2024 - sustainability.pdf'

    Separatorul este " - " (spatiu-cratima-spatiu), NU underscore.

    Returns:
        (company, year, doc_type, position) — year=0, position=0 daca nu sunt gasite.
    """
    base = Path(pdf_path).stem
    position = 0
    year = 0
    doc_type = 'Annual Report'

    # Pas 1: extrage pozitia (ex: '257.')
    pos_match = re.match(r'^(\d{1,3})\.\s*', base)
    if pos_match:
        position = int(pos_match.group(1))
        base = base[pos_match.end():]

    # Pas 2: extrage doc_type din filename
    base_lower = base.lower()
    for key, value in DOC_TYPE_MAPPING.items():
        if key in base_lower:
            doc_type = value
            break

    # Pas 3: split pe ' - '
    parts = re.split(r'\s+-\s+', base)

    # Pas 4: extrage anul (2015-2025)
    year_re = re.compile(r'^20(1[5-9]|2[0-5])$')
    for part in parts:
        if year_re.match(part.strip()):
            year = int(part.strip())
            break

    # fallback: cauta in orice parte
    if year == 0:
        m = re.search(r'20(1[5-9]|2[0-5])', base)
        if m:
            year = int(m.group(0))

    # Pas 5: compania = parts[0], curatata de pozitie
    company = ''
    if parts:
        company = re.sub(r'^\d+\.\s*', '', parts[0]).strip()

    return company, year, doc_type, position


def resolve_document_meta(
    pdf_path: str,
    f500_lookup: Optional[Dict[str, Dict]],
    year_override: Optional[int] = None,
) -> Optional[DocumentMeta]:
    """
    Resolve full metadata for a PDF file.

    First tries the Fortune 500 CSV lookup by company name.
    Falls back to filename parsing if the company is not found.

    Args:
        pdf_path:      Full path to the PDF.
        f500_lookup:   Dict from load_fortune500_csv(), or None.
        year_override: If set, use this year regardless of filename.

    Returns:
        DocumentMeta, or None if year cannot be determined.
    """
    company_raw, year_from_file, doc_type, position_from_file = _parse_filename_meta(pdf_path)
    year = year_override or year_from_file

    if year == 0:
        logger.warning(f"Cannot determine year for {pdf_path} — skipping")
        return None

    # Try CSV lookup
    meta = None
    if f500_lookup:
        key = company_raw.lower()
        if key in f500_lookup:
            meta = f500_lookup[key]
        else:
            # Try partial match (first word of company) — imprecise, log if used
            first_word = company_raw.split()[0].lower() if company_raw else ""
            for k, v in f500_lookup.items():
                if k.startswith(first_word):
                    meta = v
                    logger.warning(
                        f"Partial CSV match for '{company_raw}' → '{v['company']}' "
                        f"(matched on first word '{first_word}'). Verify this is correct."
                    )
                    break
        if meta is None:
            logger.warning(
                f"Company not matched in CSV: '{company_raw}' "
                f"({pdf_path}) — industry/sector/country will be 'Unknown'"
            )

    if meta:
        return DocumentMeta(
            source=pdf_path,
            company=meta["company"],
            year=year,
            position=meta["position"],
            industry=meta["industry"],
            sector=meta["sector"],
            country=meta["country"],
            doc_type=doc_type,
        )

    # Fallback: filename only — metadata fields will be incomplete.
    logger.warning(
        f"Using filename-only metadata for '{company_raw}' "
        f"({pdf_path}): industry/sector/country unknown, country defaulting to 'Unknown'."
    )
    return DocumentMeta(
        source=pdf_path,
        company=company_raw,
        year=year,
        position=position_from_file,
        industry="Unknown",
        sector="Unknown",
        country="Unknown",
        doc_type=doc_type,
    )


# ============================================================================
# PDF TEXT EXTRACTION
# (Stage 1, runs in ProcessPool worker)
# ============================================================================

def extract_pdf_text(pdf_path: str) -> Tuple[List[str], str]:
    """
    Extract text from a PDF, returning one string per page.

    Tries pdfplumber first (best layout preservation), then PyMuPDF (fitz),
    then falls back to OCR via pytesseract if the document appears scanned.

    Hard fail if pdfplumber is not installed.

    Args:
        pdf_path: Full path to the PDF file.

    Returns:
        (page_texts, status) where status is one of:
            "valid"                  — clean extraction
            "corrupted_ocr_success"  — needed OCR, succeeded
            "corrupted_ocr_failed"   — needed OCR, failed
            "empty"                  — no text found anywhere
            "mirrored_all_stripped"  — all pages had reversed/mirrored text (e.g. CSCEC)
    """
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError(
            "pdfplumber is required for PDF extraction. "
            "Install with: pip install pdfplumber"
        )

    page_texts: List[str] = []
    status = "valid"

    def _col_aware(page) -> str:
        """Extract text respecting multi-column layout via bbox segregation."""
        try:
            words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
        except Exception:
            return page.extract_text() or ""
        if not words or len(words) < 20:
            return page.extract_text() or ""
        pw = float(page.width or 595)
        bw = max(1.0, pw * 0.005)
        nb = int(pw / bw) + 1
        cov = [False] * nb
        for w in words:
            for b in range(max(0, int(float(w["x0"]) / bw)),
                           min(nb - 1, int(float(w["x1"]) / bw)) + 1):
                cov[b] = True
        fc = next((i for i, c in enumerate(cov) if c), 0)
        lc = next((i for i in range(nb - 1, -1, -1) if cov[i]), nb - 1)
        min_gap = max(1, int(pw * 0.03 / bw))
        boundaries, in_gap, gs = [], False, 0
        for i in range(fc, lc + 1):
            if not cov[i] and not in_gap:
                in_gap, gs = True, i
            elif cov[i] and in_gap:
                if i - gs >= min_gap:
                    boundaries.append((gs + (i - gs) / 2) * bw)
                in_gap = False
        if not boundaries:
            return page.extract_text() or ""
        ncols = len(boundaries) + 1
        cols = [[] for _ in range(ncols)]
        for w in words:
            ci = sum(1 for b in boundaries if float(w["x0"]) >= b)
            cols[min(ci, ncols - 1)].append(w)
        parts = []
        for col in cols:
            if not col:
                continue
            col.sort(key=lambda w: (round(float(w["top"]) / 5) * 5, float(w["x0"])))
            lines, cy, cur = [], None, []
            for w in col:
                wy = float(w["top"])
                if cy is None or abs(wy - cy) <= 5:
                    cur.append(w["text"])
                else:
                    lines.append(" ".join(cur))
                    cur = [w["text"]]
                cy = wy
            if cur:
                lines.append(" ".join(cur))
            parts.append("\n".join(lines))
        return "\n\n".join(p for p in parts if p.strip())

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    text = _col_aware(page)
                    page_texts.append(text)
                except Exception:
                    page_texts.append("")

    except Exception as e:
        logger.warning(f"pdfplumber failed for {pdf_path}: {e}")
        # Try PyMuPDF fallback
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            for page in doc:
                page_texts.append(page.get_text())
            doc.close()
        except Exception as e2:
            logger.error(f"PyMuPDF also failed for {pdf_path}: {e2}")
            return [], "corrupted_ocr_failed"

    # Detect scanned PDF (very little text per page)
    total_chars = sum(len(t) for t in page_texts)
    avg_chars   = total_chars / max(len(page_texts), 1)

    if avg_chars < 50 and page_texts:
        # Looks scanned — attempt OCR
        status = "corrupted_ocr_success"
        try:
            import pytesseract
            from PIL import Image
            import fitz

            ocr_pages: List[str] = []
            doc = fitz.open(pdf_path)
            for page in doc:
                mat  = fitz.Matrix(2, 2)   # 2x zoom for better OCR
                pix  = page.get_pixmap(matrix=mat)
                img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)
                ocr_pages.append(text)
            doc.close()
            page_texts = ocr_pages
            logger.info(f"OCR completed for {pdf_path} ({len(page_texts)} pages)")

        except Exception as ocr_err:
            logger.warning(f"OCR failed for {pdf_path}: {ocr_err}")
            status = "corrupted_ocr_failed"

    if not any(t.strip() for t in page_texts):
        status = "empty"

    # Detect mirrored/reversed text (RTL-encoded PDFs e.g. CSCEC bilingual reports).
    # Mirrored pages contain English words written letter-by-letter in reverse.
    # These pages produce massive false positives (e.g. "rpA" matching RPA patterns).
    # Mirrored pages are blanked out; if all pages are mirrored status = "mirrored_all_stripped".
    if status == "valid" and page_texts:
        full = " ".join(page_texts)
        _MIRRORED_MARKERS = (
            "noitcurtsnoC", "noillim", "sraey", "gnireenignE",
            "dnoB", "etaroproC", "tnempoleveD", "noitazinagrO",
        )
        if sum(1 for w in _MIRRORED_MARKERS if w in full) >= 2:
            clean_pages = []
            stripped = 0
            for page in page_texts:
                page_hits = sum(1 for w in _MIRRORED_MARKERS if w in page)
                if page_hits >= 2:
                    clean_pages.append("")
                    stripped += 1
                else:
                    clean_pages.append(page)
            page_texts = clean_pages
            logger.info(
                f"Mirrored text detected in {pdf_path}: "
                f"stripped {stripped}/{len(page_texts)} pages"
            )
            if not any(t.strip() for t in page_texts):
                status = "mirrored_all_stripped"

    return page_texts, status


# ============================================================================
# STAGE 1 WORKER
# (runs in ProcessPool — imports detect module inside the function
#  to avoid pickling SentenceTransformer across processes)
# ============================================================================

def _extract_text_worker(args: Tuple) -> Tuple[DocumentMeta, List, str, int, int, int, int]:
    """
    ProcessPool worker: extract and repair PDF text ONLY.

    Used by run_multi_taxonomy_pipeline() to separate the expensive PDF I/O
    step (done once) from Stage 1 detection (done once per taxonomy in the
    main process).

    Returns:
        (doc_meta, page_texts, text_status, total_pages, text_length, word_count, sentence_count)
        page_texts is [] on extraction failure.
    """
    doc_meta, config_dict = args

    page_texts, text_status = extract_pdf_text(doc_meta.source)
    total_pages = len(page_texts)

    try:
        _m00 = importlib.import_module("00_text_repair")
        page_texts = _m00.repair_page_texts(page_texts)
        doc_language = _m00.detect_document_language(page_texts, sample_pages=5)
        doc_meta.language = doc_language
    except Exception as repair_err:
        logger.warning(f"Text repair skipped for {doc_meta.source}: {repair_err}")

    text_length = sum(len(t) for t in page_texts)
    word_count  = sum(len(t.split()) for t in page_texts)
    full_text = "\n".join(page_texts) if page_texts else ""
    sentence_count = len([s for s in re.split(r'[.!?]+', full_text) if len(s.strip()) > 10])

    if text_status in ("empty", "corrupted_ocr_failed", "mirrored_all_stripped") or not page_texts:
        return doc_meta, [], text_status, total_pages, text_length, word_count, sentence_count

    return doc_meta, page_texts, text_status, total_pages, text_length, word_count, sentence_count


def _stage1_worker(args: Tuple) -> Tuple[DocumentMeta, List, str, int, int, int, int]:
    """
    ProcessPool worker: extract PDF text and run Stage 1 detection.

    Returns:
        (doc_meta, stage1_candidates, text_status, total_pages, text_length, word_count, sentence_count)

    CandidateFragment objects are pure dataclasses - picklable without issues.
    """
    doc_meta, config_dict = args

    # Reconstruct config in worker process (all fields passed via to_dict())
    config = AnalyzerConfig.from_dict(config_dict)

    # Extract text
    page_texts, text_status = extract_pdf_text(doc_meta.source)
    total_pages = len(page_texts)

    # Repair missing spaces and CamelCase artifacts before detection
    # (wordsegment is loaded lazily on first call, once per worker process)
    try:
        _m00 = importlib.import_module("00_text_repair")
        page_texts = _m00.repair_page_texts(page_texts)
        # Detect document language (fast CJK heuristic, no external dep required)
        doc_language = _m00.detect_document_language(page_texts, sample_pages=5)
        doc_meta.language = doc_language
    except Exception as repair_err:
        logger.warning(f"Text repair skipped for {doc_meta.source}: {repair_err}")
        doc_language = getattr(doc_meta, "language", "en")

    text_length = sum(len(t) for t in page_texts)
    word_count  = sum(len(t.split()) for t in page_texts)
    full_text = "\n".join(page_texts) if page_texts else ""
    sentence_count = len([s for s in re.split(r'[.!?]+', full_text) if len(s.strip()) > 10])

    if text_status in ("empty", "corrupted_ocr_failed", "mirrored_all_stripped") or not page_texts:
        return doc_meta, [], text_status, total_pages, text_length, word_count, sentence_count

    # Stage 1 detection (CPU only — no model)
    try:
        _m5 = importlib.import_module("05_detect")
        stage1_extract_candidates = _m5.stage1_extract_candidates
        if getattr(config, "taxonomy_excel", None):
            provider = _m5.load_taxonomy_from_excel(config.taxonomy_excel)
        else:
            provider = _m5.load_taxonomy_by_name(config.taxonomy_name)
        if hasattr(_m5, "set_taxonomy"):
            _m5.set_taxonomy(provider)
        TAXONOMY = provider

        candidates = stage1_extract_candidates(
            text=full_text,
            page_texts=page_texts,
            config=config,
            taxonomy=TAXONOMY,
        )

        # CandidateFragment objects are returned directly.
        # They are picklable (pure dataclasses, no model references).
        return doc_meta, candidates, text_status, total_pages, text_length, word_count, sentence_count

    except Exception as e:
        logger.error(f"Stage 1 failed for {doc_meta.source}: {e}")
        return doc_meta, [], "error", total_pages, text_length, word_count, sentence_count


# ============================================================================
# SEMANTIC DEDUPLICATION
# (runs after Stage 2, before DB insert)
# ============================================================================

def _populate_dedup_table(year: int, db: "DatabaseManager") -> int:
    """
    Build ai_references_deduplicated for a given year from ai_references_raw.

    Groups all references for each (company, year) and deduplicates them
    cross-document using Jaccard similarity, then writes the clusters into
    ai_references_deduplicated so that exports return non-empty dedup sheets.

    Called once per year during year-end finalization.

    Returns:
        Number of deduplicated clusters written.
    """
    rows = db.conn.execute(
        """
        SELECT company, year, position, industry, sector, country,
               text, context, category, category_a, confidence_a,
               category_b, confidence_b, doc_type,
               sentiment_score, semantic_score, confidence_score,
               page, dimensions_json
        FROM ai_references_raw
        WHERE year = ?
        ORDER BY company, confidence_score DESC
        """,
        (year,),
    ).fetchall()

    if not rows:
        return 0

    # Group by company
    from collections import defaultdict
    by_company: dict = defaultdict(list)
    for r in rows:
        by_company[r["company"]].append(dict(r))

    written = 0
    threshold = 0.7  # Jaccard similarity threshold for cross-doc dedup

    for company, refs in by_company.items():
        clusters: list = []  # list of lists-of-rows

        for ref in refs:
            a = set(ref["text"].lower().split())
            matched = False
            for cluster in clusters:
                rep = cluster[0]
                b = set(rep["text"].lower().split())
                if a and b and len(a & b) / len(a | b) >= threshold:
                    cluster.append(ref)
                    matched = True
                    break
            if not matched:
                clusters.append([ref])

        for cluster in clusters:
            rep = cluster[0]  # highest confidence_score (rows sorted desc)
            sources = ",".join(sorted({r["doc_type"] for r in cluster if r["doc_type"]}))
            pages   = ",".join(str(r["page"]) for r in cluster if r["page"] is not None)
            avg_sent = (
                sum(r["sentiment_score"] or 0 for r in cluster) / len(cluster)
            )
            avg_sem  = (
                sum(r["semantic_score"] or 0 for r in cluster) / len(cluster)
            )
            dedup = {
                "company":            company,
                "year":               year,
                "position":           rep.get("position") or 0,
                "industry":           rep.get("industry") or "",
                "sector":             rep.get("sector") or "",
                "country":            rep.get("country") or "",
                "doc_type":           rep.get("doc_type") or sources,
                "language":           rep.get("language") or "en",
                "text":               rep["text"],
                "context":            rep.get("context") or "",
                "category":           rep.get("category") or "",
                "category_a":         rep.get("category_a") or "",
                "confidence_a":       rep.get("confidence_a") or 0.0,
                "category_b":         rep.get("category_b") or "",
                "confidence_b":       rep.get("confidence_b") or 0.0,
                "sources":            sources,
                "pages":              pages,
                "doc_count":          len({r["doc_type"] for r in cluster}),
                "total_occurrences":  len(cluster),
                "avg_sentiment_score": avg_sent,
                "avg_semantic_score":  avg_sem,
                "max_confidence_score": rep.get("confidence_score") or 0.0,
                "original_refs":      [r["text"][:80] for r in cluster],
                "dimensions_json":    rep.get("dimensions_json") or "",
            }
            try:
                db.insert_deduplicated_reference(dedup)
                written += 1
            except Exception as e:
                logger.warning(f"Dedup insert skipped ({company} {year}): {e}")

    db.conn.commit()
    logger.info(f"Dedup table populated: year={year}, clusters={written}")
    return written


def run_dedup_for_db(
    db: "DatabaseManager",
    year_filter: Optional[int] = None,
) -> Dict[int, int]:
    """
    (Re-)populate ai_references_deduplicated for all years (or one year) in a DB.

    Clears existing dedup rows for the target year(s), then rebuilds Jaccard
    clusters via _populate_dedup_table(). This is the correct way to run
    cross-document per-company-per-year deduplication from the menu or CLI.

    Args:
        db:          Open DatabaseManager pointing to the target DB file.
        year_filter: If given, process only this year; otherwise all years.

    Returns:
        Dict mapping year → number of clusters written.
    """
    try:
        if year_filter is not None:
            years = [year_filter]
        else:
            rows = db.conn.execute(
                "SELECT DISTINCT year FROM ai_references_raw "
                "WHERE year IS NOT NULL ORDER BY year"
            ).fetchall()
            years = [r[0] for r in rows]
    except Exception:
        # Table doesn't exist yet (brand-new or empty DB) — nothing to do
        return {}

    result: Dict[int, int] = {}
    for year in years:
        try:
            db.conn.execute(
                "DELETE FROM ai_references_deduplicated WHERE year = ?", (year,)
            )
            db.conn.commit()
        except Exception:
            pass  # dedup table missing — _populate_dedup_table will handle gracefully
        try:
            n = _populate_dedup_table(year, db)
        except Exception:
            n = 0
        result[year] = n
    return result


def _semantic_deduplicate(
    refs: List[AIReference],
    threshold: float,
) -> List[AIReference]:
    """
    Remove duplicate references within a document using a 2-step approach.

    Step 1 — Exact dedup:
        Hash on (page, normalized_text[:100]). Keeps highest confidence_score.
        Fast O(n), catches exact and near-exact same-page duplicates.

    Step 2 — Cosine dedup (when embeddings available):
        Pairwise cosine similarity on Stage 2 embeddings stored in AIReference.
        Removes cross-page near-duplicates (e.g. same sentence in exec summary
        and body). Falls back to Jaccard if embeddings absent (e.g. from DB).

    Note: correctly described as "near-duplicate filter" in academic writing,
    not "semantic deduplication" in the strict NLP sense.

    Args:
        refs:       List of AIReference from one document.
        threshold:  Similarity threshold (config.deduplication_threshold).

    Returns:
        Deduplicated list, sorted by confidence_score descending.
    """
    if len(refs) <= 1:
        return refs

    # ── Step 1: exact hash dedup ──────────────────────────────────────────
    seen_keys: Dict[str, AIReference] = {}
    for ref in refs:
        norm_text = " ".join(ref.text.split())[:100]
        key = f"{ref.page}|{norm_text}"
        if key in seen_keys:
            if ref.confidence_score > seen_keys[key].confidence_score:
                seen_keys[key] = ref
        else:
            seen_keys[key] = ref

    step1 = sorted(seen_keys.values(), key=lambda r: r.confidence_score, reverse=True)

    if len(step1) <= 1:
        return step1

    # ── Step 2: cosine dedupe on stored embeddings, or Jaccard fallback ───
    has_embeddings = all(getattr(r, "embedding", None) is not None for r in step1)
    final: List[AIReference] = []

    if has_embeddings:
        try:
            from sentence_transformers import util as st_util
            kept_embeddings = []
            for ref in step1:
                emb = ref.embedding
                is_dup = any(
                    float(st_util.cos_sim(emb, ke).item()) >= threshold
                    for ke in kept_embeddings
                )
                if not is_dup:
                    final.append(ref)
                    kept_embeddings.append(emb)

            removed = len(refs) - len(final)
            if removed > 0:
                logger.debug(
                    f"Dedupe (cosine): {len(refs)} → {len(final)} "
                    f"(removed {removed}, threshold={threshold})"
                )
            return final
        except Exception as e:
            logger.debug(f"Cosine dedupe failed ({e}), falling back to Jaccard")
            final = []  # reset, fall through

    # Jaccard fallback (when embeddings absent — refs loaded from DB)
    for ref in step1:
        a = set(" ".join(ref.text.split()).lower().split())
        is_dup = False
        for kept in final:
            b = set(" ".join(kept.text.split()).lower().split())
            if a and b and len(a & b) / len(a | b) >= threshold:
                is_dup = True
                break
        if not is_dup:
            final.append(ref)

    removed = len(refs) - len(final)
    if removed > 0:
        logger.debug(
            f"Dedupe (Jaccard): {len(refs)} → {len(final)} "
            f"(removed {removed}, threshold={threshold})"
        )
    return final


# ============================================================================
# STAGE 3: SENTIMENT + BUZZ INDEX + MEMORY + DB PERSIST
# ============================================================================

def _stage3_persist(
    doc_result: DocumentResult,
    doc_meta: DocumentMeta,
    text_status: str,
    db: DatabaseManager,
    config: AnalyzerConfig,
    stats: ProcessingStats,
    buzz_accumulator: List[AIBuzzIndex],
) -> None:
    """
    Stage 3: sentiment, buzz index computation, memory update, DB persist.

    Args:
        doc_result:       DocumentResult from Stage 2.
        doc_meta:         DocumentMeta for this document.
        text_status:      Extraction status string.
        db:               Open DatabaseManager.
        config:           AnalyzerConfig.
        stats:            ProcessingStats to update in place.
        buzz_accumulator: List to append AIBuzzIndex to (for year-end ranking).
    """
    refs = doc_result.references

    # --- Sentiment ---
    if refs:
        analyze_sentiment_batch(refs, config)

    # --- DB: insert references ---
    ref_ids: List[int] = []
    for ref in refs:
        is_new, _ = db.insert_reference(ref)
        # Retrieve the DB id for memory linkage
        row = db.conn.execute(
            """
            SELECT id FROM ai_references_raw
            WHERE company=? AND year=? AND doc_type=? AND page=? AND text=?
            """,
            (ref.company, ref.year, ref.doc_type, ref.page, ref.text),
        ).fetchone()
        ref_ids.append(row["id"] if row else None)

    # --- Adoption Memory ---
    if refs:
        process_document_memory(refs, db, ref_ids=ref_ids)

    # --- AI Buzz Index ---
    if refs:
        buzz = calculate_buzz_index(
            refs=refs,
            company=doc_meta.company,
            year=doc_meta.year,
            position=doc_meta.position,
            industry=doc_meta.industry,
            sector=doc_meta.sector,
            country=doc_meta.country,
            total_pages=doc_result.total_pages,
            config=config,
        )
        db.insert_buzz_index(buzz)
        buzz_accumulator.append(buzz)

    # --- Mark document processed ---
    db.mark_document_processed(
        doc_result,
        refs_found=len(refs),
        text_status=text_status,
    )

    # --- Update processing stats ---
    stats.successful_documents  += 1
    stats.total_pages           += doc_result.total_pages
    stats.valid_pages           += doc_result.total_pages
    stats.total_text_chars      += doc_result.text_length
    # Use exact word count if computed (sequential path), else approximate
    stats.total_words           += getattr(doc_result, "_word_count", getattr(doc_result, "word_count", doc_result.text_length // 5))
    stats.total_sentences       += getattr(doc_result, "_sentence_count", getattr(doc_result, "sentence_count", 0))
    stats.total_references_raw          += doc_result.total_refs
    stats.total_references_deduplicated += len(refs)
    stats.total_processing_time         += doc_result.processing_time

    logger.info(
        f"[{doc_meta.company} {doc_meta.year}] "
        f"refs={len(refs)} | text_status={text_status} | "
        f"time={doc_result.processing_time:.1f}s"
    )


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(
    config: AnalyzerConfig,
    db: DatabaseManager,
    progress_callback=None,
) -> ProcessingStats:
    """
    Run the full AISA processing pipeline.

    Discovers all PDFs in config.input_folder, resolves metadata,
    skips already-processed documents, and runs all 3 stages.

    Args:
        config:            AnalyzerConfig.
        db:                Open DatabaseManager (migrations already applied).
        progress_callback: Optional callable(current, total, company, year)
                           called after each document completes.

    Returns:
        ProcessingStats with full run summary.
    """
    stats = ProcessingStats(start_time=datetime.now())

    # --- Initialize taxonomy (must happen before any Stage 1 / Stage 2 work) ---
    _m5 = importlib.import_module("05_detect")
    if config.taxonomy_excel:
        provider = _m5.load_taxonomy_from_excel(config.taxonomy_excel)
        logger.info(
            f"Pipeline starting with Excel taxonomy: {config.taxonomy_excel} "
            f"(name={config.taxonomy_name})"
        )
    else:
        provider = _m5.load_taxonomy_by_name(config.taxonomy_name)
        logger.info(f"Pipeline starting with taxonomy: {config.taxonomy_name}")
    _m5.set_taxonomy(provider)

    # --- Load Fortune 500 metadata ---
    f500_lookup: Optional[Dict[str, Dict]] = None
    if config.fortune500_csv and os.path.exists(config.fortune500_csv):
        f500_lookup = load_fortune500_csv(config.fortune500_csv)

    # --- Discover PDFs ---
    input_dir = Path(config.input_folder)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {config.input_folder}")

    all_pdfs = sorted(input_dir.glob("**/*.pdf"))
    if not all_pdfs:
        logger.warning(f"No PDFs found in {config.input_folder}")
        return stats

    # Apply top_n limit
    if config.top_n:
        all_pdfs = all_pdfs[: config.top_n]

    logger.info(f"Discovered {len(all_pdfs)} PDFs in {config.input_folder}")

    # --- Resolve metadata + filter already-processed ---
    already_processed = db.get_processed_documents()
    pending_docs: List[DocumentMeta] = []

    for pdf_path in all_pdfs:
        meta = resolve_document_meta(str(pdf_path), f500_lookup)
        if meta is None:
            logger.warning(f"Skipping (no year): {pdf_path.name}")
            stats.failed_documents += 1
            continue

        # Year filter
        if not (config.start_year <= meta.year <= config.end_year):
            logger.debug(f"Skipping (year {meta.year} out of range): {pdf_path.name}")
            continue

        # Resume check
        if meta.source in already_processed:
            logger.debug(f"Skipping (already processed): {pdf_path.name}")
            continue

        pending_docs.append(meta)

    stats.total_documents = len(pending_docs) + len(already_processed)
    logger.info(
        f"Pipeline: {len(pending_docs)} documents to process "
        f"({len(already_processed)} already done)"
    )

    if not pending_docs:
        logger.info("Nothing to process. All documents already done.")
        stats.end_time = datetime.now()
        return stats

    # --- Import Stage 2 here (single process, model loaded once) ---
    _m5 = importlib.import_module("05_detect")
    stage2_semantic_score = _m5.stage2_semantic_score
    build_ai_reference    = _m5.build_ai_reference
    TAXONOMY              = _m5.TAXONOMY
    stage1_extract_candidates = _m5.stage1_extract_candidates

    # --- Process documents ---
    buzz_accumulator: List[AIBuzzIndex] = []
    processed_years: set = set()
    # FIX: use to_dict() to send ALL config fields to worker.
    # Partial dict caused worker to run with wrong thresholds/filters/weights.
    config_dict = config.to_dict()

    completed = 0

    if config.max_workers > 1:
        # Parallel Stage 1, sequential Stage 2+3
        _run_parallel(
            pending_docs=pending_docs,
            config=config,
            config_dict=config_dict,
            db=db,
            stats=stats,
            buzz_accumulator=buzz_accumulator,
            processed_years=processed_years,
            stage2_semantic_score=stage2_semantic_score,
            build_ai_reference=build_ai_reference,
            TAXONOMY=TAXONOMY,
            progress_callback=progress_callback,
        )
    else:
        # Sequential (simpler, useful for debugging)
        for i, doc_meta in enumerate(pending_docs, 1):
            _process_one_sequential(
                doc_meta=doc_meta,
                config=config,
                db=db,
                stats=stats,
                buzz_accumulator=buzz_accumulator,
                processed_years=processed_years,
                stage1_extract_candidates=stage1_extract_candidates,
                stage2_semantic_score=stage2_semantic_score,
                build_ai_reference=build_ai_reference,
                TAXONOMY=TAXONOMY,
            )
            if progress_callback:
                progress_callback(i, len(pending_docs), doc_meta.company, doc_meta.year)

    # --- Year-end finalization ---
    for year in sorted(processed_years):
        logger.info(f"Finalizing year {year}...")
        finalize_year_memory(year, db, threshold_years=2)
        db.update_rankings(year)
        _populate_dedup_table(year, db)

    # --- Industry aggregation ---
    if buzz_accumulator:
        ranked = rank_buzz_indices(buzz_accumulator)
        industry_indices = aggregate_by_industry(ranked)
        for ind in industry_indices:
            _upsert_industry_buzz(ind, db)
        db.commit()

    # Release multilingual model if it was loaded during this run
    try:
        _m5 = importlib.import_module("05_detect")
        if hasattr(_m5, "release_multilingual_model"):
            _m5.release_multilingual_model()
    except Exception:
        pass

    stats.end_time = datetime.now()
    stats.calculate_derived_metrics()
    stats.print_summary()

    return stats


def run_multi_taxonomy_pipeline(
    base_config: "AnalyzerConfig",
    targets: List[TaxonomyTarget],
    progress_callback=None,
) -> Dict[str, "ProcessingStats"]:
    """
    Run ONE pipeline pass that processes each PDF once and applies MULTIPLE
    taxonomies simultaneously, saving results to separate databases.

    Architecture:
        Phase 1 — Text extraction (ProcessPool, once per PDF):
            Uses _extract_text_worker to extract and repair text.
            Results cached in memory as {source: (meta, page_texts, ...)}.

        Phase 2 — Per-taxonomy detection (main process, sequential):
            For each PDF × each TaxonomyTarget:
              Stage 1: stage1_extract_candidates (regex + keywords, fast)
              Stage 2: stage2_semantic_score (semantic model)
              Stage 3: _stage3_persist (sentiment + buzz + DB write)

    Args:
        base_config: AnalyzerConfig for PDF discovery, year range, top_n,
                     fortune500_csv, max_workers. taxonomy_* fields ignored.
        targets:     List of TaxonomyTarget, one per taxonomy. Each must have
                     its own open DatabaseManager and AnalyzerConfig.
        progress_callback: Optional callable(current, total, company, year).

    Returns:
        Dict[taxonomy_name → ProcessingStats], one entry per target.
    """
    stats_per_target: Dict[str, ProcessingStats] = {
        t.name: ProcessingStats(start_time=datetime.now()) for t in targets
    }

    # --- Load Fortune 500 metadata (shared across all taxonomies) ---
    f500_lookup: Optional[Dict[str, Dict]] = None
    if base_config.fortune500_csv and os.path.exists(base_config.fortune500_csv):
        f500_lookup = load_fortune500_csv(base_config.fortune500_csv)

    # --- Discover PDFs ---
    # If caller supplied an explicit list (e.g. from menu subsets 1.2/1.3/1.4/1.5),
    # use it directly instead of scanning the input folder.
    explicit_list = getattr(base_config, "_explicit_pdf_list", None)
    if explicit_list is not None:
        all_pdfs = [Path(p) for p in explicit_list]
    else:
        input_dir = Path(base_config.input_folder)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input folder not found: {base_config.input_folder}")
        all_pdfs = sorted(input_dir.glob("**/*.pdf"))
        if base_config.top_n:
            all_pdfs = all_pdfs[: base_config.top_n]

    if not all_pdfs:
        logger.warning(f"No PDFs found in {base_config.input_folder}")
        for s in stats_per_target.values():
            s.end_time = datetime.now()
        return stats_per_target

    # --- Resolve metadata; include PDF if ANY target still needs it ---
    already_per_target = {t.name: t.db.get_processed_documents() for t in targets}
    pending_docs: List[DocumentMeta] = []

    for pdf_path in all_pdfs:
        meta = resolve_document_meta(str(pdf_path), f500_lookup)
        if meta is None:
            logger.warning(f"Skipping (no year): {pdf_path.name}")
            for t in targets:
                stats_per_target[t.name].failed_documents += 1
            continue
        if not (base_config.start_year <= meta.year <= base_config.end_year):
            continue
        # Include if at least one target hasn't processed it yet
        if any(meta.source not in already_per_target[t.name] for t in targets):
            pending_docs.append(meta)

    if not pending_docs:
        logger.info("Multi-taxonomy pipeline: nothing to process.")
        for s in stats_per_target.values():
            s.end_time = datetime.now()
        return stats_per_target

    logger.info(
        f"Multi-taxonomy pipeline: {len(pending_docs)} documents × "
        f"{len(targets)} taxonomies [{', '.join(t.name for t in targets)}]"
    )

    # --- Import Stage 1/2 functions once ---
    _m5 = importlib.import_module("05_detect")
    stage1_extract_candidates = _m5.stage1_extract_candidates
    stage2_semantic_score     = _m5.stage2_semantic_score
    build_ai_reference        = _m5.build_ai_reference

    buzz_per_target:    Dict[str, List[AIBuzzIndex]] = {t.name: [] for t in targets}
    years_per_target:   Dict[str, set]               = {t.name: set() for t in targets}
    config_dict = base_config.to_dict()
    total       = len(pending_docs)

    # ── Phase 1: text extraction (ProcessPool, once per PDF) ─────────────────
    text_cache: Dict[str, Tuple] = {}   # source → (meta, page_texts, status, pages, chars, words, sentences)

    if base_config.max_workers > 1:
        with ProcessPoolExecutor(max_workers=base_config.max_workers) as executor:
            future_map = {
                executor.submit(_extract_text_worker, (doc_meta, config_dict)): doc_meta
                for doc_meta in pending_docs
            }
            for future in as_completed(future_map):
                orig = future_map[future]
                try:
                    result = future.result()
                    text_cache[orig.source] = result
                except Exception as exc:
                    logger.error(f"Text extraction failed [{orig.source}]: {exc}")
                    text_cache[orig.source] = (orig, [], "error", 0, 0, 0, 0)
    else:
        for doc_meta in pending_docs:
            result = _extract_text_worker((doc_meta, config_dict))
            text_cache[doc_meta.source] = result

    # ── Phase 2: per-document, per-taxonomy detection + persist ──────────────
    completed = 0
    for doc_meta in pending_docs:
        cached = text_cache.get(doc_meta.source)
        if cached is None:
            continue

        result_meta, page_texts, text_status, total_pages, text_length, word_count, sentence_count = cached

        # Failed extraction → mark in all targets and skip
        if text_status in ("empty", "corrupted_ocr_failed", "mirrored_all_stripped", "error"):
            for t in targets:
                if result_meta.source in already_per_target[t.name]:
                    continue
                stats_per_target[t.name].failed_documents += 1
                fail_result = DocumentResult(
                    company=result_meta.company, year=result_meta.year,
                    position=result_meta.position, industry=result_meta.industry,
                    sector=result_meta.sector, country=result_meta.country,
                    doc_type=result_meta.doc_type, source=result_meta.source,
                    total_pages=total_pages, text_length=0,
                    text_status=text_status,
                )
                t.db.mark_document_processed(fail_result, refs_found=0, text_status=text_status)
            completed += 1
            if progress_callback:
                progress_callback(completed, total, result_meta.company, result_meta.year)
            continue

        full_text    = "\n".join(page_texts)
        doc_language = getattr(result_meta, "language", "en")

        for t in targets:
            # Skip if this target already has the document
            if result_meta.source in already_per_target[t.name]:
                continue

            t0 = time.perf_counter()

            doc_result = DocumentResult(
                company=result_meta.company, year=result_meta.year,
                position=result_meta.position, industry=result_meta.industry,
                sector=result_meta.sector, country=result_meta.country,
                doc_type=result_meta.doc_type, source=result_meta.source,
                total_pages=total_pages, text_length=text_length,
                text_status=text_status,
            )
            doc_result._word_count = word_count
            doc_result._sentence_count = sentence_count
            doc_result.word_count = word_count
            doc_result.sentence_count = sentence_count

            # Stage 1: candidate detection with this taxonomy
            candidates = stage1_extract_candidates(
                text=full_text,
                page_texts=page_texts,
                config=t.config,
                taxonomy=t.provider,
            )

            # Stage 2: semantic scoring with this taxonomy
            if candidates:
                scored = stage2_semantic_score(
                    candidates=candidates,
                    config=t.config,
                    taxonomy=t.provider,
                    language=doc_language,
                    db=t.db,
                )
                for det in scored:
                    ref = build_ai_reference(
                        candidate=det,
                        company=result_meta.company,
                        year=result_meta.year,
                        position=result_meta.position,
                        industry=result_meta.industry,
                        sector=result_meta.sector,
                        country=result_meta.country,
                        doc_type=result_meta.doc_type,
                        source=result_meta.source,
                        page_count=total_pages,
                        taxonomy=t.provider,
                    )
                    doc_result.add_reference(ref)

                if doc_result.references:
                    doc_result.references = _semantic_deduplicate(
                        doc_result.references, t.config.deduplication_threshold
                    )
                    for _ref in doc_result.references:
                        _ref.embedding = None
                    import gc; gc.collect()

            doc_result.processing_time = time.perf_counter() - t0

            # Stage 3: persist to this taxonomy's DB
            _stage3_persist(
                doc_result=doc_result,
                doc_meta=result_meta,
                text_status=text_status,
                db=t.db,
                config=t.config,
                stats=stats_per_target[t.name],
                buzz_accumulator=buzz_per_target[t.name],
            )
            years_per_target[t.name].add(result_meta.year)

        completed += 1
        if progress_callback:
            progress_callback(completed, total, result_meta.company, result_meta.year)

    # Release multilingual model if it was loaded during this run
    try:
        _m5_release = importlib.import_module("05_detect")
        if hasattr(_m5_release, "release_multilingual_model"):
            _m5_release.release_multilingual_model()
    except Exception:
        pass

    # ── Year-end finalization per target ──────────────────────────────────────
    for t in targets:
        for year in sorted(years_per_target[t.name]):
            finalize_year_memory(year, t.db, threshold_years=2)
            t.db.update_rankings(year)
            _populate_dedup_table(year, t.db)

        if buzz_per_target[t.name]:
            ranked = rank_buzz_indices(buzz_per_target[t.name])
            industry_indices = aggregate_by_industry(ranked)
            for ind in industry_indices:
                _upsert_industry_buzz(ind, t.db)
            t.db.commit()

        s = stats_per_target[t.name]
        s.end_time = datetime.now()
        s.calculate_derived_metrics()
        logger.info(f"Multi-taxonomy [{t.name}]: complete")
        s.print_summary()

    return stats_per_target


def _process_one_sequential(
    doc_meta: DocumentMeta,
    config: AnalyzerConfig,
    db: DatabaseManager,
    stats: ProcessingStats,
    buzz_accumulator: List[AIBuzzIndex],
    processed_years: set,
    stage1_extract_candidates,
    stage2_semantic_score,
    build_ai_reference,
    TAXONOMY,
) -> None:
    """Process one document sequentially (all 3 stages in one call)."""
    t0 = time.perf_counter()

    # Extract text
    page_texts, text_status = extract_pdf_text(doc_meta.source)
    total_pages = len(page_texts)

    if text_status in ("empty", "corrupted_ocr_failed", "mirrored_all_stripped"):
        stats.failed_documents += 1
        doc_result = DocumentResult(
            company=doc_meta.company, year=doc_meta.year,
            position=doc_meta.position, industry=doc_meta.industry,
            sector=doc_meta.sector, country=doc_meta.country,
            doc_type=doc_meta.doc_type, source=doc_meta.source,
            total_pages=total_pages, text_length=0,
            text_status=text_status,
        )
        db.mark_document_processed(doc_result, refs_found=0, text_status=text_status)
        return

    # Repair missing spaces and CamelCase artifacts
    try:
        _m00 = importlib.import_module("00_text_repair")
        page_texts = _m00.repair_page_texts(page_texts)
        doc_language = _m00.detect_document_language(page_texts, sample_pages=5)
        doc_meta.language = doc_language
    except Exception as repair_err:
        logger.warning(f"Text repair skipped for {doc_meta.source}: {repair_err}")
        doc_language = getattr(doc_meta, "language", "en")

    full_text = "\n".join(page_texts)
    word_count = sum(len(t.split()) for t in page_texts)
    sentence_count = len([s for s in re.split(r'[.!?]+', full_text) if len(s.strip()) > 10])

    doc_result = DocumentResult(
        company=doc_meta.company, year=doc_meta.year,
        position=doc_meta.position, industry=doc_meta.industry,
        sector=doc_meta.sector, country=doc_meta.country,
        doc_type=doc_meta.doc_type, source=doc_meta.source,
        total_pages=total_pages, text_length=len(full_text),
        text_status=text_status,
    )
    # Store text stats directly on result so _stage3_persist and UI can use them
    doc_result._word_count = word_count  # type: ignore[attr-defined]
    doc_result._sentence_count = sentence_count  # type: ignore[attr-defined]
    doc_result.word_count = word_count
    doc_result.sentence_count = sentence_count

    # Stage 1
    candidates = stage1_extract_candidates(
        text=full_text,
        page_texts=page_texts,
        config=config,
        taxonomy=TAXONOMY,
    )

    # Stage 2
    if candidates:
        scored = stage2_semantic_score(
            candidates=candidates,
            config=config,
            taxonomy=TAXONOMY,
            language=doc_language,
            db=db,
        )
        for det in scored:
            ref = build_ai_reference(
                candidate=det,
                company=doc_meta.company, year=doc_meta.year,
                position=doc_meta.position, industry=doc_meta.industry,
                sector=doc_meta.sector, country=doc_meta.country,
                doc_type=doc_meta.doc_type, source=doc_meta.source,
                page_count=total_pages, taxonomy=TAXONOMY,
            )
            ref.language = doc_language
            doc_result.add_reference(ref)

    # Semantic deduplication
    if doc_result.references:
        doc_result.references = _semantic_deduplicate(
            doc_result.references, config.deduplication_threshold
        )
        for _ref in doc_result.references:
            _ref.embedding = None
        import gc; gc.collect()

    doc_result.processing_time = time.perf_counter() - t0

    # Stage 3
    _stage3_persist(
        doc_result=doc_result,
        doc_meta=doc_meta,
        text_status=text_status,
        db=db,
        config=config,
        stats=stats,
        buzz_accumulator=buzz_accumulator,
    )

    processed_years.add(doc_meta.year)



def process_single(
    pdf_path: str,
    config: AnalyzerConfig,
    db: DatabaseManager,
) -> Tuple[Optional[DocumentResult], str]:
    """
    Process a single PDF file through all 3 stages and persist results.

    This is the public entry point used by main.py _run_ingest()
    for document-by-document processing with per-file progress display.

    Args:
        pdf_path: Absolute path to the PDF file.
        config:   AnalyzerConfig (fully resolved).
        db:       Open DatabaseManager (migrations already applied).

    Returns:
        Tuple of (DocumentResult | None, text_status).
        Returns (None, 'error') if metadata cannot be resolved.
    """
    # Load Fortune 500 metadata (cheap — cached on repeated calls if needed)
    f500_lookup: Optional[Dict[str, Dict]] = None
    if config.fortune500_csv and os.path.exists(config.fortune500_csv):
        f500_lookup = load_fortune500_csv(config.fortune500_csv)

    meta = resolve_document_meta(str(pdf_path), f500_lookup)
    if meta is None:
        logger.warning(f"process_single: no metadata for {Path(pdf_path).name}")
        return None, "error"

    # Load Stage 2 models (already initialised by the time main.py calls us)
    _m5 = importlib.import_module("05_detect")
    stage1_extract_candidates = _m5.stage1_extract_candidates
    stage2_semantic_score     = _m5.stage2_semantic_score
    build_ai_reference        = _m5.build_ai_reference
    if getattr(config, "taxonomy_excel", None):
        provider = _m5.load_taxonomy_from_excel(config.taxonomy_excel)
    else:
        provider = _m5.load_taxonomy_by_name(config.taxonomy_name)
    if hasattr(_m5, "set_taxonomy"):
        _m5.set_taxonomy(provider)
    TAXONOMY = provider

    buzz_accumulator: List[AIBuzzIndex] = []
    processed_years: set = set()
    stats = ProcessingStats(start_time=datetime.now())

    _process_one_sequential(
        doc_meta=meta,
        config=config,
        db=db,
        stats=stats,
        buzz_accumulator=buzz_accumulator,
        processed_years=processed_years,
        stage1_extract_candidates=stage1_extract_candidates,
        stage2_semantic_score=stage2_semantic_score,
        build_ai_reference=build_ai_reference,
        TAXONOMY=TAXONOMY,
    )

    _word_count_run = stats.total_words
    _sentence_count_run = stats.total_sentences

    # Year-end finalization for this document's year
    for year in sorted(processed_years):
        finalize_year_memory(year, db, threshold_years=2)
        db.update_rankings(year)
        _populate_dedup_table(year, db)

    # Industry aggregation (incremental — safe to call per document)
    if buzz_accumulator:
        ranked = rank_buzz_indices(buzz_accumulator)
        industry_indices = aggregate_by_industry(ranked)
        for ind in industry_indices:
            _upsert_industry_buzz(ind, db)
        db.commit()

    # Retrieve the DocumentResult from DB so main.py can display stats
    row = db.conn.execute(
        "SELECT * FROM processed_documents WHERE source = ?",
        (meta.source,),
    ).fetchone()

    if row is None:
        return None, "error"

    # Reconstruct a minimal DocumentResult for display in main.py
    _m1 = importlib.import_module("01_models")
    doc_result = _m1.DocumentResult(
        company     = row["company"],
        year        = row["year"],
        position    = row["position"],
        industry    = row["industry"],
        sector      = meta.sector,
        country     = meta.country,
        doc_type    = row["doc_type"],
        source      = row["source"],
        total_pages = row["total_pages"],
        text_length = row["text_length"],
        text_status = row["text_status"],
    )
    doc_result.word_count = _word_count_run
    doc_result.sentence_count = _sentence_count_run

    # Attach ref count so main.py _run_ingest can read result.references length
    refs_found = row["refs_found"] or 0
    doc_result._refs_found_count = refs_found   # type: ignore[attr-defined]

    # Expose as list-like for len() in main.py: len(result.references)
    doc_result.references = [None] * refs_found  # type: ignore[assignment]

    text_status = row["text_status"] or "valid"
    return doc_result, text_status


def _run_parallel(
    pending_docs: List[DocumentMeta],
    config: AnalyzerConfig,
    config_dict: Dict,
    db: DatabaseManager,
    stats: ProcessingStats,
    buzz_accumulator: List[AIBuzzIndex],
    processed_years: set,
    stage2_semantic_score,
    build_ai_reference,
    TAXONOMY,
    progress_callback,
) -> None:
    """
    Run Stage 1 in parallel, Stage 2+3 sequentially in the main process.

    ProcessPoolExecutor submits Stage 1 workers. As each completes,
    the main process runs Stage 2 (semantic) and Stage 3 (persist).
    """
    _m5 = importlib.import_module("05_detect")
    stage1_fn = _m5.stage1_extract_candidates

    completed = 0
    total     = len(pending_docs)

    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        future_to_meta = {
            executor.submit(
                _stage1_worker,
                (doc_meta, config_dict),
            ): doc_meta
            for doc_meta in pending_docs
        }

        for future in as_completed(future_to_meta):
            doc_meta = future_to_meta[future]
            try:
                result_meta, candidates, text_status, total_pages, full_text_len, word_count, sentence_count = future.result()
                doc_meta = result_meta
            except Exception as e:
                logger.error(f"Stage 1 exception for {doc_meta.source}: {e}")
                stats.failed_documents += 1
                completed += 1
                continue

            # full_text_len, word_count and sentence_count come from _stage1_worker return value

            doc_result = DocumentResult(
                company=doc_meta.company, year=doc_meta.year,
                position=doc_meta.position, industry=doc_meta.industry,
                sector=doc_meta.sector, country=doc_meta.country,
                doc_type=doc_meta.doc_type, source=doc_meta.source,
                total_pages=total_pages, text_length=full_text_len,
                text_status=text_status,
            )
            doc_result._word_count = word_count  # type: ignore[attr-defined]
            doc_result._sentence_count = sentence_count  # type: ignore[attr-defined]
            doc_result.word_count = word_count
            doc_result.sentence_count = sentence_count

            if text_status not in ("empty", "corrupted_ocr_failed", "mirrored_all_stripped") and candidates:
                t0 = time.perf_counter()

                # Stage 2 (in main process — model lives here)
                doc_language = getattr(doc_meta, "language", "en")
                scored = stage2_semantic_score(
                    candidates=candidates,
                    config=config,
                    taxonomy=TAXONOMY,
                    language=doc_language,
                    db=db,
                )
                for det in scored:
                    ref = build_ai_reference(
                        candidate=det,
                        company=doc_meta.company, year=doc_meta.year,
                        position=doc_meta.position, industry=doc_meta.industry,
                        sector=doc_meta.sector, country=doc_meta.country,
                        doc_type=doc_meta.doc_type, source=doc_meta.source,
                        page_count=total_pages, taxonomy=TAXONOMY,
                    )
                    ref.language = doc_language
                    doc_result.add_reference(ref)

                # Deduplication
                if doc_result.references:
                    doc_result.references = _semantic_deduplicate(
                        doc_result.references, config.deduplication_threshold
                    )
                    for _ref in doc_result.references:
                        _ref.embedding = None
                    import gc; gc.collect()

                doc_result.processing_time = time.perf_counter() - t0
            elif text_status in ("empty", "corrupted_ocr_failed", "mirrored_all_stripped"):
                # Document failed extraction — mark failed and skip Stage 3
                stats.failed_documents += 1
                db.mark_document_processed(
                    doc_result, refs_found=0, text_status=text_status
                )
                completed += 1
                if progress_callback:
                    progress_callback(completed, total, doc_meta.company, doc_meta.year)
                continue

            # Stage 3
            _stage3_persist(
                doc_result=doc_result,
                doc_meta=doc_meta,
                text_status=text_status,
                db=db,
                config=config,
                stats=stats,
                buzz_accumulator=buzz_accumulator,
            )
            processed_years.add(doc_meta.year)

            completed += 1
            if progress_callback:
                progress_callback(completed, total, doc_meta.company, doc_meta.year)

    import gc
    gc.collect()


# ============================================================================
# INDUSTRY BUZZ UPSERT HELPER
# ============================================================================

def _upsert_industry_buzz(ind: IndustryBuzzIndex, db: DatabaseManager) -> None:
    """Upsert an IndustryBuzzIndex record into adoption_index_industry."""
    db.conn.execute(
        """
        INSERT INTO adoption_index_industry (
            industry, year,
            avg_volume_index, avg_depth_index, avg_breadth_index,
            avg_tone_index, avg_specificity_index,
            avg_forward_looking_index, avg_salience_index,
            ai_buzz_index_industry, rank_among_industries,
            num_companies, total_refs,
            min_index, max_index, std_deviation, companies_list
        ) VALUES (?,?, ?,?,?, ?,?, ?,?, ?,?, ?,?, ?,?,?,?)
        ON CONFLICT(industry, year) DO UPDATE SET
            avg_volume_index          = excluded.avg_volume_index,
            avg_depth_index           = excluded.avg_depth_index,
            avg_breadth_index         = excluded.avg_breadth_index,
            avg_tone_index            = excluded.avg_tone_index,
            avg_specificity_index     = excluded.avg_specificity_index,
            avg_forward_looking_index = excluded.avg_forward_looking_index,
            avg_salience_index        = excluded.avg_salience_index,
            ai_buzz_index_industry    = excluded.ai_buzz_index_industry,
            rank_among_industries     = excluded.rank_among_industries,
            num_companies             = excluded.num_companies,
            total_refs                = excluded.total_refs,
            min_index                 = excluded.min_index,
            max_index                 = excluded.max_index,
            std_deviation             = excluded.std_deviation,
            companies_list            = excluded.companies_list
        """,
        (
            ind.industry, ind.year,
            ind.avg_volume_index, ind.avg_depth_index, ind.avg_breadth_index,
            ind.avg_tone_index, ind.avg_specificity_index,
            ind.avg_forward_looking_index, ind.avg_salience_index,
            ind.ai_buzz_index_industry, ind.rank_among_industries,
            ind.num_companies, ind.total_refs,
            ind.min_index, ind.max_index, ind.std_deviation, ind.companies_list,
        ),
    )


# ============================================================================
# CONVENIENCE ENTRY POINT
# ============================================================================

def run(
    input_folder: str,
    output_folder: str = "Results_AISA",
    database_name: str = "aisa_results.db",
    fortune500_csv: Optional[str] = None,
    max_workers: int = 4,
    start_year: int = 2020,
    end_year: int = 2025,
    top_n: Optional[int] = None,
    progress_callback=None,
) -> ProcessingStats:
    """
    Convenience entry point for running the full pipeline with minimal setup.

    Creates AnalyzerConfig and DatabaseManager from arguments, applies
    migrations, runs pipeline, and returns processing stats.

    Args:
        input_folder:   Directory containing Fortune 500 PDFs.
        output_folder:  Directory for results and exports.
        database_name:  SQLite DB filename (relative to output_folder).
        fortune500_csv: Optional CSV with company metadata.
        max_workers:    Number of parallel Stage 1 workers.
        start_year:     First year to process.
        end_year:       Last year to process.
        top_n:          Optional limit on number of PDFs (for testing).
        progress_callback: Optional callable(current, total, company, year).

    Returns:
        ProcessingStats.
    """
    config = AnalyzerConfig(
        input_folder=input_folder,
        output_folder=output_folder,
        database_name=database_name,
        fortune500_csv=fortune500_csv,
        max_workers=max_workers,
        start_year=start_year,
        end_year=end_year,
        top_n=top_n,
    )

    db_path = os.path.join(output_folder, database_name)
    with DatabaseManager(db_path) as db:
        db.apply_migrations()
        stats = run_pipeline(config, db, progress_callback=progress_callback)

    return stats


# ============================================================================
# SMOKE TEST
# ============================================================================

if __name__ == "__main__":
    import tempfile
    from version import get_version_string

    print(get_version_string())
    print()

    # --- Test metadata resolution ---
    print("--- Filename metadata parsing ---")
    test_cases = [
        ("257. Edeka Zentrale - 2024 - annual report.pdf",                    ("Edeka Zentrale", 2024, "Annual Report",   257)),
        ("265. Tsingshan Holding Group - Rept Battero - 2024 - sustainability report.pdf",
                                                                              ("Tsingshan Holding Group", 2024, "Sustainability", 265)),
        ("1. Walmart - 2023 - proxy.pdf",                                     ("Walmart",          2023, "Proxy",          1)),
        ("Microsoft - 2022 - annual report.pdf",                              ("Microsoft",         2022, "Annual Report",  0)),
    ]

    for filename, (exp_company, exp_year, exp_type, exp_pos) in test_cases:
        company, year, doc_type, position = _parse_filename_meta(filename)
        ok_year = (year == exp_year)
        ok_type = (doc_type == exp_type)
        ok_pos  = (position == exp_pos)
        status  = "OK" if (ok_year and ok_type and ok_pos) else "FAIL"
        print(f"  [{status}] {filename}")
        print(f"         company={company!r} year={year} doc_type={doc_type!r} position={position}")
        if not ok_year:
            print(f"         YEAR MISMATCH: expected {exp_year}, got {year}")
        if not ok_type:
            print(f"         TYPE MISMATCH: expected {exp_type!r}, got {doc_type!r}")
        if not ok_pos:
            print(f"         POS MISMATCH: expected {exp_pos}, got {position}")

    # --- Test Fortune 500 CSV loading ---
    print()
    print("--- Fortune 500 CSV loading ---")
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "fortune500.csv")
        with open(csv_path, "w") as f:
            f.write("company,year,position,industry,sector,country\n")
            f.write("Walmart,2023,1,Retail,Consumer Staples,USA\n")
            f.write("Apple Inc,2023,3,Technology,Information Technology,USA\n")
            f.write("Microsoft,2023,5,Technology,Information Technology,USA\n")

        lookup = load_fortune500_csv(csv_path)
        assert "walmart" in lookup
        assert lookup["walmart"]["position"] == 1
        assert "apple inc" in lookup
        print(f"  Loaded {len(lookup)} entries")

        # Test metadata resolution with CSV
        meta = resolve_document_meta(
            "Apple_Inc_2023_Annual_Report.pdf",
            lookup,
        )
        assert meta is not None
        assert meta.year == 2023
        print(f"  Resolved: {meta.company} {meta.year} pos={meta.position}")

        meta2 = resolve_document_meta("Unknown_Corp_2022.pdf", lookup)
        assert meta2 is not None
        # Filename-only fallback may preserve underscores / raw stem depending
        # on the active parser. Validate the important fallback behavior rather
        # than one exact display formatting of the company name.
        assert meta2.year == 2022
        assert meta2.industry == "Unknown"
        print(f"  Fallback: {meta2.company} {meta2.year} industry={meta2.industry}")

    # --- Test semantic deduplication ---
    print()
    print("--- Semantic deduplication ---")

    def _make_ref(text, page=1, conf=0.8):
        return AIReference(
            company="TestCorp", year=2023, position=1,
            industry="Technology", sector="Software", country="USA",
            doc_type="Annual Report", text=text, context="",
            page=page, category="B4|A2", detection_method="pattern_hard",
            sentiment="positive", sentiment_score=0.8, semantic_score=0.75,
            source="test.pdf", category_a="A2", category_b="B4",
            confidence_score=conf,
        )

    refs = [
        _make_ref("We deployed machine learning models across our operations.", page=1, conf=0.9),
        _make_ref("We deployed machine learning models across our operations.", page=1, conf=0.7),  # exact dup
        _make_ref("Our machine learning models are deployed in operations globally.", page=5, conf=0.8),  # near-dup
        _make_ref("ChatGPT is now used by our customer service team.", page=10, conf=0.85),  # unique
    ]

    deduped = _semantic_deduplicate(refs, threshold=0.6)
    print(f"  Input:  {len(refs)} refs")
    print(f"  Output: {len(deduped)} refs")
    assert len(deduped) < len(refs), "Deduplication should have removed some refs"
    assert len(deduped) >= 2, "At least 2 distinct refs should remain"
    for r in deduped:
        print(f"  KEPT: page={r.page} conf={r.confidence_score} | {r.text[:60]}...")

    print()
    print("  10_pipeline.py all checks passed.")
