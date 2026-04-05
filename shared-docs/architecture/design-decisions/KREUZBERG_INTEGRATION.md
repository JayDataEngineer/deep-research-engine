# ADR: Use Kreuzberg for Document Extraction

**Status**: Accepted  
**Date**: 2026-04-05  
**Context**: Ingestion pipeline refactor  

## Problem

The ingestion node contained hand-rolled extraction code for 5 document formats:
- PDF (PyMuPDF, ~80 lines)
- DOCX (python-docx, ~20 lines)
- XLSX (openpyxl, ~25 lines)
- PPTX (python-pptx, ~50 lines)
- HTML (BeautifulSoup, ~100 lines)
- Telegram HTML (BeautifulSoup, ~120 lines)

This code was:
- **Limited**: Only 5 formats, no OCR, no table extraction, no chunking
- **Static**: No active maintenance or improvement
- **Quality gaps**: PyMuPDF text extraction has font encoding issues, no structure awareness, no layout detection
- **Maintenance burden**: ~485 lines of parsing logic we maintained ourselves

## Decision

Replace all hand-rolled document extraction with [Kreuzberg](https://github.com/kreuzberg-dev/kreuzberg).

### What Kreuzberg Provides

| Feature | Our Code | Kreuzberg |
|---------|----------|-----------|
| Formats supported | 5 | 91+ |
| OCR | None (manual note) | Tesseract/EasyOCR/PaddleOCR with quality gates |
| Table extraction | None | Native word-position + TATR model |
| Chunking | None | Text/Markdown/YAML with heading context |
| Layout detection | None | RT-DETR, YOLO, font-size clustering |
| Language detection | None | Built-in |
| Output formats | Plain text | Plain, Markdown, Djot, HTML, JSON |
| Caching | None | Built-in result caching |
| Batch extraction | Sequential | Parallel (Rayon) |

### What We Keep

Kreuzberg is **document-only**. We retain our proprietary multimodal pipeline:

| Component | Why We Keep It |
|-----------|---------------|
| ASR (VibeVoice) | Kreuzberg has zero audio support |
| Speaker diarization | Kreuzberg has zero audio support |
| Face recognition (CompreFace) | Kreuzberg does OCR on images, not face recognition |
| VLM image analysis | Kreuzberg extracts text from images, not semantic descriptions |
| Speaker clustering (ECAPA-TDNN) | Kreuzberg has zero audio support |
| Identity resolution (LLM merge) | Kreuzberg has no multimodal identity resolution |
| Telegram JSON parser | Telegram's JSON schema is app-specific; Kreuzberg doesn't understand it |
| Mega-file audio concatenation | Kreuzberg has zero audio support |

### What Changed

| File | Before | After |
|------|--------|-------|
| `nodes/ingestion.py` | 1424 lines (hand-rolled extractors) | ~950 lines (DocumentRouter orchestration) |
| `workers/chunk_processor.py` | Single-chunk classification | `workers/content_classifier.py` — batch classification of Kreuzberg chunks |
| `ingestion/kreuzberg_extractor.py` | N/A | New — wraps Kreuzberg API |
| `ingestion/telegram_parser.py` | N/A | New — Telegram JSON + HTML parsing |
| `ingestion/document_router.py` | N/A | New — input detection and routing |

## Consequences

### Positive
- **~500 lines deleted** from ingestion node
- **86 more formats** supported (5 → 91+)
- **OCR enabled** for scanned documents
- **Table extraction** added
- **Chunking** added (feeds directly into content classifier)
- **Active maintenance** from Kreuzberg upstream
- **Better extraction quality** for all existing formats

### Trade-offs
- **New dependency**: Kreuzberg (~30MB wheel with native libraries)
- **Rust dependency**: Kreuzberg's core is Rust (distributed as wheels, no build required)
- **Telegram HTML**: Still needs BeautifulSoup for structure (Kreuzberg gives flat text; we need per-message metadata)

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|-------------|
| Keep hand-rolled extractors | Limited scope, quality gaps, maintenance burden |
| Use LangChain document loaders | Fewer formats, no OCR, no table extraction |
| Use unstructured.io | Heavy dependency, API key required for best features |
| Build our own PDF parser | Multi-year effort; Kreuzberg already does it better |

## Implementation

- Added `kreuzberg>=3.13.0` to `pyproject.toml`
- Removed `python-docx`, `openpyxl`, `python-pptx` from dependencies (Kreuzberg handles these formats)
- Kept `pymupdf` for PDF image extraction (Kreuzberg doesn't extract images from PDFs to disk for downstream face recognition)
- Created `ingestion/` package with `document_router.py`, `kreuzberg_extractor.py`, `telegram_parser.py`
- Renamed `chunk_processor.py` → `content_classifier.py` (it classifies chunks, doesn't create them)
- Updated `nodes/ingestion.py` to use DocumentRouter
- Updated `nodes/content_clusterer.py` for unified input (Kreuzberg chunks + worker transcripts)
