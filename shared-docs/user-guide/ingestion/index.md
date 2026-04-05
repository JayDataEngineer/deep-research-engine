# Ingestion Pipeline

The ingestion pipeline is the entry point for all non-text data into the deep research engine. It extracts, normalizes, and enriches content from diverse sources before passing it downstream to the drafter, content clusterer, and knowledge graph builders.

## Architecture

The pipeline has **two extraction paths** that converge into unified downstream processing:

```
                    ┌─────────────────────────────────────┐
                    │           Ingestion Node             │
                    │                                     │
  Input Dir ──────→ │  Phase 0: Scan + Route              │
                    │                                     │
                    │  ┌─────────────────────────────┐    │
                    │  │ Document Path                │    │
                    │  │  (PDF, DOCX, HTML, etc.)    │    │
                    │  │       ↓                     │    │
                    │  │  Kreuzberg Extraction       │    │
                    │  │  (91+ formats, OCR, tables) │    │
                    │  │       ↓                     │    │
                    │  │  Structured Chunks          │    │
                    │  └─────────────────────────────┘    │
                    │                                     │
                    │  ┌─────────────────────────────┐    │
                    │  │ Multimodal Path              │    │
                    │  │  (Audio, Video, Photos)     │    │
                    │  │       ↓                     │    │
                    │  │  Phase 2: Visual Processing │    │
                    │  │  Phase 3: Acoustic Mega-Files│   │
                    │  │  Phase 4: ASR (VibeVoice)   │    │
                    │  │  Phase 5: Speaker Clustering│    │
                    │  │  Phase 6: Identity Resolution│   │
                    │  │  Phase 7: Enrich + Output   │    │
                    │  └─────────────────────────────┘    │
                    │                                     │
                    │        ↓ Unified Output ↓           │
                    │  document_chunks → Content Clusterer│
                    │  big_text_dump → Drafter            │
                    └─────────────────────────────────────┘
```

## Two Extraction Paths

### Document Path (Kreuzberg)

Documents flow through [Kreuzberg](https://github.com/kreuzberg-dev/kreuzberg), a high-performance document intelligence library with a Rust core:

- **91+ formats**: PDF, DOCX, XLSX, PPTX, HTML, email, archives, academic formats
- **OCR**: Tesseract (built-in), EasyOCR, PaddleOCR backends
- **Table extraction**: Native word-position + TATR model
- **Chunking**: Text/Markdown chunkers with heading context, overlap, page boundaries
- **Language detection**: Automatic document language identification

Supported document types:
| Category | Extensions |
|----------|-----------|
| PDF | `.pdf` |
| Word | `.docx`, `.odt`, `.doc` |
| Spreadsheets | `.xlsx`, `.xls`, `.ods` |
| Presentations | `.pptx`, `.ppt`, `.ppsx` |
| HTML/Markup | `.html`, `.htm`, `.xml` |
| Email | `.eml`, `.msg` |
| Archives | `.zip`, `.tar`, `.gz`, `.7z` |
| Academic | `.bib`, `.ris`, `.tex`, `.ipynb` |

### Multimodal Path (Custom Pipeline)

Audio, video, and photo content flows through our proprietary 6-phase pipeline:

| Phase | Purpose |
|-------|---------|
| **Phase 2** | Visual processing — FFmpeg scene detection, face recognition (CompreFace), VLM image analysis |
| **Phase 3** | Acoustic mega-files — Audio concatenation with silence buffers for ASR diarization |
| **Phase 4** | Long-form ASR — VibeVoice ASR-7B with built-in speaker diarization |
| **Phase 5** | Cross-mega-file speaker clustering — ECAPA-TDNN embeddings reconcile speakers across files |
| **Phase 6** | Identity resolution — LLM merges face clusters + voice clusters + text mentions into unified identities |
| **Phase 7** | Enrichment — Entity extraction, structured output, chat history generation |

### Telegram Exports

Telegram exports use a **hybrid approach**:
- **result.json** → Custom JSON parser (Telegram's schema is app-specific; Kreuzberg doesn't understand it)
- **messages.html** → Kreuzberg HTML extractor + our metadata extraction (timestamps, senders, media links)
- **Media subdirectories** (photos/, voice_messages/, video_files/) → Multimodal path

## Output

The ingestion node produces:

| Output | Location | Purpose |
|--------|----------|---------|
| `document_chunks` | `state.context.document_chunks` | Kreuzberg chunks for content classifier |
| `big_text_dump` | `output/ingested/big_text_dump.md` | Combined markdown for downstream processing |
| `resolved_identity_map` | `state.context.ingestion_structured` | Merged visual + acoustic identities |
| `entity_extractions` | `state.context.ingestion_structured` | Named entities from all content |
| `media_items` | `state.context.ingestion_structured` | Structured media with speakers, faces, descriptions |
| `video_descriptions` | `state.context.ingestion_structured` | VLM-generated video narratives |

## Configuration

```yaml
ingestion:
  enabled: true
  source: "./data/telegram_export"    # Input directory path
```

Kreuzberg extraction is configured with sensible defaults (markdown output, 4000-char chunks, Tesseract OCR). Override via `KreuzbergConfig` in the document router.

## Pipeline Features

- **Phase checkpointing**: Each phase saves artifacts to disk for resume capability
- **Dead letter queue**: Failed items (VLM failures, extraction errors) go to `dead_letter_queue.json`
- **VLM circuit breaker**: Per-batch failure tracking prevents cascading VLM errors
- **GPU memory management**: VRAM purge between phases (LLM ↔ ASR model switching)
- **Temporal normalization**: All timestamps converted to America/New_York with DST handling
