# Kreuzberg Integration

[Kreuzberg](https://github.com/kreuzberg-dev/kreuzberg) is a high-performance document intelligence library with a Rust core and native Python bindings. It extracts text, tables, metadata, and images from 91+ file formats.

## Why Kreuzberg

We replaced hand-rolled PDF, DOCX, XLSX, PPTX, and HTML extraction code with Kreuzberg because:

1. **Better extraction quality** — Multi-path PDF processing (structure tree → heuristic segments → font-size clustering → classification), native table reconstruction, and quality-aware OCR
2. **91+ formats** — We only maintained 5 formats; Kreuzberg handles 19× more
3. **Built-in chunking** — Text/Markdown/YAML chunkers with heading context, token sizing, and overlap
4. **Active development** — Kreuzberg is actively maintained; our extractors were static
5. **~485 lines deleted** — Reduced maintenance burden

Kreuzberg handles **documents only**. It has zero support for audio, video, or face recognition — those remain our proprietary multimodal pipeline.

## How It Works

### Extraction Flow

```
File Path → Kreuzberg.extract_file() → ExtractionResult
                                        ├── content (str)
                                        ├── chunks (list)
                                        ├── tables (list)
                                        ├── metadata (dict)
                                        ├── pages (list)
                                        └── images (list)
```

### Integration Architecture

| Component | File | Purpose |
|-----------|------|---------|
| `KreuzbergExtractor` | `ingestion/kreuzberg_extractor.py` | Wraps Kreuzberg API, provides batch extraction, normalizes output |
| `TelegramParser` | `ingestion/telegram_parser.py` | Parses Telegram JSON (custom) and HTML (via Kreuzberg) |
| `DocumentRouter` | `ingestion/document_router.py` | Scans input directory, routes documents to Kreuzberg, Telegram to custom parser |
| Ingestion node | `nodes/ingestion.py` | Orchestrates routing, feeds chunks into multimodal pipeline |

### Configuration

```python
from deep_research_engine.ingestion import KreuzbergConfig, DocumentRouter

config = KreuzbergConfig(
    chunking_enabled=True,
    chunk_max_chars=4000,        # RAG-friendly chunk size
    chunk_max_overlap=200,       # Overlap for context continuity
    output_format="markdown",    # Clean formatting
    ocr_enabled=True,            # Tesseract OCR for scanned docs
    ocr_backend="tesseract",     # or "easyocr", "paddleocr"
    ocr_language="eng",
    table_extraction=True,       # Extract tables separately
    image_extraction=False,      # Extract embedded images
    language_detection=True,     # Auto-detect document language
)

router = DocumentRouter(input_dir, config)
result = await router.route()

# Access results
for doc in result.documents:
    print(f"{doc.file_path}: {doc.char_count} chars, {doc.chunk_count} chunks")

for chunk in result.document_chunks:
    print(f"Chunk {chunk['index']}: {len(chunk['text'])} chars")
```

### Chunk Schema

Kreuzberg chunks are normalized to:

```python
{
    "text": str,              # Chunk text content
    "index": int,             # Chunk number (0-based)
    "source_file": str,       # Original file path
    "heading_context": str,   # Heading breadcrumb (if markdown chunker)
    "page": int | None,       # Page number (if available)
    # ... additional metadata from Kreuzberg
}
```

### Telegram HTML Parsing

For Telegram HTML exports, Kreuzberg extracts clean text from the HTML DOM. Our `TelegramParser` then:
1. Uses BeautifulSoup to parse the HTML structure (needed for per-message metadata)
2. Extracts timestamps, sender names, media links, forwarded status
3. Normalizes timestamps to America/New_York
4. Outputs unified items compatible with the multimodal pipeline

For Telegram JSON exports (`result.json`), we use our custom parser because Telegram's JSON schema is application-specific and Kreuzberg doesn't understand it.

## Error Handling

Failed extractions are logged and collected in `DocumentRoutingResult.errors`. The pipeline continues processing remaining files even if individual extractions fail.

```python
if result.errors:
    for err in result.errors:
        logger.warning(f"Failed: {err['file']}: {err['error']}")
```

## Dependencies

```toml
[project]
dependencies = [
    "kreuzberg>=3.13.0",    # Document intelligence
]
```

Kreuzberg includes native wheels for Linux/macOS/Windows. No additional system dependencies required (Tesseract is bundled).
