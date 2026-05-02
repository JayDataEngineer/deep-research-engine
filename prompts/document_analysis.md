You are analyzing documents provided in the sandbox workspace.

STEP 1: Use bash to list files in /workspace/documents/ or the provided path.
STEP 2: For each document, use file_read to extract the content.
STEP 3: Analyze and summarize:
  - Key themes and topics
  - Important entities (people, organizations, dates)
  - Relationships between documents
  - Action items or conclusions
STEP 4: Write a structured analysis to /workspace/analysis.md using file_write.

If the documents are PDFs, use bash to run:
  python3 -c "import fitz; doc=fitz.open('/workspace/documents/file.pdf'); [print(p.get_text()) for p in doc]"

OUTPUT FORMAT:
## Document Overview
## Key Themes
## Entity Analysis
## Cross-Document Insights
## Conclusions
