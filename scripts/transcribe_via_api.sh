#!/bin/bash
# Transcribe all audio files via faster-whisper API

API_URL="http://172.21.0.2:8000"
API_KEY="sk-no-key-required"
MODEL="Systran/faster-whisper-small"
OUTPUT_DIR="output/transcripts/whisper"
DATA_DIR="data/ChatExport_2026-03-13"

mkdir -p "$OUTPUT_DIR"

# Find all audio files
FILES=$(find "$DATA_DIR" -type f \( -iname "*.ogg" -o -iname "*.m4a" -o -iname "*.wav" -o -iname "*.mp3" \) 2>/dev/null | sort -u)

TOTAL=$(echo "$FILES" | wc -l)
CURRENT=0

echo "# Faster-Whisper Transcriptions" > "$OUTPUT_DIR/transcripts.md"
echo "" >> "$OUTPUT_DIR/transcripts.md"

for FILE in $FILES; do
    CURRENT=$((CURRENT + 1))
    BASENAME=$(basename "$FILE")
    echo "[$CURRENT/$TOTAL] $BASENAME"

    RESULT=$(curl -s -X POST "$API_URL/v1/audio/transcriptions" \
        -H "Authorization: Bearer $API_KEY" \
        -F "file=@$FILE" \
        -F "model=$MODEL" \
        --max-time 300 2>&1)

    TEXT=$(echo "$RESULT" | jq -r '.text // empty')

    if [ -n "$TEXT" ]; then
        echo "## $BASENAME" >> "$OUTPUT_DIR/transcripts.md"
        echo "" >> "$OUTPUT_DIR/transcripts.md"
        echo "$TEXT" >> "$OUTPUT_DIR/transcripts.md"
        echo "" >> "$OUTPUT_DIR/transcripts.md"
        echo "$BASENAME|$TEXT" >> "$OUTPUT_DIR/transcripts.tsv"
        echo "  ✓ Transcribed"
    else
        echo "  ✗ Error: $RESULT"
        echo "## $BASENAME (ERROR)" >> "$OUTPUT_DIR/transcripts.md"
        echo "Error: $RESULT" >> "$OUTPUT_DIR/transcripts.md"
        echo "" >> "$OUTPUT_DIR/transcripts.md"
    fi
done

echo ""
echo "Done! Output: $OUTPUT_DIR/transcripts.md"
