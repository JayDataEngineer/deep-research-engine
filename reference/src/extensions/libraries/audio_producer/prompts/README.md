# Audio Producer Prompts

**CRITICAL: These prompts control the LLM behavior. Soft errors only if messed up!**

## Prompt Organization

```
prompts/
├── README.md                    # This file
├── emotion_detection/           # Emotion tagging prompts
│   ├── prompts.yaml            # All emotion prompts in one place
│   ├── _metadata.yaml          # Prompt metadata (when to use what)
│   └── PROMPT_REFERENCE.md     # Quick reference guide
└── config/
    └── prompt_config.yaml      # Which prompts to use per mode
```

## Emotion Detection Prompts

| Prompt | Purpose | When to Use |
|--------|---------|-------------|
| `single_line` | Fast: one line → one emotion vector | Fast mode, simple dialogue |
| `batch_with_splitting` | Splits sentences, tags each | When sentence-level emotion needed |
| `semantic_chunking` | Groups lines into emotional batches | Overdrive/Sectioned mode phase 1 |
| `director` | Refines emotions with continuity | Overdrive mode phase 2 |
| `sectioned_director` | Same as director but for sections | Sectioned mode |

## Emotion Vector Format

```
[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
```

- **Values**: 0.0 - 1.0 for each dimension
- **All 8 dimensions**: Sent to IndexTTS
- **Natural speech**: Mostly 0.0 - 0.3 emotions, 0.7 - 1.0 calm

## Modifying Prompts

1. **DO NOT** modify prompts without understanding the emotion vector format
2. **Test changes** on a single dialogue line first
3. **Check output format** - LLM must return valid JSON
4. **Version control** - commit prompt changes with descriptive messages

## Adding New Prompts

1. Create prompt in `emotion_detection/prompts.yaml`
2. Add metadata to `emotion_detection/_metadata.yaml`
3. Update this README with purpose and usage

## Quick Reference

### For Fast Mode (single_call):
```yaml
prompt: single_line
chunking: whole_segment
parallelism: 3 workers
```

### For Overdrive Mode:
```yaml
phase1:
  prompt: semantic_chunking
  output: emotional_batches
phase2:
  prompt: director
  input: emotional_batches
  output: refined_emotions
```

### For Sectioned Mode:
```yaml
prompt: sectioned_director
chunking: by_section_markers
parallelism: 3 workers
```
