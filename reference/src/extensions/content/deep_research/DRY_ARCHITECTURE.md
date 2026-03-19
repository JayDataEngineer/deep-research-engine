# DRY Text Injection System - Architecture Summary

## The Problem We Solved

**Before (Non-DRY):**
- Podcast had its own text injection system in `text_injections.yaml`
- Twitter and Substack copied the same patterns
- Each extension reimplemented: character mapping, research directives, prefills
- Code duplication across `lab.py`, profile systems, etc.

**After (DRY):**
- **ONE** generic text injection system that ANY extension can use
- **ONE** character/personality data structure (`WorkerBehavior`, `CharacterData`)
- **ONE** mechanism for research directive injection
- **ONE** mechanism for "true prefill" (role="assistant")
- Extensions provide **CONTENT** (YAML), system provides **MECHANISM** (Python)

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            EXTENSIONS                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                    │
│  │   Podcast    │  │    Twitter   │  │   Substack   │                    │
│  │              │  │              │  │              │                    │
│  │ character_   │  │ character_   │  │ character_   │                    │
│  │   mappings   │  │   mappings   │  │   mappings   │                    │
│  └───────┬──────┘  └───────┬──────┘  └───────┬──────┘                    │
└──────────┼──────────────────┼──────────────────┼─────────────────────────────┘
           │                  │                  │
           │                  │                  │
           ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     GENERIC TEXT INJECTION SYSTEM                        │
│                    (profiles/injection.py + loader.py)                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  DATA LAYER (The Interface)                                       │   │
│  │                                                                   │   │
│  │  @dataclass WorkerBehavior                                        │   │
│  │    - research_directive: str                                       │   │
│  │    - prefill_prompt: str                                           │   │
│  │    - research_sources: List[str]                                   │   │
│  │    - speech_patterns: List[str]                                    │   │
│  │    - emotion_bias: Optional[List[float]]                           │   │
│  │                                                                   │   │
│  │  @dataclass CharacterData                                          │   │
│  │    - name: str                                                     │   │
│  │    - role: str                                                     │   │
│  │    - description: str                                              │   │
│  │    - speech_patterns: List[str]                                    │   │
│  │    - ideological_framework: str                                    │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  FUNCTIONS (The Mechanism)                                        │   │
│  │                                                                   │   │
│  │  get_worker_behavior()         → Extract behavior from personality │   │
│  │  get_character_data()           → Extract character for prompts    │   │
│  │  apply_research_directive()     → Override specialist_prompt        │   │
│  │  build_simple_system_messages() → Build messages with prefill      │   │
│  │  load_personality_for_worker()  → Unified personality loading       │   │
│  │  resolve_character_mapping()    → Generic mapping resolution       │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────┘
           │
           │ Uses
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         PERSONALITY STORAGE                               │
│                                                                             │
│  Priority (User → System):                                               │
│  1. data/user_profiles/characters/    (User customizations)              │
│  2. profiles/personalities/           (System defaults)                  │
│  3. podcast/text_injections.yaml      (Legacy fallback)                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## How Each Extension Uses This System

### Podcast (Original Implementation)
```yaml
# podcast/variants/2worker.yaml
character_card_mappings:
  left_character: "EMMA"
  right_character: "OLIVIA"
```

```python
# Generic code that ANY extension can now use:
from profiles.injection import load_personality_for_worker, get_worker_behavior

personality = load_personality_for_worker(
    worker_type="left_character",
    character_card_mappings=config["character_card_mappings"],
    profile_loader=loader,
    legacy_patterns=PODCAST_WORKER_PATTERNS
)

behavior = get_worker_behavior(personality)
task = apply_research_directive(specialist_config, behavior)
messages = build_simple_system_messages(base_prompt, character=personality, prefill=behavior.prefill_prompt)
```

### Twitter (New - Can Use Same System)
```yaml
# soc_media_gen/variants/fitness_club.yaml
character_card_mappings:
  researcher_0: "MOTIVATION_HAMMER"
  researcher_1: "ANTI_EXCUSE"
```

```python
# SAME GENERIC CODE:
personality = load_personality_for_worker(
    worker_type="researcher_0",
    character_card_mappings=config["character_card_mappings"],
    profile_loader=loader,
    legacy_patterns=TWITTER_WORKER_PATTERNS  # Different pattern, same system
)

behavior = get_worker_behavior(personality)
# ... same as above
```

### Substack (New - Can Use Same System)
```yaml
# longform_gen/variants/fitness_club.yaml
character_card_mappings:
  philosopher_connector: "NIETZSCHE_SCHOLAR"
  fitness_researcher: "FITNESS_EXPERT"
```

```python
# SAME GENERIC CODE:
# ... same as above
```

---

## Key Insight: Separation of Mechanism from Content

| Layer | What | Example | Who Provides |
|-------|------|---------|--------------|
| **Engine** | The mechanism | `get_worker_behavior()`, `build_simple_system_messages()` | System (Python code) |
| **Soul** | The content | "Research from far-left perspectives", "Understood. I will portray..." | User (YAML files) |
| **Config** | The mapping | `researcher_0: "MOTIVATION_HAMMER"` | Extension (YAML variant) |

---

## The DRY Pattern

### BEFORE: Each extension reimplemented
```python
# podcast/profile.py
def _get_worker_behavior(self, personality_id):
    personality = self._get_personality(personality_id)
    return {
        "research_directive": personality.get("research_directive", ""),
        "prefill_prompt": personality.get("prefill_prompt", ""),
        # ...
    }

# twitter/profile.py (copied!)
def _get_worker_behavior(self, personality_id):
    personality = self._get_personality(personality_id)
    return {
        "research_directive": personality.get("research_directive", ""),
        "prefill_prompt": personality.get("prefill_prompt", ""),
        # ...
    }

# substack/profile.py (copied again!)
def _get_worker_behavior(self, personality_id):
    # ... same code ...
```

### AFTER: One generic system
```python
# profiles/injection.py (ONE implementation)
def get_worker_behavior(personality_data, worker_id=None):
    return WorkerBehavior(
        research_directive=personality_data.get("research_directive", ""),
        prefill_prompt=personality_data.get("prefill_prompt", ""),
        # ...
    )

# podcast/profile.py
from profiles.injection import get_worker_behavior

# twitter/profile.py
from profiles.injection import get_worker_behavior

# substack/profile.py
from profiles.injection import get_worker_behavior
```

---

## Personality Storage Priority

The system tries THREE locations in order:

```
1. data/user_profiles/characters/EMMA.yaml
   ↓ (not found)
2. profiles/personalities/EMMA.yaml
   ↓ (not found)
3. podcast/text_injections.yaml (legacy fallback)
```

This means:
- **Users** can override ANY personality by dropping a file in `data/user_profiles/`
- **System** provides defaults in `profiles/personalities/`
- **Legacy** podcast personalities still work without migration

---

## Complete DRY Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER LAYER                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐       │
│  │ User Characters │  │ User Brands     │  │ User Tools     │       │
│  │ (data/user_     │  │ (data/user_     │  │ (data/user_     │       │
│  │  profiles/)     │  │  profiles/)     │  │  profiles/)     │       │
│  └────────────────┘  └────────────────┘  └────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      GENERIC SYSTEM LAYER                             │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Text Injection (profiles/injection.py)                        │ │
│  │  - get_worker_behavior()                                       │ │
│  │  - build_simple_system_messages()                              │ │
│  │  - load_personality_for_worker()                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Profile Loader (profiles/loader.py)                            │ │
│  │  - load_brand_profile()                                         │ │
│  │  - load_personality()                                           │ │
│  │  - get_boss_strategy_from_profile()                             │ │
│  │  - get_critic_criteria_from_profile()                            │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Prompt Engine (libraries/prompt_engine)                        │ │
│  │  - StrictPromptRegistry for multi-layer validation              │ │
│  │  - Variable contracts + preflight rendering                     │ │
│  │  - Jinja2 templates with crash-only semantics                   │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        EXTENSION LAYER                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                    │
│  │   Podcast    │  │    Twitter   │  │   Substack   │                    │
│  │              │  │              │  │              │                    │
│  │ Provides:    │  │ Provides:    │  │ Provides:    │                    │
│  │ - Mappings   │  │ - Mappings   │  │ - Mappings   │                    │
│  │ - Variants   │  │ - Variants   │  │ - Variants   │                    │
│  └──────────────┘  └──────────────┘  └──────────────┘                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Example: Adding a New Character

### Before (Non-DRY)
1. Edit `podcast/text_injections.yaml`
2. Add personality definition with all fields
3. Rebuild system if code changes needed

### After (DRY)
1. Create `data/user_profiles/characters/my_waifu.yaml`
```yaml
name: "My Waifu"
role: "podcast host"
description: "A custom character I created"
research_directive: "Research from my unique perspective..."
prefill_prompt: "Understood. I will portray this character authentically..."
```
2. Add to variant YAML:
```yaml
character_card_mappings:
  my_worker: "my_waifu"
```
3. Done! No code changes, works for ANY extension

---

## Summary: What's Now DRY

| Component | Before | After |
|-----------|--------|-------|
| **Worker Behavior** | Each extension reimplemented | `WorkerBehavior` dataclass |
| **Character Data** | Each extension reimplemented | `CharacterData` dataclass |
| **Research Directive** | Hardcoded in lab.py | `apply_research_directive()` function |
| **True Prefill** | Podcast-specific | `build_simple_system_messages()` function |
| **Character Mapping** | Multiple patterns in lab.py | `resolve_character_mapping()` function |
| **Personality Loading** | Extension-specific | `load_personality_for_worker()` function |
| **Storage Priority** | Single location | User → System → Legacy |

**All extensions now use the SAME mechanism, just with different content (YAML).**
