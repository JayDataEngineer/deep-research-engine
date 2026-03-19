# Extension Import Patterns

## Acceptable Imports

### 1. Core Infrastructure (✅ OK)
Extensions MAY import from core extension infrastructure:
```python
from src.extensions.core.base import LogicExtension, ExtensionStatus
from src.extensions.core.factory import register_extension_factory
from src.extensions.core.context import ExtensionContext
```

### 2. Libraries (✅ OK)
Extensions MAY import from libraries (designed for sharing):
```python
from src.extensions.libraries.character_engine import CharacterCard
from src.extensions.libraries.tts_engine import get_tts_service
from src.extensions.libraries.prompt_engine import build_system_messages
from src.extensions.libraries.tool_manager import ToolManager
```

### 3. Same Extension (✅ OK)
Extensions MAY import from their own submodules:
```python
# In features/text_modes/roleplay/cli.py:
from src.extensions.modes.text_modes.roleplay.storage import RPStorageService
```

### 4. Shared Infrastructure (✅ OK)
Extensions MAY import from shared infrastructure (not extensions):
```python
from src.shared.llm.factory import get_llm
from src.shared.config import get_config
from src.api.shared.helpers import get_inputs_root_dir
```

## Forbidden Imports

### 1. Cross-Feature Imports (❌ FORBIDDEN)
Extensions MUST NOT import from other features:
```python
# ❌ WRONG: characters feature importing from roleplay feature
from src.extensions.modes.text_modes.roleplay.storage import RPStorageService

# ✅ RIGHT: Both should use the library
from src.extensions.libraries.character_engine import CharacterStorageService
```

### 2. Bypassing ExtensionContext (❌ FORBIDDEN when service available)
Extensions MUST use ExtensionContext to access other extension services:
```python
# ❌ WRONG: Direct import of another extension's service
from src.extensions.features.some_feature.service import SomeService

# ✅ RIGHT: Use context
some_service = context.get_extension_service("some_feature")
```

## Migration Plan

### Phase 1: Move Shared Storage Service (High Priority)
**Current Issue:** `characters` feature imports from `roleplay` feature

**Action:** Move `RPStorageService` to `character_engine` library

**Files to modify:**
- Move: `src/extensions/features/text_modes/roleplay/storage.py` → `src/extensions/libraries/character_engine/storage.py`
- Update: `src/extensions/features/characters/manager.py`
- Update: `src/extensions/features/text_modes/roleplay/*`

**Impact:** Moderate - need to update imports in 2-3 files

### Phase 2: Create Extension Service Registry (Medium Priority)
**Goal:** Allow extensions to discover each other's services through context

**Implementation:**
1. Add `register_service()` to ExtensionContext
2. Extensions register their services during initialization
3. Other extensions query services via context

**Example:**
```python
# In extension.py
class MyExtension(LogicExtension):
    async def initialize(self, context):
        # Register service
        context.register_service("my_service", self.my_service)

# In another extension
other_service = context.get_extension_service("my_extension", "my_service")
```

### Phase 3: Refactor Direct Imports (Low Priority)
**Goal:** Replace remaining cross-feature imports with context-based access

**Current violations:**
- None identified beyond Phase 1

## Summary

**Current State:** Mostly clean with 1 known violation

**Acceptable Patterns:**
- Core infrastructure imports
- Library imports (by design)
- Same-extension imports
- Shared infrastructure imports (src.shared, src.api.shared)

**Action Items:**
1. ✅ Document acceptable patterns (this file)
2. ⏳ Move RPStorageService to character_engine library
3. ⏳ Create extension service registry
4. ⏳ Update imports to use context where appropriate
