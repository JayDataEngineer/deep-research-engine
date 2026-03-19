# Deep Agents Mode Extension

**Deep agents all the way down** - An intelligent conversational agent with router-based subgraph delegation, ultimate memory system, and tool integration.

## Overview

Deep Agents is a chat mode extension for the Unified LLM Interface that provides:

- **Intelligent Routing**: LLM-based router for delegating to appropriate subgraphs
- **Ultimate Memory**: Three-tier memory system (semantic, episodic, procedural)
- **Tool Integration**: Web search, vector search, and more
- **Infinite Conversation**: Persistent context via LangGraph checkpointing
- **Larger Context Window**: 20-turn context with 8000 token window

## Features

| Feature | Description |
|---------|-------------|
| **Router System** | LLM-based routing between chat, web research, vector search, and tools |
| **Semantic Memory** | Long-term fact storage and retrieval |
| **Episodic Memory** | Conversation history and few-shot examples |
| **Procedural Memory** | Self-improving instructions and guidelines |
| **Tool Support** | web_search, web_get, vector_search |
| **Context Window** | 20 turns with 8000 token capacity |
| **Checkpointing** | Persistent conversation threads |

## Installation

The extension is included in the Unified LLM Interface. No additional installation required.

To enable the extension, ensure it's enabled in `config.yaml`:

```yaml
extension:
  enabled: true
```

## Usage

### Via Web Interface

1. Select **Deep Agents** mode from the mode dropdown
2. Start chatting!

### Via CLI

```bash
# Start interactive chat
uv run backend/manage.py deep_agents chat --user-id alice

# Manage memory
uv run backend/manage.py deep_agents memory list --user-id alice
uv run backend/manage.py deep_agents memory clear --user-id alice --memory-type all

# Manage instructions (procedural memory)
uv run backend/manage.py deep_agents instructions show --agent-id deep_agents
uv run backend/manage.py deep_agents instructions update --instruction-type system_prompt --content "Your new instructions"

# View statistics
uv run backend/manage.py deep_agents stats --user-id alice
```

### Via API

```python
from langchain_core.messages import HumanMessage
from src.extensions.modes.text_modes.deep_agents.extension import DeepAgentsExtension

# Create extension
config = {
    'extension': {'name': 'deep_agents', 'category': 'mode'},
    'enabled_tools': ['web_search', 'web_get', 'vector_search'],
}
ext = DeepAgentsExtension(config, Path('src/extensions/features/text_modes/deep_agents'))
graph = ext.build_graph()

# Chat
config = {'configurable': {'thread_id': 'my_conversation'}}
result = await graph.ainvoke(
    {'messages': [HumanMessage(content='Hello!')]},
    config
)
```

## Configuration

### config.yaml

```yaml
extension:
  name: deep_agents
  type: mode
  display_name: Deep Agents
  description: Deep agents all the way down - infinite conversation with router, subgraphs, and ultimate memory system
  version: 1.0.0
  category: mode
  api_prefix: /api/v1/deep_agents
  enabled: true

system:
  lazy_load: true  # Load on first use for faster startup

# Available tools
enabled_tools:
  - web_search
  - web_get
  - vector_search
```

### Brain Configuration

The extension uses `BrainConfig` for advanced customization:

```python
brain_config = BrainConfig(
    mode="deep_agents",
    enable_tool_routing=True,
    enable_conversation_summary=True,
    enable_mood_tracking=False,
    max_context_turns=20,
    summarize_interval=10,
    context_window_size=20,
    available_tools=['web_search', 'web_get', 'vector_search'],
    max_tokens=1000,
    temperature=0.7,
)
```

## Architecture

### Memory System (CoALA-based)

```
┌─────────────────────────────────────────────────────────┐
│                  ULTIMATE MEMORY SYSTEM                 │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   SEMANTIC   │  │   EPISODIC   │  │  PROCEDURAL  │  │
│  │              │  │              │  │              │  │
│  │  • Facts     │  │  • Episodes  │  │  • System    │  │
│  │  • Knowledge │  │  • Examples  │  │    Prompt    │  │
│  │  • Concepts  │  │  • History   │  │  • Guidelines │  │
│  │  (Postgres)  │  │  (Checkpointer)│  │  • Reflections│  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### Router System

```
                    ┌─────────────┐
                    │   USER      │
                    │  MESSAGE    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   ROUTER    │
                    │  (LLM-based)│
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼──────┐  ┌────────▼────────┐  ┌────▼─────────┐
│    CHAT      │  │  WEB RESEARCH   │  │   TOOLS      │
│  SUBGRAPH    │  │    SUBGRAPH     │  │   SUBGRAPH   │
└──────────────┘  └─────────────────┘  └──────────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    ┌──────▼──────┐
                    │   RESPONSE   │
                    └───────────────┘
```

## API Endpoints

### Health Check

```bash
GET /api/v1/deep_agents/health
```

**Response:**
```json
{
  "status": "healthy",
  "mode": "deep_agents",
  "features": {
    "router_system": true,
    "ultimate_memory": true,
    "subgraphs": ["chat", "web_research", "vector_search", "tools"]
  },
  "message": "Deep Agents mode is operational"
}
```

## CLI Commands

### Chat

```bash
# Start interactive chat
uv run backend/manage.py deep_agents chat \
  --user-id alice \
  --agent-id deep_agents \
  --router-type llm_based
```

### Memory Management

```bash
# List memories
uv run backend/manage.py deep_agents memory list \
  --user-id alice \
  --memory-type all

# Clear memories
uv run backend/manage.py deep_agents memory clear \
  --user-id alice \
  --memory-type semantic \
  --confirm

# Export/Import memories
uv run backend/manage.py deep_agents memory export \
  --user-id alice \
  --output backup.json

uv run backend/manage.py deep_agents memory import \
  --input backup.json \
  --confirm
```

### Instructions (Procedural Memory)

```bash
# Show instructions
uv run backend/manage.py deep_agents instructions show \
  --agent-id deep_agents \
  --instruction-type system_prompt

# Update instructions
uv run backend/manage.py deep_agents instructions update \
  --agent-id deep_agents \
  --instruction-type guidelines \
  --content "New guidelines here" \
  --feedback "Reasoning for change"

# Reset instructions
uv run backend/manage.py deep_agents instructions reset \
  --agent-id deep_agents \
  --instruction-type system_prompt \
  --confirm
```

### Statistics

```bash
uv run backend/manage.py deep_agents stats \
  --user-id alice \
  --agent-id deep_agents
```

## Development

### Project Structure

```
deep_agents/
├── __init__.py
├── extension.py       # Main extension class
├── graph.py           # Graph building logic
├── config.yaml        # Extension configuration
├── routes.py          # FastAPI routes
├── cli.py             # CLI commands
└── README.md          # This file
```

### Adding New Tools

1. Add tool to `config.yaml`:

```yaml
enabled_tools:
  - web_search
  - your_new_tool
```

2. Ensure tool is registered in the tool manager.

### Customizing Brain Configuration

Edit `graph.py` to modify the `BrainConfig`:

```python
brain_config = BrainConfig(
    mode="deep_agents",
    enable_tool_routing=True,
    max_context_turns=30,  # Increase context
    temperature=0.8,       # More creative responses
    # ... other settings
)
```

## Troubleshooting

### Extension Not Loading

Check logs:
```bash
tail -f data/outputs/logs/server/<latest>/backend.log | grep deep_agents
```

### Memory Not Persisting

Ensure PostgreSQL is running:
```bash
make lang-tools  # Starts Docker services
```

### Tools Not Working

Verify tools are enabled in config and registered:
```bash
uv run backend/manage.py tools list
```

## License

Part of the Unified LLM Interface project.

## Credits

Built with:
- LangGraph for agent orchestration
- LangChain for LLM integration
- Pydantic for configuration validation
- FastAPI for API endpoints
- PostgreSQL for memory persistence
