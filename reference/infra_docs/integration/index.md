# Integration Guides

Complete guides for integrating your applications with the Shared Docker Infrastructure.

## Integration Patterns

### [LangChain + LiteLLM + Langfuse](langchain.md)

Use LangChain with LiteLLM proxy and automatic Langfuse tracing.

**Best for:** Production applications with LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://litellm.local/v1",
    api_key="sk-litellm-master",
    model="qwen-35-27"
)

response = llm.invoke("What is the capital of France?")
```

**Documentation:** [LangChain Integration](langchain.md)

### [LiteLLM SDK + Langfuse](litellm-sdk.md)

Direct LiteLLM SDK usage with Langfuse integration.

**Best for:** Simple scripts, lightweight applications

```python
import litellm

litellm.success_callback = ["langfuse"]

response = litellm.completion(
    model="openai/qwen-35-27",
    api_base="http://litellm.local/v1",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**Documentation:** [LiteLLM SDK](litellm-sdk.md)

### [Deep Agents / LangGraph](deep-agents.md)

LangGraph and complex multi-step agents with full tracing.

**Best for:** Agentic applications, complex workflows

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model="openai:gpt-4o",
    tools=[my_tool]
)

result = agent.invoke({"messages": [...]})
```

**Documentation:** [Deep Agents](deep-agents.md)

### [OpenAI SDK](openai-sdk.md)

Drop-in replacement for OpenAI API.

**Best for:** Existing OpenAI applications

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://litellm.local/v1",
    api_key="sk-litellm-master"
)

response = client.chat.completions.create(
    model="qwen-35-27",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**Documentation:** [OpenAI SDK](openai-sdk.md)

### [MCP Tools](mcp-tools.md)

Using web tools (search, scraping, crawling) via MCP.

**Best for:** Applications needing web access

```python
response = client.chat.completions.create(
    model="qwen-35-27",
    messages=[{"role": "user", "content": "Search for AI news"}],
    tools=[{"type": "function", "function": {"name": "search_web"}}]
)
```

**Documentation:** [MCP Tools](mcp-tools.md)

## Setup

### Environment Variables

```bash
# .env in your application
LITELLM_BASE_URL=http://litellm.local/v1
LITELLM_API_KEY=sk-litellm-master
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://langfuse.local
```

### Required Packages

```bash
# LangChain
pip install langchain-openai langchain-core

# LiteLLM
pip install litellm langfuse

# LangGraph
pip install langgraph langchain langchain-openai

# OpenAI SDK
pip install openai
```

## Integration Checklist

- [ ] Set environment variables
- [ ] Configure client with LiteLLM base URL
- [ ] Set up Langfuse for tracing
- [ ] Test basic connectivity
- [ ] Add error handling
- [ ] Configure session tracking
- [ ] Set up monitoring

## Common Patterns

### Session Tracking

```python
response = client.chat.completions.create(
    model="qwen-35-27",
    messages=[{"role": "user", "content": "Hello!"}],
    metadata={
        "user_id": "user-123",
        "session_id": "session-456"
    }
)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="qwen-35-27",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Error Handling

```python
from openai import APIError, APIConnectionError

try:
    response = client.chat.completions.create(
        model="qwen-35-27",
        messages=[{"role": "user", "content": "Hello!"}]
    )
except APIConnectionError:
    print("LiteLLM unavailable")
except APIError as e:
    if e.status_code == 503:
        print("Model loading, please wait")
    else:
        raise
```

## Next Steps

- **[LangChain](langchain.md)** - LangChain integration
- **[LiteLLM SDK](litellm-sdk.md)** - Direct SDK usage
- **[Deep Agents](deep-agents.md)** - Agent frameworks
- **[OpenAI SDK](openai-sdk.md)** - OpenAI replacement
- **[MCP Tools](mcp-tools.md)** - Web tools
