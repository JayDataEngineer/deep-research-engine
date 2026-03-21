"""
Central Tool Registry

All LangChain tools are registered here in one place.
Extensions request tools by name, and the registry provides them.

LANGGRAPH V1 PATTERN (NEW):
- V2 tools use @tool decorator and return Command(update={...})
- State is passed via runtime: ToolRuntime parameter
- No in-place state mutations
- No need for set_state() calls

V1 PATTERN (LEGACY):
- V1 tools inherit from BaseTool class
- State is injected via set_state() method
- In-place state mutations (anti-pattern)

Usage:
    from .tools.registry import load_tools

    # In your extension's initialize()
    tool_names = self.config.get("enabled_tools", [])
    tools = load_tools(tool_names)
"""

from .logging_config import get_logger
from typing import Callable, List, Dict, Any

logger = get_logger(__name__)


# =============================================================================
# Import All Tools
# =============================================================================

# Web Tools (V2 - LangGraph V1 pattern with @tool decorator)
try:
    from .ulli_langchain.web import (
        web_search,  # V2 - @tool decorator
        web_get,     # V2 - @tool decorator
        web_research_agent,  # V2 - @tool decorator
        community_scrape,  # V2 - @tool decorator
        get_source_content,  # V2 - @tool decorator
        generate_source_id,  # Helper function
        web_map,     # V1 - BaseTool class
        web_crawl,   # V1 - BaseTool class
        fact_check,  # V1 - function
    )
    WEB_TOOLS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Web tools not available: {e}")
    WEB_TOOLS_AVAILABLE = False
    web_search = web_get = web_map = web_crawl = fact_check = community_scrape = None
    web_research_agent = get_source_content = generate_source_id = None

# Image Tools - ComfyUI Extension Based
try:
    from .ulli_langchain.image.edit import set_character_reference, clear_character_reference
    # Character reference tools are kept as-is
    CHARACTER_TOOLS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Character tools not available: {e}")
    CHARACTER_TOOLS_AVAILABLE = False
    set_character_reference = clear_character_reference = None

# ComfyUI Extension Tools
try:
    from langchain_core.tools import tool
    from src.extensions.integrations.comfyui.image_generation.z_image_turbo.extension import register_extension as register_z_image_turbo
    from src.extensions.integrations.comfyui.image_edit.qwen_image_edit.extension import register_extension as register_qwen_image_edit
    COMFYUI_EXTENSIONS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ComfyUI extensions not available: {e}")
    COMFYUI_EXTENSIONS_AVAILABLE = False

# Memory/RAG Tools
try:
    from .ulli_langchain.memory.rag import vector_search, add_documents, list_collections, get_rag_client
    MEMORY_TOOLS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Memory tools not available: {e}")
    MEMORY_TOOLS_AVAILABLE = False
    vector_search = add_documents = list_collections = get_rag_client = None


def _should_register_pdf_agent() -> bool:
    """Check if pdf_deep_agent should be registered.

    Only register pdf_deep_agent if:
    1. Vector store is initialized (get_rag_client() returns not None)
    2. Collection has documents (count > 0)

    This prevents workers from getting a useless tool that returns empty results.
    """
    try:
        if get_rag_client is None:
            return False

        client = get_rag_client()
        if client is None:
            return False

        # Check if collection has any documents
        count = client._collection.count()
        has_documents = count > 0

        if has_documents:
            logger.info(f"[REGISTRY] pdf_deep_agent registered: vector store has {count} documents")
        else:
            logger.info("[REGISTRY] pdf_deep_agent NOT registered: vector store is empty")

        return has_documents

    except Exception as e:
        logger.warning(f"[REGISTRY] Error checking vector store for pdf_deep_agent: {e}")
        return False

# Vision Tools
try:
    from .ulli_langchain.vision.vision_tools import (
        image_analysis as analyze_image,
        video_analysis as analyze_video,
        extract_text_from_screenshot as extract_text_from_image,
        diagnose_error_screenshot as diagnose_error,
        understand_technical_diagram as understand_diagram,
        analyze_data_visualization as analyze_data_viz
    )
    from .ulli_langchain.vision.caption import (
        image_caption,
        batch_caption,
    )
    VISION_TOOLS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Vision tools not available: {e}")
    VISION_TOOLS_AVAILABLE = False
    analyze_image = analyze_video = extract_text_from_image = None
    diagnose_error = understand_diagram = analyze_data_viz = None
    image_caption = batch_caption = None

# OSINT Tools
try:
    from .ulli_langchain.osint.osint_tools import (
        osint_username_search as username_search,
        osint_email_investigation as email_investigation,
        osint_domain_whois as domain_whois,
        osint_breach_lookup as breach_lookup,
        osint_generate_dorks as google_dork
    )
    OSINT_TOOLS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OSINT tools not available: {e}")
    OSINT_TOOLS_AVAILABLE = False
    username_search = email_investigation = domain_whois = None
    breach_lookup = google_dork = None

# Agent Tools
try:
    from .ulli_langchain.agent.deep_agent import deep_agent
    from .ulli_langchain.agent.subprocess_tools import (
        deep_research_subprocess,
        index_tts_subprocess,
        run_cli_script
    )
    AGENT_TOOLS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Agent tools not available: {e}")
    AGENT_TOOLS_AVAILABLE = False
    deep_agent = deep_research_subprocess = index_tts_subprocess = run_cli_script = None


# PDF Agent (conditional on vector store having documents)
try:
    from .ulli_langchain.memory.pdf_agent import pdf_deep_agent
    PDF_AGENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PDF agent not available: {e}")
    PDF_AGENT_AVAILABLE = False
    pdf_deep_agent = None

# Context Engine Tools (LangGraph Store + ArXiv)
try:
    # Import search_memories from context_engine extension
    from src.extensions.features.tools.context_engine.tools.search_memories import search_memories
    from src.extensions.features.tools.context_engine.tools.arxiv import (
        search_arxiv,
        get_arxiv_abstract,
    )
    CONTEXT_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Context Engine tools not available: {e}")
    CONTEXT_ENGINE_AVAILABLE = False
    search_memories = search_arxiv = get_arxiv_abstract = None

# Context Engine Observability Tools (logging, metrics)
try:
    from src.extensions.features.tools.context_engine.tools.observability import (
        log_observation,
        save_metrics,
        get_logs,
        get_run_history,
        save_output,
    )
    OBSERVABILITY_TOOLS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Observability tools not available: {e}")
    OBSERVABILITY_TOOLS_AVAILABLE = False
    log_observation = save_metrics = get_logs = get_run_history = save_output = None

# Context Engine Session Tools (session management)
try:
    from src.extensions.features.tools.context_engine.tools.session import (
        create_research_session,
        get_session_status,
        update_session_status,
        list_sessions,
        cleanup_session,
    )
    SESSION_TOOLS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Session tools not available: {e}")
    SESSION_TOOLS_AVAILABLE = False
    create_research_session = get_session_status = update_session_status = None
    list_sessions = cleanup_session = None


# =============================================================================
# Tool Catalog
# =============================================================================

TOOL_CATALOG: Dict[str, Callable] = {}

# Web Tools (V2 - @tool decorator, no factory needed)
if WEB_TOOLS_AVAILABLE and web_search:
    TOOL_CATALOG["web_search"] = web_search
if WEB_TOOLS_AVAILABLE and web_get:
    TOOL_CATALOG["web_get"] = web_get
if WEB_TOOLS_AVAILABLE and web_research_agent:
    TOOL_CATALOG["web_research_agent"] = web_research_agent
if WEB_TOOLS_AVAILABLE and community_scrape:
    TOOL_CATALOG["community_scrape"] = community_scrape
if WEB_TOOLS_AVAILABLE and get_source_content:
    TOOL_CATALOG["get_source_content"] = get_source_content
if WEB_TOOLS_AVAILABLE and web_map:
    TOOL_CATALOG["web_map"] = web_map
if WEB_TOOLS_AVAILABLE and web_crawl:
    TOOL_CATALOG["web_crawl"] = web_crawl
if WEB_TOOLS_AVAILABLE and fact_check:
    TOOL_CATALOG["fact_check"] = fact_check

# =============================================================================
# ComfyUI Extension-Based Image Tools
# =============================================================================

# Cache for extension instances
_z_image_turbo_ext = None
_qwen_image_edit_ext = None


def _get_z_image_turbo_extension():
    """Get or create the z_image_turbo extension instance."""
    global _z_image_turbo_ext
    if _z_image_turbo_ext is None and COMFYUI_EXTENSIONS_AVAILABLE:
        from pathlib import Path
        ext_dir = Path("src/extensions/services/comfyui/image_generation/z_image_turbo")
        _z_image_turbo_ext = register_z_image_turbo()
    return _z_image_turbo_ext


def _get_qwen_image_edit_extension():
    """Get or create the qwen_image_edit extension instance."""
    global _qwen_image_edit_ext
    if _qwen_image_edit_ext is None and COMFYUI_EXTENSIONS_AVAILABLE:
        from pathlib import Path
        ext_dir = Path("src/extensions/services/comfyui/image_edit/qwen_image_edit")
        _qwen_image_edit_ext = register_qwen_image_edit()
    return _qwen_image_edit_ext


# =============================================================================
# ComfyUI Tool Wrappers (LangChain-compatible)
# =============================================================================

if COMFYUI_EXTENSIONS_AVAILABLE:
    from langchain_core.tools import tool

    @tool
    async def generate_image(
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 9,
        negative_prompt: str = "",
        seed: int = -1,
    ) -> str:
        """Generate an image from text prompt using ComfyUI.

        Args:
            prompt: Image description/prompt
            width: Image width in pixels (512-2048, default 1024)
            height: Image height in pixels (512-2048, default 1024)
            steps: Number of sampling steps (1-20, default 9)
            negative_prompt: Things to avoid in the image
            seed: Random seed (-1 for random)

        Returns:
            JSON string with success status and output path
        """
        ext = _get_z_image_turbo_extension()
        if ext:
            result = await ext.send_request("generate", {
                "prompt": prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "negative_prompt": negative_prompt,
                "seed": seed,
            })
            return str(result)
        return '{"success": false, "error": "Extension not available"}'

    @tool
    async def edit_image(
        image: str,
        prompt: str,
        strength: float = 0.5,
        negative_prompt: str = "",
    ) -> str:
        """Edit an image using ComfyUI with Qwen GGUF model.

        Args:
            image: Path to source image or base64 data
            prompt: Edit instruction (e.g., "make the person smile")
            strength: Edit strength 0.1-1.0 (lower=more faithful, higher=more change)
            negative_prompt: Things to avoid in the edit

        Returns:
            JSON string with success status and output path
        """
        ext = _get_qwen_image_edit_extension()
        if ext:
            result = await ext.send_request("edit", {
                "image": image,
                "prompt": prompt,
                "strength": strength,
                "negative_prompt": negative_prompt,
            })
            return str(result)
        return '{"success": false, "error": "Extension not available"}'

    @tool
    async def edit_character_image(
        prompt: str,
        character_name: str = "default",
        strength: float = 0.3,
    ) -> str:
        """Edit a character image while preserving appearance using ComfyUI.

        Args:
            prompt: Edit instruction for the character image
            character_name: Name of the character (uses reference if set)
            strength: Edit strength 0.1-0.5 (lower to preserve more character features)

        Returns:
            JSON string with success status and output path
        """
        # Use character reference if available
        from .ulli_langchain.image.edit import get_character_reference
        ref_image = get_character_reference(character_name)

        ext = _get_qwen_image_edit_extension()
        if ext:
            result = await ext.send_request("edit", {
                "image": ref_image if ref_image else "",
                "prompt": f"{prompt} (preserve character appearance)",
                "strength": strength,
            })
            return str(result)
        return '{"success": false, "error": "Extension not available"}'

    TOOL_CATALOG["generate_image"] = generate_image
    TOOL_CATALOG["edit_image"] = edit_image
    TOOL_CATALOG["edit_character_image"] = edit_character_image

# Character Reference Tools (kept as-is)
if CHARACTER_TOOLS_AVAILABLE and set_character_reference:
    TOOL_CATALOG["set_character_reference"] = set_character_reference
if CHARACTER_TOOLS_AVAILABLE and clear_character_reference:
    TOOL_CATALOG["clear_character_reference"] = clear_character_reference

# Memory Tools
if MEMORY_TOOLS_AVAILABLE and vector_search:
    TOOL_CATALOG["vector_search"] = vector_search
if MEMORY_TOOLS_AVAILABLE and add_documents:
    TOOL_CATALOG["add_documents"] = add_documents
if MEMORY_TOOLS_AVAILABLE and list_collections:
    TOOL_CATALOG["list_collections"] = list_collections
# Only register pdf_deep_agent if vector store has documents
if PDF_AGENT_AVAILABLE and pdf_deep_agent and _should_register_pdf_agent():
    TOOL_CATALOG["pdf_deep_agent"] = pdf_deep_agent

# Vision Tools
if VISION_TOOLS_AVAILABLE and analyze_image:
    TOOL_CATALOG["analyze_image"] = analyze_image
if VISION_TOOLS_AVAILABLE and analyze_video:
    TOOL_CATALOG["analyze_video"] = analyze_video
if VISION_TOOLS_AVAILABLE and extract_text_from_image:
    TOOL_CATALOG["extract_text_from_image"] = extract_text_from_image
if VISION_TOOLS_AVAILABLE and diagnose_error:
    TOOL_CATALOG["diagnose_error"] = diagnose_error
if VISION_TOOLS_AVAILABLE and understand_diagram:
    TOOL_CATALOG["understand_diagram"] = understand_diagram
if VISION_TOOLS_AVAILABLE and analyze_data_viz:
    TOOL_CATALOG["analyze_data_viz"] = analyze_data_viz
if VISION_TOOLS_AVAILABLE and image_caption:
    TOOL_CATALOG["image_caption"] = image_caption
if VISION_TOOLS_AVAILABLE and batch_caption:
    TOOL_CATALOG["batch_caption"] = batch_caption

# OSINT Tools
if OSINT_TOOLS_AVAILABLE and username_search:
    TOOL_CATALOG["username_search"] = username_search
if OSINT_TOOLS_AVAILABLE and email_investigation:
    TOOL_CATALOG["email_investigation"] = email_investigation
if OSINT_TOOLS_AVAILABLE and domain_whois:
    TOOL_CATALOG["domain_whois"] = domain_whois
if OSINT_TOOLS_AVAILABLE and breach_lookup:
    TOOL_CATALOG["breach_lookup"] = breach_lookup
if OSINT_TOOLS_AVAILABLE and google_dork:
    TOOL_CATALOG["google_dork"] = google_dork

# Agent Tools
if AGENT_TOOLS_AVAILABLE and deep_agent:
    TOOL_CATALOG["deep_agent"] = deep_agent
if AGENT_TOOLS_AVAILABLE and deep_research_subprocess:
    TOOL_CATALOG["deep_research_subprocess"] = deep_research_subprocess
if AGENT_TOOLS_AVAILABLE and index_tts_subprocess:
    TOOL_CATALOG["index_tts_subprocess"] = index_tts_subprocess
if AGENT_TOOLS_AVAILABLE and run_cli_script:
    TOOL_CATALOG["run_cli_script"] = run_cli_script

# Context Engine Tools (LangGraph Store + ArXiv)
if CONTEXT_ENGINE_AVAILABLE and search_memories:
    TOOL_CATALOG["search_memories"] = search_memories
if CONTEXT_ENGINE_AVAILABLE and search_arxiv:
    TOOL_CATALOG["search_arxiv"] = search_arxiv
if CONTEXT_ENGINE_AVAILABLE and get_arxiv_abstract:
    TOOL_CATALOG["get_arxiv_abstract"] = get_arxiv_abstract

# Context Engine Observability Tools
if OBSERVABILITY_TOOLS_AVAILABLE and log_observation:
    TOOL_CATALOG["log_observation"] = log_observation
if OBSERVABILITY_TOOLS_AVAILABLE and save_metrics:
    TOOL_CATALOG["save_metrics"] = save_metrics
if OBSERVABILITY_TOOLS_AVAILABLE and get_logs:
    TOOL_CATALOG["get_logs"] = get_logs
if OBSERVABILITY_TOOLS_AVAILABLE and get_run_history:
    TOOL_CATALOG["get_run_history"] = get_run_history
if OBSERVABILITY_TOOLS_AVAILABLE and save_output:
    TOOL_CATALOG["save_output"] = save_output

# Context Engine Session Tools
if SESSION_TOOLS_AVAILABLE and create_research_session:
    TOOL_CATALOG["create_research_session"] = create_research_session
if SESSION_TOOLS_AVAILABLE and get_session_status:
    TOOL_CATALOG["get_session_status"] = get_session_status
if SESSION_TOOLS_AVAILABLE and update_session_status:
    TOOL_CATALOG["update_session_status"] = update_session_status
if SESSION_TOOLS_AVAILABLE and list_sessions:
    TOOL_CATALOG["list_sessions"] = list_sessions
if SESSION_TOOLS_AVAILABLE and cleanup_session:
    TOOL_CATALOG["cleanup_session"] = cleanup_session


# =============================================================================
# Registry Functions
# =============================================================================

def get_all_tools() -> List[str]:
    """Get list of all available tool names."""
    return list(TOOL_CATALOG.keys())


def get_tool(name: str) -> Callable | None:
    """Get a specific tool by name."""
    return TOOL_CATALOG.get(name)


def load_tools(tool_names: List[str]) -> List[Callable]:
    """Load tools from the registry by name.

    Args:
        tool_names: List of tool names to load (e.g., ["web_search", "edit_character_image"])

    Returns:
        List of tool functions

    Example:
        tools = load_tools(["web_search", "edit_character_image"])
    """
    tools = []
    for name in tool_names:
        tool = get_tool(name)
        if tool:
            tools.append(tool)
            logger.info(f"[REGISTRY] Loaded tool: {name}")
        else:
            logger.warning(f"[REGISTRY] Requested unknown tool: {name}")
    return tools


def load_tools_by_category(category: str) -> List[Callable]:
    """Load all tools in a category.

    Args:
        category: Category name ("web", "image", "memory", "vision", "osint", "agent")

    Returns:
        List of tool functions in that category

    Example:
        web_tools = load_tools_by_category("web")
    """
    category_map = {
        "web": ["web_search", "web_get", "web_map", "web_crawl", "community_scrape", "fact_check", "get_source_content", "web_research_agent"],
        "image": ["generate_image", "edit_image", "edit_character_image", "set_character_reference", "clear_character_reference"],  # ComfyUI extension-based
        "memory": ["vector_search", "add_documents", "list_collections", "pdf_deep_agent"],
        "vision": ["analyze_image", "analyze_video", "extract_text_from_image", "diagnose_error",
                  "understand_diagram", "analyze_data_viz", "image_caption", "batch_caption"],
        "osint": ["username_search", "email_investigation", "domain_whois", "breach_lookup", "google_dork"],
        "agent": ["deep_agent", "deep_research_subprocess", "index_tts_subprocess", "run_cli_script"],
    }

    tool_names = category_map.get(category, [])
    return load_tools(tool_names)


__all__ = [
    "TOOL_CATALOG",
    "get_all_tools",
    "get_tool",
    "load_tools",
    "load_tools_by_category",
]
