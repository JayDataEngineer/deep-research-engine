"""Vision tools using LiteLLM proxy.

Integrates with LiteLLM for image and video analysis.
Uses vision-capable models via OpenAI-compatible API.

Requires: LiteLLM proxy running (http://litellm.local/v1)
Vision Model: Configured via LiteLLM (e.g., qwen-35-27-vision, zai-glm-4v)
"""

import base64
from .logging_config import get_logger
import os
from typing import Optional, List, Dict, Any

from langchain_core.tools import tool

logger = get_logger(__name__)

# =============================================================================
# VISION MODEL CONFIGURATION
# =============================================================================
# Vision model name via LiteLLM (configured in LiteLLM proxy)
# Can be overridden via LLM_VISION_MODEL environment variable
VISION_MODEL = os.getenv("LLM_VISION_MODEL", "qwen-35-27-vision")


# =============================================================================
# VISION TOOLS
# =============================================================================

@tool
async def image_analysis(
    image_source: str,
    prompt: str,
) -> str:
    """General-purpose image understanding and analysis.

    Use this tool to:
    - Analyze image content and context
    - Extract information from images
    - Understand visual content
    - Describe what's in an image

    Args:
        image_source: URL or file path to the image (supports PNG, JPG, JPEG)
        prompt: Specific instructions on what to analyze or extract from the image

    Returns:
        Analysis results as text
    """
    vision = _VisionTools()
    return await vision.image_analysis(image_source, prompt)


@tool
async def video_analysis(
    video_source: str,
    prompt: str,
) -> str:
    """Analyze video content (frame-based analysis).

    Note: This tool provides guidance for frame-based video analysis.
    For actual video analysis, extract key frames and use image_analysis.

    Args:
        video_source: URL or file path to the video (MP4, MOV, M4V)
        prompt: What to analyze or look for in the video

    Returns:
        Analysis guidance and frame extraction instructions
    """
    vision = _VisionTools()
    return await vision.video_analysis(video_source, prompt)


@tool
async def ui_to_artifact(
    image_source: str,
    artifact_type: str,
) -> str:
    """Convert UI screenshots into artifacts (code, prompts, specs, or descriptions).

    Use this tool to:
    - Generate code from UI screenshots
    - Create detailed prompts for UI regeneration
    - Extract design specifications
    - Get comprehensive UI descriptions

    Args:
        image_source: URL or file path to the UI screenshot
        artifact_type: Type of artifact to generate - "code", "prompt", "spec", or "description"

    Returns:
        Generated artifact based on the UI screenshot
    """
    vision = _VisionTools()
    return await vision.ui_to_artifact(image_source, artifact_type)


@tool
async def extract_text_from_screenshot(
    image_source: str,
    prompt: str = "Extract all text from this image",
) -> str:
    """OCR extraction from screenshots with high accuracy.

    Use this tool to:
    - Extract text from code screenshots
    - Get terminal output
    - Read documentation screenshots
    - Extract any visible text with structure preserved

    Args:
        image_source: URL or file path to the screenshot
        prompt: Specific text extraction instructions (default: extract all text)

    Returns:
        Extracted text with structure preserved
    """
    vision = _VisionTools()
    return await vision.extract_text(image_source, prompt)


@tool
async def diagnose_error_screenshot(
    image_source: str,
    context: str = "",
) -> str:
    """Analyze error screenshots and propose actionable fixes.

    Use this tool to:
    - Understand error messages from screenshots
    - Get specific fix suggestions
    - Diagnose stack traces
    - Troubleshoot failures

    Args:
        image_source: URL or file path to the error screenshot
        context: Additional context about when the error occurred (e.g., "during npm install")

    Returns:
        Error analysis with specific actionable steps to fix
    """
    vision = _VisionTools()
    return await vision.diagnose_error(image_source, context)


@tool
async def understand_technical_diagram(
    image_source: str,
    prompt: str = "Explain this technical diagram in detail",
) -> str:
    """Interpret technical diagrams (architecture, flowchart, UML, ER, sequence).

    Use this tool to:
    - Understand system architecture diagrams
    - Analyze flowcharts and UML diagrams
    - Interpret ER diagrams and database schemas
    - Explain sequence diagrams

    Args:
        image_source: URL or file path to the technical diagram
        prompt: What to understand about the diagram (default: detailed explanation)

    Returns:
        Detailed explanation of diagram structure, components, relationships, and flows
    """
    vision = _VisionTools()
    return await vision.understand_diagram(image_source, prompt)


@tool
async def analyze_data_visualization(
    image_source: str,
    prompt: str = "Analyze this data visualization and extract insights",
) -> str:
    """Read charts and dashboards to surface insights and trends.

    Use this tool to:
    - Extract data from charts and graphs
    - Identify trends and patterns
    - Analyze bar charts, line graphs, pie charts
    - Get statistical insights from dashboards

    Args:
        image_source: URL or file path to the chart or dashboard
        prompt: What to analyze (default: extract insights and trends)

    Returns:
        Extracted data points, trends, patterns, and conclusions
    """
    vision = _VisionTools()
    return await vision.analyze_chart(image_source, prompt)


@tool
async def ui_diff_check(
    image_source_before: str,
    image_source_after: str,
    prompt: str = "Compare these two UIs and list differences",
) -> str:
    """Compare two UI screenshots to identify visual or implementation differences.

    Use this tool to:
    - Detect visual changes between UI versions
    - Identify missing or added elements
    - Find text/content changes
    - Flag bugs or inconsistencies

    Args:
        image_source_before: URL or file path to the 'before' UI
        image_source_after: URL or file path to the 'after' UI
        prompt: What aspects to compare (default: list all differences)

    Returns:
        Detailed comparison of visual and functional differences
    """
    vision = _VisionTools()
    return await vision.ui_diff_check(image_source_before, image_source_after, prompt)


# =============================================================================
# INTERNAL VISION API CLIENT (via LiteLLM)
# =============================================================================

class _VisionTools:
    """Internal vision API client via LiteLLM proxy."""

    def __init__(self):
        self.enabled = True  # LiteLLM is always available when infrastructure is running

    async def _call_vision_llm(self, messages: List[Dict[str, Any]]) -> str:
        """Call vision model via LiteLLM proxy.
        
        Uses the shared LiteLLM infrastructure for all vision tasks.
        Automatically traced to Langfuse for observability.
        
        Args:
            messages: List of message dicts with content (text + images)
            
        Returns:
            Model response content
        """
        try:
            # Import here to avoid circular imports
            from .litellm_factory import get_llm
            
            # Get vision LLM instance
            llm = await get_llm(
                model=VISION_MODEL,
                enable_thinking=False,  # Vision models typically don't need thinking
                max_tokens=4096,
                temperature=0.3,
            )
            
            # Call the model
            response = await llm.ainvoke(messages)
            
            # Extract content from response
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                logger.warning(f"Unexpected response type: {type(response)}")
                return str(response)
                
        except Exception as e:
            logger.error(f"Error calling vision model via LiteLLM: {e}")
            return f"Error calling vision model: {str(e)}"

    def _prepare_image_content(self, image_source: str) -> Dict[str, Any]:
        """Prepare image content for API call.
        
        Args:
            image_source: URL or file path to image
            
        Returns:
            Dict in OpenAI vision format
        """
        if image_source.startswith(("http://", "https://")):
            return {
                "type": "image_url",
                "image_url": {"url": image_source}
            }

        # Local file - encode as base64
        try:
            with open(image_source, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            import mimetypes
            mime_type, _ = mimetypes.guess_type(image_source)
            if not mime_type:
                mime_type = "image/png"

            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
            }
        except FileNotFoundError:
            logger.warning(f"Image file not found: {image_source}")
            return {
                "type": "text",
                "text": f"[Error: Image file not found: {image_source}]"
            }
        except Exception as e:
            logger.error(f"Error loading image {image_source}: {e}")
            return {
                "type": "text",
                "text": f"[Error loading image: {str(e)}]"
            }

    async def image_analysis(self, image_source: str, prompt: str) -> str:
        """General-purpose image understanding."""
        image_content = self._prepare_image_content(image_source)

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                image_content
            ]
        }]

        return await self._call_vision_llm(messages)

    async def video_analysis(self, video_source: str, prompt: str) -> str:
        """Analyze video content."""
        return f"Video analysis for: {video_source}\nPrompt: {prompt}\n\n[Note: Video analysis requires frame extraction. Please use image_analysis with key frames.]"

    async def ui_to_artifact(self, image_source: str, artifact_type: str) -> str:
        """Turn UI screenshots into code, prompts, specs, or descriptions."""
        prompts = {
            "code": "Analyze this UI screenshot and generate complete code to recreate it. Include HTML, CSS, and any necessary JavaScript.",
            "prompt": "Analyze this UI and write a detailed prompt that could be used to regenerate it.",
            "spec": "Create a detailed technical specification document for this UI, including components, layout, colors, typography, and interactions.",
            "description": "Provide a comprehensive description of this UI, including layout, styling, functionality, and user experience elements."
        }

        prompt = prompts.get(artifact_type.lower(), prompts["description"])
        return await self.image_analysis(image_source, prompt)

    async def extract_text(self, image_source: str, prompt: str = "Extract all text from this image") -> str:
        """OCR extraction from screenshots."""
        enhanced_prompt = f"{prompt}\n\nPlease extract ALL text visible in the image, preserving the structure and layout as much as possible. Include text from buttons, labels, code, terminal output, etc."
        return await self.image_analysis(image_source, enhanced_prompt)

    async def diagnose_error(self, image_source: str, context: str = "") -> str:
        """Analyze error screenshots and propose fixes."""
        prompt = "Analyze this error screenshot and provide:\n1. The specific error message\n2. The likely cause\n3. Specific actionable steps to fix it\n\n"
        if context:
            prompt += f"\nContext: {context}\n"
        prompt += "\nBe as specific and helpful as possible."
        return await self.image_analysis(image_source, prompt)

    async def understand_diagram(self, image_source: str, prompt: str = "Explain this technical diagram in detail") -> str:
        """Interpret technical diagrams."""
        enhanced_prompt = f"""{prompt}

Please provide:
1. The type of diagram (architecture, flowchart, UML, ER, sequence, etc.)
2. All components/entities and their relationships
3. The flow or logic shown
4. Any patterns or anti-patterns visible
5. A clear explanation of what the diagram communicates
"""
        return await self.image_analysis(image_source, enhanced_prompt)

    async def analyze_chart(self, image_source: str, prompt: str = "Analyze this data visualization and extract insights") -> str:
        """Read charts and dashboards to surface insights and trends."""
        enhanced_prompt = f"""{prompt}

Please extract:
1. The chart type (bar, line, pie, etc.)
2. All data points and values visible
3. Axes labels and scales
4. Key trends, patterns, or anomalies
5. Statistical insights if applicable
6. Any conclusions that can be drawn
"""
        return await self.image_analysis(image_source, enhanced_prompt)

    async def ui_diff_check(self, image_source_before: str, image_source_after: str, prompt: str = "Compare these two UIs and list differences") -> str:
        """Compare two UI screenshots to flag visual or implementation drift."""
        content_before = self._prepare_image_content(image_source_before)
        content_after = self._prepare_image_content(image_source_after)

        enhanced_prompt = f"""{prompt}

Please identify:
1. Visual differences (layout, colors, spacing, typography)
2. Missing or added elements
3. Text/content changes
4. Functional differences visible in the UI
5. Any bugs or inconsistencies
"""

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "BEFORE:"},
                content_before,
                {"type": "text", "text": "\n\nAFTER:"},
                content_after,
                {"type": "text", "text": f"\n\n{enhanced_prompt}"}
            ]
        }]

        return await self._call_vision_llm(messages)
