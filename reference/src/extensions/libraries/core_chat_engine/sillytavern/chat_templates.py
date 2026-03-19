"""
Chat Templates

Different LLMs expect different chat message formats.
This module provides templates for common formats.
"""

from typing import Dict, List, Optional, Any
from enum import Enum
import re


class ChatTemplate(Enum):
    """Supported chat template types."""

    AUTO = "auto"  # Detect from model name
    RAW = "raw"  # No formatting
    CHATML = "chatml"  # OpenAI ChatML format
    ALPACA = "alpaca"  # Alpaca format
    VICUNA = "vicuna"  # Vicuna format
    ZEPHYR = "zephyr"  # Zephyr format
    PHI = "phi"  # Phi format
    LLAMA2 = "llama2"  # Llama 2 format
    MISTRAL = "mistral"  # Mistral format
    GEMMA = "gemma"  # Gemma format


class ChatTemplateManager:
    """
    Manage chat message formatting for different LLMs.

    Converts standard message format to model-specific format.
    """

    # Template definitions
    TEMPLATES = {
        ChatTemplate.CHATML: {
            "prefix": "",
            "message_templates": {
                "system": "<|im_start|>system\n{content}<|im_end|>",
                "user": "<|im_start|>user\n{content}<|im_end|>",
                "assistant": "<|im_start|>assistant\n{content}<|im_end|>",
            },
            "suffix": "",
            "add_bos": False,
            "add_eos": False,
        },
        ChatTemplate.ALPACA: {
            "prefix": "",
            "message_templates": {
                "system": "### Instruction:\n{content}\n",
                "user": "### Input:\n{content}\n",
                "assistant": "### Response:\n{content}",
            },
            "suffix": "",
            "add_bos": False,
            "add_eos": True,
        },
        ChatTemplate.VICUNA: {
            "prefix": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n",
            "message_templates": {
                "system": "",
                "user": "USER: {content}",
                "assistant": "ASSISTANT: {content}",
            },
            "suffix": "",
            "add_bos": True,
            "add_eos": False,
        },
        ChatTemplate.LLAMA2: {
            "prefix": "[INST] ",
            "message_templates": {
                "system": "<<SYS>>\n{content}\n<</SYS>>\n\n",
                "user": "{content} [/INST]",
                "assistant": "{content}</s>",
            },
            "suffix": "",
            "add_bos": True,
            "add_eos": False,
        },
        ChatTemplate.MISTRAL: {
            "prefix": "[INST] ",
            "message_templates": {
                "system": "",
                "user": "{content} [/INST] ",
                "assistant": "",
            },
            "suffix": " </s>",
            "add_bos": False,
            "add_eos": False,
        },
        ChatTemplate.ZEPHYR: {
            "prefix": "<|system|>\n{content}</s>\n<|user|>\n",
            "message_templates": {
                "system": "{content}",
                "user": "{content}",
                "assistant": "{content}",
            },
            "suffix": "</s>",
            "add_bos": True,
            "add_eos": False,
        },
        ChatTemplate.PHI: {
            "prefix": "",
            "message_templates": {
                "system": "Instruct: {content}\n",
                "user": "Input: {content}\n",
                "assistant": "Output: {content}",
            },
            "suffix": "",
            "add_bos": False,
            "add_eos": True,
        },
        ChatTemplate.GEMMA: {
            "prefix": "<start_of_turn>user\n",
            "message_templates": {
                "system": "",
                "user": "{content}<end_of_turn>\n<start_of_turn>model\n",
                "assistant": "{content}<end_of_turn>\n",
            },
            "suffix": "",
            "add_bos": False,
            "add_eos": False,
        },
        ChatTemplate.RAW: {
            "prefix": "",
            "message_templates": {
                "system": "{content}",
                "user": "{content}",
                "assistant": "{content}",
            },
            "suffix": "",
            "add_bos": False,
            "add_eos": False,
        },
    }

    # Model to template mapping
    MODEL_TEMPLATE_MAP = {
        "gpt-4": ChatTemplate.CHATML,
        "gpt-3.5-turbo": ChatTemplate.CHATML,
        "gpt-35-turbo": ChatTemplate.CHATML,
        "gpt-4o": ChatTemplate.CHATML,
        "chatgpt": ChatTemplate.CHATML,
        # Llama models
        "llama-2": ChatTemplate.LLAMA2,
        "llama2": ChatTemplate.LLAMA2,
        "llama-3": ChatTemplate.RAW,
        "llama3": ChatTemplate.RAW,
        "vicuna": ChatTemplate.VICUNA,
        # Mistral
        "mistral": ChatTemplate.MISTRAL,
        "mixtral": ChatTemplate.MISTRAL,
        # Others
        "alpaca": ChatTemplate.ALPACA,
        "zephyr": ChatTemplate.ZEPHYR,
        "phi": ChatTemplate.PHI,
        "gemma": ChatTemplate.GEMMA,
    }

    def __init__(self, template: ChatTemplate = ChatTemplate.AUTO):
        """
        Initialize chat template manager.

        Args:
            template: Template to use (AUTO detects from model name)
        """
        self.template = template

    def detect_template(self, model: str) -> ChatTemplate:
        """Detect appropriate template from model name."""
        model_lower = model.lower()

        for pattern, template in self.MODEL_TEMPLATE_MAP.items():
            if pattern in model_lower:
                return template

        return ChatTemplate.RAW  # Default to raw

    def format_messages(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
    ) -> str:
        """
        Format messages according to template.

        Args:
            messages: List of message dicts with "role" and "content"
            model: Optional model name for auto-detection

        Returns:
            Formatted prompt string
        """
        # Determine template
        template = self.template
        if template == ChatTemplate.AUTO:
            if model:
                template = self.detect_template(model)
            else:
                template = ChatTemplate.RAW

        # Get template definition
        template_def = self.TEMPLATES.get(template, self.TEMPLATES[ChatTemplate.RAW])

        # Build formatted prompt
        output = template_def["prefix"]

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            # Get template for role
            role_template = template_def["message_templates"].get(role, "{content}")

            # Format and add
            formatted = role_template.format(content=content)
            output += formatted + "\n"

        # Add suffix if needed
        if template_def["suffix"]:
            output += template_def["suffix"]

        # Add BOS/EOS tokens if needed
        if template_def["add_bos"]:
            output = "<s>" + output
        if template_def["add_eos"]:
            output = output + "</s>"

        return output

    def parse_template_from_string(self, template_str: str) -> Dict[str, Any]:
        """
        Parse a custom template from string.

        Supports simple placeholder syntax: {role}, {content}, etc.
        """
        # This would parse a custom template string
        # For now, return basic structure
        return {
            "prefix": "",
            "message_templates": {
                "system": template_str,
                "user": template_str,
                "assistant": template_str,
            },
            "suffix": "",
            "add_bos": False,
            "add_eos": False,
        }


def create_chat_template_manager(
    template: ChatTemplate = ChatTemplate.AUTO,
) -> ChatTemplateManager:
    """Factory function to create ChatTemplateManager."""
    return ChatTemplateManager(template=template)
