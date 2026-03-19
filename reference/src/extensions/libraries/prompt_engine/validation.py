"""Crash-only prompt validation for prompt_engine.

This module provides validation utilities for ensuring required prompts
exist in configuration files. It follows the crash-only philosophy:
if a prompt is required, its absence immediately crashes the system
with a clear error message - NO SILENT FALLBACKS.

Usage:
    from prompt_engine import PromptValidator

    # Create validator for your namespace
    validator = PromptValidator("deep_research")

    # Declare required prompts
    validator.require("scout", "boss", "critic", "synthesis")

    # Declare known/valid keys (for detecting unknown prompts)
    validator.known_keys("scout", "boss", "boss.structured", "critic", ...)

    # Validate config - crashes if required prompts missing
    validator.validate_or_crash(config)

    # Or get validation results without crashing
    is_valid, errors, warnings = validator.validate(config)

Strict Validation (Multi-Layer):
    from prompt_engine import StrictPromptRegistry

    # Create strict registry with multi-layer validation
    registry = StrictPromptRegistry("deep_research")

    # Declare required prompts with variable contracts
    registry.require("boss")
    registry.declare_variables("boss",
        required=["research_brief", "iteration_count"],
        optional=["previous_findings"]
    )

    # Set mock data for preflight validation
    registry.set_mock_data({
        "research_brief": "[MOCK BRIEF]",
        "iteration_count": 1,
        "previous_findings": "[MOCK FINDINGS]"
    })

    # Validate with ALL layers (syntax, variables, preflight render)
    registry.validate(config)  # Crashes on ANY issue

    # Get prompt with contract validation
    prompt = registry.get_prompt(config, "boss",
        research_brief="...",
        iteration_count=3
    )
"""

from typing import Dict, Any, List, Set, Tuple, Optional, FrozenSet


class PromptValidator:
    """Validates prompt configs with crash-only semantics.

    Features:
    - Declare required prompts upfront
    - Validate all required prompts exist
    - Detect unknown/unused prompt keys
    - Crash immediately on validation failure

    Example:
        validator = PromptValidator("my_extension")
        validator.require("system_prompt", "user_prompt")
        validator.known_keys("system_prompt", "user_prompt", "optional_prompt")

        # Crashes if system_prompt or user_prompt missing
        validator.validate_or_crash(config)
    """

    def __init__(self, namespace: str):
        """Initialize validator for a specific namespace.

        Args:
            namespace: Name of the extension/module using this validator
                      (used in error messages for clarity)
        """
        self.namespace = namespace
        self._required: Set[str] = set()
        self._known_keys: Set[str] = set()
        self._allow_unknown = False
        self._validated = False  # Cache validation result

    def require(self, *keys: str) -> "PromptValidator":
        """Declare required prompts. Chainable.

        Args:
            *keys: Prompt keys that must exist in config

        Returns:
            Self for method chaining

    Example:
        validator.require("worker", "boss", "critic")
36
    # Crashes if worker_prompt or user_prompt missing
        validator.validate_or_crash(config)
    """
        self._required.update(keys)
        self._validated = False  # Reset cache
        return self

    def known_keys(self, *keys: str) -> "PromptValidator":
        """Declare known/valid prompt keys. Chainable.

        When set, validation will warn about keys in config that
        are not in this set (potential typos or unused prompts).

        Args:
            *keys: All valid prompt keys for this namespace

        Returns:
            Self for method chaining

        Example:
            validator.known_keys("worker", "boss", "boss.structured", "critic")
        """
        self._known_keys.update(keys)
        self._validated = False  # Reset cache
        return self

    def allow_unknown(self, allow: bool = True) -> "PromptValidator":
        """Allow unknown prompt keys in config.

        By default, if known_keys is set and a config has keys not
        in known_keys, a warning is generated. Call this to suppress
        those warnings.

        Args:
            allow: True to allow unknown keys (default: True)

        Returns:
            Self for method chaining
        """
        self._allow_unknown = allow
        return self

    def _prompt_exists(self, prompts: Dict[str, Any], key: str) -> bool:
        """Check if a prompt exists in the prompts dict.

        Supports multiple formats:
        1. Flat dotted keys: prompts["boss.structured"]
        2. Nested paths: prompts["boss"]["structured"]

        Args:
            prompts: The prompts section of config
            key: Prompt key (can be nested with dots)

        Returns:
            True if prompt exists and is not empty
        """
        # First, try flat key (most common in YAML with dotted names)
        if key in prompts:
            return self._is_valid_prompt_value(prompts[key])

        # Second, try nested path
        if "." in key:
            parts = key.split(".")
            current = prompts
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    # Part not found in nested structure
                    break
            else:
                # All parts found
                return self._is_valid_prompt_value(current)

        return False

    def _is_valid_prompt_value(self, value: Any) -> bool:
        """Check if a prompt value is valid (not empty/None).

        A value is considered valid if:
        - It's a non-empty string (template)
        - It's a dict with at least one template field (template, instructions, etc.)
        - It's a dict with metadata fields (boss_role, domain, etc.) - these are
          used to provide variables for base templates

        Args:
            value: The prompt value to check

        Returns:
            True if value is a valid prompt
        """
        if value is None:
            return False
        if isinstance(value, str):
            return len(value.strip()) > 0
        if isinstance(value, dict):
            # Check for template fields
            template_fields = ("template", "instructions", "prompt", "content", "system")
            for field in template_fields:
                if field in value and isinstance(value[field], str) and value[field].strip():
                    return True

            # Check for metadata fields (these provide variables for base templates)
            # Common metadata fields that indicate this is a valid config dict
            metadata_fields = (
                "boss_role", "domain", "worker_role", "operation_type",
                "synthesis_role", "citation_instructions", "output_instructions",
                "evaluation_focus", "critic_role", "output_type",
                "task_allocation_guidance", "dependency_graph_prompt",
                "citation_requirements", "available_skills", "research_brief",
            )
            for field in metadata_fields:
                if field in value:
                    return True

            # Empty dict
            if len(value) == 0:
                return False

            # Non-empty dict without known fields - still consider valid
            # (might be custom metadata for specific use cases)
            return True
        return True

    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """Validate config. Returns (is_valid, errors, warnings).

        Args:
            config: Full configuration dictionary (must contain 'prompts' key)

        Returns:
            Tuple of:
            - is_valid: True if all required prompts exist
            - errors: List of error messages (missing required prompts)
            - warnings: List of warning messages (unknown keys, etc.)
        """
        errors = []
        warnings = []
        prompts = config.get("prompts", {})

        # Check required prompts exist
        for key in sorted(self._required):
            if not self._prompt_exists(prompts, key):
                errors.append(f"Missing required prompt: '{key}'")

        # Check for unknown keys (if strict mode)
        if not self._allow_unknown and self._known_keys:
            unknown_keys = self._find_unknown_keys(prompts)
            for key in unknown_keys:
                if not key.startswith("_"):  # Allow private keys
                    warnings.append(f"Unknown prompt key: '{key}' (not in known_keys)")

        self._validated = len(errors) == 0
        return self._validated, errors, warnings

    def _find_unknown_keys(self, prompts: Dict[str, Any], prefix: str = "") -> List[str]:
        """Find all keys not in known_keys.

        Recursively checks nested dicts.

        Args:
            prompts: Prompts dict to check
            prefix: Current key prefix for nested paths

        Returns:
            List of unknown key paths
        """
        unknown = []
        for key, value in prompts.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if full_key in self._known_keys or key in self._known_keys:
                continue

            # If it's a dict, check if it's a prompt dict or a namespace
            if isinstance(value, dict):
                # Check if this looks like a prompt dict (has template-like keys)
                prompt_fields = {"template", "instructions", "prompt", "content", "system"}
                if any(k in value for k in prompt_fields):
                    # This is a prompt dict, check the parent key
                    unknown.append(full_key)
                else:
                    # This might be a namespace, recurse
                    unknown.extend(self._find_unknown_keys(value, full_key))
            else:
                unknown.append(full_key)

        return unknown

    def validate_or_crash(self, config: Dict[str, Any]) -> None:
        """Validate config. CRASH if invalid.

        This is the primary validation method for crash-only philosophy.
        If any required prompt is missing, immediately raises RuntimeError
        with a clear, formatted error message.

        Args:
            config: Full configuration dictionary

        Raises:
            RuntimeError: If any required prompt is missing
        """
        is_valid, errors, warnings = self.validate(config)

        if not is_valid:
            raise RuntimeError(
                f"\n"
                f"╔═══════════════════════════════════════════════════════════════════╗\n"
                f"║         PROMPT VALIDATION FAILED: {self.namespace:<33} ║\n"
                f"╚═══════════════════════════════════════════════════════════════════╝\n"
                f"\n"
                f"ERRORS:\n"
                f"{chr(10).join(f'  - {e}' for e in errors)}\n"
                f"\n"
                f"Required prompts for {self.namespace}:\n"
                f"  {', '.join(sorted(self._required))}\n"
                f"\n"
                f"Add the missing prompts to your YAML config:\n"
                f"  prompts:\n"
                f"    <key>: |\n"
                f"      Your prompt template here\n"
            )


class RequiredPromptRegistry:
    """Registry for required prompts with crash-only validation.

    This is an alternative interface to PromptValidator that provides
    a slightly different API more focused on get-or-crash patterns.

    Usage:
        registry = RequiredPromptRegistry("deep_research")
        registry.require("worker", "boss", "critic", "synthesis")

        # Validate config has all required prompts (crashes if missing)
        registry.validate(config)

        # Get a prompt (crashes if missing)
        prompt = registry.get_prompt(config, "boss", **kwargs)
    """

    def __init__(self, namespace: str):
        """Initialize registry for a specific namespace.

        Args:
            namespace: Name of the extension/module using this registry
        """
        self.namespace = namespace
        self._validator = PromptValidator(namespace)
        self._descriptions: Dict[str, str] = {}

    def require(self, key: str, description: str = "") -> "RequiredPromptRegistry":
        """Declare a required prompt. Chainable.

        Args:
            key: Prompt key that must exist
            description: Optional description of what this prompt is for

        Returns:
            Self for method chaining
        """
        self._validator.require(key)
        self._descriptions[key] = description or f"Required prompt: {key}"
        return self

    def known_keys(self, *keys: str) -> "RequiredPromptRegistry":
        """Declare known/valid prompt keys. Chainable."""
        self._validator.known_keys(*keys)
        return self

    def allow_unknown(self, allow: bool = True) -> "RequiredPromptRegistry":
        """Allow unknown prompt keys. Chainable."""
        self._validator.allow_unknown(allow)
        return self

    def validate(self, config: Dict[str, Any]) -> None:
        """Validate all required prompts exist. CRASH if any missing.

        Args:
            config: Full configuration dictionary

        Raises:
            RuntimeError: If any required prompt is missing
        """
        self._validator.validate_or_crash(config)

    def get_prompt(self, config: Dict[str, Any], key: str, **kwargs) -> str:
        """Get and render a prompt. CRASH if missing.

        Args:
            config: Full configuration dictionary
            key: Prompt key to retrieve
            **kwargs: Variables for Jinja2 rendering

        Returns:
            Rendered prompt string

        Raises:
            RuntimeError: If prompt is missing
        """
        # Validate all required prompts first
        self.validate(config)

        prompts = config.get("prompts", {})
        prompt = self._resolve_prompt(prompts, key)

        if prompt is None:
            raise RuntimeError(
                f"\n"
                f"╔═══════════════════════════════════════════════════════════════════╗\n"
                f"║         REQUIRED PROMPT MISSING: {key:<34} ║\n"
                f"╚═══════════════════════════════════════════════════════════════════╝\n"
                f"\n"
                f"The prompt '{key}' is required but not found in config.\n"
                f"\n"
                f"Required prompts for {self.namespace}:\n"
                f"  {', '.join(sorted(self._validator._required))}\n"
            )

        # Handle dict prompts - extract template
        if isinstance(prompt, dict):
            prompt = self._extract_template_from_dict(prompt, key)

        # Render with Jinja2
        return self._render_prompt(prompt, key, kwargs)

    def _resolve_prompt(self, prompts: Dict[str, Any], key: str) -> Optional[Any]:
        """Resolve a prompt from the prompts dict.

        Supports nested paths like "boss.guidance".

        Args:
            prompts: The prompts section of config
            key: Prompt key (can be nested with dots)

        Returns:
            Prompt value or None if not found
        """
        if "." in key:
            parts = key.split(".")
            current = prompts
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            return current
        else:
            return prompts.get(key)

    def _extract_template_from_dict(self, prompt_dict: Dict[str, Any], key: str) -> str:
        """Extract template string from a dict-style prompt.

        Args:
            prompt_dict: Dict containing prompt fields
            key: Prompt key for error messages

        Returns:
            Template string

        Raises:
            RuntimeError: If no template field found
        """
        # Try common template fields in order
        for field in ("template", "instructions", "prompt", "content", "system"):
            if field in prompt_dict and isinstance(prompt_dict[field], str):
                return prompt_dict[field]

        raise RuntimeError(
            f"Prompt '{key}' is a dict but has no template field. "
            f"Expected one of: template, instructions, prompt, content, system"
        )

    def _render_prompt(self, template: str, key: str, variables: Dict[str, Any]) -> str:
        """Render a prompt template with Jinja2.

        Args:
            template: Jinja2 template string
            key: Prompt key for error messages
            variables: Variables to render

        Returns:
            Rendered prompt string
        """
        from jinja2 import Template, TemplateError, StrictUndefined

        try:
            t = Template(
                template,
                undefined=StrictUndefined,  # Fail on missing variables
                trim_blocks=True,
                lstrip_blocks=True,
            )
            return t.render(**variables)
        except TemplateError as e:
            raise RuntimeError(
                f"Failed to render prompt '{key}' with Jinja2: {e}\n"
                f"Variables provided: {list(variables.keys())}"
            ) from e


# =============================================================================
# STRICT PROMPT REGISTRY - Multi-Layer Validation
# =============================================================================

class StrictPromptRegistry:
    """Multi-layer prompt validation with zero tolerance for errors.

    This registry provides comprehensive validation across multiple layers:
    1. Syntax: Validates Jinja2 template syntax
    2. Variables: Extracts and validates required/optional variables
    3. Contract: Enforces variable contracts at render time
    4. Preflight: Tests templates with mock data before runtime
    5. Output: Validates rendered output meets constraints

    NO HIDDEN ERRORS. If anything is wrong, the system crashes immediately
    with a clear error message.

    Example:
        registry = StrictPromptRegistry("deep_research")

        # Declare required prompts
        registry.require("boss", "critic", "synthesis")

        # Declare variable contracts
        registry.declare_variables("boss",
            required=["research_brief", "iteration_count"],
            optional=["previous_findings", "worker_results"]
        )

        # Set mock data for preflight validation
        registry.set_mock_data({
            "research_brief": "[MOCK RESEARCH BRIEF]",
            "iteration_count": 1,
            "previous_findings": "[MOCK FINDINGS]",
            "worker_results": []
        })

        # Validate config with ALL layers
        registry.validate(config)  # Crashes on ANY issue

        # Get prompt with contract validation
        prompt = registry.get_prompt(config, "boss",
            research_brief="My research topic",
            iteration_count=1
        )
    """

    def __init__(self, namespace: str):
        """Initialize strict registry.

        Args:
            namespace: Name of the extension/module for error messages
        """
        self.namespace = namespace
        self._required: Set[str] = set()
        self._known_keys: Set[str] = set()
        self._allow_unknown = False

        # Variable contracts: key -> {"required": set, "optional": set}
        self._variable_contracts: Dict[str, Dict[str, Set[str]]] = {}

        # Mock data for preflight validation
        self._mock_data: Dict[str, Any] = {}
        self._preflight_enabled = False

        # Output constraints
        self._output_constraints: Dict[str, Dict[str, int]] = {}

        # Cache for extracted variables
        self._extracted_variables: Dict[str, Set[str]] = {}

    # =========================================================================
    # Configuration Methods (Chainable)
    # =========================================================================

    def require(self, *keys: str) -> "StrictPromptRegistry":
        """Declare required prompts. Chainable.

        Args:
            *keys: Prompt keys that must exist in config

        Returns:
            Self for method chaining
        """
        self._required.update(keys)
        return self

    def known_keys(self, *keys: str) -> "StrictPromptRegistry":
        """Declare known/valid prompt keys. Chainable.

        Args:
            *keys: All valid prompt keys for this namespace

        Returns:
            Self for method chaining
        """
        self._known_keys.update(keys)
        return self

    def allow_unknown(self, allow: bool = True) -> "StrictPromptRegistry":
        """Allow unknown prompt keys. Chainable.

        Args:
            allow: True to allow unknown keys

        Returns:
            Self for method chaining
        """
        self._allow_unknown = allow
        return self

    def declare_variables(
        self,
        key: str,
        required: Optional[Set[str]] = None,
        optional: Optional[Set[str]] = None
    ) -> "StrictPromptRegistry":
        """Declare variable contract for a prompt. Chainable.

        This creates a contract that will be validated at render time.
        If a template uses a variable not in required or optional,
        validation will fail (catches typos).

        Args:
            key: Prompt key
            required: Set of required variable names
            optional: Set of optional variable names

        Returns:
            Self for method chaining

        Example:
            registry.declare_variables("boss",
                required={"research_brief", "iteration_count"},
                optional={"previous_findings"}
            )
        """
        self._variable_contracts[key] = {
            "required": required or set(),
            "optional": optional or set()
        }
        return self

    def set_mock_data(self, mock_data: Dict[str, Any]) -> "StrictPromptRegistry":
        """Set mock data for preflight validation. Chainable.

        When enabled, validate() will attempt to render all templates
        with this mock data to catch errors before runtime.

        Args:
            mock_data: Dictionary of mock variable values

        Returns:
            Self for method chaining
        """
        self._mock_data = mock_data
        self._preflight_enabled = True
        return self

    def set_output_constraints(
        self,
        key: str,
        min_length: int = 10,
        max_length: int = 100000
    ) -> "StrictPromptRegistry":
        """Set output length constraints for a prompt. Chainable.

        Args:
            key: Prompt key
            min_length: Minimum rendered length (default: 10)
            max_length: Maximum rendered length (default: 100000)

        Returns:
            Self for method chaining
        """
        self._output_constraints[key] = {
            "min_length": min_length,
            "max_length": max_length
        }
        return self

    # =========================================================================
    # Static Analysis Methods
    # =========================================================================

    @staticmethod
    def extract_template_variables(template: str) -> Set[str]:
        """Extract all variables used in a Jinja2 template.

        Uses AST analysis - does not render the template.

        Args:
            template: Jinja2 template string

        Returns:
            Set of variable names used in the template
        """
        from jinja2 import Environment, meta

        try:
            env = Environment()
            ast = env.parse(template)
            return meta.find_undeclared_variables(ast)
        except Exception:
            # If parsing fails, return empty set (will be caught by syntax validation)
            return set()

    @staticmethod
    def validate_jinja2_syntax(template: str) -> Tuple[bool, Optional[str]]:
        """Validate Jinja2 template syntax.

        Args:
            template: Jinja2 template string

        Returns:
            Tuple of (is_valid, error_message)
        """
        from jinja2 import Environment, TemplateSyntaxError

        try:
            Environment().from_string(template)
            return True, None
        except TemplateSyntaxError as e:
            return False, f"Line {e.lineno}: {e.message}"

    # =========================================================================
    # Core Validation Methods
    # =========================================================================

    def validate(self, config: Dict[str, Any]) -> None:
        """Validate config with ALL layers. CRASH on ANY issue.

        Runs all validation layers:
        1. Existence: All required prompts exist
        2. Unknown keys: No typo'd prompt names (if allow_unknown=False)
        3. Syntax: All templates have valid Jinja2 syntax
        4. Variables: All template variables are in contracts
        5. Preflight: All templates render with mock data (if enabled)

        Args:
            config: Full configuration dictionary

        Raises:
            RuntimeError: If any validation fails
        """
        prompts = config.get("prompts", {})
        errors: List[str] = []

        # Layer 1: Check required prompts exist
        for key in sorted(self._required):
            template = self._get_template(prompts, key)
            if template is None:
                errors.append(f"MISSING PROMPT: '{key}'")
                continue

            # Layer 3: Validate Jinja2 syntax
            is_valid, syntax_error = self.validate_jinja2_syntax(template)
            if not is_valid:
                errors.append(f"INVALID SYNTAX in '{key}': {syntax_error}")
                continue

            # Layer 4: Extract and validate variables
            variables_used = self.extract_template_variables(template)
            self._extracted_variables[key] = variables_used

            contract = self._variable_contracts.get(key)
            if contract:
                declared = contract["required"] | contract["optional"]
                undeclared = variables_used - declared
                if undeclared:
                    errors.append(
                        f"UNDECLARED VARIABLES in '{key}': {undeclared}\n"
                        f"    Used in template but not in contract.\n"
                        f"    Add to required or optional: {undeclared}"
                    )

            # Layer 5: Preflight render (if enabled)
            if self._preflight_enabled and self._mock_data:
                mock_context = self._build_mock_context(key, variables_used)
                if mock_context is not None:
                    missing = variables_used - set(mock_context.keys())
                    if missing:
                        errors.append(
                            f"MOCK DATA INCOMPLETE for '{key}':\n"
                            f"    Missing variables: {missing}\n"
                            f"    Update set_mock_data() to include these variables"
                        )
                    else:
                        preflight_error = self._preflight_render(key, template, mock_context)
                        if preflight_error:
                            errors.append(preflight_error)

        # Layer 2: Check for unknown keys (typo detection)
        if not self._allow_unknown and self._known_keys:
            unknown_keys = self._find_unknown_keys(prompts)
            for key in unknown_keys:
                if not key.startswith("_"):  # Allow private keys
                    errors.append(
                        f"UNKNOWN PROMPT KEY: '{key}'\n"
                        f"    This may be a typo. Known keys: {sorted(self._known_keys)[:10]}...\n"
                        f"    If intentional, add to known_keys() or set allow_unknown(True)"
                    )

        if errors:
            raise RuntimeError(
                f"\n"
                f"{'═'*70}\n"
                f"STRICT VALIDATION FAILED: {self.namespace}\n"
                f"{'═'*70}\n\n"
                + "\n\n".join(f"❌ {e}" for e in errors)
                + f"\n\n{'═'*70}\n"
            )

    def _find_unknown_keys(self, prompts: Dict[str, Any], prefix: str = "") -> List[str]:
        """Find all keys not in known_keys.

        Recursively checks nested dicts.

        Args:
            prompts: Prompts dict to check
            prefix: Current key prefix for nested paths

        Returns:
            List of unknown key paths
        """
        unknown = []
        for key, value in prompts.items():
            full_key = f"{prefix}.{key}" if prefix else key

            # Check if this key or its full path is known
            if full_key in self._known_keys or key in self._known_keys:
                continue

            # If it's a dict, check if it's a prompt dict or a namespace
            if isinstance(value, dict):
                # Check if this looks like a prompt dict (has template-like keys)
                prompt_fields = {"template", "instructions", "prompt", "content", "system"}
                if any(k in value for k in prompt_fields):
                    # This is a prompt dict, check the parent key
                    unknown.append(full_key)
                else:
                    # This might be a namespace, recurse
                    unknown.extend(self._find_unknown_keys(value, full_key))
            else:
                unknown.append(full_key)

        return unknown

    def get_prompt(self, config: Dict[str, Any], key: str, **kwargs) -> str:
        """Get and render a prompt with full validation. CRASH on ANY issue.

        Args:
            config: Full configuration dictionary
            key: Prompt key to retrieve
            **kwargs: Variables for rendering

        Returns:
            Rendered prompt string

        Raises:
            RuntimeError: If validation fails or variable contract is violated
        """
        # Run full validation first
        self.validate(config)

        prompts = config.get("prompts", {})
        template = self._get_template(prompts, key)

        if template is None:
            raise RuntimeError(
                f"\n{'═'*70}\n"
                f"REQUIRED PROMPT MISSING: {key}\n"
                f"{'═'*70}\n"
            )

        # Validate variable contract
        provided = set(kwargs.keys())
        contract = self._variable_contracts.get(key)

        if contract:
            missing = contract["required"] - provided
            if missing:
                raise RuntimeError(
                    f"\n{'═'*70}\n"
                    f"CONTRACT VIOLATION: {key}\n"
                    f"{'═'*70}\n"
                    f"Missing required variables: {missing}\n"
                    f"Required: {contract['required']}\n"
                    f"Optional: {contract['optional']}\n"
                    f"Provided: {provided}\n"
                    f"{'═'*70}\n"
                )

        # Auto-fill optional variables with empty string if not provided
        render_kwargs = dict(kwargs)
        if contract:
            for var in contract["optional"]:
                if var not in render_kwargs:
                    render_kwargs[var] = ""

        # Render with StrictUndefined
        rendered = self._render_template(template, key, render_kwargs)

        # Validate output constraints
        constraints = self._output_constraints.get(key)
        if constraints:
            length = len(rendered)
            if length < constraints["min_length"]:
                raise RuntimeError(
                    f"\n{'═'*70}\n"
                    f"OUTPUT TOO SHORT: {key}\n"
                    f"{'═'*70}\n"
                    f"Rendered length: {length}\n"
                    f"Minimum required: {constraints['min_length']}\n"
                    f"{'═'*70}\n"
                )
            if length > constraints["max_length"]:
                raise RuntimeError(
                    f"\n{'═'*70}\n"
                    f"OUTPUT TOO LONG: {key}\n"
                    f"{'═'*70}\n"
                    f"Rendered length: {length}\n"
                    f"Maximum allowed: {constraints['max_length']}\n"
                    f"{'═'*70}\n"
                )

        return rendered

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _get_template(self, prompts: Dict[str, Any], key: str) -> Optional[str]:
        """Get template string from prompts dict.

        Supports nested paths and dict-style prompts.

        Args:
            prompts: Prompts section of config
            key: Prompt key (can be nested with dots)

        Returns:
            Template string or None if not found
        """
        value = None

        # Try flat key first
        if key in prompts:
            value = prompts[key]
        # Try nested path
        elif "." in key:
            parts = key.split(".")
            current = prompts
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    current = None
                    break
            value = current

        if value is None:
            return None

        # Extract template from dict if needed
        if isinstance(value, dict):
            for field in ("template", "instructions", "prompt", "content", "system"):
                if field in value and isinstance(value[field], str):
                    return value[field]
            return None

        return value if isinstance(value, str) else None

    def _build_mock_context(self, key: str, variables: Set[str]) -> Optional[Dict[str, Any]]:
        """Build mock context for preflight rendering.

        Args:
            key: Prompt key
            variables: Variables needed by template

        Returns:
            Mock context dict or None if cannot build
        """
        if not self._mock_data:
            return None

        # Start with all mock data
        context = dict(self._mock_data)

        # Add empty values for any optional variables not in mock data
        contract = self._variable_contracts.get(key)
        if contract:
            for var in contract["optional"]:
                if var not in context:
                    context[var] = ""

        return context

    def _preflight_render(self, key: str, template: str, context: Dict[str, Any]) -> Optional[str]:
        """Attempt preflight render. Returns error message if failed.

        Args:
            key: Prompt key
            template: Template string
            context: Mock context

        Returns:
            Error message string or None if successful
        """
        try:
            from jinja2 import Template, StrictUndefined, TemplateError

            t = Template(
                template,
                undefined=StrictUndefined,
                trim_blocks=True,
                lstrip_blocks=True,
            )
            t.render(**context)
            return None
        except TemplateError as e:
            return f"PREFLIGHT FAILED for '{key}': {e}"
        except Exception as e:
            return f"PREFLIGHT FAILED for '{key}': {type(e).__name__}: {e}"

    def _render_template(self, template: str, key: str, variables: Dict[str, Any]) -> str:
        """Render template with Jinja2.

        Args:
            template: Template string
            key: Prompt key for error messages
            variables: Variables to render

        Returns:
            Rendered string

        Raises:
            RuntimeError: If rendering fails
        """
        from jinja2 import Template, StrictUndefined, TemplateError

        try:
            t = Template(
                template,
                undefined=StrictUndefined,
                trim_blocks=True,
                lstrip_blocks=True,
            )
            return t.render(**variables)
        except TemplateError as e:
            raise RuntimeError(
                f"\n{'═'*70}\n"
                f"RENDER FAILED: {key}\n"
                f"{'═'*70}\n"
                f"Error: {e}\n"
                f"Variables provided: {list(variables.keys())}\n"
                f"{'═'*70}\n"
            ) from e
