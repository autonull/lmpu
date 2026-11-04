"""LMPU Host - Two-phase constrained decoding."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

from lmpu.protocol.messages import ModelMessage, ModelMessageKind


class ConstrainedDecoder(ABC):
    """Abstract constrained decoder interface."""

    @abstractmethod
    async def decode_constrained(
        self,
        prompt: str,
        schema: dict[str, Any],
        max_tokens: int,
        stop_sequence: str,
    ) -> str:
        """Decode with JSON schema constraints."""
        pass

    @abstractmethod
    async def decode_free(
        self,
        prompt: str,
        max_tokens: int,
        stop_sequence: str,
    ) -> str:
        """Decode free text until stop sequence."""
        pass


class OutlinesDecoder(ConstrainedDecoder):
    """Constrained decoder using Outlines library with FSM-based generation.

    Uses Outlines' JSON schema guided generation which compiles the schema
    to an FSM, avoiding per-token prefix parsing (unlike lm-format-enforcer).
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._outlines_model = None

    async def _ensure_outlines(self):
        if self._outlines_model is None:
            import outlines

            self._outlines_model = outlines.models.Transformers(self.model, self.tokenizer)

    def _schema_to_regex(self, schema: dict[str, Any]) -> str:
        """Convert JSON schema to regex pattern for Outlines."""
        # Simplified: handle basic object properties
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        
        parts = []
        for prop_name, prop_schema in props.items():
            prop_type = prop_schema.get("type", "string")
            if prop_type == "string":
                # String pattern
                if "enum" in prop_schema:
                    enum_vals = "|".join(re.escape(v) for v in prop_schema["enum"])
                    parts.append(f'"{prop_name}"\\s*:\\s*"({enum_vals})"')
                else:
                    parts.append(f'"{prop_name}"\\s*:\\s*"[^"]*"')
            elif prop_type == "integer":
                parts.append(f'"{prop_name}"\\s*:\\s*-?\\d+')
            elif prop_type == "boolean":
                parts.append(f'"{prop_name}"\\s*:\\s*(true|false)')
            elif prop_type == "array":
                parts.append(f'"{prop_name}"\\s*:\\s*\\[[^\\]]*\\]')
            elif prop_type == "object":
                nested = self._schema_to_regex(prop_schema)
                parts.append(f'"{prop_name}"\\s*:\\s*\\{{{nested}\\}}')
            else:
                parts.append(f'"{prop_name}"\\s*:\\s*.*?')
        
        return ",".join(parts)

    async def decode_constrained(
        self,
        prompt: str,
        schema: dict[str, Any],
        max_tokens: int,
        stop_sequence: str,
    ) -> str:
        await self._ensure_outlines()

        import outlines

        # Use Outlines' JSON schema guided generation
        # Compile schema to regex for FSM-based generation
        regex_pattern = self._schema_to_regex(schema)
        full_pattern = f"\\{{{regex_pattern}\\}}"

        try:
            generator = outlines.generate.regex(self._outlines_model, full_pattern)
            result = generator(prompt)
            return result
        except Exception:
            # Fallback: free generation with schema hint
            schema_hint = f"\nRespond with valid JSON matching this schema: {json.dumps(schema)}\n"
            full_prompt = prompt + schema_hint
            return await self.decode_free(full_prompt, max_tokens, stop_sequence)

    async def decode_free(
        self,
        prompt: str,
        max_tokens: int,
        stop_sequence: str,
    ) -> str:
        # Free generation with stop sequence
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            stop_strings=[stop_sequence],
            tokenizer=self.tokenizer,
            do_sample=False,
        )

        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        # Truncate at stop sequence
        if stop_sequence in generated:
            generated = generated[: generated.index(stop_sequence)]

        return generated


class LMFormatEnforcerDecoder(ConstrainedDecoder):
    """Constrained decoder using lm-format-enforcer."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    async def decode_constrained(
        self,
        prompt: str,
        schema: dict[str, Any],
        max_tokens: int,
        stop_sequence: str,
    ) -> str:
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )

        parser = JsonSchemaParser(schema)
        prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn(
            self.tokenizer,
            parser,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            stop_strings=[stop_sequence] if stop_sequence else None,
            tokenizer=self.tokenizer,
            do_sample=False,
        )

        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        if stop_sequence and stop_sequence in generated:
            generated = generated[: generated.index(stop_sequence)]

        return generated

    async def decode_free(
        self,
        prompt: str,
        max_tokens: int,
        stop_sequence: str,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            stop_strings=[stop_sequence],
            tokenizer=self.tokenizer,
            do_sample=False,
        )

        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        if stop_sequence in generated:
            generated = generated[: generated.index(stop_sequence)]

        return generated


class FallbackDecoder(ConstrainedDecoder):
    """Fallback decoder: free generation + validation + retry."""

    def __init__(self, model, tokenizer, max_retries: int = 1):
        self.model = model
        self.tokenizer = tokenizer
        self.max_retries = max_retries

    async def decode_constrained(
        self,
        prompt: str,
        schema: dict[str, Any],
        max_tokens: int,
        stop_sequence: str,
    ) -> str:
        # Add schema hint to prompt
        schema_hint = f"\nRespond with valid JSON matching this schema: {json.dumps(schema)}\n"
        full_prompt = prompt + schema_hint

        for attempt in range(self.max_retries + 1):
            generated = await self.decode_free(full_prompt, max_tokens, stop_sequence)

            # Try to parse and validate
            try:
                data = json.loads(generated)
                # Basic validation
                if "kind" in data and data["kind"] in [m.value for m in ModelMessageKind]:
                    return generated
            except json.JSONDecodeError:
                pass

            if attempt < self.max_retries:
                # Retry with error feedback
                full_prompt = (
                    prompt
                    + f"\nPrevious response was invalid JSON. Please respond with valid JSON only.\n"
                )

        # Return last attempt even if invalid (caller will handle validation error)
        return generated

    async def decode_free(
        self,
        prompt: str,
        max_tokens: int,
        stop_sequence: str,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            stop_strings=[stop_sequence],
            tokenizer=self.tokenizer,
            do_sample=False,
        )

        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        if stop_sequence in generated:
            generated = generated[: generated.index(stop_sequence)]

        return generated


def create_decoder(decoder_type: str, model, tokenizer) -> ConstrainedDecoder:
    """Factory function to create decoder."""
    if decoder_type == "outlines":
        return OutlinesDecoder(model, tokenizer)
    elif decoder_type == "lm-format-enforcer":
        return LMFormatEnforcerDecoder(model, tokenizer)
    elif decoder_type == "fallback":
        return FallbackDecoder(model, tokenizer)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")
