"""DeepSeek LLM client — retry with backoff + validate-repair-retry loop.

Capabilities:
- call_structured(): JSON output with Pydantic validation + repair loop
- call_json(): Untyped JSON output (backwards-compatible)
- call_with_tools(): Native tool calling for ReAct agent loop
- Retry transient errors (429, 503, timeout) with exponential backoff
- JSON repair for common LLM formatting mistakes
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import settings
from app.core.http_client import get_http_client

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# HTTP status codes that are safe to retry (transient errors).
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


# ── Tool calling data types ───────────────────────────────────────────

@dataclass
class ToolCallInfo:
    """A single tool call from the LLM response."""
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMToolResponse:
    """Response from a tool-calling LLM call."""
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=lambda: {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
    })


class LLMClient:
    """Async DeepSeek client with production-grade error handling."""

    def __init__(self) -> None:
        self.api_key = (settings.DEEPSEEK_API_KEY or "").strip()
        self.model = (settings.DEEPSEEK_MODEL or "deepseek-chat").strip()
        self.base_url = (settings.DEEPSEEK_BASE_URL or "https://api.deepseek.com").strip().rstrip("/")
        self.timeout_seconds = max(5, int(settings.LLM_TIMEOUT_SECONDS))
        self.input_price_per_1m = float(settings.CHAT_DEEPSEEK_INPUT_PRICE_PER_1M)
        self.output_price_per_1m = float(settings.CHAT_DEEPSEEK_OUTPUT_PRICE_PER_1M)
        self.debug_log_prompts = bool(settings.CHAT_DEBUG_LOG_PROMPTS)
        self.debug_log_max_chars = max(500, int(settings.CHAT_DEBUG_LOG_MAX_CHARS))

    # ── Core API: Typed output with validation ────────────────────────

    async def call_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        response_model: type[T],
        stage: str = "llm",
        max_retries: int = 2,
        model_override: str | None = None,
        enable_thinking: bool = False,
    ) -> tuple[T, dict[str, int]]:
        """Call LLM and validate response against a Pydantic model.

        Implements the Validate → Repair → Retry loop:
        1. Call LLM → parse JSON → validate with Pydantic
        2. If validation fails → send error back to LLM to self-correct
        3. Retry up to max_retries times → then raise
        """
        accumulated_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        last_error: str | None = None
        last_raw_text: str | None = None

        for attempt in range(max_retries + 1):
            # Build prompt — include error feedback on retry.
            if attempt == 0 or last_error is None:
                effective_prompt = user_prompt
            else:
                effective_prompt = (
                    f"Your previous JSON output was invalid.\n"
                    f"Error: {last_error}\n"
                    f"Previous output (first 500 chars): {(last_raw_text or '')[:500]}\n\n"
                    f"Please fix and output valid JSON conforming to the schema.\n\n"
                    f"Original request:\n{user_prompt}"
                )

            raw_text, usage = await self._raw_call(
                system_prompt=system_prompt,
                user_prompt=effective_prompt,
                max_output_tokens=max_output_tokens,
                stage=f"{stage}_attempt{attempt}" if attempt > 0 else stage,
                model_override=model_override,
                enable_thinking=enable_thinking,
            )
            _accumulate_usage(accumulated_usage, usage)

            # Step 1: Parse JSON.
            parsed = self._repair_and_parse_json(raw_text)
            if parsed is None:
                last_error = f"JSON parse error on raw text"
                last_raw_text = raw_text
                logger.warning(
                    "[CHAT][%s] JSON parse failed (attempt %d/%d)",
                    stage, attempt + 1, max_retries + 1,
                )
                continue

            # Step 2: Validate with Pydantic.
            try:
                validated = response_model.model_validate(parsed)
                if attempt > 0:
                    logger.info("[CHAT][%s] Validation succeeded on retry %d", stage, attempt)
                return validated, accumulated_usage
            except ValidationError as exc:
                last_error = "; ".join(
                    f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}"
                    for e in exc.errors()[:3]
                )
                last_raw_text = raw_text
                logger.warning(
                    "[CHAT][%s] Pydantic validation failed (attempt %d/%d): %s",
                    stage, attempt + 1, max_retries + 1, last_error,
                )

        # All retries exhausted — return default instance.
        logger.error(
            "[CHAT][%s] All %d attempts failed, returning default %s",
            stage, max_retries + 1, response_model.__name__,
        )
        return response_model(), accumulated_usage

    async def call_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        stage: str = "llm",
        model_override: str | None = None,
        enable_thinking: bool = False,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """Backwards-compatible untyped JSON call (single attempt, no validation).

        Use call_structured() for production paths.
        """
        raw_text, usage = await self._raw_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
            stage=stage,
            model_override=model_override,
            enable_thinking=enable_thinking,
        )
        parsed = self._repair_and_parse_json(raw_text) or {}
        return parsed, usage

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        in_cost = (input_tokens / 1_000_000.0) * self.input_price_per_1m
        out_cost = (output_tokens / 1_000_000.0) * self.output_price_per_1m
        return round(in_cost + out_cost, 8)

    # ── Tool calling API (ReAct agent) ────────────────────────────────

    async def call_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_output_tokens: int = 4096,
        stage: str = "agent",
        max_retries: int = 2,
        model_override: str | None = None,
        enable_thinking: bool = True,
    ) -> LLMToolResponse:
        """Make a tool-calling LLM call for the ReAct agent loop.

        Unlike call_structured/call_json, this method:
        - Accepts a full messages array (not just system+user)
        - Sends tool definitions for native function calling
        - Does NOT use response_format: json_object (conflicts with tool calling)
        - Returns either tool_calls or content (final answer)
        """
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is missing")

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload: dict[str, Any] = {
            "model": model_override or self.model,
            "temperature": 0.0,
            "max_tokens": max_output_tokens,
            "messages": messages,
        }
        if enable_thinking:
            payload["thinking"] = {"type": "enabled"}
            # Reasoner generally prefers higher timeout and max_tokens but follows payload rules
        
        if tools:
            payload["tools"] = tools

        if self.debug_log_prompts:
            last_msg = messages[-1] if messages else {}
            logger.warning(
                "[CHAT][%s] %d messages, last_role=%s, %d tools",
                stage, len(messages), last_msg.get("role", "?"), len(tools or []),
            )

        last_exception: BaseException | None = None
        timeout = httpx.Timeout(self.timeout_seconds)
        client = get_http_client()

        for attempt in range(max_retries + 1):
            try:
                response = await client.post(
                    endpoint, headers=headers, json=payload, timeout=timeout,
                )

                if response.status_code in _RETRYABLE_STATUS_CODES and attempt < max_retries:
                    wait = min(2 ** attempt, 8)
                    logger.warning(
                        "[CHAT][%s] HTTP %d, retrying in %ds (attempt %d/%d)",
                        stage, response.status_code, wait, attempt + 1, max_retries + 1,
                    )
                    await asyncio.sleep(wait)
                    continue

                response.raise_for_status()
                body = response.json() if response.content else {}
                return self._parse_tool_response(body, stage)

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exception = exc
                if attempt < max_retries:
                    wait = min(2 ** attempt, 8)
                    logger.warning(
                        "[CHAT][%s] %s, retrying in %ds (attempt %d/%d)",
                        stage, type(exc).__name__, wait, attempt + 1, max_retries + 1,
                    )
                    await asyncio.sleep(wait)
                    continue

        raise last_exception or RuntimeError(f"LLM call failed after {max_retries + 1} attempts")

    def _parse_tool_response(self, body: dict[str, Any], stage: str) -> LLMToolResponse:
        """Parse OpenAI-compatible response into LLMToolResponse."""
        usage = self._extract_usage(body)

        choices = body.get("choices", [])
        if not choices:
            return LLMToolResponse(content="", usage=usage)

        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        finish_reason = choice.get("finish_reason", "stop")

        content = message.get("content")
        reasoning_content = message.get("reasoning_content")
        raw_tool_calls = message.get("tool_calls") or []

        tool_calls: list[ToolCallInfo] = []
        for tc in raw_tool_calls:
            func = tc.get("function", {}) if isinstance(tc, dict) else {}
            try:
                args = json.loads(func.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append(ToolCallInfo(
                id=tc.get("id", "") if isinstance(tc, dict) else "",
                name=func.get("name", "") if isinstance(func, dict) else "",
                arguments=args,
            ))

        if reasoning_content and self.debug_log_prompts:
            logger.warning("[CHAT][%s] REASONING: %s", stage, reasoning_content)

        if self.debug_log_prompts:
            if tool_calls:
                tc_summary = ", ".join(f"{tc.name}({tc.arguments})" for tc in tool_calls)
                logger.warning("[CHAT][%s] TOOL_CALLS: %s", stage, self._truncate(tc_summary))
            elif content:
                logger.warning("[CHAT][%s] CONTENT: %s", stage, self._truncate(content))

        return LLMToolResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    # ── Raw HTTP call with retry ──────────────────────────────────────

    async def _raw_call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        stage: str,
        max_retries: int = 2,
        model_override: str | None = None,
        enable_thinking: bool = False,
    ) -> tuple[str, dict[str, int]]:
        """Make the actual HTTP call with exponential backoff for transient errors."""
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is missing")

        self._log_prompt(stage=stage, prompt=user_prompt)

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload: dict[str, Any] = {
            "model": model_override or self.model,
            "temperature": 0.0,
            "max_tokens": max_output_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if enable_thinking:
            # Thinking mode is NOT compatible with response_format.
            payload["thinking"] = {"type": "enabled"}
        else:
            payload["response_format"] = {"type": "json_object"}

        last_exception: BaseException | None = None
        timeout = httpx.Timeout(self.timeout_seconds)
        client = get_http_client()

        for attempt in range(max_retries + 1):
            try:
                response = await client.post(
                    endpoint, headers=headers, json=payload, timeout=timeout,
                )

                # Handle response_format rejection (some endpoints don't support it).
                if response.status_code in {400, 422}:
                    fallback_payload = dict(payload)
                    fallback_payload.pop("response_format", None)
                    response = await client.post(
                        endpoint, headers=headers, json=fallback_payload, timeout=timeout,
                    )

                if response.status_code in _RETRYABLE_STATUS_CODES and attempt < max_retries:
                    wait = min(2 ** attempt, 8)
                    logger.warning(
                        "[CHAT][%s] HTTP %d, retrying in %ds (attempt %d/%d)",
                        stage, response.status_code, wait, attempt + 1, max_retries + 1,
                    )
                    await asyncio.sleep(wait)
                    continue

                response.raise_for_status()
                body = response.json() if response.content else {}

                reasoning = self._extract_reasoning_content(body)
                if reasoning and self.debug_log_prompts:
                    logger.warning("[CHAT][%s] REASONING: %s", stage, self._truncate(reasoning))

                text = self._extract_message_content(body)
                self._log_raw_response(stage=stage, text=text)
                usage = self._extract_usage(body)
                return text, usage

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exception = exc
                if attempt < max_retries:
                    wait = min(2 ** attempt, 8)
                    logger.warning(
                        "[CHAT][%s] %s, retrying in %ds (attempt %d/%d)",
                        stage, type(exc).__name__, wait, attempt + 1, max_retries + 1,
                    )
                    await asyncio.sleep(wait)
                    continue

        raise last_exception or RuntimeError(f"LLM call failed after {max_retries + 1} attempts")

    # ── JSON repair ───────────────────────────────────────────────────

    @staticmethod
    def _repair_and_parse_json(text: str) -> dict[str, Any] | None:
        """Parse JSON with common LLM format repair.

        Handles:
        - Markdown code fences (```json ... ```)
        - Trailing commas
        - Single quotes instead of double quotes
        - Missing outer braces
        """
        if not text or not text.strip():
            return None

        raw = text.strip()

        # Strip reasoning blocks if present (supports <think>, </think>, <|think|>).
        if "think" in raw.lower():
            raw = re.sub(r"<\|?think\|?>.*?<\|?/?think\|?>\s*", "", raw, flags=re.DOTALL|re.IGNORECASE)
            raw = raw.strip()

        # Strip markdown code fences.
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```\s*$", "", raw)
            raw = raw.strip()

        # First attempt: raw parse.
        try:
            result = json.loads(raw)
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            pass

        # Repair attempt 1: remove trailing commas before } or ].
        repaired = re.sub(r",\s*([}\]])", r"\1", raw)
        try:
            result = json.loads(repaired)
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            pass

        # Repair attempt 2: wrap in braces if missing.
        if not raw.startswith("{"):
            try:
                result = json.loads("{" + raw + "}")
                return result if isinstance(result, dict) else None
            except json.JSONDecodeError:
                pass

        return None

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _extract_message_content(response_json: dict[str, Any]) -> str:
        """Extract only the content field (not reasoning) from the API response."""
        if not isinstance(response_json, dict):
            return ""
        choices = response_json.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        content = message.get("content")
        return str(content) if content is not None else ""

    @staticmethod
    def _extract_reasoning_content(response_json: dict[str, Any]) -> str | None:
        """Extract reasoning_content from DeepSeek thinking mode response."""
        if not isinstance(response_json, dict):
            return None
        choices = response_json.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        rc = message.get("reasoning_content")
        return str(rc) if rc else None

    @staticmethod
    def _extract_usage(response_json: Any) -> dict[str, int]:
        usage = response_json.get("usage") if isinstance(response_json, dict) else None
        if not isinstance(usage, dict):
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        def _to_int(v: Any) -> int:
            try:
                return int(v)
            except Exception:
                return 0

        input_tokens = _to_int(usage.get("prompt_tokens"))
        output_tokens = _to_int(usage.get("completion_tokens"))
        total_tokens = _to_int(usage.get("total_tokens"))
        if total_tokens <= 0:
            total_tokens = input_tokens + output_tokens
        return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}

    def _log_prompt(self, *, stage: str, prompt: str) -> None:
        if not self.debug_log_prompts:
            return
        logger.warning("[CHAT][PROMPT][%s] %s", stage, self._truncate(prompt))

    def _log_raw_response(self, *, stage: str, text: str) -> None:
        if not self.debug_log_prompts:
            return
        logger.warning("[CHAT][RESPONSE][%s] %s", stage, self._truncate(text))

    def _truncate(self, text: str) -> str:
        if len(text) <= self.debug_log_max_chars:
            return text
        return text[: self.debug_log_max_chars] + "...[truncated]"


def _accumulate_usage(acc: dict[str, int], new: dict[str, int]) -> None:
    """Sum token counts across multiple LLM calls."""
    acc["input_tokens"] += new.get("input_tokens", 0)
    acc["output_tokens"] += new.get("output_tokens", 0)
    acc["total_tokens"] += new.get("total_tokens", 0)
