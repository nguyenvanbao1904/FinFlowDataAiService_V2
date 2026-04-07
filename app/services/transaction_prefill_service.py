from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from app.core.config import settings
from app.domain.entities.transaction_prefill import (
    TransactionPrefillRequest,
    TransactionPrefillResponse,
)


class TransactionPrefillService:
    def __init__(self) -> None:
        self.base_url = settings.LOCAL_LLM_BASE_URL.strip()
        self.api_key = settings.LOCAL_LLM_API_KEY.strip() or "no-key-required"
        self.model = settings.LOCAL_LLM_MODEL.strip()
        self.thinking_mode = self._normalize_thinking_mode(settings.LOCAL_LLM_THINKING_MODE)
        self.timeout = max(5, int(settings.GEMINI_TIMEOUT_SECONDS))

    async def prefill(self, request: TransactionPrefillRequest) -> TransactionPrefillResponse:
        if not self.base_url or not self.model:
            return TransactionPrefillResponse(
                warnings=["LOCAL_LLM_BASE_URL hoặc LOCAL_LLM_MODEL chưa được cấu hình"],
                missingFields=["amount", "type", "categoryId", "accountId", "transactionDate"],
            )

        try:
            model_output = await self._call_local_model(request)
            return self._normalize_output(model_output, request)
        except APIStatusError as exc:
            # Lỗi HTTP từ local model endpoint.
            print(f"[LocalLLM][APIStatusError] code={exc.status_code} message={exc}")
            return TransactionPrefillResponse(
                warnings=[f"Local LLM API error code={exc.status_code}: {exc}"],
                missingFields=["amount", "type", "categoryId", "accountId", "transactionDate"],
            )
        except (APITimeoutError, APIConnectionError) as exc:
            print(f"[LocalLLM][ConnectionError] {type(exc).__name__}: {exc}")
            return TransactionPrefillResponse(
                warnings=[f"Local LLM connection error: {type(exc).__name__}: {exc}"],
                missingFields=["amount", "type", "categoryId", "accountId", "transactionDate"],
            )
        except Exception as exc:
            # Bug khác (network, parse, v.v.) – log thô, vẫn trả JSON an toàn.
            print(f"[LocalLLM][UnexpectedError] {type(exc).__name__}: {exc}")
            return TransactionPrefillResponse(
                warnings=[f"Local LLM unexpected error: {type(exc).__name__}: {exc}"],
                missingFields=["amount", "type", "categoryId", "accountId", "transactionDate"],
            )

    async def _call_local_model(self, request: TransactionPrefillRequest) -> dict[str, Any]:
        prompt = self._build_prompt(request)
        retry_prompt = self._build_retry_prompt(request)
        print(f"[LocalLLM] Model: {self.model}")
        print(f"[LocalLLM] Thinking mode: {self.thinking_mode}")
        print("[LocalLLM] Prompt:", prompt)

        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        try:
            attempts = self._build_attempts(prompt=prompt, retry_prompt=retry_prompt)
            for idx, (current_prompt, enable_thinking, system_prompt) in enumerate(attempts, start=1):
                response = await client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,
                    top_p=0.2,
                    max_tokens=300,
                    extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": current_prompt,
                        },
                    ],
                )

                text = ""
                if response.choices and response.choices[0].message:
                    text = response.choices[0].message.content or ""

                preview_limit = 700
                is_preview_truncated = len(text) > preview_limit
                preview = text[:preview_limit] + (" ...[preview-truncated]" if is_preview_truncated else "")
                print(
                    f"[LocalLLM] Raw text (attempt {idx}, thinking={enable_thinking}, len={len(text)}, preview_truncated={is_preview_truncated}):",
                    preview,
                )
                parsed = self._parse_first_json_object(text)
                if parsed:
                    return parsed

            return {}
        finally:
            await client.close()

    def _build_attempts(
        self,
        *,
        prompt: str,
        retry_prompt: str,
    ) -> list[tuple[str, bool, str]]:
        if self.thinking_mode == "on":
            return [(prompt, True, self._system_prompt_with_thinking())]
        if self.thinking_mode == "off":
            return [(prompt, False, self._system_prompt_json_only())]
        # auto: ưu tiên think để tăng chất lượng, fallback no-think nếu chưa parse được JSON.
        return [
            (prompt, True, self._system_prompt_with_thinking()),
            (retry_prompt, False, self._system_prompt_json_only()),
        ]

    def _normalize_thinking_mode(self, mode: str) -> str:
        normalized = (mode or "").strip().lower()
        if normalized in {"on", "off", "auto"}:
            return normalized
        return "auto"

    def _system_prompt_with_thinking(self) -> str:
        return (
            "<|think|>\n"
            "You extract one transaction into JSON. "
            "You may reason in thought channel, but keep thought concise. "
            "Final visible output must be exactly one JSON object and nothing else. "
            "Use exact keys: amount,type,categoryId,accountId,note,transactionDate,confidence,warnings."
        )

    def _system_prompt_json_only(self) -> str:
        return (
            "You extract one transaction into JSON. "
            "Do not output reasoning, channel tags, markdown, or explanations. "
            "Output must be exactly one JSON object and nothing else. "
            "Use exact keys: amount,type,categoryId,accountId,note,transactionDate,confidence,warnings."
        )

    def _parse_first_json_object(self, text: str) -> dict[str, Any]:
        if not text:
            return {}

        # Try parsing text with thought-channel sections stripped first.
        cleaned = self._strip_thought_channel_text(text)
        for candidate_text in (cleaned, text):
            try:
                parsed = json.loads(candidate_text)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        # Fallback: scan all JSON objects and pick the best match by expected schema.
        candidates: list[tuple[int, int, dict[str, Any]]] = []
        decoder = json.JSONDecoder()

        for source_priority, candidate_text in enumerate((cleaned, text)):
            if not candidate_text:
                continue
            for idx, ch in enumerate(candidate_text):
                if ch != "{":
                    continue
                try:
                    parsed, _end = decoder.raw_decode(candidate_text[idx:])
                    if isinstance(parsed, dict):
                        score = self._score_json_candidate(parsed)
                        # Prefer candidates from cleaned text in tie-breakers.
                        if source_priority == 0:
                            score += 100
                        candidates.append((score, idx, parsed))
                except json.JSONDecodeError:
                    continue

        if not candidates:
            return {}

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[0][2]

    def _strip_thought_channel_text(self, text: str) -> str:
        if not text:
            return ""

        cleaned = text.strip()
        # Remove content inside thought channel blocks in common gemma channel formats.
        cleaned = re.sub(
            r"(?is)<\|channel\|>thought.*?(?:<\|channel\|>|<channel\|>)",
            " ",
            cleaned,
        )
        cleaned = re.sub(
            r"(?is)<\|channel>thought.*?(?:<\|channel\|>|<channel\|>)",
            " ",
            cleaned,
        )
        # If a final channel marker exists, keep only content after that marker.
        cleaned = re.sub(r"(?is)^.*?<\|channel\|>final", "", cleaned)
        cleaned = re.sub(r"(?is)^.*?<\|channel>final", "", cleaned)
        return cleaned.strip()

    def _score_json_candidate(self, candidate: dict[str, Any]) -> int:
        expected_keys = {
            "amount",
            "type",
            "categoryId",
            "accountId",
            "note",
            "transactionDate",
            "confidence",
            "warnings",
        }

        present = sum(1 for key in expected_keys if key in candidate)
        extras = len(set(candidate.keys()) - expected_keys)
        score = (present * 10) - extras

        tx_type = str(candidate.get("type") or "").upper()
        if tx_type in {"INCOME", "EXPENSE"}:
            score += 2
        if isinstance(candidate.get("warnings"), list):
            score += 1
        if candidate.get("categoryId"):
            score += 1
        if candidate.get("accountId"):
            score += 1

        return score

    def _build_prompt(self, request: TransactionPrefillRequest) -> str:
        categories = [
            {"id": c.id, "name": c.name, "type": c.type}
            for c in request.categories
        ]
        accounts = [
            {
                "id": a.id,
                "name": a.name,
                "transactionEligible": a.transactionEligible,
            }
            for a in request.accounts
        ]
        tz = self._safe_zoneinfo(request.timezone)
        now_local = datetime.now(tz)
        current_date = now_local.date().isoformat()

        raw_text_block = self._sanitize_text(request.rawText)
        return self._build_base_model_prompt(
            request=request,
            current_date=current_date,
            categories=categories,
            accounts=accounts,
            raw_text=raw_text_block,
        )

    def _build_base_model_prompt(
        self,
        *,
        request: TransactionPrefillRequest,
        current_date: str,
        categories: list[dict[str, Any]],
        accounts: list[dict[str, Any]],
        raw_text: str,
    ) -> str:
        return (
            "TASK: Extract one finance transaction from RAW_TEXT.\n"
            "OUTPUT: Return exactly one JSON object with keys "
            "amount,type,categoryId,accountId,note,transactionDate,confidence,warnings.\n"
            "CONSTRAINTS: No markdown, no code fence, no extra keys. "
            "type in {INCOME,EXPENSE}. transactionDate is ISO8601. "
            "categoryId/accountId must come from provided lists.\n"
            f"CURRENT_DATE={current_date}; timezone={request.timezone}; locale={request.locale}; source={request.source}\n"
            "CATEGORIES_JSON="
            + json.dumps(categories, ensure_ascii=False, separators=(",", ":"))
            + "\nACCOUNTS_JSON="
            + json.dumps(accounts, ensure_ascii=False, separators=(",", ":"))
            + "\nRAW_TEXT="
            + raw_text
            + "\nJSON:"
        )

    def _build_retry_prompt(self, request: TransactionPrefillRequest) -> str:
        categories = [
            {"id": c.id, "name": c.name, "type": c.type}
            for c in request.categories
        ]
        accounts = [
            {
                "id": a.id,
                "name": a.name,
                "transactionEligible": a.transactionEligible,
            }
            for a in request.accounts
        ]
        tz = self._safe_zoneinfo(request.timezone)
        current_date = datetime.now(tz).date().isoformat()
        raw_text = self._sanitize_text(request.rawText)

        return (
            "RETRY: previous output invalid. Return JSON only.\n"
            "Fill this template and keep keys unchanged:\n"
            '{"amount":null,"type":null,"categoryId":null,"accountId":null,'
            '"note":null,"transactionDate":null,"confidence":0.0,"warnings":[]}\n'
            "Rules: categoryId/accountId must be from lists below; type is INCOME or EXPENSE; "
            "transactionDate ISO8601; if no date in text use CURRENT_DATE at local midnight.\n"
            f"CURRENT_DATE={current_date}\n"
            "CATEGORIES_JSON="
            + json.dumps(categories, ensure_ascii=False, separators=(",", ":"))
            + "\nACCOUNTS_JSON="
            + json.dumps(accounts, ensure_ascii=False, separators=(",", ":"))
            + "\nRAW_TEXT="
            + raw_text
            + "\nJSON:"
        )

    def _sanitize_text(self, text: str) -> str:
        cleaned = text.replace("\x00", " ").replace("\r", " ").strip()
        if len(cleaned) > 4000:
            cleaned = cleaned[:4000]
        return cleaned

    def _safe_zoneinfo(self, timezone_name: str) -> ZoneInfo:
        try:
            return ZoneInfo(timezone_name)
        except Exception:
            return ZoneInfo("UTC")

    def _normalize_output(
        self, output: dict[str, Any], request: TransactionPrefillRequest
    ) -> TransactionPrefillResponse:
        warnings: list[str] = []
        allowed_categories = {c.id for c in request.categories}
        allowed_accounts = {a.id for a in request.accounts}

        amount = self._to_positive_float(output.get("amount"))
        tx_type = self._normalize_type(output.get("type"))
        category_id = self._to_str(output.get("categoryId"))
        account_id = self._to_str(output.get("accountId"))
        note = self._to_note(output.get("note"))
        tx_date = self._normalize_date(output.get("transactionDate"))
        confidence = self._to_confidence(output.get("confidence"))

        if category_id and category_id not in allowed_categories:
            warnings.append("categoryId không nằm trong danh sách cho phép")
            category_id = None
        if account_id and account_id not in allowed_accounts:
            warnings.append("accountId không nằm trong danh sách cho phép")
            account_id = None

        raw_warnings = output.get("warnings")
        if isinstance(raw_warnings, list):
            warnings.extend([str(w) for w in raw_warnings[:10]])

        missing: list[str] = []
        if amount is None:
            missing.append("amount")
        if tx_type is None:
            missing.append("type")
        if category_id is None:
            missing.append("categoryId")
        if account_id is None:
            missing.append("accountId")
        if tx_date is None:
            missing.append("transactionDate")

        return TransactionPrefillResponse(
            amount=amount,
            type=tx_type,
            categoryId=category_id,
            accountId=account_id,
            note=note,
            transactionDate=tx_date,
            confidence=confidence,
            missingFields=missing,
            warnings=warnings,
        )

    def _to_positive_float(self, value: Any) -> float | None:
        try:
            val = float(value)
            return val if val > 0 else None
        except (TypeError, ValueError):
            return None

    def _normalize_type(self, value: Any) -> str | None:
        if value is None:
            return None
        t = str(value).strip().upper()
        return t if t in {"INCOME", "EXPENSE"} else None

    def _to_str(self, value: Any) -> str | None:
        if value is None:
            return None
        s = str(value).strip()
        return s if s else None

    def _to_note(self, value: Any) -> str | None:
        note = self._to_str(value)
        if not note:
            return None
        return note[:500]

    def _normalize_date(self, value: Any) -> str | None:
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None

        # Accept ISO-like values from model and normalize.
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            # Accept date-only and normalize to UTC midnight ISO.
            try:
                dt = datetime.fromisoformat(f"{s}T00:00:00+00:00")
            except ValueError:
                return None

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    def _to_confidence(self, value: Any) -> float:
        try:
            c = float(value)
        except (TypeError, ValueError):
            return 0.0
        if c < 0:
            return 0.0
        if c > 1:
            return 1.0
        return c
