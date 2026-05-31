from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModelSettings

from app.infrastructure.llm_agent import get_deepseek_model
from app.models.transaction import (
    TransactionPrefillRequest,
    TransactionPrefillResponse,
)
from app.services.chat.utils.json_io import parse_llm_json

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are a helpful financial assistant that extracts transaction details from text.\n"
    "You must output exactly one JSON object and nothing else. "
    "Do not output markdown, code format tags, or reasoning."
)


class TransactionPrefillService:
    def __init__(self) -> None:
        self._agent = Agent(
            get_deepseek_model(),
            system_prompt=_SYSTEM_PROMPT,
            model_settings=OpenAIChatModelSettings(
                temperature=0.0,
                max_tokens=300,
                extra_body={"response_format": {"type": "json_object"}},
            ),
            output_type=str,
            retries=2,
        )

    async def prefill(self, request: TransactionPrefillRequest) -> TransactionPrefillResponse:
        try:
            result = await self._agent.run(self._build_user_prompt(request))
        except Exception as exc:
            logger.error("[Prefill/Service] LLM failed: %s", exc)
            raise

        parsed = parse_llm_json(result.output)
        if not isinstance(parsed, dict):
            logger.warning("[Prefill/Service] output không phải dict (got %s) — dùng dict rỗng", type(parsed).__name__)
            parsed = {}

        response = TransactionPrefillResponse.model_validate(parsed)
        normalized = self._normalize_output(response, request)
        return normalized

    def _build_user_prompt(self, request: TransactionPrefillRequest) -> str:
        categories = [
            {"id": c.id, "name": c.name, "type": c.type}
            for c in request.categories
        ]
        accounts = [
            {
                "id": a.id,
                "name": a.name,
                "transactionEligible": a.transactionEligible,
                "typeCode": a.typeCode,
                "typeDisplayName": a.typeDisplayName,
                "balance": a.balance,
                "debt": a.debt,
            }
            for a in request.accounts
        ]
        recent_history = [
            {
                "amount": item.get("amount"),
                "type": item.get("type"),
                "categoryId": item.get("categoryId"),
                "accountId": item.get("accountId"),
                "note": self._sanitize_text(str(item.get("note") or ""))[:160],
                "transactionDate": str(item.get("transactionDate") or ""),
            }
            for item in request.recentHistory[:20]
            if isinstance(item, dict)
        ]
        tz = self._safe_zoneinfo(request.timezone)
        current_date_local = datetime.now(tz).isoformat(timespec="seconds")

        raw_text_block = self._sanitize_text(request.rawText)

        return (
            "TASK: Extract one finance transaction from RAW_TEXT.\n"
            "OUTPUT: Return JSON with keys: amount (float), type ('INCOME' or 'EXPENSE'), categoryId, accountId, note, transactionDate, confidence (0.0-1.0), warnings (list).\n"
            "CONSTRAINTS:\n"
            "- amount: MUST be the full integer value in VND. Apply multipliers: 'k', 'nghìn' = x 1,000; 'lít', 'xị' = x 100,000; 'củ', 'triệu' = x 1,000,000 (e.g., '30k' -> 30000, '2 xị' -> 200000).\n"
            "- transactionDate must be an ISO8601 string.\n"
            "- categoryId and accountId MUST exactly match the id from the provided lists below. Match by name or context.\n"
            "- If RAW_TEXT does not explicitly mention an account, infer accountId from RECENT_HISTORY_JSON by preferring the most recent transaction with the same categoryId, then the same type. If there is still no match, choose the first transaction-eligible account.\n"
            "- If a required field cannot be reasonably deduced, leave it null and list the field name in missingFields.\n"
            "- **note**: Must be a concise, clean description of the transaction, extracted from RAW_TEXT BUT:\n"
            "    * Remove any amount, numeric value, currency unit (e.g., '2 xị', '200k', '30 nghìn', '10 triệu').\n"
            "    * Remove redundant verbs like 'hết', 'mất', 'tốn', 'chi', 'tiêu' if they are just indicating expense.\n"
            "    * Keep the essential action and object (e.g., 'đổ xăng', 'mua cà phê', 'rút tiền', 'nhận lương').\n"
            "    * Do NOT copy RAW_TEXT literally. Derive a meaningful short note.\n"
            "    * If the cleaned note would be empty, fallback to a generic term like 'giao dịch' or keep the first meaningful word.\n"
            f"CURRENT_TIME={current_date_local}; timezone={request.timezone}; locale={request.locale}\n\n"
            "CATEGORIES_JSON=\n"
            f"{json.dumps(categories, ensure_ascii=False, separators=(',', ':'))}\n\n"
            "ACCOUNTS_JSON=\n"
            f"{json.dumps(accounts, ensure_ascii=False, separators=(',', ':'))}\n\n"
            "RECENT_HISTORY_JSON=\n"
            f"{json.dumps(recent_history, ensure_ascii=False, separators=(',', ':'))}\n\n"
            "RAW_TEXT=\n"
            f"{raw_text_block}\n"
        )

    @staticmethod
    def _sanitize_text(text: str) -> str:
        cleaned = text.replace("\x00", " ").replace("\r", " ").strip()
        if len(cleaned) > 4000:
            cleaned = cleaned[:4000]
        return cleaned

    @staticmethod
    def _safe_zoneinfo(timezone_name: str) -> ZoneInfo:
        try:
            return ZoneInfo(timezone_name)
        except Exception:
            logger.warning("Invalid timezone '%s', falling back to UTC", timezone_name)
            return ZoneInfo("UTC")

    def _normalize_output(
        self, output: TransactionPrefillResponse, request: TransactionPrefillRequest
    ) -> TransactionPrefillResponse:
        allowed_categories = {c.id for c in request.categories}
        allowed_accounts = {a.id for a in request.accounts}

        warnings = output.warnings or []
        missing = output.missingFields or []

        if output.amount is not None:
            if output.amount <= 0:
                logger.warning("[Prefill/Normalize] amount=%s <= 0 — xóa", output.amount)
                output.amount = None
        else:
            logger.warning("[Prefill/Normalize] LLM trả amount=None")

        if output.categoryId and output.categoryId not in allowed_categories:
            logger.warning("[Prefill/Normalize] categoryId '%s' không hợp lệ (có %d items) — xóa",
                          output.categoryId, len(allowed_categories))
            warnings.append("Hệ thống: categoryId không nằm trong danh sách cho phép")
            output.categoryId = None

        if output.accountId and output.accountId not in allowed_accounts:
            logger.warning("[Prefill/Normalize] accountId '%s' không hợp lệ (có %d items) — xóa",
                          output.accountId, len(allowed_accounts))
            warnings.append("Hệ thống: accountId không nằm trong danh sách cho phép")
            output.accountId = None

        if output.accountId is None:
            output.accountId = self._fallback_account_id(output, request)
            if output.accountId is not None and "accountId" in missing:
                missing.remove("accountId")

        if output.note:
            output.note = output.note[:500]

        if output.transactionDate:
            try:
                dt = datetime.fromisoformat(output.transactionDate.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                output.transactionDate = dt.isoformat()
            except ValueError:
                output.transactionDate = None

        output.confidence = max(0.0, min(1.0, float(output.confidence or 0.0)))

        required_checks = {
            "amount": output.amount,
            "type": output.type,
            "categoryId": output.categoryId,
            "accountId": output.accountId,
            "transactionDate": output.transactionDate,
        }
        for field_name, value in required_checks.items():
            if value is None and field_name not in missing:
                missing.append(field_name)

        output.missingFields = missing
        output.warnings = warnings[:10]

        return output

    @staticmethod
    def _fallback_account_id(
        output: TransactionPrefillResponse, request: TransactionPrefillRequest
    ) -> str | None:
        eligible_accounts = {
            account.id
            for account in request.accounts
            if account.transactionEligible is not False
        }
        if not eligible_accounts:
            return None

        if output.categoryId:
            for item in request.recentHistory:
                if not isinstance(item, dict):
                    continue
                account_id = str(item.get("accountId") or "")
                if item.get("categoryId") == output.categoryId and account_id in eligible_accounts:
                    return account_id

        if output.type:
            for item in request.recentHistory:
                if not isinstance(item, dict):
                    continue
                account_id = str(item.get("accountId") or "")
                if item.get("type") == output.type and account_id in eligible_accounts:
                    return account_id

        return next((account.id for account in request.accounts if account.id in eligible_accounts), None)
