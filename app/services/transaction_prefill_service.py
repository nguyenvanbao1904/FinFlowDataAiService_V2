from __future__ import annotations

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any

from app.core.config import settings
from app.domain.entities.transaction_prefill import (
    TransactionPrefillRequest,
    TransactionPrefillResponse,
)
from app.services.chat.llm_client import LLMClient


class TransactionPrefillService:
    def __init__(self) -> None:
        self.llm_client = LLMClient()

    async def prefill(self, request: TransactionPrefillRequest) -> TransactionPrefillResponse:
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(request)
        
        # Use LLMClient's structured output with Pydantic validation + retry loop
        raw_response, usage = await self.llm_client.call_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=TransactionPrefillResponse,
            max_output_tokens=300,
            stage="prefill",
            max_retries=2,
        )

        return self._normalize_output(raw_response, request)

    def _build_system_prompt(self) -> str:
        return (
            "You are a helpful financial assistant that extracts transaction details from text.\n"
            "You must output exactly one JSON object and nothing else. "
            "Do not output markdown, code format tags, or reasoning."
        )

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
            }
            for a in request.accounts
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
            "RAW_TEXT=\n"
            f"{raw_text_block}\n"
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
        self, output: TransactionPrefillResponse, request: TransactionPrefillRequest
    ) -> TransactionPrefillResponse:
        allowed_categories = {c.id for c in request.categories}
        allowed_accounts = {a.id for a in request.accounts}

        warnings = output.warnings or []
        missing = output.missingFields or []
        
        if output.amount is not None:
            if output.amount <= 0:
                output.amount = None
        
        if output.categoryId and output.categoryId not in allowed_categories:
            warnings.append("Hệ thống: categoryId không nằm trong danh sách cho phép")
            output.categoryId = None
            
        if output.accountId and output.accountId not in allowed_accounts:
            warnings.append("Hệ thống: accountId không nằm trong danh sách cho phép")
            output.accountId = None

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

        # Re-evaluate missing fields
        missing.clear()
        if output.amount is None:
            missing.append("amount")
        if output.type is None:
            missing.append("type")
        if output.categoryId is None:
            missing.append("categoryId")
        if output.accountId is None:
            missing.append("accountId")
        if output.transactionDate is None:
            missing.append("transactionDate")
            
        output.missingFields = missing
        output.warnings = warnings[:10]

        return output
