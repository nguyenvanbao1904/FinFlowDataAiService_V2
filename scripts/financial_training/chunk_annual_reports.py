from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import fitz
import requests

try:
    import kreuzberg
except Exception:  # pragma: no cover
    kreuzberg = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from app.services.vision_ocr import (
        AppleVisionOcr,
        build_vision_ocr,
        can_use_vision_ocr,
        format_unsupported_languages,
        is_macos_arm64,
    )
except Exception:  # pragma: no cover
    AppleVisionOcr = Any  # type: ignore[misc,assignment]
    build_vision_ocr = None  # type: ignore[assignment]

    def can_use_vision_ocr() -> bool:  # type: ignore[no-redef]
        return False

    def is_macos_arm64() -> bool:  # type: ignore[no-redef]
        return False

    def format_unsupported_languages(tokens: Iterable[str]) -> str:  # type: ignore[no-redef]
        return ", ".join(tokens)


ROOT = PROJECT_ROOT
DEFAULT_INPUT_DIR = ROOT / "artifacts" / "rag" / "annual_reports" / "raw_pdfs"
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "rag" / "annual_reports" / "chunks" / "annual_reports_chunks.json"


@dataclass
class ChunkDraft:
    heading: str
    text: str
    page_start: int
    page_end: int


class AnnualReportChunker:
    def __init__(
        self,
        min_chars: int = 300,
        max_chars: int = 2400,
        overlap_chars: int = 250,
        enable_ocr: bool = False,
        ocr_language: str = "vie+eng",
        ocr_dpi: int = 200,
        ocr_force_all_pages: bool = False,
        ocr_fix_garbled: bool = False,
        ocr_garbled_page_ratio_threshold: float = 0.30,
        ocr_backend: str = "auto",
        drop_garbled_chunks: bool = True,
        llm_repair_garbled_chunks: bool = False,
        llm_repair_base_url: str = "http://127.0.0.1:9090/v1",
        llm_repair_api_key: str = "no-key-required",
        llm_repair_model: str = "mlx-community/gemma-4-e2b-it-4bit",
        llm_repair_enable_thinking: bool = False,
        llm_repair_timeout_seconds: int = 45,
        llm_repair_max_chunks_per_file: int = 8,
        keep_focus_only: bool = True,
        parser_backend: str = "kreuzberg",
    ) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.enable_ocr = enable_ocr
        self.ocr_language = ocr_language
        self.ocr_dpi = ocr_dpi
        self.ocr_force_all_pages = ocr_force_all_pages
        self.ocr_fix_garbled = ocr_fix_garbled
        self.ocr_garbled_page_ratio_threshold = min(1.0, max(0.0, float(ocr_garbled_page_ratio_threshold)))
        self.ocr_backend = self._resolve_ocr_backend(ocr_backend)
        self.drop_garbled_chunks = drop_garbled_chunks
        self.llm_repair_garbled_chunks = llm_repair_garbled_chunks
        self.llm_repair_base_url = str(llm_repair_base_url).strip().rstrip("/")
        self.llm_repair_api_key = str(llm_repair_api_key).strip() or "no-key-required"
        self.llm_repair_model = str(llm_repair_model).strip()
        self.llm_repair_enable_thinking = bool(llm_repair_enable_thinking)
        self.llm_repair_timeout_seconds = max(5, int(llm_repair_timeout_seconds))
        self.llm_repair_max_chunks_per_file = max(0, int(llm_repair_max_chunks_per_file))
        self._llm_repair_disabled = False
        self._llm_repair_warned = False
        self.keep_focus_only = keep_focus_only
        self.parser_backend = self._resolve_parser_backend(parser_backend)
        self._vision_ocr: AppleVisionOcr | None = None
        self._vision_ocr_init_attempted = False
        self._vision_ocr_error: str | None = None
        self._warned_ocr_backend_none = False
        self.focus_categories = {
            "mdna",
            "strategy",
            "risk",
            "governance",
            "sustainability",
            "business_overview",
        }
        self.category_priority_score = {
            "mdna": 5,
            "strategy": 5,
            "risk": 4,
            "governance": 3,
            "sustainability": 3,
            "business_overview": 2,
            "other": 1,
        }
        self.category_priority_label = {
            "mdna": "highest",
            "strategy": "highest",
            "risk": "high",
            "governance": "medium",
            "sustainability": "medium",
            "business_overview": "lower",
            "other": "lowest",
        }

        # Taxonomy for value investing oriented retrieval.
        self.category_map = {
            "mdna": [
                "bao cao ban giam doc",
                "bao cao cua ban dieu hanh",
                "ban dieu hanh",
                "danh gia ket qua",
                "bao cao va danh gia cua ban dieu hanh",
                "thu cua chu tich",
                "thong diep cua chu tich",
                "tong giam doc",
                "giam doc dieu hanh",
                "md&a",
                "mda",
            ],
            "strategy": [
                "ke hoach",
                "dinh huong",
                "chien luoc",
                "tam nhin",
                "su menh",
                "muc tieu",
                "ke hoach sxkd",
                "ke hoach tuong lai",
                "dinh huong phat trien",
            ],
            "risk": ["rui ro", "quan ly rui ro", "thach thuc", "khung quan ly rui ro"],
            "governance": [
                "quan tri cong ty",
                "hoi dong quan tri",
                "bao cao quan tri",
                "hdqt",
                "ban kiem soat",
                "bks",
            ],
            "sustainability": [
                "phat trien ben vung",
                "esg",
                "csr",
                "trach nhiem xa hoi",
                "moi truong",
            ],
            "business_overview": [
                "thong tin chung",
                "ve chung toi",
                "tong quan",
                "thong tin co ban",
                "gioi thieu",
                "lich su",
                "nganh nghe",
            ],
        }

        if self.enable_ocr:
            print(
                f"[INFO] OCR enabled: backend={self.ocr_backend} language={self.ocr_language} dpi={self.ocr_dpi}"
            )
        else:
            print("[INFO] OCR disabled")

        self._reset_run_stats()

    def _resolve_parser_backend(self, parser_backend: str) -> str:
        backend = (parser_backend or "").strip().lower()
        if backend not in {"kreuzberg", "pymupdf"}:
            backend = "kreuzberg"
        if backend == "kreuzberg" and kreuzberg is None:
            print("[WARN] Kreuzberg is not installed. Fallback to PyMuPDF backend.")
            return "pymupdf"
        return backend

    @staticmethod
    def _resolve_ocr_backend(ocr_backend: str) -> str:
        backend = (ocr_backend or "auto").strip().lower()
        if backend not in {"auto", "vision", "none"}:
            backend = "auto"
        if backend == "auto":
            return "vision" if is_macos_arm64() else "none"
        return backend

    def _ensure_vision_ocr_ready(self) -> bool:
        if self.ocr_backend != "vision":
            return False
        if self._vision_ocr is not None:
            return True
        if self._vision_ocr_init_attempted:
            return False

        self._vision_ocr_init_attempted = True
        if not can_use_vision_ocr() or build_vision_ocr is None:
            self._vision_ocr_error = "Vision OCR runtime unavailable"
            print(
                "[WARN] Vision OCR backend requested but unavailable. "
                "Install pyobjc Vision frameworks and run on macOS ARM64."
            )
            return False

        try:
            ocr, unsupported_tokens = build_vision_ocr(self.ocr_language)
            self._vision_ocr = ocr
            if unsupported_tokens:
                unsupported = format_unsupported_languages(unsupported_tokens)
                print(
                    f"[WARN] Unsupported OCR language token(s) for Vision ignored: {unsupported}. "
                    f"Using languages={getattr(self._vision_ocr, 'languages', [])}"
                )
            return True
        except Exception as exc:
            self._vision_ocr_error = str(exc)
            print(f"[WARN] Cannot initialize Vision OCR backend: {exc}")
            return False

    def _reset_run_stats(self) -> None:
        self._run_pages_total = 0
        self._run_pages_with_text = 0
        self._run_pages_with_ocr = 0
        self._run_files_processed = 0
        self._run_files_with_ocr = 0
        self._run_chunks_dropped_garbled = 0
        self._run_llm_repair_attempted = 0
        self._run_llm_repair_succeeded = 0

    def get_run_summary(self) -> dict[str, float | int]:
        total_pages = self._run_pages_total
        text_pages = self._run_pages_with_text
        ocr_pages = self._run_pages_with_ocr

        ocr_rate_on_total = (ocr_pages / total_pages * 100.0) if total_pages else 0.0
        ocr_rate_on_text = (ocr_pages / text_pages * 100.0) if text_pages else 0.0

        return {
            "files_processed": self._run_files_processed,
            "files_with_ocr": self._run_files_with_ocr,
            "chunks_dropped_garbled": self._run_chunks_dropped_garbled,
            "pages_total": total_pages,
            "pages_with_text": text_pages,
            "pages_with_ocr": ocr_pages,
            "llm_repair_attempted": self._run_llm_repair_attempted,
            "llm_repair_succeeded": self._run_llm_repair_succeeded,
            "ocr_rate_on_total_pages_pct": ocr_rate_on_total,
            "ocr_rate_on_text_pages_pct": ocr_rate_on_text,
        }

    def _clean_llm_text_response(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text.strip()
        cleaned = re.sub(r"(?is)<\|channel\|>thought.*?(?=<\|channel\|>final|$)", "", cleaned)
        cleaned = re.sub(r"(?is)^.*?<\|channel\|>final", "", cleaned)

        # Remove simple markdown fences if model wraps output.
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)

        return cleaned.strip()

    def _repair_text_with_local_llm(self, text: str) -> str | None:
        if not self.llm_repair_garbled_chunks or self._llm_repair_disabled:
            return None
        if not text or not text.strip():
            return None
        if not self.llm_repair_base_url or not self.llm_repair_model:
            return None

        base_url = self.llm_repair_base_url.rstrip("/")
        endpoints = [f"{base_url}/chat/completions"]
        if not base_url.endswith("/v1"):
            endpoints.append(f"{base_url}/v1/chat/completions")
        endpoints.append(f"{base_url}/completions")

        prompt = (
            "Sửa lỗi OCR tiếng Việt trong đoạn văn dưới đây. "
            "Giữ nguyên số liệu, đơn vị, mã chứng khoán, tên riêng nếu không chắc. "
            "Không thêm thông tin mới, không giải thích. "
            "Trả về đúng một đoạn văn bản đã sửa, không markdown.\n\n"
            f"OCR_TEXT:\n{text.strip()}"
        )

        chat_payload: dict[str, object] = {
            "model": self.llm_repair_model,
            "temperature": 0.0,
            "top_p": 0.2,
            "max_tokens": min(1400, max(300, len(text) // 2)),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Bạn là bộ sửa lỗi OCR cho báo cáo thường niên tiếng Việt. "
                        "Không trả lời lan man. Chỉ trả về văn bản đã sửa."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        }
        if self.llm_repair_enable_thinking:
            chat_payload["chat_template_kwargs"] = {"enable_thinking": True}

        completion_payload: dict[str, object] = {
            "model": self.llm_repair_model,
            "temperature": 0.0,
            "top_p": 0.2,
            "max_tokens": min(1400, max(300, len(text) // 2)),
            "prompt": (
                "Bạn là bộ sửa lỗi OCR cho báo cáo thường niên tiếng Việt. "
                "Không trả lời lan man. Chỉ trả về văn bản đã sửa.\n\n"
                + prompt
            ),
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm_repair_api_key}",
        }

        errors: list[str] = []
        content = ""
        for endpoint in list(dict.fromkeys(endpoints)):
            is_chat_endpoint = endpoint.endswith("/chat/completions")
            payload = chat_payload if is_chat_endpoint else completion_payload
            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.llm_repair_timeout_seconds,
                )
                if response.status_code == 404:
                    errors.append(f"{endpoint}:404")
                    continue

                response.raise_for_status()
                data = response.json()
                if is_chat_endpoint:
                    choice = (data.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    content = str(message.get("content") or choice.get("text") or "")
                else:
                    choice = (data.get("choices") or [{}])[0]
                    content = str(choice.get("text") or "")

                if content.strip():
                    break
                errors.append(f"{endpoint}:empty")
            except Exception as exc:
                errors.append(f"{endpoint}:{exc}")
                continue

        if not content.strip():
            if not self._llm_repair_warned:
                print(
                    "[WARN] LLM garbled-text repair unavailable: "
                    + "; ".join(errors[:3])
                )
                self._llm_repair_warned = True
            self._llm_repair_disabled = True
            return None

        repaired = self._clean_llm_text_response(str(content or ""))
        if not repaired:
            return None
        return repaired

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text:
            return ""
        normalized = unicodedata.normalize("NFKD", text)
        without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        compact = re.sub(r"\s+", " ", without_accents.strip().lower())
        return compact

    def _infer_stock_and_year(self, file_path: Path) -> tuple[str, int]:
        stem = file_path.stem.strip()
        name_normalized = self._normalize_text(stem)

        stock = "UNKNOWN"
        tokens = [token for token in re.split(r"[_\-\s]+", stem) if token]
        if tokens:
            prefix = tokens[0].upper()
            if (
                2 <= len(prefix) <= 6
                and re.fullmatch(r"[A-Z0-9]+", prefix)
                and re.search(r"[A-Z]", prefix)
            ):
                stock = prefix

        if stock == "UNKNOWN":
            if "hpg" in name_normalized or "hoa phat" in name_normalized:
                stock = "HPG"
            elif "acb" in name_normalized or "a chau" in name_normalized or "asia commercial" in name_normalized:
                stock = "ACB"
            else:
                found = re.search(r"([A-Za-z][A-Za-z0-9]{1,5})", stem)
                if found:
                    stock = found.group(1).upper()

        year = 0
        year_match = re.search(r"\b(20\d{2})\b", stem)
        if year_match:
            year = int(year_match.group(1))
        else:
            two_digit_year = re.search(r"(?:^|[_\-])(\d{2})(?:CN|TC|TN|BCTN|BC|$)", stem, flags=re.IGNORECASE)
            if two_digit_year:
                yy = int(two_digit_year.group(1))
                year = 2000 + yy if yy <= 30 else 1900 + yy

        return stock, year

    def _detect_category(self, heading: str, text: str = "") -> str:
        heading_normalized = self._normalize_text(heading)
        preview_lines = " ".join(line.strip() for line in text.splitlines()[:3])
        preview_normalized = self._normalize_text(preview_lines)
        signal = f"{heading_normalized} {preview_normalized}".strip()

        # Fast rules for common annual-report chapter structures.
        if re.search(r"\b1\.7(\.\d+)?\b", signal):
            return "strategy"
        if re.search(r"\b1\.8(\.\d+)?\b", signal):
            return "risk"
        if re.search(r"\b1\.10(\.\d+)?\b", signal) and "rui ro" in signal:
            return "risk"
        if "chuong 3" in signal and (
            "ban dieu hanh" in signal or "danh gia" in signal
        ):
            return "mdna"
        if ("chuong 5" in signal or "phan 4" in signal) and "quan tri" in signal:
            return "governance"
        if ("chuong 6" in signal or "phan 6" in signal) and (
            "ben vung" in signal or "esg" in signal or "bao cao phat trien ben vung" in signal
        ):
            return "sustainability"
        if ("chuong 1" in signal or "phan 1" in signal) and (
            "ve chung toi" in signal
            or "tong quan" in signal
            or "thong tin chung" in signal
        ):
            return "business_overview"

        # Strong domain anchors for value-investing taxonomy.
        if any(term in signal for term in [
            "bao cao ban giam doc",
            "bao cao va danh gia cua ban dieu hanh",
            "bao cao cua ban dieu hanh",
            "danh gia ket qua hoat dong",
            "ban dieu hanh",
        ]):
            return "mdna"

        for category, keywords in self.category_map.items():
            if any(keyword in signal for keyword in keywords):
                return category
        return "other"

    def _extract_chapter_hint(self, heading: str) -> str | None:
        normalized = self._normalize_text(heading)
        if not normalized:
            return None

        chapter = re.search(r"\b(chuong|phan)\s+([ivx]+|\d{1,2})\b", normalized)
        if chapter:
            return f"{chapter.group(1)}_{chapter.group(2)}"

        section_no = re.search(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\b", normalized)
        if section_no:
            return section_no.group(1)

        return None

    def _is_noise_heading(self, heading: str) -> bool:
        h = (heading or "").strip()
        hn = self._normalize_text(h)
        if not hn:
            return True
        if h in {":", "-", "--"}:
            return True
        if re.fullmatch(r"\d{1,3}", hn):
            return True
        if re.fullmatch(r"\d{1,3}\s*/\s*\d{1,3}", hn):
            return True
        if re.fullmatch(r"\d+(\.\d+){1,3}", hn) and len(hn) <= 7:
            return True
        return False

    def _looks_like_toc_or_table(self, text: str) -> bool:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) < 6:
            return False

        short_lines = sum(1 for ln in lines if len(ln) <= 45)
        starts_number_like = sum(1 for ln in lines if re.match(r"^(\d+|[ivx]+|[-•])", self._normalize_text(ln)))
        dotted_leaders = sum(1 for ln in lines if re.search(r"\.{3,}", ln))

        if short_lines / len(lines) >= 0.70 and starts_number_like / len(lines) >= 0.45:
            return True
        if dotted_leaders >= 3:
            return True
        return False

    def _is_heading(self, line: str) -> bool:
        normalized = self._normalize_text(line)
        if not normalized:
            return False
        if len(normalized) > 160:
            return False

        heading_patterns = [
            r"^(chuong|phan|muc)\b",
            r"^\d{1,2}(\.\d{1,2}){0,3}\b",
            r"^(i|ii|iii|iv|v|vi|vii|viii|ix|x)\b",
        ]
        if any(re.match(pattern, normalized) for pattern in heading_patterns):
            return True

        # Very short all-uppercase lines are often section headers in annual reports.
        compact = re.sub(r"[^A-Za-z]", "", line)
        if 4 <= len(compact) <= 80 and compact.isupper():
            return True

        if line.strip().endswith(":") and len(line.strip()) <= 120:
            return True

        return False

    def _non_investing_disclosure(self, heading: str, text: str) -> bool:
        block = self._normalize_text(f"{heading}\n{text}")
        patterns = [
            r"thuyet minh bao cao tai chinh",
            r"bao cao tai chinh hop nhat",
            r"bang can doi ke toan",
            r"bao cao luu chuyen tien te",
            r"bao cao kiem toan doc lap",
        ]
        return any(re.search(p, block) for p in patterns)

    def _is_finance_like_mixed_token(self, token: str) -> bool:
        t = token.strip().strip(".,;:!?()[]{}\"'`")
        if not t:
            return False

        patterns = [
            r"^[Qq][1-4](?:[/\-]?\d{2,4})?$",  # Q1/2025
            r"^(?:FY|fy)\d{2,4}$",  # FY2025
            r"^[A-Z]{2,8}\d{2,4}$",  # ROE2025, NPL2025
            r"^\d+(?:[.,]\d+)?[xX]$",  # 12x
        ]
        return any(re.fullmatch(pattern, t) for pattern in patterns)

    def _is_garbled_text(self, text: str) -> bool:
        if not text:
            return False
        stripped = text.strip()
        if not stripped:
            return False

        tokens = re.findall(r"\S+", stripped)
        if not tokens:
            return False

        mixed_alpha_digit = 0
        suspicious_mixed_alpha_digit = 0
        for token in tokens:
            cleaned = token.strip().strip(".,;:!?()[]{}\"'`")
            if not cleaned:
                continue
            if not (re.search(r"[A-Za-z]", cleaned) and re.search(r"\d", cleaned)):
                continue

            mixed_alpha_digit += 1
            if self._is_finance_like_mixed_token(cleaned):
                continue

            groups = re.findall(r"[A-Za-z]+|\d+", cleaned)
            transitions = max(0, len(groups) - 1)
            alpha_count = sum(1 for ch in cleaned if ch.isalpha())
            digit_count = sum(1 for ch in cleaned if ch.isdigit())
            digit_ratio = digit_count / max(1, alpha_count + digit_count)

            if (
                transitions >= 2
                or re.search(r"[A-Za-z]\d[A-Za-z]", cleaned)
                or re.search(r"\d[A-Za-z]\d", cleaned)
                or (len(cleaned) >= 6 and digit_ratio >= 0.30)
            ):
                suspicious_mixed_alpha_digit += 1

        weird_symbol = sum(1 for token in tokens if re.search(r"[\^~`|{}<>]", token))
        replacement_char = stripped.count("\ufffd")

        mixed_ratio = mixed_alpha_digit / len(tokens)
        suspicious_mixed_ratio = suspicious_mixed_alpha_digit / len(tokens)
        weird_ratio = weird_symbol / len(tokens)
        replacement_ratio = replacement_char / max(1, len(stripped))

        # High-confidence corruption signals.
        if replacement_ratio >= 0.01:
            return True
        if replacement_char >= 3 and replacement_ratio >= 0.002:
            return True
        if suspicious_mixed_alpha_digit >= 8 and suspicious_mixed_ratio >= 0.12:
            return True

        # Length-aware thresholds to reduce false positives on finance-heavy text.
        token_count = len(tokens)
        if token_count >= 20:
            if suspicious_mixed_ratio >= 0.10:
                return True
            if suspicious_mixed_ratio >= 0.07 and weird_ratio >= 0.03:
                return True
            if mixed_ratio >= 0.40 and suspicious_mixed_alpha_digit >= 5:
                return True
            return False

        if token_count >= 8:
            if suspicious_mixed_alpha_digit >= 3:
                return True
            if suspicious_mixed_ratio >= 0.20 and weird_symbol >= 1:
                return True
            return False

        if suspicious_mixed_alpha_digit >= 2:
            return True
        if weird_symbol >= 2 and replacement_char >= 1:
            return True
        return False

    def _prefer_ocr_text(self, original_text: str, ocr_text: str) -> bool:
        if not ocr_text or not ocr_text.strip():
            return False
        if not original_text or not original_text.strip():
            return True

        original_is_garbled = self._is_garbled_text(original_text)
        ocr_is_garbled = self._is_garbled_text(ocr_text)

        if original_is_garbled and not ocr_is_garbled:
            return True
        if original_is_garbled and ocr_is_garbled and len(ocr_text.strip()) > len(original_text.strip()) * 1.2:
            return True
        if not original_is_garbled and not ocr_is_garbled and len(ocr_text.strip()) > len(original_text.strip()) * 1.5:
            return True
        return False

    def _build_overlap_prefix(self, previous_piece: str) -> str:
        if self.overlap_chars <= 0 or not previous_piece:
            return ""
        tail = previous_piece[-self.overlap_chars :]

        # Prefer sentence boundary, fallback to first whitespace.
        boundaries = list(re.finditer(r"[\.!?;:]\s+", tail))
        if boundaries:
            start = boundaries[-1].end()
            prefix = tail[start:].strip()
            if prefix:
                return prefix

        ws = re.search(r"\s+", tail)
        if ws:
            prefix = tail[ws.end() :].strip()
            if prefix:
                return prefix

        return tail.strip()

    def _apply_piece_overlap(self, pieces: list[str]) -> list[str]:
        if len(pieces) <= 1 or self.overlap_chars <= 0:
            return pieces

        merged: list[str] = [pieces[0]]
        for i in range(1, len(pieces)):
            prefix = self._build_overlap_prefix(pieces[i - 1])
            current = pieces[i]
            candidate = f"{prefix}\n{current}".strip() if prefix else current

            if len(candidate) > self.max_chars and prefix:
                overshoot = len(candidate) - self.max_chars
                if overshoot < len(prefix):
                    prefix = prefix[overshoot:].lstrip()
                    candidate = f"{prefix}\n{current}".strip()
                else:
                    candidate = current[-self.max_chars :]

            merged.append(candidate)

        return merged

    def _split_long_text(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        if not paragraphs:
            return []

        pieces: list[str] = []
        current = ""
        for para in paragraphs:
            candidate = f"{current}\n\n{para}".strip() if current else para
            if len(candidate) <= self.max_chars:
                current = candidate
                continue

            if current:
                pieces.append(current)

            if len(para) <= self.max_chars:
                current = para
                continue

            # Hard split extremely long paragraph to keep embedding chunks bounded.
            start = 0
            while start < len(para):
                stop = min(start + self.max_chars, len(para))
                pieces.append(para[start:stop].strip())
                if stop >= len(para):
                    break
                start = max(0, stop - self.overlap_chars)
            current = ""

        if current:
            pieces.append(current)

        pieces = [p for p in pieces if len(p) >= self.min_chars]
        return self._apply_piece_overlap(pieces)

    def _flush_draft(
        self,
        drafts: list[ChunkDraft],
        current_heading: str,
        current_lines: list[str],
        page_start: int,
        page_end: int,
    ) -> None:
        text = "\n".join(current_lines).strip()
        if len(text) < self.min_chars:
            return
        drafts.append(
            ChunkDraft(
                heading=current_heading.strip() or "Gioi thieu",
                text=text,
                page_start=page_start,
                page_end=page_end,
            )
        )

    def _extract_pages_pymupdf(self, pdf_path: Path) -> tuple[list[str], int]:
        doc = fitz.open(pdf_path)
        try:
            page_texts: list[str] = []

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                text = page.get_text("text")
                page_texts.append(str(text or ""))

            return page_texts, len(doc)
        finally:
            doc.close()

    def _extract_pages_kreuzberg(self, pdf_path: Path) -> tuple[list[str], int]:
        if kreuzberg is None:
            raise RuntimeError("Kreuzberg backend requested but package is unavailable")

        config_kwargs: dict[str, object] = {
            "pages": kreuzberg.PageConfig(extract_pages=True, insert_page_markers=False),
            "disable_ocr": True,
        }

        config = kreuzberg.ExtractionConfig(**config_kwargs)
        result = kreuzberg.extract_file_sync(pdf_path, config=config)

        page_texts: list[str] = []
        pages_data = result.pages if isinstance(result.pages, list) else []
        for page in pages_data:
            if isinstance(page, dict):
                text = str(page.get("content") or "")
            else:
                text = ""
            page_texts.append(text)

        if not page_texts:
            # Fallback to full content as a single pseudo-page if page extraction is unavailable.
            page_texts = [str(result.content or "")]

        return page_texts, len(page_texts)

    def _apply_vision_ocr(self, pdf_path: Path, page_texts: list[str]) -> tuple[list[str], int]:
        if not self.enable_ocr or self.ocr_backend != "vision":
            return page_texts, 0
        if not self._ensure_vision_ocr_ready() or self._vision_ocr is None:
            return page_texts, 0

        pages_with_ocr = 0
        doc = fitz.open(pdf_path)
        try:
            total_pages = len(doc)
            if len(page_texts) < total_pages:
                page_texts.extend([""] * (total_pages - len(page_texts)))

            force_repair_on_text_pages = False
            if self.ocr_fix_garbled:
                non_empty_pages = [text for text in page_texts[:total_pages] if str(text).strip()]
                if non_empty_pages:
                    garbled_pages = sum(1 for text in non_empty_pages if self._is_garbled_text(str(text)))
                    garbled_ratio = garbled_pages / len(non_empty_pages)
                    if garbled_ratio >= self.ocr_garbled_page_ratio_threshold:
                        force_repair_on_text_pages = True
                        print(
                            f"[WARN] Garbled text ratio={garbled_ratio:.2f} on {pdf_path.name}; "
                            "Vision OCR will be applied to all text pages."
                        )

            for page_idx in range(total_pages):
                original_text = page_texts[page_idx] if page_idx < len(page_texts) else ""
                needs_ocr = self.ocr_force_all_pages or (not str(original_text).strip())
                if (
                    not needs_ocr
                    and self.ocr_fix_garbled
                    and (force_repair_on_text_pages or self._is_garbled_text(str(original_text)))
                ):
                    needs_ocr = True

                if not needs_ocr:
                    continue

                try:
                    page = doc[page_idx]
                    ocr_text = self._vision_ocr.extract_text_from_fitz_page(
                        page,
                        dpi=self.ocr_dpi,
                    )
                except Exception as exc:
                    print(f"[WARN] Vision OCR failed on {pdf_path.name} page {page_idx + 1}: {exc}")
                    continue

                if self.ocr_force_all_pages:
                    if ocr_text and ocr_text.strip():
                        page_texts[page_idx] = ocr_text
                        pages_with_ocr += 1
                else:
                    if self._prefer_ocr_text(str(original_text), str(ocr_text)):
                        page_texts[page_idx] = str(ocr_text)
                        pages_with_ocr += 1
            return page_texts, pages_with_ocr
        finally:
            doc.close()

    def _extract_pages(self, pdf_path: Path) -> tuple[list[str], int, int, int, str]:
        backend_label = self.parser_backend
        if self.parser_backend == "kreuzberg":
            try:
                page_texts, total_pages = self._extract_pages_kreuzberg(pdf_path)
            except Exception as exc:
                print(f"[WARN] Kreuzberg extraction failed for {pdf_path.name}: {exc}. Fallback to PyMuPDF.")
                page_texts, total_pages = self._extract_pages_pymupdf(pdf_path)
                backend_label = "pymupdf-fallback"
        else:
            page_texts, total_pages = self._extract_pages_pymupdf(pdf_path)

        pages_with_ocr = 0
        if self.enable_ocr and self.ocr_backend == "vision":
            page_texts, pages_with_ocr = self._apply_vision_ocr(pdf_path, page_texts)
            if pages_with_ocr > 0:
                backend_label = f"{backend_label}+vision"
        elif self.enable_ocr and self.ocr_backend == "none":
            if not self._warned_ocr_backend_none:
                print("[WARN] OCR requested but backend is 'none'; OCR stage skipped.")
                self._warned_ocr_backend_none = True

        pages_with_text = sum(1 for text in page_texts if text and str(text).strip())
        return page_texts, total_pages, pages_with_text, pages_with_ocr, backend_label

    def _build_drafts_from_page_texts(self, page_texts: list[str], total_pages: int) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        current_heading = "Gioi thieu"
        current_lines: list[str] = []
        page_start = 1

        for page_idx, text in enumerate(page_texts):
            page_no = page_idx + 1
            if not text or not text.strip():
                continue

            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue

                if self._is_heading(line):
                    if current_lines:
                        self._flush_draft(
                            drafts=drafts,
                            current_heading=current_heading,
                            current_lines=current_lines,
                            page_start=page_start,
                            page_end=page_no,
                        )
                    current_heading = line
                    current_lines = [line]
                    page_start = page_no
                else:
                    current_lines.append(line)

        if current_lines:
            self._flush_draft(
                drafts=drafts,
                current_heading=current_heading,
                current_lines=current_lines,
                page_start=page_start,
                page_end=total_pages,
            )

        return drafts

    def process_pdf(self, pdf_path: Path) -> list[dict]:
        if not pdf_path.exists():
            print(f"[WARN] Skip unreadable PDF {pdf_path.name}: file does not exist")
            return []

        try:
            page_texts, total_pages, pages_with_text, pages_with_ocr, backend_used = self._extract_pages(pdf_path)
        except Exception as exc:
            print(f"[WARN] Skip unreadable PDF {pdf_path.name}: {exc}")
            return []

        print(f"[INFO] Extract backend for {pdf_path.name}: {backend_used}")

        self._run_files_processed += 1
        self._run_pages_total += total_pages
        stock_code, year = self._infer_stock_and_year(pdf_path)
        drafts = self._build_drafts_from_page_texts(page_texts, total_pages)

        if pages_with_text == 0:
            print(
                f"[WARN] No extractable text found in {pdf_path.name}. "
                "The PDF may be image-only and requires OCR before chunking."
            )
        elif pages_with_ocr > 0:
            print(f"[INFO] OCR extracted text for {pages_with_ocr}/{total_pages} pages in {pdf_path.name}")
        elif backend_used == "kreuzberg" and self.enable_ocr:
            print(
                f"[INFO] Kreuzberg OCR mode enabled for {pdf_path.name}; page-level OCR counts were not reported."
            )

        self._run_pages_with_text += pages_with_text
        self._run_pages_with_ocr += pages_with_ocr
        if pages_with_ocr > 0:
            self._run_files_with_ocr += 1

        chunks: list[dict] = []
        chunk_index = 0
        dropped_garbled = 0
        repaired_with_llm = 0
        llm_attempted_for_file = 0
        for draft in drafts:
            raw_text = draft.text.strip()
            if len(raw_text) < self.min_chars:
                continue

            split_texts = self._split_long_text(raw_text)
            if not split_texts:
                continue
            category = self._detect_category(draft.heading, raw_text)

            if self.keep_focus_only and category not in self.focus_categories:
                continue

            if self._is_noise_heading(draft.heading):
                continue

            if self._non_investing_disclosure(draft.heading, raw_text):
                continue

            for split_rank, piece in enumerate(split_texts, start=1):
                if (
                    self.llm_repair_garbled_chunks
                    and self._is_garbled_text(piece)
                    and llm_attempted_for_file < self.llm_repair_max_chunks_per_file
                ):
                    llm_attempted_for_file += 1
                    self._run_llm_repair_attempted += 1
                    repaired = self._repair_text_with_local_llm(piece)
                    if repaired and not self._is_garbled_text(repaired):
                        piece = repaired
                        repaired_with_llm += 1
                        self._run_llm_repair_succeeded += 1

                if self.drop_garbled_chunks and self._is_garbled_text(piece):
                    dropped_garbled += 1
                    continue
                if self._looks_like_toc_or_table(piece):
                    continue
                chunk_index += 1
                priority_score = int(self.category_priority_score.get(category, 1))
                chapter_hint = self._extract_chapter_hint(draft.heading)
                year_for_id = year if year > 0 else 0
                chunks.append(
                    {
                        "chunk_id": f"{stock_code}_{year_for_id}_{chunk_index:04d}",
                        "stock_code": stock_code,
                        "year": year,
                        "category": category,
                        "section": category,
                        "category_priority_score": priority_score,
                        "category_priority_label": self.category_priority_label.get(category, "lowest"),
                        "value_importance_stars": "*" * priority_score,
                        "taxonomy_version": "value_investing_v2",
                        "chapter_hint": chapter_hint,
                        "subsection_title": draft.heading,
                        "document_type": "annual_report",
                        "text": piece,
                        "page_start": draft.page_start,
                        "page_end": draft.page_end,
                        "child_rank": split_rank,
                        "source_file": pdf_path.name,
                    }
                )

        if dropped_garbled > 0:
            self._run_chunks_dropped_garbled += dropped_garbled
            print(
                f"[INFO] Dropped {dropped_garbled} garbled chunks in {pdf_path.name}"
            )

        if repaired_with_llm > 0:
            print(
                f"[INFO] LLM repaired {repaired_with_llm} chunks in {pdf_path.name}"
            )

        return chunks

    def process_many(self, pdf_paths: Iterable[Path]) -> list[dict]:
        self._reset_run_stats()
        all_chunks: list[dict] = []
        for pdf_path in pdf_paths:
            if not pdf_path.exists():
                print(f"[WARN] File not found: {pdf_path}")
                continue
            if pdf_path.stat().st_size == 0:
                print(f"[WARN] Skip empty PDF file: {pdf_path.name}")
                continue
            print(f"[INFO] Processing {pdf_path.name}")
            try:
                chunks = self.process_pdf(pdf_path)
            except Exception as exc:
                print(f"[WARN] Failed while processing {pdf_path.name}: {exc}")
                continue
            all_chunks.extend(chunks)
            print(f"[INFO] Finished {pdf_path.name}: {len(chunks)} chunks")
        return all_chunks


def _collect_pdf_paths(input_dir: Path, pattern: str, explicit_files: list[str]) -> list[Path]:
    explicit_paths = [Path(p).expanduser().resolve() for p in explicit_files]
    if explicit_paths:
        return explicit_paths
    if not input_dir.exists():
        return []
    return sorted(p.resolve() for p in input_dir.glob(pattern) if p.is_file())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk annual report PDFs into RAG-ready JSON")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing annual report PDFs",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.pdf",
        help="Glob pattern used when scanning --input-dir",
    )
    parser.add_argument(
        "--pdf-files",
        nargs="*",
        default=[],
        help="Optional explicit PDF paths. If provided, --input-dir scan is skipped",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON path for produced chunks",
    )
    parser.add_argument("--min-chars", type=int, default=300, help="Minimum chunk length")
    parser.add_argument("--max-chars", type=int, default=2400, help="Maximum chunk length before split")
    parser.add_argument("--overlap-chars", type=int, default=250, help="Overlap when hard splitting long chunks")
    parser.add_argument(
        "--ocr-image-only",
        action="store_true",
        help="Apply OCR fallback for pages without extractable text (useful for scanned PDFs)",
    )
    parser.add_argument(
        "--ocr-force-all-pages",
        action="store_true",
        help="Force OCR on every page (slower but useful for heavily garbled PDFs)",
    )
    parser.add_argument(
        "--ocr-fix-garbled",
        action="store_true",
        help="OCR-repair pages that appear garbled (mixed alpha-digit artifacts)",
    )
    parser.add_argument(
        "--ocr-garbled-page-ratio-threshold",
        type=float,
        default=0.30,
        help="When --ocr-fix-garbled is enabled, OCR all text pages if garbled-page ratio exceeds this threshold",
    )
    parser.add_argument(
        "--keep-garbled-chunks",
        action="store_true",
        help="Keep chunks with heavily garbled text (default behavior is to drop them)",
    )
    parser.add_argument(
        "--llm-repair-garbled-chunks",
        action="store_true",
        help="Use local LLM (Gemma/OpenAI-compatible API) to repair garbled OCR chunks before dropping",
    )
    parser.add_argument(
        "--llm-repair-base-url",
        type=str,
        default=os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:9090/v1"),
        help="Base URL for local OpenAI-compatible API used for OCR text repair",
    )
    parser.add_argument(
        "--llm-repair-api-key",
        type=str,
        default=os.getenv("LOCAL_LLM_API_KEY", "no-key-required"),
        help="API key for local OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--llm-repair-model",
        type=str,
        default=os.getenv("LOCAL_LLM_REPAIR_MODEL", os.getenv("LOCAL_LLM_MODEL", "mlx-community/gemma-4-e2b-it-4bit")),
        help="Model name for local LLM OCR repair",
    )
    parser.add_argument(
        "--llm-repair-thinking",
        action="store_true",
        help="Enable thinking mode when calling local Gemma repair (opt-in, slower)",
    )
    parser.add_argument(
        "--llm-repair-timeout",
        type=int,
        default=45,
        help="Timeout seconds for each local LLM repair call",
    )
    parser.add_argument(
        "--llm-repair-max-chunks-per-file",
        type=int,
        default=8,
        help="Maximum number of garbled chunks per file to send to local LLM for repair",
    )
    parser.add_argument(
        "--ocr-language",
        type=str,
        default="vie+eng",
        help="Vision OCR language tokens (supports vie+eng or explicit BCP-47 like vi-VN,en-US)",
    )
    parser.add_argument("--ocr-dpi", type=int, default=200, help="Render DPI used for Vision OCR")
    parser.add_argument(
        "--ocr-backend",
        type=str,
        default="auto",
        choices=["auto", "vision", "none"],
        help="OCR backend (auto=vision on macOS ARM64, none elsewhere)",
    )
    parser.add_argument(
        "--parser-backend",
        type=str,
        default="kreuzberg",
        choices=["kreuzberg", "pymupdf"],
        help="Document extraction backend. Default uses Kreuzberg with fallback to PyMuPDF.",
    )
    parser.add_argument(
        "--include-other",
        action="store_true",
        help="Include chunks outside core value-investing categories",
    )
    parser.add_argument(
        "--allow-empty-output",
        action="store_true",
        help="Allow writing output JSON even when no chunks are generated",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    pdf_paths = _collect_pdf_paths(args.input_dir, args.glob, args.pdf_files)
    if not pdf_paths:
        print("[WARN] No PDF files found. Put files in artifacts/rag/annual_reports/raw_pdfs or pass --pdf-files.")
        return 1

    chunker = AnnualReportChunker(
        min_chars=max(100, int(args.min_chars)),
        max_chars=max(500, int(args.max_chars)),
        overlap_chars=max(0, int(args.overlap_chars)),
        enable_ocr=bool(args.ocr_image_only),
        ocr_language=str(args.ocr_language),
        ocr_dpi=max(72, int(args.ocr_dpi)),
        ocr_backend=str(args.ocr_backend),
        ocr_force_all_pages=bool(args.ocr_force_all_pages),
        ocr_fix_garbled=bool(args.ocr_fix_garbled),
        ocr_garbled_page_ratio_threshold=float(args.ocr_garbled_page_ratio_threshold),
        drop_garbled_chunks=not bool(args.keep_garbled_chunks),
        llm_repair_garbled_chunks=bool(args.llm_repair_garbled_chunks),
        llm_repair_base_url=str(args.llm_repair_base_url),
        llm_repair_api_key=str(args.llm_repair_api_key),
        llm_repair_model=str(args.llm_repair_model),
        llm_repair_enable_thinking=bool(args.llm_repair_thinking),
        llm_repair_timeout_seconds=max(5, int(args.llm_repair_timeout)),
        llm_repair_max_chunks_per_file=max(0, int(args.llm_repair_max_chunks_per_file)),
        keep_focus_only=not bool(args.include_other),
        parser_backend=str(args.parser_backend),
    )
    all_chunks = chunker.process_many(pdf_paths)

    if not all_chunks and not args.allow_empty_output:
        print("[WARN] No chunks generated. Output file is not overwritten. Check PDF integrity and retry.")
        return 2

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as fh:
        json.dump(all_chunks, fh, ensure_ascii=False, indent=2)

    summary = chunker.get_run_summary()
    print(
        "[OCR] "
        f"files={summary['files_processed']} "
        f"files_with_ocr={summary['files_with_ocr']} "
        f"pages_total={summary['pages_total']} "
        f"pages_with_text={summary['pages_with_text']} "
        f"pages_with_ocr={summary['pages_with_ocr']} "
        f"llm_repair_attempted={summary['llm_repair_attempted']} "
        f"llm_repair_succeeded={summary['llm_repair_succeeded']} "
        f"ocr_rate_total={summary['ocr_rate_on_total_pages_pct']:.2f}% "
        f"ocr_rate_text={summary['ocr_rate_on_text_pages_pct']:.2f}%"
    )

    print(f"[DONE] Saved {len(all_chunks)} chunks -> {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
