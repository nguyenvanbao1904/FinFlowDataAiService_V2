from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
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
    ) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.enable_ocr = enable_ocr
        self.ocr_language = ocr_language
        self.ocr_dpi = ocr_dpi

        # Taxonomy for value investing oriented retrieval.
        self.category_map = {
            "mdna": [
                "bao cao ban giam doc",
                "bao cao cua ban dieu hanh",
                "ban dieu hanh",
                "danh gia ket qua",
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
            ],
            "risk": ["rui ro", "quan ly rui ro", "thach thuc"],
            "governance": [
                "quan tri cong ty",
                "hoi dong quan tri",
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
                "lich su",
                "nganh nghe",
            ],
        }

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text:
            return ""
        normalized = unicodedata.normalize("NFKD", text)
        without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        compact = re.sub(r"\s+", " ", without_accents.strip().lower())
        return compact

    def _infer_stock_and_year(self, file_path: Path) -> tuple[str, int]:
        name_normalized = self._normalize_text(file_path.stem)

        stock = "UNKNOWN"
        if "hpg" in name_normalized or "hoa phat" in name_normalized:
            stock = "HPG"
        elif "acb" in name_normalized or "a chau" in name_normalized or "asia commercial" in name_normalized:
            stock = "ACB"
        else:
            found = re.search(r"([A-Za-z]{3,5})", file_path.stem)
            if found:
                stock = found.group(1).upper()

        year = 2025
        year_match = re.search(r"(20\d{2})", file_path.stem)
        if year_match:
            year = int(year_match.group(1))

        return stock, year

    def _detect_category(self, heading: str) -> str:
        heading_normalized = self._normalize_text(heading)
        for category, keywords in self.category_map.items():
            if any(keyword in heading_normalized for keyword in keywords):
                return category
        return "other"

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

        return [p for p in pieces if len(p) >= self.min_chars]

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

    def process_pdf(self, pdf_path: Path) -> list[dict]:
        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            print(f"[WARN] Skip unreadable PDF {pdf_path.name}: {exc}")
            return []
        stock_code, year = self._infer_stock_and_year(pdf_path)

        drafts: list[ChunkDraft] = []
        current_heading = "Gioi thieu"
        current_lines: list[str] = []
        page_start = 1
        pages_with_text = 0
        pages_with_ocr = 0

        for page_idx in tqdm(range(len(doc)), desc=f"Chunking {pdf_path.name}"):
            page_no = page_idx + 1
            page = doc[page_idx]
            text = page.get_text("text")
            if not text or not text.strip():
                if self.enable_ocr:
                    try:
                        textpage = page.get_textpage_ocr(language=self.ocr_language, dpi=self.ocr_dpi)
                        text = page.get_text("text", textpage=textpage)
                        if text and text.strip():
                            pages_with_ocr += 1
                    except Exception as exc:
                        print(f"[WARN] OCR failed on {pdf_path.name} page {page_no}: {exc}")
                if not text or not text.strip():
                    continue
            pages_with_text += 1

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
                page_end=len(doc),
            )

        if pages_with_text == 0:
            print(
                f"[WARN] No extractable text found in {pdf_path.name}. "
                "The PDF may be image-only and requires OCR before chunking."
            )
        elif pages_with_ocr > 0:
            print(f"[INFO] OCR extracted text for {pages_with_ocr}/{len(doc)} pages in {pdf_path.name}")

        doc.close()

        chunks: list[dict] = []
        chunk_index = 0
        for draft in drafts:
            split_texts = self._split_long_text(draft.text)
            if not split_texts:
                continue
            category = self._detect_category(draft.heading)
            for split_rank, piece in enumerate(split_texts, start=1):
                chunk_index += 1
                chunks.append(
                    {
                        "chunk_id": f"{stock_code}_{year}_{chunk_index:04d}",
                        "stock_code": stock_code,
                        "year": year,
                        "category": category,
                        "section": category,
                        "subsection_title": draft.heading,
                        "document_type": "annual_report",
                        "text": piece,
                        "page_start": draft.page_start,
                        "page_end": draft.page_end,
                        "child_rank": split_rank,
                        "source_file": pdf_path.name,
                    }
                )

        return chunks

    def process_many(self, pdf_paths: Iterable[Path]) -> list[dict]:
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
    parser.add_argument("--ocr-language", type=str, default="vie+eng", help="Tesseract language codes")
    parser.add_argument("--ocr-dpi", type=int, default=200, help="Render DPI used for OCR")
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
    )
    all_chunks = chunker.process_many(pdf_paths)

    if not all_chunks and not args.allow_empty_output:
        print("[WARN] No chunks generated. Output file is not overwritten. Check PDF integrity and retry.")
        return 2

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as fh:
        json.dump(all_chunks, fh, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved {len(all_chunks)} chunks -> {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
