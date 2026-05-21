from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = ROOT / "artifacts" / "rag" / "annual_reports" / "raw_pdfs"
DEFAULT_CHUNKS_JSON = ROOT / "artifacts" / "rag" / "annual_reports" / "chunks" / "annual_reports_chunks.json"
DEFAULT_CRAWL_MANIFEST = ROOT / "artifacts" / "rag" / "annual_reports" / "raw_pdfs" / "crawl_manifest.json"


@dataclass
class CrawlResult:
    downloaded: int = 0
    deduplicated: int = 0
    failed: int = 0
    discovered_pdf_links: int = 0
    source_urls: int = 0


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", without_accents.strip().lower())


def _infer_stock_code_from_filename(file_name: str) -> str:
    stem = Path(file_name).stem
    normalized = _normalize_text(stem)

    if "hpg" in normalized or "hoa phat" in normalized:
        return "HPG"
    if "acb" in normalized or "a chau" in normalized or "asia commercial" in normalized:
        return "ACB"

    token_stopwords = {
        "annual",
        "report",
        "bao",
        "cao",
        "nam",
        "tai",
        "chinh",
        "hop",
        "nhat",
        "investor",
        "relations",
        "bctn",
        "bcb",
        "bdl",
        "cn",
        "bc",
    }

    prefix_token = re.split(r"[_\-\s]+", stem.strip())[0].upper()
    if (
        2 <= len(prefix_token) <= 6
        and re.fullmatch(r"[A-Z0-9]+", prefix_token)
        and re.search(r"[A-Z]", prefix_token)
        and prefix_token.lower() not in token_stopwords
    ):
        return prefix_token

    tokens = re.split(r"[^A-Za-z0-9]+", stem)
    for token in tokens:
        if not token:
            continue
        upper = token.upper()
        if (
            2 <= len(upper) <= 6
            and re.fullmatch(r"[A-Z0-9]+", upper)
            and re.search(r"[A-Z]", upper)
            and upper.lower() not in token_stopwords
        ):
            return upper

    match = re.search(r"([A-Za-z][A-Za-z0-9]{1,5})", stem)
    return match.group(1).upper() if match else "UNKNOWN"


def _collect_code_counter_from_pdfs(pdf_paths: list[Path]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for pdf_path in pdf_paths:
        code = _infer_stock_code_from_filename(pdf_path.name)
        counter[code] += 1
    return counter


def _format_known_code_counts(counter: Counter[str]) -> str:
    known_codes = sorted(code for code in counter.keys() if code != "UNKNOWN")
    if not known_codes:
        return "{}"
    body = ", ".join(f"{code}:{counter[code]}" for code in known_codes)
    return "{" + body + "}"


def _safe_filename(name: str, fallback: str) -> str:
    candidate = unquote(name or "").strip()
    candidate = candidate.split("?")[0].split("#")[0]
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate)
    candidate = re.sub(r"_+", "_", candidate).strip("._")
    if not candidate:
        candidate = fallback
    if not candidate.lower().endswith(".pdf"):
        candidate = f"{candidate}.pdf"
    return candidate


def _filename_from_url(url: str, fallback: str) -> str:
    parsed = urlparse(url)
    base = Path(parsed.path).name
    return _safe_filename(base, fallback)


def _is_pdf_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    query = parsed.query.lower()
    return path.endswith(".pdf") or "pdf" in query


def _is_pdf_response(url: str, content_type: str) -> bool:
    ct = (content_type or "").lower()
    return "application/pdf" in ct or _is_pdf_url(url)


def _ensure_unique_path(raw_dir: Path, filename: str) -> Path:
    base = Path(filename).stem
    suffix = Path(filename).suffix or ".pdf"
    target = raw_dir / f"{base}{suffix}"
    index = 1
    while target.exists():
        target = raw_dir / f"{base}_{index}{suffix}"
        index += 1
    return target


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {"items": []}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {"items": []}
    if not isinstance(data, dict):
        return {"items": []}
    items = data.get("items")
    if not isinstance(items, list):
        data["items"] = []
    return data


def _save_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _dedup_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for url in urls:
        normalized = url.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _extract_year(raw: Any, fallback_text: str) -> int:
    try:
        year = int(raw)
        if 1900 <= year <= 2100:
            return year
    except (TypeError, ValueError):
        pass
    m = re.search(r"(20\d{2})", fallback_text)
    return int(m.group(1)) if m else 0


_VIETSTOCK_ANNUAL_RE = re.compile(
    r"(bao\s*_?cao\s*_?thuong\s*_?nien|baocaothuongnien|bctn|"
    r"bc\s*_?thuong\s*_?nien|annual\s*_?report|annualreport)",
    re.IGNORECASE,
)
_VIETSTOCK_BAD_RE = re.compile(
    r"(cbtt|cvcbtt|cv\s*_?cbtt|cong\s*_?bo\s*_?thong\s*_?tin|"
    r"information\s*_?disclosure|phu\s*_?luc|"
    r"bao\s*_?cao\s*_?phat\s*_?trien\s*_?ben\s*_?vung|sustainability|"
    r"bao\s*_?cao\s*_?tai\s*_?chinh|bctc)",
    re.IGNORECASE,
)
_VIETSTOCK_CHAPTER_RE = re.compile(
    r"(chapter|chuong|part|[_\-\s]c\d+(?:[_\-\s]|$))",
    re.IGNORECASE,
)


def _vietstock_token_from_html(html: str) -> str:
    match = re.search(
        r"name=__RequestVerificationToken\s+type=hidden\s+value=([^>\s]+)",
        html or "",
    )
    if match:
        return match.group(1)
    match = re.search(
        r'name=["\']__RequestVerificationToken["\'][^>]*value=["\']([^"\']+)["\']',
        html or "",
    )
    return match.group(1) if match else ""


def _vietstock_is_annual_text(text: str) -> bool:
    return bool(_VIETSTOCK_ANNUAL_RE.search(_normalize_text(text)))


def _vietstock_is_bad_child(text: str) -> bool:
    return bool(_VIETSTOCK_BAD_RE.search(_normalize_text(text)))


def _vietstock_is_chapter_child(text: str) -> bool:
    return bool(_VIETSTOCK_CHAPTER_RE.search(_normalize_text(text)))


def _fetch_vietstock_json(
    session: requests.Session,
    url: str,
    data: dict[str, Any],
    timeout_seconds: int,
) -> Any:
    response = session.post(url, data=data, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()


def _select_vietstock_archive_pdf_children(
    children: list[dict[str, Any]],
    *,
    year: int,
) -> list[dict[str, Any]]:
    pdfs = [
        child
        for child in children
        if isinstance(child, dict)
        and (
            str(child.get("FileName") or "").lower().endswith(".pdf")
            or str(child.get("Url") or "").lower().endswith(".pdf")
        )
    ]
    if not pdfs:
        return []

    def child_text(child: dict[str, Any]) -> str:
        return f"{child.get('FileName') or ''} {child.get('Url') or ''}"

    annual_pdfs = [
        child
        for child in pdfs
        if str(year) in _normalize_text(child_text(child))
        and _vietstock_is_annual_text(child_text(child))
        and not _vietstock_is_bad_child(child_text(child))
    ]
    chapter_pdfs = [
        child
        for child in annual_pdfs
        if _vietstock_is_chapter_child(child_text(child))
    ]
    if len(chapter_pdfs) >= 3:
        return sorted(chapter_pdfs, key=lambda child: str(child.get("FileName") or ""))

    pool = annual_pdfs or pdfs
    return [
        max(
            pool,
            key=lambda child: int(child.get("FileSize") or 0),
        )
    ]


def _fetch_vietstock_annual_reports_for_symbol(
    session: requests.Session,
    *,
    symbol: str,
    years: int,
    target_year: int = 0,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    """Fetch annual-report PDF URLs from VietstockFinance document API.

    VietstockFinance document type 2 is "Bao cao thuong nien". Archive
    documents are expanded through /Data/ViewDocument, which returns direct PDF
    children; when an annual report is split into chapters, all chapter PDFs are
    returned as one logical report via the "links" field.
    """

    stock = str(symbol).strip().upper()
    if not stock:
        return []

    old_headers = dict(session.headers)
    referer = f"https://finance.vietstock.vn/{stock}/tai-tai-lieu.htm?doctype=2"
    session.headers.update(
        {
            "Referer": referer,
            "X-Requested-With": "XMLHttpRequest",
        }
    )
    try:
        try:
            page = session.get(referer, timeout=timeout_seconds)
            page.raise_for_status()
            token = _vietstock_token_from_html(page.text)
        except requests.RequestException as exc:
            print(f"[VIETSTOCK][WARN] symbol={stock} page_failed={exc}")
            return []

        try:
            docs = _fetch_vietstock_json(
                session,
                "https://finance.vietstock.vn/data/getdocument",
                {
                    "code": stock,
                    "page": 1,
                    "type": 2,
                    "__RequestVerificationToken": token,
                },
                timeout_seconds,
            )
        except (requests.RequestException, ValueError) as exc:
            print(f"[VIETSTOCK][WARN] symbol={stock} document_api_failed={exc}")
            return []

        if not isinstance(docs, list):
            print(f"[VIETSTOCK][WARN] symbol={stock} invalid_document_rows")
            return []

        selected_by_year: dict[int, dict[str, Any]] = {}
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            title = str(doc.get("Title") or doc.get("FullName") or "").strip()
            url = str(doc.get("Url") or "").strip().replace("http://", "https://")
            ext = str(doc.get("FileExt") or "").strip().lower()
            if not title or not url:
                continue

            report_year = _extract_year(None, f"{title} {url}")
            if report_year <= 0:
                continue
            if target_year > 0 and report_year != target_year:
                continue
            if report_year in selected_by_year:
                continue

            links: list[str] = []
            child_names: list[str] = []
            if ext == ".pdf":
                links = [url]
            elif ext in {".zip", ".rar"}:
                try:
                    children = _fetch_vietstock_json(
                        session,
                        "https://finance.vietstock.vn/Data/ViewDocument",
                        {
                            "id": doc.get("FileInfoID"),
                            "__RequestVerificationToken": token,
                        },
                        timeout_seconds,
                    )
                except (requests.RequestException, ValueError) as exc:
                    print(
                        f"[VIETSTOCK][WARN] symbol={stock} year={report_year} "
                        f"archive_api_failed={exc}"
                    )
                    continue
                if not isinstance(children, list):
                    continue
                chosen_children = _select_vietstock_archive_pdf_children(
                    [child for child in children if isinstance(child, dict)],
                    year=report_year,
                )
                links = [
                    str(child.get("Url") or "").strip().replace("http://", "https://")
                    for child in chosen_children
                    if str(child.get("Url") or "").strip()
                ]
                child_names = [
                    str(child.get("FileName") or "").strip()
                    for child in chosen_children
                    if str(child.get("FileName") or "").strip()
                ]
            else:
                continue

            links = _dedup_urls(links)
            if not links:
                continue

            selected_by_year[report_year] = {
                "symbol": stock,
                "year": report_year,
                "name": title,
                "link": links[0],
                "links": links,
                "source": "vietstock",
                "file_names": child_names,
            }

        selected = [
            selected_by_year[year]
            for year in sorted(selected_by_year.keys(), reverse=True)
        ]
        if years > 0:
            selected = selected[:years]

        if not selected:
            print(f"[VIETSTOCK][WARN] symbol={stock} no_annual_report_link")
        else:
            print(f"[VIETSTOCK][OK] symbol={stock} annual_links={len(selected)}")
        return selected
    finally:
        session.headers.clear()
        session.headers.update(old_headers)


def _discover_pdf_links_from_html(base_url: str, html: str, max_links_per_page: int) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        if not href:
            continue
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            continue
        if not _is_pdf_url(absolute):
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        links.append(absolute)
        if len(links) >= max_links_per_page:
            break
    return links


def _download_pdf(
    session: requests.Session,
    url: str,
    raw_dir: Path,
    timeout_seconds: int,
    known_hashes: set[str],
    manifest_items: list[dict[str, Any]],
) -> tuple[str, Path | None, str | None]:
    try:
        response = session.get(url, timeout=timeout_seconds)
        response.raise_for_status()
    except requests.RequestException as exc:
        return "failed", None, f"request_error={exc}"

    if not _is_pdf_response(url, response.headers.get("Content-Type", "")):
        return "failed", None, "not_pdf"

    payload = response.content
    if not payload:
        return "failed", None, "empty_body"

    checksum = hashlib.sha256(payload).hexdigest()
    if checksum in known_hashes:
        return "deduplicated", None, "same_sha256"

    fallback_name = f"downloaded_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    filename = _filename_from_url(response.url or url, fallback=fallback_name)
    target_path = _ensure_unique_path(raw_dir, filename)
    target_path.write_bytes(payload)

    known_hashes.add(checksum)
    manifest_items.append(
        {
            "url": url,
            "final_url": response.url,
            "file_name": target_path.name,
            "sha256": checksum,
            "size_bytes": len(payload),
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    return "downloaded", target_path, None


def crawl_pdfs(
    source_urls: list[str],
    raw_dir: Path,
    manifest_path: Path,
    timeout_seconds: int,
    max_links_per_page: int,
    user_agent: str,
) -> CrawlResult:
    raw_dir.mkdir(parents=True, exist_ok=True)
    result = CrawlResult(source_urls=len(source_urls))

    manifest = _load_manifest(manifest_path)
    manifest_items = manifest.get("items", [])
    known_hashes = {str(item.get("sha256", "")) for item in manifest_items if item.get("sha256")}

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    for source_url in source_urls:
        try:
            response = session.get(source_url, timeout=timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"[CRAWL][WARN] cannot open source url={source_url}: {exc}")
            result.failed += 1
            continue

        content_type = response.headers.get("Content-Type", "")
        if _is_pdf_response(source_url, content_type):
            candidate_pdf_links = [source_url]
        else:
            candidate_pdf_links = _discover_pdf_links_from_html(
                base_url=source_url,
                html=response.text,
                max_links_per_page=max_links_per_page,
            )

        result.discovered_pdf_links += len(candidate_pdf_links)
        if not candidate_pdf_links:
            print(f"[CRAWL][WARN] no pdf link found in source url={source_url}")

        for pdf_url in candidate_pdf_links:
            status, saved_path, reason = _download_pdf(
                session=session,
                url=pdf_url,
                raw_dir=raw_dir,
                timeout_seconds=timeout_seconds,
                known_hashes=known_hashes,
                manifest_items=manifest_items,
            )
            if status == "downloaded":
                result.downloaded += 1
                print(f"[CRAWL][OK] downloaded {pdf_url} -> {saved_path.name}")
            elif status == "deduplicated":
                result.deduplicated += 1
                print(f"[CRAWL][SKIP] duplicate content {pdf_url}")
            else:
                result.failed += 1
                print(f"[CRAWL][WARN] failed {pdf_url}: {reason}")

    _save_manifest(manifest_path, manifest)
    return result


def _load_chunks(chunks_json: Path) -> list[dict[str, Any]]:
    if not chunks_json.exists():
        return []
    try:
        data = json.loads(chunks_json.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]
