from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
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
DEFAULT_INDEX_JSON = ROOT / "artifacts" / "rag" / "annual_reports" / "index" / "annual_reports_bm25_index.json"
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


def _tokenize(text: str) -> list[str]:
    normalized = _normalize_text(text)
    return re.findall(r"[a-z0-9]{2,}", normalized)


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


def _fetch_cafef_annual_links_for_symbol(
    session: requests.Session,
    *,
    symbol: str,
    market: str,
    years: int,
    timeout_seconds: int,
) -> list[str]:
    endpoint = (
        "https://cafef.vn/du-lieu/Ajax/PageNew/FileBCTC.ashx"
        f"?Symbol={symbol.lower()}&Type=3&Year=0"
    )
    referer = f"https://cafef.vn/du-lieu/{market.lower()}/{symbol.lower()}-bao-cao-tai-chinh.chn"

    try:
        response = session.get(endpoint, timeout=timeout_seconds, headers={"Referer": referer})
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[CAFEF][WARN] symbol={symbol} request_failed={exc}")
        return []

    try:
        payload = response.json()
    except ValueError:
        print(f"[CAFEF][WARN] symbol={symbol} invalid_json")
        return []

    rows = payload.get("Data")
    if not isinstance(rows, list):
        print(f"[CAFEF][WARN] symbol={symbol} no_data_rows")
        return []

    annual_rows: list[tuple[int, str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("Name") or "").strip()
        link = str(row.get("Link") or "").strip()
        if not name or not link:
            continue

        normalized_name = _normalize_text(name)
        if "bao cao thuong nien" not in normalized_name:
            continue
        if any(term in normalized_name for term in ("ban dieu le", "ban cao bach")):
            continue

        year = _extract_year(row.get("Year"), name)
        if year <= 0:
            continue
        annual_rows.append((year, name, link))

    annual_rows.sort(key=lambda item: item[0], reverse=True)

    selected: list[str] = []
    used_years: set[int] = set()
    for year, _name, link in annual_rows:
        if year in used_years:
            continue
        used_years.add(year)
        selected.append(link)
        if years > 0 and len(selected) >= years:
            break

    if not selected:
        print(f"[CAFEF][WARN] symbol={symbol} no_annual_report_link")
    else:
        print(f"[CAFEF][OK] symbol={symbol} annual_links={len(selected)}")
    return selected


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


def _build_bm25_index(chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> dict[str, Any]:
    doc_term_freqs: list[dict[str, int]] = []
    doc_lengths: list[int] = []
    doc_freq: Counter[str] = Counter()
    documents: list[dict[str, Any]] = []

    for chunk in chunks:
        title = str(chunk.get("subsection_title") or "")
        body = str(chunk.get("text") or "")
        full_text = f"{title}\n{body}".strip()
        tokens = _tokenize(full_text)
        tf = Counter(tokens)
        doc_len = sum(tf.values())
        doc_term_freqs.append(dict(tf))
        doc_lengths.append(doc_len)
        doc_freq.update(tf.keys())
        documents.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "stock_code": chunk.get("stock_code"),
                "year": chunk.get("year"),
                "category": chunk.get("category"),
                "subsection_title": chunk.get("subsection_title"),
                "source_file": chunk.get("source_file"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "text_preview": body[:280],
            }
        )

    total_docs = len(chunks)
    avg_doc_len = (sum(doc_lengths) / total_docs) if total_docs else 0.0

    idf: dict[str, float] = {}
    for token, df in doc_freq.items():
        idf[token] = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))

    inverted_index: dict[str, list[list[int]]] = defaultdict(list)
    for doc_idx, tf_map in enumerate(doc_term_freqs):
        for token, tf in tf_map.items():
            inverted_index[token].append([doc_idx, tf])

    return {
        "schema": "annual_report_bm25_v1",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "documents_count": total_docs,
        "avg_doc_len": avg_doc_len,
        "k1": k1,
        "b": b,
        "doc_lengths": doc_lengths,
        "idf": idf,
        "inverted_index": dict(inverted_index),
        "documents": documents,
    }
