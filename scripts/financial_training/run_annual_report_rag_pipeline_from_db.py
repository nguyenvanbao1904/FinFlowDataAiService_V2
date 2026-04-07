from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pymysql
import requests

try:
    from chunk_annual_reports import AnnualReportChunker
    from annual_report_rag_shared import (
        DEFAULT_CHUNKS_JSON,
        DEFAULT_CRAWL_MANIFEST,
        DEFAULT_INDEX_JSON,
        DEFAULT_RAW_DIR,
        _build_bm25_index,
        _collect_code_counter_from_pdfs,
        _download_pdf,
        _dedup_urls,
        _fetch_cafef_annual_links_for_symbol,
        _format_known_code_counts,
        _load_manifest,
        _load_chunks,
        _save_manifest,
        crawl_pdfs,
    )
except ImportError:  # pragma: no cover
    from scripts.financial_training.chunk_annual_reports import AnnualReportChunker
    from scripts.financial_training.annual_report_rag_shared import (
        DEFAULT_CHUNKS_JSON,
        DEFAULT_CRAWL_MANIFEST,
        DEFAULT_INDEX_JSON,
        DEFAULT_RAW_DIR,
        _build_bm25_index,
        _collect_code_counter_from_pdfs,
        _download_pdf,
        _dedup_urls,
        _fetch_cafef_annual_links_for_symbol,
        _format_known_code_counts,
        _load_manifest,
        _load_chunks,
        _save_manifest,
        crawl_pdfs,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings

DEFAULT_CHUNKS_DB = (
    PROJECT_ROOT
    / "artifacts"
    / "rag"
    / "annual_reports"
    / "chunks"
    / "annual_reports_chunks.sqlite"
)

_ENSURED_CHUNKS_DB: set[str] = set()


def _write_chunks_checkpoint(output_chunks: Path, chunks: list[dict[str, Any]]) -> None:
    output_chunks.parent.mkdir(parents=True, exist_ok=True)
    output_chunks.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")


def _open_chunks_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def _ensure_chunks_db_schema(db_path: Path) -> None:
    key = str(db_path.resolve())
    if key in _ENSURED_CHUNKS_DB:
        return

    last_exc: Exception | None = None
    for attempt in range(1, 16):
        try:
            with _open_chunks_db(db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id TEXT PRIMARY KEY,
                        stock_code TEXT,
                        year INTEGER,
                        category TEXT,
                        source_file TEXT,
                        page_start INTEGER,
                        page_end INTEGER,
                        worker_id TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        chunk_json TEXT NOT NULL
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_stock_year ON chunks(stock_code, year)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source_file ON chunks(source_file)")
                conn.commit()
            _ENSURED_CHUNKS_DB.add(key)
            return
        except sqlite3.OperationalError as exc:
            last_exc = exc
            if "locked" in str(exc).lower() and attempt < 15:
                time.sleep(min(0.25 * attempt, 2.0))
                continue
            raise

    if last_exc is not None:
        raise last_exc


def _write_chunks_checkpoint_sqlite(
    db_path: Path,
    chunks: list[dict[str, Any]],
    *,
    worker_id: str,
) -> int:
    _ensure_chunks_db_schema(db_path)

    if not chunks:
        return 0

    rows: list[tuple[Any, ...]] = []
    for item in chunks:
        chunk_id = str(item.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        rows.append(
            (
                chunk_id,
                str(item.get("stock_code") or ""),
                int(item.get("year") or 0),
                str(item.get("category") or "other"),
                str(item.get("source_file") or ""),
                int(item.get("page_start") or 0),
                int(item.get("page_end") or 0),
                worker_id,
                json.dumps(item, ensure_ascii=False),
            )
        )

    if not rows:
        return 0

    with _open_chunks_db(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO chunks (
                chunk_id,
                stock_code,
                year,
                category,
                source_file,
                page_start,
                page_end,
                worker_id,
                chunk_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                stock_code = excluded.stock_code,
                year = excluded.year,
                category = excluded.category,
                source_file = excluded.source_file,
                page_start = excluded.page_start,
                page_end = excluded.page_end,
                worker_id = excluded.worker_id,
                chunk_json = excluded.chunk_json
            """,
            rows,
        )
        conn.commit()

    return len(rows)


def _count_chunks_in_db(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    _ensure_chunks_db_schema(db_path)
    with _open_chunks_db(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
    return int(row[0]) if row else 0


def _load_chunks_from_db(db_path: Path) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []

    _ensure_chunks_db_schema(db_path)

    with _open_chunks_db(db_path) as conn:
        rows = conn.execute(
            """
            SELECT chunk_json
            FROM chunks
            ORDER BY stock_code ASC, year ASC, chunk_id ASC
            """
        ).fetchall()

    chunks: list[dict[str, Any]] = []
    for row in rows:
        if not row or row[0] is None:
            continue
        try:
            value = json.loads(str(row[0]))
        except Exception:
            continue
        if isinstance(value, dict):
            chunks.append(value)
    return chunks


def _cleanup_outputs_for_reset(
    *,
    output_chunks: Path,
    output_chunks_db: Path,
    checkpoint_backend: str,
) -> None:
    def _remove_sqlite_with_sidecars(path: Path) -> None:
        if path.exists():
            path.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(path) + suffix)
            if sidecar.exists():
                sidecar.unlink()

    if output_chunks.exists():
        try:
            output_chunks.unlink()
            print(f"[STREAM][RESET] removed_existing_output={output_chunks}")
        except Exception as exc:
            print(f"[STREAM][WARN] cannot remove output file {output_chunks}: {exc}")

    if checkpoint_backend == "sqlite" and output_chunks_db.exists():
        try:
            _remove_sqlite_with_sidecars(output_chunks_db)
            print(f"[STREAM][RESET] removed_existing_checkpoint_db={output_chunks_db}")
        except Exception as exc:
            print(f"[STREAM][WARN] cannot remove checkpoint db {output_chunks_db}: {exc}")


def _build_worker_command(
    args: argparse.Namespace,
    *,
    worker_id: str,
    worker_raw_dir: Path,
    worker_manifest: Path,
    worker_output_chunks: Path,
    worker_output_chunks_db: Path,
    exchange_filter_value: str,
    shard_count: int = 1,
    shard_index: int = 0,
) -> list[str]:

    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker-mode",
        "single",
        "--worker-id",
        worker_id,
        "--cafef-years",
        str(args.cafef_years),
        "--exchange-filter",
        exchange_filter_value,
        "--shard-count",
        str(shard_count),
        "--shard-index",
        str(shard_index),
        "--raw-dir",
        str(worker_raw_dir),
        "--crawl-manifest",
        str(worker_manifest),
        "--crawl-timeout",
        str(args.crawl_timeout),
        "--user-agent",
        str(args.user_agent),
        "--output-chunks",
        str(worker_output_chunks),
        "--output-chunks-db",
        str(worker_output_chunks_db),
        "--checkpoint-backend",
        str(args.checkpoint_backend),
        "--output-index",
        str(args.output_index),
        "--min-chars",
        str(args.min_chars),
        "--max-chars",
        str(args.max_chars),
        "--overlap-chars",
        str(args.overlap_chars),
        "--ocr-language",
        str(args.ocr_language),
        "--ocr-dpi",
        str(args.ocr_dpi),
        "--ocr-garbled-page-ratio-threshold",
        str(args.ocr_garbled_page_ratio_threshold),
        "--ocr-backend",
        str(args.ocr_backend),
        "--parser-backend",
        str(args.parser_backend),
        "--llm-repair-base-url",
        str(args.llm_repair_base_url),
        "--llm-repair-api-key",
        str(args.llm_repair_api_key),
        "--llm-repair-model",
        str(args.llm_repair_model),
        "--llm-repair-timeout",
        str(args.llm_repair_timeout),
        "--llm-repair-max-chunks-per-file",
        str(args.llm_repair_max_chunks_per_file),
        "--limit-symbols",
        str(args.limit_symbols),
        "--skip-index",
        "--skip-final-export",
    ]

    if args.streaming:
        cmd.append("--streaming")
    if args.delete_pdf_after_chunk:
        cmd.append("--delete-pdf-after-chunk")
    if args.skip_crawl:
        cmd.append("--skip-crawl")
    if args.ocr_image_only:
        cmd.append("--ocr-image-only")
    if args.ocr_force_all_pages:
        cmd.append("--ocr-force-all-pages")
    if args.ocr_fix_garbled:
        cmd.append("--ocr-fix-garbled")
    if args.include_other:
        cmd.append("--include-other")
    if args.keep_garbled_chunks:
        cmd.append("--keep-garbled-chunks")
    if args.llm_repair_garbled_chunks:
        cmd.append("--llm-repair-garbled-chunks")
    if args.llm_repair_thinking:
        cmd.append("--llm-repair-thinking")
    if args.allow_empty_output:
        cmd.append("--allow-empty-output")

    return cmd


def _start_worker_process(
    *,
    args: argparse.Namespace,
    cmd: list[str],
    worker_label: str,
) -> tuple[subprocess.Popen[bytes], Any]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    if str(args.worker_log_mode) == "inherit":
        return subprocess.Popen(cmd, env=env), None

    log_dir = args.workers_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in worker_label).strip("_")
    if not safe_label:
        safe_label = "worker"
    log_path = log_dir / f"{safe_label}.log"
    log_handle = log_path.open("ab")
    header = f"\n=== worker={worker_label} started_at={time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
    log_handle.write(header.encode("utf-8", errors="ignore"))
    log_handle.flush()
    proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
    log_handle.close()
    return proc, log_path


def _finalize_from_storage(args: argparse.Namespace) -> int:
    checkpoint_backend = str(args.checkpoint_backend)
    if checkpoint_backend == "sqlite":
        chunks_for_output = _load_chunks_from_db(args.output_chunks_db)
    else:
        chunks_for_output = _load_chunks(args.output_chunks)

    if not chunks_for_output and not args.allow_empty_output:
        print("[WARN] Chunking produced 0 chunks. Stop before writing outputs.")
        return 2

    if not args.skip_final_export:
        _write_chunks_checkpoint(args.output_chunks, chunks_for_output)
        print(f"[CHUNK][DONE] saved_chunks={len(chunks_for_output)} output={args.output_chunks}")
    else:
        print(
            "[CHUNK][DONE] "
            f"saved_chunks={len(chunks_for_output)} "
            f"export_json_skipped=true checkpoint_backend={checkpoint_backend}"
        )

    if args.skip_index:
        print("[INDEX] skipped by --skip-index")
        return 0

    index_data = _build_bm25_index(chunks_for_output)
    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    args.output_index.write_text(json.dumps(index_data, ensure_ascii=False), encoding="utf-8")
    print(
        "[INDEX][DONE] "
        f"documents={index_data['documents_count']} "
        f"vocab={len(index_data.get('idf', {}))} "
        f"output={args.output_index}"
    )
    return 0


def _run_exchange_partition(args: argparse.Namespace) -> int:
    if str(args.checkpoint_backend) != "sqlite":
        print(
            "[WORKER][WARN] exchange-partition mode requires --checkpoint-backend sqlite "
            "to avoid concurrent write conflicts."
        )
        return 2

    if args.skip_crawl:
        print("[WORKER][WARN] worker-mode exchange-partition requires crawl stage. Use --worker-mode single for skip-crawl.")
        return 2

    exchanges = sorted(_parse_exchange_filter(str(args.exchange_filter)) or {"HOSE", "HNX", "UPCOM"})
    if not exchanges:
        print("[WORKER][WARN] no exchanges resolved for exchange-partition mode")
        return 2

    args.workers_dir.mkdir(parents=True, exist_ok=True)

    if args.reset_output:
        _cleanup_outputs_for_reset(
            output_chunks=args.output_chunks,
            output_chunks_db=args.output_chunks_db,
            checkpoint_backend=str(args.checkpoint_backend),
        )

    _ensure_chunks_db_schema(args.output_chunks_db)

    processes: dict[str, tuple[subprocess.Popen[bytes], Any]] = {}
    for exchange in exchanges:
        worker_dir = args.workers_dir / exchange.lower()
        cmd = _build_worker_command(
            args,
            worker_id=exchange,
            worker_raw_dir=worker_dir / "raw_pdfs",
            worker_manifest=worker_dir / "crawl_manifest.json",
            worker_output_chunks=worker_dir / "chunks.json",
            worker_output_chunks_db=args.output_chunks_db,
            exchange_filter_value=exchange,
            shard_count=1,
            shard_index=0,
        )
        proc, log_path = _start_worker_process(
            args=args,
            cmd=cmd,
            worker_label=f"exchange_{exchange.lower()}",
        )
        if log_path is None:
            print(f"[WORKER][START] exchange={exchange} log_mode=inherit cmd={' '.join(cmd)}")
        else:
            print(f"[WORKER][START] exchange={exchange} log={log_path} cmd={' '.join(cmd)}")
        processes[exchange] = (proc, log_path)

    failures: dict[str, int] = {}
    for exchange, (proc, log_path) in processes.items():
        code = proc.wait()
        if code != 0:
            failures[exchange] = int(code)
            print(
                f"[WORKER][FAIL] exchange={exchange} exit_code={code}"
                + (f" log={log_path}" if log_path is not None else "")
            )
        else:
            print(f"[WORKER][DONE] exchange={exchange}" + (f" log={log_path}" if log_path is not None else ""))

    if failures:
        print(f"[WORKER][SUMMARY] failures={failures}")
        return max(failures.values())

    print(f"[WORKER][SUMMARY] completed_exchanges={exchanges}")
    return _finalize_from_storage(args)


def _run_symbol_shard_partition(args: argparse.Namespace) -> int:
    if str(args.checkpoint_backend) != "sqlite":
        print(
            "[WORKER][WARN] symbol-shard-partition mode requires --checkpoint-backend sqlite "
            "for worker-local DB merge flow."
        )
        return 2

    if args.skip_crawl:
        print("[WORKER][WARN] symbol-shard-partition requires crawl stage. Use --worker-mode single for skip-crawl.")
        return 2

    shard_count = max(1, int(args.shard_count))
    if shard_count < 2:
        print("[WORKER][WARN] symbol-shard-partition requires --shard-count >= 2.")
        return 2

    if int(args.shard_index) != 0:
        print("[WORKER][WARN] --shard-index is ignored in symbol-shard-partition mode.")

    args.workers_dir.mkdir(parents=True, exist_ok=True)

    if args.reset_output:
        _cleanup_outputs_for_reset(
            output_chunks=args.output_chunks,
            output_chunks_db=args.output_chunks_db,
            checkpoint_backend=str(args.checkpoint_backend),
        )

    processes: dict[int, tuple[subprocess.Popen[bytes], Any]] = {}
    worker_db_paths: dict[int, Path] = {}
    for shard_idx in range(shard_count):
        worker_name = f"shard_{shard_idx:02d}_of_{shard_count:02d}"
        worker_dir = args.workers_dir / worker_name
        worker_raw_dir = worker_dir / "raw_pdfs"
        worker_manifest = worker_dir / "crawl_manifest.json"
        worker_output_db = worker_dir / "chunks.sqlite"
        worker_output_json = worker_dir / "chunks.json"
        worker_db_paths[shard_idx] = worker_output_db

        if args.reset_output:
            if worker_manifest.exists():
                try:
                    worker_manifest.unlink()
                except Exception:
                    pass
            if worker_raw_dir.exists():
                for pdf_path in worker_raw_dir.glob("*.pdf"):
                    try:
                        pdf_path.unlink()
                    except Exception:
                        pass
            if worker_output_json.exists():
                try:
                    worker_output_json.unlink()
                except Exception:
                    pass
            if worker_output_db.exists():
                try:
                    worker_output_db.unlink()
                except Exception:
                    pass
            for suffix in ("-wal", "-shm"):
                sidecar = Path(str(worker_output_db) + suffix)
                if sidecar.exists():
                    try:
                        sidecar.unlink()
                    except Exception:
                        pass

        cmd = _build_worker_command(
            args,
            worker_id=worker_name,
            worker_raw_dir=worker_raw_dir,
            worker_manifest=worker_manifest,
            worker_output_chunks=worker_output_json,
            worker_output_chunks_db=worker_output_db,
            exchange_filter_value=str(args.exchange_filter),
            shard_count=shard_count,
            shard_index=shard_idx,
        )
        proc, log_path = _start_worker_process(
            args=args,
            cmd=cmd,
            worker_label=worker_name,
        )
        if log_path is None:
            print(f"[WORKER][START] shard={shard_idx}/{shard_count} log_mode=inherit cmd={' '.join(cmd)}")
        else:
            print(f"[WORKER][START] shard={shard_idx}/{shard_count} log={log_path} cmd={' '.join(cmd)}")
        processes[shard_idx] = (proc, log_path)

    failures: dict[int, int] = {}
    for shard_idx, (proc, log_path) in processes.items():
        code = proc.wait()
        if code != 0:
            failures[shard_idx] = int(code)
            print(
                f"[WORKER][FAIL] shard={shard_idx} exit_code={code}"
                + (f" log={log_path}" if log_path is not None else "")
            )
        else:
            db_count = _count_chunks_in_db(worker_db_paths[shard_idx])
            print(
                f"[WORKER][DONE] shard={shard_idx} chunks={db_count} db={worker_db_paths[shard_idx]}"
                + (f" log={log_path}" if log_path is not None else "")
            )

    if failures:
        print(f"[WORKER][SUMMARY] failures={failures}")
        return max(failures.values())

    merged_rows = _merge_worker_chunk_dbs(
        worker_db_paths=[worker_db_paths[idx] for idx in sorted(worker_db_paths.keys())],
        output_db_path=args.output_chunks_db,
    )
    final_count = _count_chunks_in_db(args.output_chunks_db)
    print(
        "[WORKER][MERGE] "
        f"workers={shard_count} merged_rows={merged_rows} final_unique_chunks={final_count} "
        f"output_db={args.output_chunks_db}"
    )
    return _finalize_from_storage(args)


def _exchange_to_market(exchange: str | None) -> str:
    value = (exchange or "").strip().upper()
    if value in {"HOSE", "HSX"}:
        return "hsx"
    if value == "HNX":
        return "hnx"
    if value == "UPCOM":
        return "upcom"
    return "hsx"


def _parse_exchange_filter(raw: str) -> set[str]:
    if not raw.strip():
        return set()
    return {item.strip().upper() for item in raw.split(",") if item.strip()}


def _compute_symbol_shard(symbol: str, shard_count: int) -> int:
    if shard_count <= 1:
        return 0
    digest = hashlib.sha1(symbol.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value % shard_count


def _apply_symbol_shard_filter(
    pairs: list[tuple[str, str]],
    *,
    shard_count: int,
    shard_index: int,
) -> list[tuple[str, str]]:
    if shard_count <= 1:
        return pairs
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"Invalid shard_index={shard_index} for shard_count={shard_count}")

    selected: list[tuple[str, str]] = []
    for symbol, exchange in pairs:
        if _compute_symbol_shard(symbol, shard_count) == shard_index:
            selected.append((symbol, exchange))
    return selected


def _merge_worker_chunk_dbs(
    *,
    worker_db_paths: list[Path],
    output_db_path: Path,
) -> int:
    _ensure_chunks_db_schema(output_db_path)
    total_merged = 0
    with _open_chunks_db(output_db_path) as conn:
        for idx, worker_db in enumerate(worker_db_paths):
            if not worker_db.exists():
                continue
            alias = f"w{idx}"
            conn.execute(f"ATTACH DATABASE ? AS {alias}", (str(worker_db),))
            try:
                try:
                    row = conn.execute(f"SELECT COUNT(*) FROM {alias}.chunks").fetchone()
                except sqlite3.OperationalError:
                    row = None
                worker_rows = int(row[0]) if row else 0
                if worker_rows <= 0:
                    continue

                conn.execute(
                    f"""
                    INSERT INTO chunks (
                        chunk_id,
                        stock_code,
                        year,
                        category,
                        source_file,
                        page_start,
                        page_end,
                        worker_id,
                        chunk_json
                    )
                    SELECT
                        chunk_id,
                        stock_code,
                        year,
                        category,
                        source_file,
                        page_start,
                        page_end,
                        worker_id,
                        chunk_json
                    FROM {alias}.chunks
                    WHERE 1=1
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        stock_code = excluded.stock_code,
                        year = excluded.year,
                        category = excluded.category,
                        source_file = excluded.source_file,
                        page_start = excluded.page_start,
                        page_end = excluded.page_end,
                        worker_id = excluded.worker_id,
                        chunk_json = excluded.chunk_json
                    """
                )
                total_merged += worker_rows
                conn.commit()
            finally:
                conn.execute(f"DETACH DATABASE {alias}")
    return total_merged


def _fetch_symbols_from_companies(
    *,
    exchange_filter: set[str],
    limit_symbols: int,
) -> list[tuple[str, str]]:
    host = settings.MYSQL_HOST
    port = int(settings.MYSQL_PORT)
    user = settings.MYSQL_USER
    password = settings.MYSQL_PASSWORD
    database = settings.MYSQL_DATABASE

    if not user or not database:
        raise RuntimeError("MYSQL_USER/MYSQL_DATABASE is missing in .env")

    conn = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id AS symbol, exchange
                FROM companies
                WHERE id IS NOT NULL AND TRIM(id) <> ''
                ORDER BY id ASC
                """
            )
            rows: list[dict[str, Any]] = cur.fetchall()
    finally:
        conn.close()

    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        exchange = str(row.get("exchange") or "").strip().upper()
        if not symbol:
            continue
        if exchange_filter and exchange not in exchange_filter:
            continue
        if symbol in seen:
            continue
        seen.add(symbol)
        pairs.append((symbol, exchange))

    if limit_symbols > 0:
        pairs = pairs[:limit_symbols]

    return pairs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run annual report RAG pipeline for all symbols from DB companies table"
    )

    parser.add_argument(
        "--cafef-years",
        type=int,
        default=5,
        help="Number of latest annual reports to collect per symbol",
    )
    parser.add_argument(
        "--exchange-filter",
        type=str,
        default="",
        help="Optional CSV exchanges filter (e.g. HOSE,HNX,UPCOM). Empty = all exchanges in companies table",
    )
    parser.add_argument(
        "--limit-symbols",
        type=int,
        default=0,
        help="Optional limit for first N symbols (0 = all)",
    )

    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--crawl-manifest", type=Path, default=DEFAULT_CRAWL_MANIFEST)
    parser.add_argument("--crawl-timeout", type=int, default=30)
    parser.add_argument(
        "--user-agent",
        type=str,
        default="FinFlow-RAG-Crawler/1.0 (+https://localhost)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Process per downloaded file immediately during crawl (recommended for large-scale runs)",
    )
    parser.add_argument(
        "--delete-pdf-after-chunk",
        action="store_true",
        help="Delete each downloaded PDF right after successful chunking in streaming mode",
    )
    parser.add_argument("--skip-crawl", action="store_true")
    parser.add_argument("--skip-index", action="store_true")

    parser.add_argument("--output-chunks", type=Path, default=DEFAULT_CHUNKS_JSON)
    parser.add_argument(
        "--output-chunks-db",
        type=Path,
        default=DEFAULT_CHUNKS_DB,
        help="SQLite checkpoint path for concurrent-safe chunk storage",
    )
    parser.add_argument(
        "--checkpoint-backend",
        type=str,
        default="sqlite",
        choices=["sqlite", "json"],
        help="Checkpoint backend for chunk persistence (sqlite recommended for concurrent runs)",
    )
    parser.add_argument("--output-index", type=Path, default=DEFAULT_INDEX_JSON)
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Start a clean run by clearing existing output checkpoint/output before streaming",
    )
    parser.add_argument(
        "--skip-final-export",
        action="store_true",
        help="Skip final JSON export and keep chunks only in checkpoint backend",
    )
    parser.add_argument(
        "--worker-mode",
        type=str,
        default="single",
        choices=["single", "exchange-partition", "symbol-shard-partition"],
        help="Execution mode: single process, exchange-partition, or symbol-shard-partition orchestration",
    )
    parser.add_argument(
        "--workers-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "rag" / "annual_reports" / "workers",
        help="Base directory for worker artifacts (exchange-partition or symbol-shard-partition)",
    )
    parser.add_argument(
        "--worker-log-mode",
        type=str,
        default="file",
        choices=["file", "inherit"],
        help="Worker stdout/stderr handling in partition modes: file (default) writes to workers/logs, inherit prints to parent terminal",
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        default="main",
        help="Logical worker id used in chunk checkpoint metadata",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Total symbol shards. In single mode only symbols where hash(symbol) %% shard_count == shard_index are processed.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Shard index used in single mode with --shard-count > 1.",
    )

    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--max-chars", type=int, default=2400)
    parser.add_argument("--overlap-chars", type=int, default=250)
    parser.add_argument("--ocr-image-only", action="store_true")
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
        "--ocr-language",
        type=str,
        default="vie+eng",
        help="Vision OCR language tokens (supports vie+eng or explicit BCP-47 like vi-VN,en-US)",
    )
    parser.add_argument("--ocr-dpi", type=int, default=200, help="Render DPI used for Vision OCR")
    parser.add_argument(
        "--ocr-garbled-page-ratio-threshold",
        type=float,
        default=0.30,
        help="When --ocr-fix-garbled is enabled, OCR all text pages if garbled-page ratio exceeds this threshold",
    )
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
        help="Document extraction backend for chunking",
    )
    parser.add_argument("--include-other", action="store_true")
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
        default="http://127.0.0.1:9090/v1",
        help="Base URL for local OpenAI-compatible API used for OCR text repair",
    )
    parser.add_argument(
        "--llm-repair-api-key",
        type=str,
        default="no-key-required",
        help="API key for local OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--llm-repair-model",
        type=str,
        default="mlx-community/gemma-4-e2b-it-4bit",
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
    parser.add_argument("--allow-empty-output", action="store_true")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if str(args.worker_mode) == "exchange-partition":
        return _run_exchange_partition(args)
    if str(args.worker_mode) == "symbol-shard-partition":
        return _run_symbol_shard_partition(args)

    raw_dir: Path = args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    before_pdf_paths = sorted(p for p in raw_dir.glob("*.pdf") if p.is_file())
    before_code_counter = _collect_code_counter_from_pdfs(before_pdf_paths)
    before_codes = set(before_code_counter.keys())

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

    chunks: list[dict[str, Any]] = []
    streaming_mode = bool(args.streaming and not args.skip_crawl)
    existing_chunk_ids: set[str] = set()
    checkpoint_backend = str(args.checkpoint_backend)

    if streaming_mode and args.reset_output:
        _cleanup_outputs_for_reset(
            output_chunks=args.output_chunks,
            output_chunks_db=args.output_chunks_db,
            checkpoint_backend=checkpoint_backend,
        )

    if checkpoint_backend == "sqlite":
        _ensure_chunks_db_schema(args.output_chunks_db)

    if streaming_mode:
        if checkpoint_backend == "sqlite":
            existing_count = _count_chunks_in_db(args.output_chunks_db)
            if existing_count:
                print(
                    "[STREAM][RESUME] "
                    f"loaded_existing_chunks={existing_count} "
                    f"checkpoint={args.output_chunks_db}"
                )
        elif args.output_chunks.exists():
            existing_chunks = _load_chunks(args.output_chunks)
            if existing_chunks:
                chunks.extend(existing_chunks)
                existing_chunk_ids.update(
                    str(item.get("chunk_id"))
                    for item in existing_chunks
                    if item.get("chunk_id") is not None
                )
                print(
                    "[STREAM][RESUME] "
                    f"loaded_existing_chunks={len(existing_chunks)} "
                    f"output={args.output_chunks}"
                )

    if not args.skip_crawl:
        exchange_filter = _parse_exchange_filter(str(args.exchange_filter))
        symbol_exchange_pairs = _fetch_symbols_from_companies(
            exchange_filter=exchange_filter,
            limit_symbols=max(0, int(args.limit_symbols)),
        )
        shard_count = max(1, int(args.shard_count))
        shard_index = int(args.shard_index)
        if shard_count > 1:
            if shard_index < 0 or shard_index >= shard_count:
                print(
                    f"[DB][WARN] invalid shard settings: shard_index={shard_index}, shard_count={shard_count}"
                )
                return 2
            symbol_exchange_pairs = _apply_symbol_shard_filter(
                symbol_exchange_pairs,
                shard_count=shard_count,
                shard_index=shard_index,
            )

        print(
            "[DB][SYMBOLS] "
            f"loaded={len(symbol_exchange_pairs)} "
            f"exchange_filter={sorted(exchange_filter) if exchange_filter else 'ALL'} "
            f"shard={shard_index}/{shard_count}"
        )

        if streaming_mode:
            print(
                "[STREAM] enabled=true "
                f"delete_pdf_after_chunk={bool(args.delete_pdf_after_chunk)} "
                "manifest_dedup_scope=current_run_only"
            )

        session = requests.Session()
        session.headers.update({"User-Agent": str(args.user_agent)})

        years = max(1, int(args.cafef_years))
        timeout_seconds = max(5, int(args.crawl_timeout))

        all_links: list[str] = []
        symbols_with_links = 0
        if streaming_mode:
            manifest = _load_manifest(args.crawl_manifest)
            manifest_items = manifest.get("items", [])
            if not isinstance(manifest_items, list):
                manifest_items = []
                manifest["items"] = manifest_items

            # Streaming mode deduplicates only in current run to avoid skipping files
            # that were downloaded in previous runs and then cleaned from raw_pdfs.
            known_hashes: set[str] = set()
            seen_urls: set[str] = set()

            crawl_summary = {
                "sources": 0,
                "discovered_pdf_links": 0,
                "downloaded": 0,
                "deduplicated": 0,
                "failed": 0,
            }

            for symbol, exchange in symbol_exchange_pairs:
                market = _exchange_to_market(exchange)
                links = _fetch_cafef_annual_links_for_symbol(
                    session,
                    symbol=symbol,
                    market=market,
                    years=years,
                    timeout_seconds=timeout_seconds,
                )
                if links:
                    symbols_with_links += 1
                all_links.extend(links)

                for url in links:
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    crawl_summary["sources"] += 1
                    crawl_summary["discovered_pdf_links"] += 1

                    status, saved_path, reason = _download_pdf(
                        session=session,
                        url=url,
                        raw_dir=raw_dir,
                        timeout_seconds=timeout_seconds,
                        known_hashes=known_hashes,
                        manifest_items=manifest_items,
                    )
                    if status == "downloaded":
                        crawl_summary["downloaded"] += 1
                        if saved_path is None:
                            continue

                        print(f"[CRAWL][OK] downloaded {url} -> {saved_path.name}")

                        try:
                            file_chunks = chunker.process_pdf(saved_path)
                            added = 0
                            new_file_chunks: list[dict[str, Any]] = []
                            for item in file_chunks:
                                if checkpoint_backend == "json":
                                    chunk_id = item.get("chunk_id")
                                    key = str(chunk_id) if chunk_id is not None else ""
                                    if key and key in existing_chunk_ids:
                                        continue
                                    if key:
                                        existing_chunk_ids.add(key)
                                    chunks.append(item)
                                new_file_chunks.append(item)
                                added += 1
                            print(
                                f"[STREAM][CHUNK] file={saved_path.name} chunks={len(file_chunks)} added={added}"
                            )
                            if checkpoint_backend == "sqlite":
                                saved = _write_chunks_checkpoint_sqlite(
                                    args.output_chunks_db,
                                    new_file_chunks,
                                    worker_id=str(args.worker_id),
                                )
                                print(
                                    "[STREAM][CHECKPOINT] "
                                    f"saved_chunks={saved} checkpoint={args.output_chunks_db}"
                                )
                            else:
                                _write_chunks_checkpoint(args.output_chunks, chunks)
                                print(
                                    "[STREAM][CHECKPOINT] "
                                    f"saved_chunks={len(chunks)} output={args.output_chunks}"
                                )
                            if args.delete_pdf_after_chunk:
                                try:
                                    saved_path.unlink()
                                    print(f"[STREAM][CLEAN] deleted {saved_path.name}")
                                except Exception as exc:
                                    print(f"[STREAM][WARN] cannot delete {saved_path.name}: {exc}")
                        except KeyboardInterrupt:
                            if checkpoint_backend == "sqlite":
                                pass
                            else:
                                _write_chunks_checkpoint(args.output_chunks, chunks)
                            print(
                                "[STREAM][STOP] interrupted_by_user=true "
                                f"saved_chunks={len(chunks) if checkpoint_backend == 'json' else _count_chunks_in_db(args.output_chunks_db)} "
                                f"checkpoint={'output_json=' + str(args.output_chunks) if checkpoint_backend == 'json' else 'db=' + str(args.output_chunks_db)}"
                            )
                            return 130
                        except Exception as exc:
                            print(f"[STREAM][WARN] chunk_failed file={saved_path.name}: {exc}")
                    elif status == "deduplicated":
                        crawl_summary["deduplicated"] += 1
                        print(f"[CRAWL][SKIP] duplicate content {url}")
                    else:
                        crawl_summary["failed"] += 1
                        print(f"[CRAWL][WARN] failed {url}: {reason}")

            _save_manifest(args.crawl_manifest, manifest)

            print(
                "[CAFEF][SUMMARY] "
                f"symbols={len(symbol_exchange_pairs)} "
                f"symbols_with_links={symbols_with_links} "
                f"links_total={len(_dedup_urls(all_links))}"
            )
            print(
                "[CRAWL][SUMMARY] "
                f"sources={crawl_summary['sources']} "
                f"discovered_pdf_links={crawl_summary['discovered_pdf_links']} "
                f"downloaded={crawl_summary['downloaded']} "
                f"deduplicated={crawl_summary['deduplicated']} "
                f"failed={crawl_summary['failed']}"
            )
        else:
            for symbol, exchange in symbol_exchange_pairs:
                market = _exchange_to_market(exchange)
                links = _fetch_cafef_annual_links_for_symbol(
                    session,
                    symbol=symbol,
                    market=market,
                    years=years,
                    timeout_seconds=timeout_seconds,
                )
                if links:
                    symbols_with_links += 1
                    all_links.extend(links)

            source_urls = _dedup_urls(all_links)
            print(
                "[CAFEF][SUMMARY] "
                f"symbols={len(symbol_exchange_pairs)} "
                f"symbols_with_links={symbols_with_links} "
                f"links_total={len(source_urls)}"
            )

            if source_urls:
                crawl_result = crawl_pdfs(
                    source_urls=source_urls,
                    raw_dir=raw_dir,
                    manifest_path=args.crawl_manifest,
                    timeout_seconds=timeout_seconds,
                    max_links_per_page=1,
                    user_agent=str(args.user_agent),
                )
                print(
                    "[CRAWL][SUMMARY] "
                    f"sources={crawl_result.source_urls} "
                    f"discovered_pdf_links={crawl_result.discovered_pdf_links} "
                    f"downloaded={crawl_result.downloaded} "
                    f"deduplicated={crawl_result.deduplicated} "
                    f"failed={crawl_result.failed}"
                )
            else:
                print("[CRAWL][WARN] no annual-report links resolved from DB symbols")

        after_pdf_paths = sorted(p for p in raw_dir.glob("*.pdf") if p.is_file())
        after_code_counter = _collect_code_counter_from_pdfs(after_pdf_paths)
        after_codes = set(after_code_counter.keys())
        new_codes = sorted(code for code in (after_codes - before_codes) if code != "UNKNOWN")
        known_codes_before = len([c for c in before_codes if c != "UNKNOWN"])
        known_codes_after = len([c for c in after_codes if c != "UNKNOWN"])
        unknown_file_count_after = int(after_code_counter.get("UNKNOWN", 0))
        print(
            "[CRAWL][CODES] "
            f"known_codes_before={known_codes_before} "
            f"known_codes_after={known_codes_after} "
            f"unknown_file_count={unknown_file_count_after} "
            f"new_codes={new_codes if new_codes else []} "
            f"known_code_file_counts={_format_known_code_counts(after_code_counter)}"
        )

    pdf_paths = sorted(p for p in raw_dir.glob("*.pdf") if p.is_file())
    raw_code_counter = _collect_code_counter_from_pdfs(pdf_paths)
    known_code_count = len([code for code in raw_code_counter.keys() if code != "UNKNOWN"])
    unknown_file_count = int(raw_code_counter.get("UNKNOWN", 0))
    print(
        "[RAW][CODES] "
        f"known_codes={known_code_count} "
        f"unknown_file_count={unknown_file_count} "
        f"known_code_file_counts={_format_known_code_counts(raw_code_counter)}"
    )

    if not streaming_mode:
        if not pdf_paths:
            print("[WARN] No PDF found in raw-dir after crawl.")
            return 1
        chunks = chunker.process_many(pdf_paths)

    if checkpoint_backend == "sqlite":
        if not streaming_mode:
            saved = _write_chunks_checkpoint_sqlite(
                args.output_chunks_db,
                chunks,
                worker_id=str(args.worker_id),
            )
            print(f"[CHUNK][CHECKPOINT] saved_chunks={saved} checkpoint={args.output_chunks_db}")
        chunks_for_output = _load_chunks_from_db(args.output_chunks_db)
    else:
        if not chunks and not args.allow_empty_output:
            print("[WARN] Chunking produced 0 chunks. Stop before writing outputs.")
            return 2
        _write_chunks_checkpoint(args.output_chunks, chunks)
        chunks_for_output = chunks

    ocr_summary = chunker.get_run_summary()
    print(
        "[CHUNK][OCR] "
        f"files={ocr_summary['files_processed']} "
        f"files_with_ocr={ocr_summary['files_with_ocr']} "
        f"chunks_dropped_garbled={ocr_summary['chunks_dropped_garbled']} "
        f"llm_repair_attempted={ocr_summary['llm_repair_attempted']} "
        f"llm_repair_succeeded={ocr_summary['llm_repair_succeeded']} "
        f"pages_total={ocr_summary['pages_total']} "
        f"pages_with_text={ocr_summary['pages_with_text']} "
        f"pages_with_ocr={ocr_summary['pages_with_ocr']} "
        f"ocr_rate_total={ocr_summary['ocr_rate_on_total_pages_pct']:.2f}% "
        f"ocr_rate_text={ocr_summary['ocr_rate_on_text_pages_pct']:.2f}%"
    )
    if not chunks_for_output and not args.allow_empty_output:
        print("[WARN] Chunking produced 0 chunks. Stop before writing outputs.")
        return 2

    if not args.skip_final_export:
        _write_chunks_checkpoint(args.output_chunks, chunks_for_output)
        print(f"[CHUNK][DONE] saved_chunks={len(chunks_for_output)} output={args.output_chunks}")
    else:
        print(
            "[CHUNK][DONE] "
            f"saved_chunks={len(chunks_for_output)} export_json_skipped=true "
            f"checkpoint_backend={checkpoint_backend}"
        )

    if args.skip_index:
        print("[INDEX] skipped by --skip-index")
        return 0

    index_data = _build_bm25_index(chunks_for_output)
    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    args.output_index.write_text(json.dumps(index_data, ensure_ascii=False), encoding="utf-8")
    print(
        "[INDEX][DONE] "
        f"documents={index_data['documents_count']} "
        f"vocab={len(index_data.get('idf', {}))} "
        f"output={args.output_index}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
