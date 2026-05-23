from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)
DEFAULT_CHUNKS_DB = PROJECT_ROOT / "artifacts" / "rag" / "annual_reports" / "chunks" / "annual_reports_chunks.sqlite"
DEFAULT_EMBEDDINGS_DB = (
    PROJECT_ROOT / "artifacts" / "rag" / "annual_reports" / "embeddings" / "annual_reports_embeddings.sqlite"
)
DEFAULT_EMBED_MODEL = "voyage-3.5-lite"
DEFAULT_BASE_URL = os.getenv("VOYAGE_EMBED_BASE_URL", "https://api.voyageai.com/v1")
DEFAULT_API_KEY = os.getenv("VOYAGE_API_KEY", "")
DEFAULT_QDRANT_URL = os.getenv("CHAT_QDRANT_URL", os.getenv("QDRANT_URL", "http://127.0.0.1:6333"))
DEFAULT_QDRANT_COLLECTION = os.getenv(
    "CHAT_QDRANT_COLLECTION",
    os.getenv("QDRANT_COLLECTION", "annual_report_chunks_voyage_3_5_lite"),
)
DEFAULT_QDRANT_DISTANCE = os.getenv("QDRANT_DISTANCE", "cosine")


def _open_sqlite(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=60)
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _ensure_embeddings_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id TEXT NOT NULL,
            model TEXT NOT NULL,
            dim INTEGER NOT NULL,
            stock_code TEXT,
            year INTEGER,
            category TEXT,
            source_file TEXT,
            page_start INTEGER,
            page_end INTEGER,
            text_sha256 TEXT NOT NULL,
            vector_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (chunk_id, model)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_stock_year ON embeddings(stock_code, year)")
    conn.commit()


def _build_input_text(title: str, text: str, max_input_chars: int) -> str:
    merged = f"{title.strip()}\n{text.strip()}".strip()
    if not merged:
        merged = text.strip() or title.strip()
    if max_input_chars > 0 and len(merged) > max_input_chars:
        return merged[:max_input_chars]
    return merged


def _create_client(base_url: str, api_key: str, timeout_seconds: int) -> tuple[requests.Session, str, str, int]:
    session = requests.Session()
    key = str(api_key or "").strip()
    if key:
        session.headers.update({"Authorization": f"Bearer {key}"})
    session.headers.update({"Content-Type": "application/json"})
    return session, base_url.rstrip("/"), key, max(10, int(timeout_seconds))


def _embed_batch_with_retry(
    client: tuple[requests.Session, str, str, int],
    *,
    model: str,
    inputs: list[str],
    input_type: str,
    max_retries: int,
    retry_sleep_seconds: float,
) -> list[list[float]]:
    if not inputs:
        return []

    last_exc: Exception | None = None
    attempts = max(1, int(max_retries))
    for attempt in range(1, attempts + 1):
        try:
            session, base_url, _api_key, timeout_seconds = client
            payload: dict[str, Any] = {"model": model, "input": inputs}
            if input_type:
                payload["input_type"] = input_type
            response = session.post(
                f"{base_url}/embeddings",
                json=payload,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data")
            if not isinstance(data, list):
                raise RuntimeError(f"Invalid embeddings response schema: {payload}")
            rows = sorted(
                [item for item in data if isinstance(item, dict)],
                key=lambda item: int(item.get("index", 0)),
            )
            vectors = [list(item.get("embedding") or []) for item in rows]
            if len(vectors) != len(inputs):
                raise RuntimeError(
                    f"Embedding API returned mismatched batch size. expected={len(inputs)} actual={len(vectors)}"
                )
            if any(not vector for vector in vectors):
                raise RuntimeError("Embedding API returned empty vector in batch")
            return vectors
        except requests.RequestException as exc:  # pragma: no cover
            last_exc = RuntimeError(f"HTTP error calling embeddings endpoint: {exc}")
            if attempt >= attempts:
                break
            sleep_seconds = retry_sleep_seconds * attempt
            print(
                f"[EMBED][WARN] batch_failed attempt={attempt}/{attempts} "
                f"error={last_exc}; retry_in={sleep_seconds:.1f}s"
            )
            time.sleep(sleep_seconds)
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            if attempt >= attempts:
                break
            sleep_seconds = retry_sleep_seconds * attempt
            print(f"[EMBED][WARN] batch_failed attempt={attempt}/{attempts} error={exc}; retry_in={sleep_seconds:.1f}s")
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Embedding request failed after {attempts} attempts: {last_exc}")


def _create_qdrant_client(url: str, api_key: str):
    try:
        from qdrant_client import QdrantClient  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "qdrant-client is required for --qdrant-upsert. Install in venv with: ./venv/bin/pip install qdrant-client"
        ) from exc
    key = api_key.strip()
    return QdrantClient(url=url.rstrip("/"), api_key=key or None, timeout=60.0)


def _normalize_distance(distance: str):
    value = str(distance or "").strip().lower()
    try:
        from qdrant_client.http.models import Distance  # type: ignore
    except Exception:  # pragma: no cover
        return value, None
    mapping = {
        "cosine": Distance.COSINE,
        "dot": Distance.DOT,
        "euclid": Distance.EUCLID,
        "manhattan": Distance.MANHATTAN,
    }
    return value, mapping.get(value, Distance.COSINE)


def _ensure_qdrant_collection(
    client: Any,
    *,
    collection: str,
    vector_dim: int,
    distance: str,
    recreate: bool,
) -> None:
    try:
        from qdrant_client.http.models import VectorParams  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Cannot import qdrant-client models. Reinstall qdrant-client inside venv."
        ) from exc

    distance_key, distance_enum = _normalize_distance(distance)
    if distance_enum is None:
        raise RuntimeError(f"Unsupported Qdrant distance='{distance_key}'. Use: cosine|dot|euclid|manhattan")

    exists = False
    if hasattr(client, "collection_exists"):
        try:
            exists = bool(client.collection_exists(collection_name=collection))
        except Exception:
            exists = False
    else:
        try:
            client.get_collection(collection_name=collection)
            exists = True
        except Exception:
            exists = False

    if exists and recreate:
        client.delete_collection(collection_name=collection)
        exists = False
        print(f"[QDRANT][RESET] deleted_collection={collection}")

    if not exists:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=int(vector_dim), distance=distance_enum),
        )
        print(
            "[QDRANT][CREATE] "
            f"collection={collection} dim={vector_dim} distance={distance_key}"
        )
        return

    info = client.get_collection(collection_name=collection)
    config = getattr(info, "config", None)
    params = getattr(config, "params", None)
    vectors = getattr(params, "vectors", None)

    existing_dim: int | None = None
    if hasattr(vectors, "size"):
        existing_dim = int(getattr(vectors, "size"))
    elif isinstance(vectors, dict) and vectors:
        first = next(iter(vectors.values()))
        if hasattr(first, "size"):
            existing_dim = int(getattr(first, "size"))

    if existing_dim is not None and int(existing_dim) != int(vector_dim):
        raise RuntimeError(
            f"Qdrant collection dim mismatch: collection={collection} has={existing_dim}, embedding_dim={vector_dim}. "
            "Use --qdrant-recreate-collection or another --qdrant-collection."
        )


def _upsert_qdrant_batch(
    client: Any,
    *,
    collection: str,
    rows: list[tuple[Any, ...]],
    vectors: list[list[float]],
    model: str,
    text_sha256_values: list[str],
) -> int:
    if not rows:
        return 0
    try:
        from qdrant_client.http.models import PointStruct  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Cannot import PointStruct from qdrant-client. Reinstall qdrant-client inside venv."
        ) from exc

    points = []
    for row, vector, text_sha256 in zip(rows, vectors, text_sha256_values):
        chunk_id = str(row[0] or "").strip()
        if not chunk_id:
            continue
        # Qdrant point id must be uint64 or UUID. Use deterministic UUID5 for stable upserts.
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"finflow://annual-report-chunk/{chunk_id}"))
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "qdrant_point_id": point_id,
                    "chunk_id": chunk_id,
                    "stock_code": str(row[1] or ""),
                    "year": int(row[2] or 0),
                    "category": str(row[3] or ""),
                    "source_file": str(row[4] or ""),
                    "page_start": int(row[5] or 0),
                    "page_end": int(row[6] or 0),
                    "embed_model": model,
                    "text_sha256": text_sha256,
                },
            )
        )
    if not points:
        return 0
    client.upsert(collection_name=collection, points=points, wait=True)
    return len(points)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Voyage embeddings for annual-report chunks from SQLite checkpoint")
    parser.add_argument("--chunks-db", type=Path, default=DEFAULT_CHUNKS_DB)
    parser.add_argument("--output-db", type=Path, default=DEFAULT_EMBEDDINGS_DB)
    parser.add_argument("--embed-base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--embed-api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--embed-model", type=str, default=os.getenv("VOYAGE_EMBED_MODEL", DEFAULT_EMBED_MODEL))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-input-chars", type=int, default=3500)
    parser.add_argument(
        "--target-year",
        type=int,
        default=0,
        help="Only embed chunks for this report year. Default 0 scans all chunks.",
    )
    parser.add_argument(
        "--embed-input-type",
        type=str,
        default=os.getenv("VOYAGE_EMBED_INPUT_TYPE", "document"),
        choices=["", "query", "document"],
        help="Voyage embedding input_type. Use document when indexing chunks.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of pending chunks to embed (0 = all)")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0)
    parser.add_argument("--sleep-ms", type=int, default=0, help="Optional sleep between batches to reduce thermal pressure")
    parser.add_argument("--qdrant-upsert", action="store_true", help="Upsert embeddings into Qdrant in the same run")
    parser.add_argument("--qdrant-url", type=str, default=DEFAULT_QDRANT_URL)
    parser.add_argument("--qdrant-api-key", type=str, default=os.getenv("CHAT_QDRANT_API_KEY", os.getenv("QDRANT_API_KEY", "")))
    parser.add_argument("--qdrant-collection", type=str, default=DEFAULT_QDRANT_COLLECTION)
    parser.add_argument(
        "--qdrant-distance",
        type=str,
        default=DEFAULT_QDRANT_DISTANCE,
        choices=["cosine", "dot", "euclid", "manhattan"],
    )
    parser.add_argument(
        "--qdrant-recreate-collection",
        action="store_true",
        help="Delete and recreate collection before upsert (dangerous, collection data will be lost)",
    )
    parser.add_argument(
        "--rebuild-model",
        action="store_true",
        help="Delete existing embeddings for --embed-model in output DB before recomputing",
    )
    return parser


def _chunk_rows_query(target_year: int = 0) -> str:
    year_filter = "AND c.year = ?" if int(target_year or 0) > 0 else ""
    return f"""
        SELECT
            c.chunk_id,
            c.stock_code,
            c.year,
            c.category,
            c.source_file,
            c.page_start,
            c.page_end,
            COALESCE(json_extract(c.chunk_json, '$.subsection_title'), ''),
            COALESCE(json_extract(c.chunk_json, '$.text'), ''),
            e.text_sha256
        FROM chunks_db.chunks c
        LEFT JOIN embeddings e
            ON e.chunk_id = c.chunk_id
           AND e.model = ?
        WHERE c.chunk_id > ?
        {year_filter}
        ORDER BY c.chunk_id ASC
        LIMIT ?
    """


def _collect_pending_rows(
    conn: sqlite3.Connection,
    *,
    model: str,
    last_chunk_id: str,
    wanted: int,
    scan_limit: int,
    max_input_chars: int,
    target_year: int = 0,
) -> tuple[list[tuple[Any, ...]], list[str], list[str], str, bool]:
    pending_rows: list[tuple[Any, ...]] = []
    pending_inputs: list[str] = []
    pending_hashes: list[str] = []
    cursor_after = last_chunk_id
    exhausted = False

    while len(pending_rows) < wanted:
        params: tuple[Any, ...]
        if int(target_year or 0) > 0:
            params = (model, cursor_after, int(target_year), scan_limit)
        else:
            params = (model, cursor_after, scan_limit)
        rows = conn.execute(_chunk_rows_query(target_year), params).fetchall()
        if not rows:
            exhausted = True
            break

        for row in rows:
            cursor_after = str(row[0] or cursor_after)
            title = str(row[7] or "")
            text = str(row[8] or "")
            payload = _build_input_text(title, text, max_input_chars=max_input_chars)
            if not payload:
                payload = str(row[0] or "")
            text_sha256 = hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()
            existing_sha = str(row[9] or "")
            if existing_sha == text_sha256:
                continue
            pending_rows.append(row[:9])
            pending_inputs.append(payload)
            pending_hashes.append(text_sha256)
            if len(pending_rows) >= wanted:
                break

    return pending_rows, pending_inputs, pending_hashes, cursor_after, exhausted


def _count_pending_rows(
    conn: sqlite3.Connection,
    *,
    model: str,
    max_input_chars: int,
    limit: int,
    target_year: int = 0,
) -> int:
    total = 0
    last_chunk_id = ""
    while True:
        params: tuple[Any, ...]
        if int(target_year or 0) > 0:
            params = (model, last_chunk_id, int(target_year), 1000)
        else:
            params = (model, last_chunk_id, 1000)
        rows = conn.execute(_chunk_rows_query(target_year), params).fetchall()
        if not rows:
            break
        last_chunk_id = str(rows[-1][0] or last_chunk_id)
        for row in rows:
            title = str(row[7] or "")
            text = str(row[8] or "")
            payload = _build_input_text(title, text, max_input_chars=max_input_chars)
            if not payload:
                payload = str(row[0] or "")
            text_sha256 = hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()
            if str(row[9] or "") != text_sha256:
                total += 1
                if limit > 0 and total >= limit:
                    return limit
    return total


def main() -> int:
    args = _build_parser().parse_args()
    chunks_db: Path = args.chunks_db
    output_db: Path = args.output_db
    model = str(args.embed_model).strip()
    batch_size = max(1, int(args.batch_size))
    max_input_chars = max(0, int(args.max_input_chars))
    embed_input_type = str(args.embed_input_type or "").strip().lower()
    sleep_ms = max(0, int(args.sleep_ms))
    target_year = max(0, int(args.target_year or 0))

    if not chunks_db.exists():
        print(f"[EMBED][ERR] chunks_db_not_found={chunks_db}")
        return 2
    if not model:
        print("[EMBED][ERR] --embed-model is empty")
        return 2

    client = _create_client(
        base_url=str(args.embed_base_url),
        api_key=str(args.embed_api_key),
        timeout_seconds=int(args.timeout_seconds),
    )
    qdrant_client: Any | None = None
    qdrant_collection = str(args.qdrant_collection).strip()
    qdrant_upserted = 0
    qdrant_collection_ready = False
    if args.qdrant_upsert:
        if not qdrant_collection:
            print("[QDRANT][ERR] --qdrant-collection is empty")
            return 2
        qdrant_client = _create_qdrant_client(
            url=str(args.qdrant_url),
            api_key=str(args.qdrant_api_key),
        )
        print(
            "[QDRANT][START] "
            f"url={str(args.qdrant_url)} collection={qdrant_collection} distance={str(args.qdrant_distance)}"
        )

    with _open_sqlite(output_db) as conn:
        _ensure_embeddings_schema(conn)
        conn.execute("ATTACH DATABASE ? AS chunks_db", (str(chunks_db),))
        try:
            if target_year > 0:
                row = conn.execute("SELECT COUNT(*) FROM chunks_db.chunks WHERE year = ?", (target_year,)).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM chunks_db.chunks").fetchone()
            total_chunks = int(row[0]) if row else 0

            if args.rebuild_model:
                if target_year > 0:
                    conn.execute("DELETE FROM embeddings WHERE model = ? AND year = ?", (model, target_year))
                else:
                    conn.execute("DELETE FROM embeddings WHERE model = ?", (model,))
                conn.commit()
                print(f"[EMBED][RESET] removed_existing_model_rows model={model} target_year={target_year or 'all'}")

            if target_year > 0:
                existing_row = conn.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE model = ? AND year = ?",
                    (model, target_year),
                ).fetchone()
            else:
                existing_row = conn.execute("SELECT COUNT(*) FROM embeddings WHERE model = ?", (model,)).fetchone()
            existing_for_model = int(existing_row[0]) if existing_row else 0

            pending_total = _count_pending_rows(
                conn,
                model=model,
                max_input_chars=max_input_chars,
                limit=max(0, int(args.limit)),
                target_year=target_year,
            )

            print(
                "[EMBED][START] "
                f"model={model} target_year={target_year or 'all'} total_chunks={total_chunks} existing_for_model={existing_for_model} pending={pending_total} "
                f"batch_size={batch_size} "
                f"embed_base_url={str(args.embed_base_url)} output_db={output_db}"
            )
            if pending_total <= 0:
                print("[EMBED][DONE] no pending chunks")
                return 0

            inserted = 0
            started_at = time.time()
            last_chunk_id = ""
            limit_remaining: int | None = int(args.limit) if int(args.limit) > 0 else None

            while True:
                if limit_remaining is not None and limit_remaining <= 0:
                    break

                current_limit = batch_size
                if limit_remaining is not None:
                    current_limit = min(current_limit, limit_remaining)

                rows, inputs, text_sha256_values, last_chunk_id, exhausted = _collect_pending_rows(
                    conn,
                    model=model,
                    last_chunk_id=last_chunk_id,
                    wanted=current_limit,
                    scan_limit=max(1000, current_limit * 10),
                    max_input_chars=max_input_chars,
                    target_year=target_year,
                )

                if not rows:
                    if exhausted:
                        break
                    break

                vectors = _embed_batch_with_retry(
                    client,
                    model=model,
                    inputs=inputs,
                    input_type=embed_input_type,
                    max_retries=max(1, int(args.max_retries)),
                    retry_sleep_seconds=max(0.5, float(args.retry_sleep_seconds)),
                )

                records: list[tuple[Any, ...]] = []
                qdrant_rows: list[tuple[Any, ...]] = []
                qdrant_vectors: list[list[float]] = []
                for row, text_sha256, vector in zip(rows, text_sha256_values, vectors):
                    chunk_id = str(row[0] or "").strip()
                    if not chunk_id:
                        continue
                    records.append(
                        (
                            chunk_id,
                            model,
                            int(len(vector)),
                            str(row[1] or ""),
                            int(row[2] or 0),
                            str(row[3] or ""),
                            str(row[4] or ""),
                            int(row[5] or 0),
                            int(row[6] or 0),
                            text_sha256,
                            json.dumps(vector, ensure_ascii=False, separators=(",", ":")),
                        )
                    )
                    qdrant_rows.append(row)
                    qdrant_vectors.append(vector)

                conn.executemany(
                    """
                    INSERT INTO embeddings (
                        chunk_id,
                        model,
                        dim,
                        stock_code,
                        year,
                        category,
                        source_file,
                        page_start,
                        page_end,
                        text_sha256,
                        vector_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id, model) DO UPDATE SET
                        dim = excluded.dim,
                        stock_code = excluded.stock_code,
                        year = excluded.year,
                        category = excluded.category,
                        source_file = excluded.source_file,
                        page_start = excluded.page_start,
                        page_end = excluded.page_end,
                        text_sha256 = excluded.text_sha256,
                        vector_json = excluded.vector_json,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    records,
                )
                conn.commit()

                if qdrant_client is not None:
                    if not qdrant_collection_ready and vectors:
                        _ensure_qdrant_collection(
                            qdrant_client,
                            collection=qdrant_collection,
                            vector_dim=len(vectors[0]),
                            distance=str(args.qdrant_distance),
                            recreate=bool(args.qdrant_recreate_collection),
                        )
                        qdrant_collection_ready = True
                    qdrant_batch = _upsert_qdrant_batch(
                        qdrant_client,
                        collection=qdrant_collection,
                        rows=qdrant_rows,
                        vectors=qdrant_vectors,
                        model=model,
                        text_sha256_values=text_sha256_values,
                    )
                    qdrant_upserted += qdrant_batch

                batch_inserted = len(records)
                inserted += batch_inserted
                if limit_remaining is not None:
                    limit_remaining = max(0, limit_remaining - batch_inserted)

                elapsed = max(0.001, time.time() - started_at)
                speed = inserted / elapsed
                remain = max(0, pending_total - inserted)
                eta = (remain / speed) if speed > 0 else 0.0
                print(
                    "[EMBED][PROGRESS] "
                    f"inserted={inserted}/{pending_total} "
                    f"batch={batch_inserted} "
                    f"speed={speed:.2f}_chunks_per_sec "
                    f"eta_sec={eta:.1f} "
                    f"qdrant_upserted={qdrant_upserted}"
                )

                if sleep_ms > 0:
                    time.sleep(sleep_ms / 1000.0)

                # Reduce long-run memory pressure by dropping large batch objects promptly.
                del vectors
                del records
                del qdrant_rows
                del qdrant_vectors
                del text_sha256_values
                del rows
                del inputs

            if target_year > 0:
                done_row = conn.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE model = ? AND year = ?",
                    (model, target_year),
                ).fetchone()
            else:
                done_row = conn.execute("SELECT COUNT(*) FROM embeddings WHERE model = ?", (model,)).fetchone()
            total_for_model = int(done_row[0]) if done_row else 0
            print(
                "[EMBED][DONE] "
                f"model={model} newly_inserted={inserted} total_for_model={total_for_model} output_db={output_db}"
            )
            if qdrant_client is not None:
                print(
                    "[QDRANT][DONE] "
                    f"collection={qdrant_collection} upserted_points={qdrant_upserted}"
                )
            return 0
        finally:
            conn.execute("DETACH DATABASE chunks_db")


if __name__ == "__main__":
    raise SystemExit(main())
