from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)
DEFAULT_CHUNKS_DB = PROJECT_ROOT / "artifacts" / "rag" / "annual_reports" / "chunks" / "annual_reports_chunks.sqlite"
DEFAULT_EMBEDDINGS_DB = (
    PROJECT_ROOT / "artifacts" / "rag" / "annual_reports" / "embeddings" / "annual_reports_embeddings.sqlite"
)
DEFAULT_EMBED_MODEL = "mlx-community/bge-m3-mlx-fp16"
DEFAULT_BASE_URL = os.getenv("LOCAL_EMBEDDING_BASE_URL", os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:9090/v1"))
DEFAULT_API_KEY = os.getenv("LOCAL_EMBEDDING_API_KEY", os.getenv("LOCAL_LLM_API_KEY", "no-key-required"))
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
DEFAULT_QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "annual_report_chunks_bge_m3")
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


def _create_local_mlx_embedder(
    model_ref: str,
    *,
    max_tokens: int,
    clear_cache_every: int = 0,
) -> Callable[[list[str]], list[list[float]]]:
    try:
        import mlx.core as mx  # type: ignore
        from mlx_embeddings.utils import load  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Local MLX embedding mode requires mlx-embeddings. Install in venv: ./venv/bin/pip install mlx-embeddings==0.1.0"
        ) from exc

    model, tokenizer = load(model_ref)
    call_count = 0

    def _embed(texts: list[str]) -> list[list[float]]:
        nonlocal call_count
        if not texts:
            return []
        inputs = tokenizer.batch_encode_plus(
            texts,
            return_tensors="mlx",
            padding="max_length",
            truncation=True,
            max_length=max(32, int(max_tokens)),
        )
        outputs = model(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        values = getattr(outputs, "text_embeds", None)
        if values is None:
            values = getattr(outputs, "pooler_output", None)
        if values is None:
            raise RuntimeError("MLX embedding output missing text_embeds/pooler_output")
        mx.eval(values)
        rows = values.tolist() if hasattr(values, "tolist") else []
        del values
        del outputs
        del inputs

        vectors: list[list[float]] = []
        for row in rows:
            vectors.append([float(x) for x in row])
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"MLX local embedding returned mismatched batch size. expected={len(texts)} actual={len(vectors)}"
            )
        call_count += 1
        if clear_cache_every > 0 and call_count % clear_cache_every == 0:
            try:
                clear_fn = getattr(mx, "clear_cache", None)
                if clear_fn is None:
                    metal = getattr(mx, "metal", None)
                    clear_fn = getattr(metal, "clear_cache", None) if metal is not None else None
                if callable(clear_fn):
                    clear_fn()
            except Exception:
                pass
            gc.collect()
        return vectors

    return _embed


def _embed_batch_with_retry(
    client: tuple[requests.Session, str, str, int],
    *,
    model: str,
    inputs: list[str],
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
            response = session.post(
                f"{base_url}/embeddings",
                json={"model": model, "input": inputs},
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


def _embed_batch_local(
    local_embedder: Callable[[list[str]], list[list[float]]],
    *,
    inputs: list[str],
) -> list[list[float]]:
    vectors = local_embedder(inputs)
    if len(vectors) != len(inputs):
        raise RuntimeError(
            f"Local embedding returned mismatched batch size. expected={len(inputs)} actual={len(vectors)}"
        )
    return vectors


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
    parser = argparse.ArgumentParser(description="Build MLX embeddings for annual-report chunks from SQLite checkpoint")
    parser.add_argument("--chunks-db", type=Path, default=DEFAULT_CHUNKS_DB)
    parser.add_argument("--output-db", type=Path, default=DEFAULT_EMBEDDINGS_DB)
    parser.add_argument("--embed-base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--embed-api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--embed-model", type=str, default=os.getenv("LOCAL_EMBEDDING_MODEL", DEFAULT_EMBED_MODEL))
    parser.add_argument(
        "--embed-mode",
        type=str,
        default=os.getenv("LOCAL_EMBEDDING_MODE", "http"),
        choices=["http", "mlx-local"],
        help="Embedding execution mode: http (OpenAI-compatible endpoint) or mlx-local (direct local MLX inference)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-input-chars", type=int, default=3500)
    parser.add_argument(
        "--mlx-max-tokens",
        type=int,
        default=512,
        help="Max tokens per text for mlx-local mode tokenizer truncation",
    )
    parser.add_argument(
        "--mlx-clear-cache-every",
        type=int,
        default=20,
        help="In mlx-local mode, clear MLX cache and trigger GC every N batches (0 = disable)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of pending chunks to embed (0 = all)")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0)
    parser.add_argument("--sleep-ms", type=int, default=0, help="Optional sleep between batches to reduce thermal pressure")
    parser.add_argument("--qdrant-upsert", action="store_true", help="Upsert embeddings into Qdrant in the same run")
    parser.add_argument("--qdrant-url", type=str, default=DEFAULT_QDRANT_URL)
    parser.add_argument("--qdrant-api-key", type=str, default=os.getenv("QDRANT_API_KEY", ""))
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


def main() -> int:
    args = _build_parser().parse_args()
    chunks_db: Path = args.chunks_db
    output_db: Path = args.output_db
    model = str(args.embed_model).strip()
    batch_size = max(1, int(args.batch_size))
    max_input_chars = max(0, int(args.max_input_chars))
    sleep_ms = max(0, int(args.sleep_ms))

    if not chunks_db.exists():
        print(f"[EMBED][ERR] chunks_db_not_found={chunks_db}")
        return 2
    if not model:
        print("[EMBED][ERR] --embed-model is empty")
        return 2

    embed_mode = str(args.embed_mode).strip().lower()
    client: tuple[requests.Session, str, str, int] | None = None
    local_embedder: Callable[[list[str]], list[list[float]]] | None = None
    if embed_mode == "http":
        client = _create_client(
            base_url=str(args.embed_base_url),
            api_key=str(args.embed_api_key),
            timeout_seconds=int(args.timeout_seconds),
        )
    elif embed_mode == "mlx-local":
        local_embedder = _create_local_mlx_embedder(
            model,
            max_tokens=max(32, int(args.mlx_max_tokens)),
            clear_cache_every=max(0, int(args.mlx_clear_cache_every)),
        )
    else:
        print(f"[EMBED][ERR] unsupported embed_mode={embed_mode}")
        return 2
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
            row = conn.execute("SELECT COUNT(*) FROM chunks_db.chunks").fetchone()
            total_chunks = int(row[0]) if row else 0

            if args.rebuild_model:
                conn.execute("DELETE FROM embeddings WHERE model = ?", (model,))
                conn.commit()
                print(f"[EMBED][RESET] removed_existing_model_rows model={model}")

            existing_row = conn.execute("SELECT COUNT(*) FROM embeddings WHERE model = ?", (model,)).fetchone()
            existing_for_model = int(existing_row[0]) if existing_row else 0

            pending_sql = """
                SELECT COUNT(*)
                FROM chunks_db.chunks c
                LEFT JOIN embeddings e
                    ON e.chunk_id = c.chunk_id
                   AND e.model = ?
                WHERE e.chunk_id IS NULL
            """
            pending_row = conn.execute(pending_sql, (model,)).fetchone()
            pending_total = int(pending_row[0]) if pending_row else 0
            if int(args.limit) > 0:
                pending_total = min(pending_total, int(args.limit))

            print(
                "[EMBED][START] "
                f"model={model} total_chunks={total_chunks} existing_for_model={existing_for_model} pending={pending_total} "
                f"batch_size={batch_size} embed_mode={embed_mode} "
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

                rows = conn.execute(
                    """
                    SELECT
                        c.chunk_id,
                        c.stock_code,
                        c.year,
                        c.category,
                        c.source_file,
                        c.page_start,
                        c.page_end,
                        COALESCE(json_extract(c.chunk_json, '$.subsection_title'), ''),
                        COALESCE(json_extract(c.chunk_json, '$.text'), '')
                    FROM chunks_db.chunks c
                    LEFT JOIN embeddings e
                        ON e.chunk_id = c.chunk_id
                       AND e.model = ?
                    WHERE c.chunk_id > ?
                      AND e.chunk_id IS NULL
                    ORDER BY c.chunk_id ASC
                    LIMIT ?
                    """,
                    (model, last_chunk_id, current_limit),
                ).fetchall()

                if not rows:
                    break

                last_chunk_id = str(rows[-1][0] or last_chunk_id)

                inputs: list[str] = []
                for row in rows:
                    title = str(row[7] or "")
                    text = str(row[8] or "")
                    payload = _build_input_text(title, text, max_input_chars=max_input_chars)
                    if not payload:
                        payload = str(row[0] or "")
                    inputs.append(payload)

                if embed_mode == "mlx-local":
                    if local_embedder is None:
                        raise RuntimeError("Local embedder is not initialized")
                    vectors = _embed_batch_local(
                        local_embedder,
                        inputs=inputs,
                    )
                else:
                    if client is None:
                        raise RuntimeError("HTTP embedding client is not initialized")
                    vectors = _embed_batch_with_retry(
                        client,
                        model=model,
                        inputs=inputs,
                        max_retries=max(1, int(args.max_retries)),
                        retry_sleep_seconds=max(0.5, float(args.retry_sleep_seconds)),
                    )

                records: list[tuple[Any, ...]] = []
                qdrant_rows: list[tuple[Any, ...]] = []
                qdrant_vectors: list[list[float]] = []
                text_sha256_values: list[str] = []
                for row, payload, vector in zip(rows, inputs, vectors):
                    chunk_id = str(row[0] or "").strip()
                    if not chunk_id:
                        continue
                    text_sha256 = hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()
                    text_sha256_values.append(text_sha256)
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
