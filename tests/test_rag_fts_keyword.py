from __future__ import annotations

import json
import sqlite3


def _insert_chunk(
    conn: sqlite3.Connection,
    *,
    chunk_id: str,
    stock_code: str,
    year: int,
    title: str,
    text: str,
) -> None:
    payload = {
        "chunk_id": chunk_id,
        "stock_code": stock_code,
        "year": year,
        "subsection_title": title,
        "text": text,
        "page_start": 7,
        "page_end": 8,
        "source_file": f"{stock_code}_{year}.pdf",
        "category": "risk",
    }
    conn.execute(
        """
        INSERT INTO chunks (
            chunk_id, stock_code, year, category, source_file,
            page_start, page_end, worker_id, chunk_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            chunk_id,
            stock_code,
            year,
            "risk",
            f"{stock_code}_{year}.pdf",
            7,
            8,
            "test",
            json.dumps(payload, ensure_ascii=False),
        ),
    )


def _create_chunks_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE chunks (
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


def _create_fts_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id UNINDEXED,
            stock_code UNINDEXED,
            year UNINDEXED,
            title,
            text,
            category UNINDEXED,
            source_file UNINDEXED,
            page_start UNINDEXED,
            page_end UNINDEXED,
            tokenize = 'unicode61 remove_diacritics 2'
        )
        """
    )


def test_build_fts_query_uses_prefix_for_long_tokens() -> None:
    from app.infrastructure.rag_client import RagRetrievalService

    query = RagRetrievalService._build_fts_query('Rủi ro chiến lược "HPG"')

    assert '"rủi"*' in query
    assert '"ro"' in query
    assert '"chiến"*' in query
    assert '""' not in query


def test_keyword_search_uses_fts5_bm25_with_ticker_and_year_filters(tmp_path, monkeypatch) -> None:
    from app.core.config import settings

    db_path = tmp_path / "chunks.sqlite"
    with sqlite3.connect(db_path) as conn:
        _create_chunks_table(conn)
        _create_fts_table(conn)
        _insert_chunk(
            conn,
            chunk_id="HPG_2024_0001",
            stock_code="HPG",
            year=2024,
            title="Quản trị rủi ro",
            text="Công ty tăng cường quản trị rủi ro thị trường thép và rủi ro tỷ giá.",
        )
        _insert_chunk(
            conn,
            chunk_id="VCB_2024_0001",
            stock_code="VCB",
            year=2024,
            title="Rủi ro tín dụng",
            text="Ngân hàng quản lý rủi ro tín dụng và an toàn vốn.",
        )
        conn.execute(
            """
            INSERT INTO chunks_fts (
                chunk_id, stock_code, year, title, text, category,
                source_file, page_start, page_end
            )
            SELECT
                chunk_id,
                stock_code,
                year,
                json_extract(chunk_json, '$.subsection_title'),
                json_extract(chunk_json, '$.text'),
                category,
                source_file,
                page_start,
                page_end
            FROM chunks
            """
        )

    monkeypatch.setattr(settings, "CHAT_RAG_CHUNKS_DB", str(db_path))
    monkeypatch.setattr(settings, "CHAT_RAG_TOPK_KEYWORD", 10)

    from app.infrastructure.rag_client import RagRetrievalService

    service = RagRetrievalService()
    hits = service._keyword_search_sync("rủi ro thép", ticker="HPG", years=[2024])

    assert [hit["chunk_id"] for hit in hits] == ["HPG_2024_0001"]
    assert hits[0]["source"] == "keyword"
    assert hits[0]["score"] > 0


def test_keyword_search_falls_back_when_fts_table_is_missing(tmp_path, monkeypatch) -> None:
    from app.core.config import settings

    db_path = tmp_path / "chunks.sqlite"
    with sqlite3.connect(db_path) as conn:
        _create_chunks_table(conn)
        _insert_chunk(
            conn,
            chunk_id="HPG_2024_0001",
            stock_code="HPG",
            year=2024,
            title="Rủi ro",
            text="Rủi ro thị trường và rủi ro tỷ giá là trọng tâm quản trị.",
        )

    monkeypatch.setattr(settings, "CHAT_RAG_CHUNKS_DB", str(db_path))
    monkeypatch.setattr(settings, "CHAT_RAG_TOPK_KEYWORD", 10)

    from app.infrastructure.rag_client import RagRetrievalService

    service = RagRetrievalService()
    hits = service._keyword_search_sync("rủi ro", ticker="HPG", years=[2024])

    assert [hit["chunk_id"] for hit in hits] == ["HPG_2024_0001"]
    assert hits[0]["score"] >= 2
