from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import sqlite3
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from app.core.config import settings
from app.infrastructure.rag_client import RagRetrievalService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "rag_eval" / "ragas"
PRODUCTION_RAG_CONFIG = {
    "mode": "optimized",
    "embedding_model": "voyage-3.5-lite",
    "rerank_model": "rerank-2.5-lite",
    "qdrant_collection": "annual_report_chunks_voyage_3_5_lite",
    "vector_topk": 50,
    "keyword_topk": 50,
    "rerank_candidates": 64,
    "final_topk": 5,
    "context_max_chars": 1800,
}
VECTOR_BASELINE_CONFIG = {
    "mode": "vector",
    "embedding_model": "voyage-3.5-lite",
    "rerank_model": None,
    "qdrant_collection": "annual_report_chunks_voyage_3_5_lite",
    "vector_topk": 5,
    "keyword_topk": 0,
    "rerank_candidates": 0,
    "final_topk": 5,
    "context_max_chars": 1800,
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _clean_text(value: Any, max_chars: int = 0) -> str:
    text = " ".join(str(value or "").split())
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rsplit(" ", 1)[0].strip()
    return text


def _rag_config_for_mode(mode: str) -> dict[str, Any]:
    if mode == "optimized":
        return dict(PRODUCTION_RAG_CONFIG)
    if mode == "vector":
        return dict(VECTOR_BASELINE_CONFIG)
    raise ValueError(f"Unsupported RAG mode: {mode}")


def _ensure_rag_config(mode: str) -> dict[str, Any]:
    config = _rag_config_for_mode(mode)
    settings.VOYAGE_EMBED_MODEL = config["embedding_model"]
    settings.VOYAGE_EMBED_INPUT_TYPE = "query"
    settings.VOYAGE_RERANK_MODEL = config["rerank_model"] or "rerank-2.5-lite"
    settings.CHAT_QDRANT_COLLECTION = config["qdrant_collection"]
    settings.CHAT_RAG_TOPK_VECTOR = config["vector_topk"]
    settings.CHAT_RAG_TOPK_KEYWORD = max(1, int(config["keyword_topk"] or 1))
    settings.CHAT_RAG_RERANK_CANDIDATES = max(0, int(config["rerank_candidates"] or 0))
    settings.CHAT_RAG_TOPK_FINAL = config["final_topk"]
    settings.CHAT_RAG_CONTEXT_MAX_CHARS = config["context_max_chars"]
    settings.CHAT_RAG_RERANK_ENABLED = mode == "optimized"
    return config


def _load_source_chunks(limit: int, seed: int) -> list[dict[str, Any]]:
    db_path = Path(settings.CHAT_RAG_CHUNKS_DB)
    if not db_path.exists():
        raise FileNotFoundError(f"RAG chunks DB not found: {db_path}")

    sql = """
        SELECT
            chunk_id,
            stock_code,
            year,
            category,
            source_file,
            page_start,
            page_end,
            COALESCE(json_extract(chunk_json, '$.subsection_title'), ''),
            COALESCE(json_extract(chunk_json, '$.text'), '')
        FROM chunks
        WHERE LENGTH(COALESCE(json_extract(chunk_json, '$.text'), '')) BETWEEN 600 AND 5000
          AND stock_code IS NOT NULL
          AND year IS NOT NULL
        ORDER BY chunk_id ASC
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(sql).fetchall()

    candidates: list[dict[str, Any]] = []
    for row in rows:
        text = _clean_text(row[8], max_chars=3500)
        if len(text) < 600:
            continue
        candidates.append({
            "id": str(row[0]),
            "ticker": str(row[1] or "").upper(),
            "year": int(row[2] or 0),
            "category": str(row[3] or ""),
            "source_file": str(row[4] or ""),
            "page_start": int(row[5] or 0),
            "page_end": int(row[6] or 0),
            "source_chunk_id": str(row[0]),
            "source_title": str(row[7] or ""),
            "source_text": text,
        })

    if len(candidates) < limit:
        raise RuntimeError(f"Not enough source chunks. requested={limit} available={len(candidates)}")

    rng = random.Random(seed)
    return rng.sample(candidates, limit)


def _deepseek_endpoint() -> str:
    return f"{settings.DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"


async def _deepseek_json(client: httpx.AsyncClient, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": settings.DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    response = await client.post(_deepseek_endpoint(), headers=headers, json=payload)
    response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    return parsed if isinstance(parsed, dict) else {}


async def _deepseek_text(client: httpx.AsyncClient, messages: list[dict[str, str]], max_tokens: int) -> str:
    payload = {
        "model": settings.DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    response = await client.post(_deepseek_endpoint(), headers=headers, json=payload)
    response.raise_for_status()
    body = response.json()
    return str(body["choices"][0]["message"]["content"] or "").strip()


def _fallback_case(chunk: dict[str, Any]) -> dict[str, Any]:
    title = chunk.get("source_title") or chunk.get("category") or "báo cáo thường niên"
    reference = _clean_text(chunk["source_text"], max_chars=420)
    return {
        **chunk,
        "question": f"Theo báo cáo thường niên {chunk['year']}, {chunk['ticker']} đề cập gì về {title}?",
        "reference": reference,
    }


async def _make_case(client: httpx.AsyncClient, chunk: dict[str, Any]) -> dict[str, Any]:
    system = (
        "Bạn tạo bộ câu hỏi đánh giá RAG từ báo cáo thường niên doanh nghiệp Việt Nam. "
        "Chỉ trả về JSON thô, không markdown."
    )
    user = f"""
Tạo 1 câu hỏi tiếng Việt và 1 câu trả lời chuẩn để đánh giá RAG.

Yêu cầu:
- Câu hỏi phải trả lời được CHỈ từ CONTEXT.
- Câu hỏi nên nhắc mã cổ phiếu và năm nếu tự nhiên.
- Reference answer ngắn gọn, trung thành với context, không bịa số liệu.
- JSON schema: {{"question": string, "reference": string}}

METADATA:
ticker={chunk['ticker']}
year={chunk['year']}
category={chunk.get('category', '')}
title={chunk.get('source_title', '')}

CONTEXT:
{chunk['source_text']}
""".strip()
    try:
        parsed = await _deepseek_json(
            client,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=450,
        )
        question = _clean_text(parsed.get("question"), max_chars=260)
        reference = _clean_text(parsed.get("reference"), max_chars=700)
        if not question or not reference:
            return _fallback_case(chunk)
        return {**chunk, "question": question, "reference": reference}
    except Exception:
        return _fallback_case(chunk)


async def _build_testset(limit: int, seed: int, concurrency: int) -> list[dict[str, Any]]:
    if not settings.DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is required to generate a testset")

    chunks = _load_source_chunks(limit=limit, seed=seed)
    timeout = httpx.Timeout(max(30, int(settings.LLM_TIMEOUT_SECONDS)))
    sem = asyncio.Semaphore(max(1, concurrency))
    async with httpx.AsyncClient(timeout=timeout) as client:
        async def _one(chunk: dict[str, Any]) -> dict[str, Any]:
            async with sem:
                return await _make_case(client, chunk)

        return await asyncio.gather(*[_one(chunk) for chunk in chunks])


async def _answer_case(client: httpx.AsyncClient, service: RagRetrievalService, case: dict[str, Any]) -> dict[str, Any]:
    chunks = await service.retrieve(
        query=case["question"],
        ticker=case.get("ticker"),
        years=[int(case["year"])] if case.get("year") else None,
    )
    contexts = [_clean_text(chunk.get("text"), max_chars=0) for chunk in chunks]
    context_block = "\n\n".join(f"[{idx + 1}] {text}" for idx, text in enumerate(contexts))
    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là trợ lý phân tích báo cáo thường niên chuyên nghiệp cho nhà đầu tư. "
                "Nhiệm vụ của bạn là trả lời câu hỏi một cách trung thực, rõ ràng và có tính phân tích "
                "dựa trên CONTEXT được cung cấp. "
                "Trả lời trực diện vào câu hỏi, ưu tiên các dữ kiện, số liệu, rủi ro, nguyên nhân, "
                "biện pháp hoặc nhận định được nêu trong CONTEXT. "
                "Có thể diễn giải theo phong cách phân tích tài chính/CFO, nhưng mọi nhận định thực tế "
                "phải được hỗ trợ bởi CONTEXT. "
                "Không bịa số liệu, không thêm nguyên nhân, dự báo, kết luận đầu tư hoặc đánh giá cá nhân "
                "nằm ngoài CONTEXT. "
                "Nếu CONTEXT chỉ hỗ trợ một phần câu hỏi, hãy trả lời phần được hỗ trợ và nêu rõ phần còn thiếu. "
                "Nếu CONTEXT không chứa thông tin liên quan, hãy nói rằng tài liệu được cung cấp chưa đề cập "
                "đủ thông tin để kết luận."
            ),
        },
        {
            "role": "user",
            "content": (
                f"QUESTION:\n{case['question']}\n\n"
                f"CONTEXT:\n{context_block}\n\n"
                "Yêu cầu trả lời:\n"
                "- Trả lời ngắn gọn, thường từ 1-2 đoạn ngắn.\n"
                "- Bám sát các dữ kiện trong CONTEXT.\n"
                "- Có thể tổng hợp/liên kết ý, nhưng không suy luận vượt ra ngoài CONTEXT.\n"
                "- Khi có số liệu, rủi ro, biện pháp, hoặc nhận định trong CONTEXT, hãy nêu lại chính xác."
            ),
        },
    ]
    try:
        answer = await _deepseek_text(client, messages=messages, max_tokens=550)
    except Exception as exc:
        answer = f"Không đủ thông tin để trả lời. ({type(exc).__name__})"

    return {
        **case,
        "answer": answer,
        "retrieved_contexts": contexts,
        "retrieved_chunks": chunks,
    }


async def _run_rag(testset: list[dict[str, Any]], concurrency: int, mode: str) -> list[dict[str, Any]]:
    if not settings.DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is required to answer RAG cases")

    service = RagRetrievalService()
    if mode == "vector":
        async def _skip_keyword(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
            return []

        service._keyword_search = _skip_keyword  # type: ignore[method-assign]
    timeout = httpx.Timeout(max(30, int(settings.LLM_TIMEOUT_SECONDS)))
    sem = asyncio.Semaphore(max(1, concurrency))
    async with httpx.AsyncClient(timeout=timeout) as client:
        async def _one(case: dict[str, Any]) -> dict[str, Any]:
            async with sem:
                return await _answer_case(client, service, case)

        return await asyncio.gather(*[_one(case) for case in testset])


def _evaluate_with_ragas(rows: list[dict[str, Any]], batch_size: int):
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import (
            AnswerRelevancy,
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
        )

    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    dataset = EvaluationDataset.from_list([
        {
            "user_input": row["question"],
            "response": row["answer"],
            "retrieved_contexts": row["retrieved_contexts"],
            "reference": row["reference"],
        }
        for row in rows
    ])
    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.DEEPSEEK_MODEL,
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL.rstrip("/"),
            temperature=0,
            timeout=float(settings.LLM_TIMEOUT_SECONDS),
            max_retries=2,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=settings.VOYAGE_EMBED_MODEL,
            api_key=settings.VOYAGE_API_KEY,
            base_url=settings.VOYAGE_EMBED_BASE_URL.rstrip("/"),
            timeout=float(settings.LLM_TIMEOUT_SECONDS),
            max_retries=2,
            check_embedding_ctx_length=False,
            tiktoken_enabled=False,
        )
    )
    return evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(strictness=1),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        batch_size=max(1, int(batch_size)),
    )


def _scores_to_rows(result: Any) -> list[dict[str, Any]]:
    try:
        frame = result.to_pandas()
        return frame.to_dict(orient="records")
    except Exception:
        return []


def _metric_summary(score_rows: list[dict[str, Any]]) -> dict[str, float | None]:
    aliases = {
        "faithfulness": ["faithfulness"],
        "answer_relevancy": ["answer_relevancy"],
        "context_precision": ["llm_context_precision_with_reference", "context_precision"],
        "context_recall": ["context_recall"],
    }
    summary: dict[str, float | None] = {}
    for metric, keys in aliases.items():
        values: list[float] = []
        for row in score_rows:
            for key in keys:
                value = row.get(key)
                if isinstance(value, (int, float)):
                    metric_value = float(value)
                    if math.isfinite(metric_value):
                        values.append(metric_value)
                        break
        summary[metric] = round(sum(values) / len(values), 6) if values else None
    return summary


def _write_scores_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the production RAGAS benchmark for FinFlow annual-report RAG."
    )
    parser.add_argument("limit", type=int, nargs="?", default=50, help="Number of eval questions, e.g. 30 or 50.")
    parser.add_argument("--testset", type=Path, default=None, help="Reuse an existing testset.json.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory. Defaults to artifacts/rag_eval/ragas/<timestamp>.")
    parser.add_argument("--seed", type=int, default=1904)
    parser.add_argument("--testset-concurrency", type=int, default=3)
    parser.add_argument("--rag-concurrency", type=int, default=1)
    parser.add_argument("--ragas-batch-size", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=["optimized", "vector"],
        default="optimized",
        help="optimized = hybrid retrieval + Voyage rerank; vector = traditional vector-only RAG baseline.",
    )
    parser.add_argument("--skip-ragas", action="store_true", help="Only generate testset + RAG answers; skip RAGAS scoring.")
    return parser


async def _main_async(args: argparse.Namespace) -> int:
    started = time.time()
    rag_config = _ensure_rag_config(args.mode)

    if not settings.VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY is required")
    if not settings.DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is required")

    out_dir = args.out_dir or DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.testset:
        testset = _read_json(args.testset)
        if not isinstance(testset, list):
            raise RuntimeError(f"Invalid testset format: {args.testset}")
        testset = testset[: int(args.limit)]
    else:
        print(f"[RAGAS][TESTSET] generating {args.limit} cases")
        testset = await _build_testset(
            limit=int(args.limit),
            seed=int(args.seed),
            concurrency=int(args.testset_concurrency),
        )
    _write_json(out_dir / "testset.json", testset)

    print(f"[RAGAS][RAG] answering {len(testset)} cases with {args.mode} Voyage RAG")
    rag_rows = await _run_rag(testset, concurrency=int(args.rag_concurrency), mode=args.mode)
    ragas_dataset = [
        {
            "user_input": row["question"],
            "response": row["answer"],
            "retrieved_contexts": row["retrieved_contexts"],
            "reference": row["reference"],
        }
        for row in rag_rows
    ]
    _write_json(out_dir / "ragas_dataset.json", ragas_dataset)
    _write_json(out_dir / "rag_outputs.json", rag_rows)

    metrics: dict[str, float | None] = {}
    score_rows: list[dict[str, Any]] = []
    if not args.skip_ragas:
        print("[RAGAS][SCORE] running RAGAS metrics")
        result = _evaluate_with_ragas(rag_rows, batch_size=int(args.ragas_batch_size))
        score_rows = _scores_to_rows(result)
        _write_scores_csv(out_dir / "scores.csv", score_rows)
        metrics = _metric_summary(score_rows)

    summary = {
        "limit": len(testset),
        "duration_seconds": round(time.time() - started, 2),
        "config": rag_config,
        "metrics": metrics,
        "output_dir": str(out_dir),
    }
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = _build_parser().parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
