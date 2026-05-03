"""Local embedding + reranker server for FinFlow RAG.

Endpoints:
- POST /v1/embeddings  — bge-m3 via MLX (Apple Neural Engine)
- POST /v1/rerank      — bge-reranker-v2-m3 via sentence-transformers (Apple MPS)

The two models live on different stacks (MLX vs PyTorch+MPS) because
mlx_embeddings doesn't ship a cross-encoder API yet — but both run on the
M-series GPU/ANE so latency is fine for production RAG.
"""
import os
from typing import List, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import mlx_embeddings
from sentence_transformers import CrossEncoder

# ── Config ────────────────────────────────────────────────────────────
EMBED_MODEL = "mlx-community/bge-m3-mlx-fp16"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
HOST = "127.0.0.1"
PORT = 9091

# Pick device for the cross-encoder. M4 has MPS; CPU fallback otherwise.
RERANK_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

app = FastAPI(title="Local MLX Embedding + Rerank Server", version="2.0")

# ── Load models ───────────────────────────────────────────────────────
print(f"🔄 Loading embedding model: {EMBED_MODEL}")
embed_model, embed_tokenizer = mlx_embeddings.load(EMBED_MODEL)
print(f"✅ Embedding ready (Apple Neural Engine)")

print(f"🔄 Loading rerank model: {RERANK_MODEL} (device={RERANK_DEVICE})")
rerank_model = CrossEncoder(RERANK_MODEL, device=RERANK_DEVICE, max_length=512)
print(f"✅ Rerank ready (device={RERANK_DEVICE})")
print(f"\n🚀 Server listening on http://{HOST}:{PORT}")


# ── Schemas ───────────────────────────────────────────────────────────


class EmbedRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = EMBED_MODEL


class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_n: int | None = None
    model: str = RERANK_MODEL


# ── Endpoints ─────────────────────────────────────────────────────────


@app.post("/v1/embeddings")
def get_embeddings(req: EmbedRequest):
    """OpenAI-compatible embeddings endpoint."""
    texts = [req.input] if isinstance(req.input, str) else req.input
    arr = mlx_embeddings.generate(embed_model, embed_tokenizer, texts)
    raw_embeddings = getattr(arr, "text_embeds", arr.last_hidden_state).tolist()

    data = [
        {"object": "embedding", "index": idx, "embedding": vec}
        for idx, vec in enumerate(raw_embeddings)
    ]
    return {
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


@app.post("/v1/rerank")
def rerank(req: RerankRequest):
    """Cohere-compatible rerank endpoint.

    Body:
        {"query": "...", "documents": ["doc1", "doc2", ...], "top_n": 6}

    Returns:
        {"results": [{"index": int, "relevance_score": float, "document": {"text": str}}]}
    """
    if not req.documents:
        return {"results": [], "model": req.model}
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    pairs = [(req.query, doc) for doc in req.documents]
    scores = rerank_model.predict(pairs, show_progress_bar=False).tolist()

    ranked = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True,
    )
    if req.top_n:
        ranked = ranked[: req.top_n]

    return {
        "model": req.model,
        "results": [
            {
                "index": idx,
                "relevance_score": float(score),
                "document": {"text": req.documents[idx]},
            }
            for idx, score in ranked
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "embed_model": EMBED_MODEL,
        "rerank_model": RERANK_MODEL,
        "rerank_device": RERANK_DEVICE,
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
