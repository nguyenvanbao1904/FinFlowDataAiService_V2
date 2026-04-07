import json
import sqlite3
import sys
from pathlib import Path

import mlx.core as mx
from mlx_embeddings.utils import load
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

# 1. Cấu hình y hệt script chính
MODEL_REF = "mlx-community/bge-m3-mlx-fp16"
QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION_NAME = "annual_report_chunks_bge_m3"
CHUNKS_DB = Path("artifacts/rag/annual_reports/chunks/annual_reports_chunks.sqlite")

# 2. Khởi tạo Qdrant
client = QdrantClient(url=QDRANT_URL)

# 3. Load model và tokenizer (Dùng hàm load từ mlx_embeddings)
print(f"--- 🧠 Đang nạp model {MODEL_REF} vào M4... ---")
model, tokenizer = load(MODEL_REF)

def get_query_vector(text: str):
    # Dùng đúng logic batch_encode_plus của ông
    inputs = tokenizer.batch_encode_plus(
        [text],
        return_tensors="mlx",
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    outputs = model(inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
    
    # Lấy vector (ưu tiên text_embeds như script của ông)
    values = getattr(outputs, "text_embeds", None)
    if values is None:
        values = getattr(outputs, "pooler_output", None)
    
    mx.eval(values)
    return [float(x) for x in values.tolist()[0]]


def load_chunk_text_by_ids(chunk_ids: list[str]) -> dict[str, dict]:
    if not chunk_ids or not CHUNKS_DB.exists():
        return {}

    placeholders = ",".join("?" for _ in chunk_ids)
    sql = f"""
        SELECT chunk_id, chunk_json
        FROM chunks
        WHERE chunk_id IN ({placeholders})
    """
    with sqlite3.connect(str(CHUNKS_DB)) as conn:
        rows = conn.execute(sql, chunk_ids).fetchall()

    out: dict[str, dict] = {}
    for chunk_id, chunk_json in rows:
        try:
            payload = json.loads(chunk_json or "{}")
            if isinstance(payload, dict):
                out[str(chunk_id)] = payload
        except Exception:
            continue
    return out

# 4. Thực hiện Search
query_text = "khó khăn của pnj trong năm 2024 là gì"
print(f"--- 🔍 Đang tìm kiếm: {query_text} ---")

vector = get_query_vector(query_text)

query_filter = Filter(
    must=[
        FieldCondition(key="stock_code", match=MatchValue(value="PNJ")),
        FieldCondition(key="year", match=MatchValue(value=2024)),
    ]
)

response = client.query_points(
    collection_name=COLLECTION_NAME,
    query=vector,
    query_filter=query_filter,
    limit=3,
    with_payload=True,
)
results = list(response.points or [])
chunk_ids = [str((hit.payload or {}).get("chunk_id", "")) for hit in results]
chunk_details = load_chunk_text_by_ids([cid for cid in chunk_ids if cid])
is_full_text = "--full-text" in sys.argv

# 5. Show kết quả
print("\n--- 📈 KẾT QUẢ TỪ QDRANT ---")
for hit in results:
    payload = hit.payload or {}
    chunk_id = str(payload.get("chunk_id", "-"))
    detail = chunk_details.get(chunk_id, {})
    title = str(detail.get("subsection_title", "") or payload.get("category", "") or "-")
    text = str(detail.get("text", "")).strip()
    if is_full_text:
        display_text = text
    else:
        display_text = (text[:480] + "...") if len(text) > 480 else text
    print(
        f"Score: {hit.score:.4f} | ID: {chunk_id}"
        f" | Trang: {payload.get('page_start', '-')}"
    )
    print(f"Title: {title}")
    print(f"Text: {display_text if display_text else '[Không tìm thấy text trong chunks DB]'}")
    print("-" * 120)
