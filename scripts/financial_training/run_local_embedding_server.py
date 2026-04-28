import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union, List
import mlx_embeddings

# --- 1. CONFIGURATION ---
MODEL_NAME = "mlx-community/bge-m3-mlx-fp16"
HOST = "127.0.0.1"
PORT = 9091

# --- 2. FASTAPI APP & MLX ENGINE ---
app = FastAPI(title="Local MLX Embedding Server", version="1.0")

print(f"🔄 Loading MLX model: {MODEL_NAME} (This takes a few seconds on first run...)")
model, tokenizer = mlx_embeddings.load(MODEL_NAME)
print(f"✅ Model loaded onto Apple Neural Engine! Ready at port {PORT}.")

class EmbedRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = MODEL_NAME

# --- 3. OPENAI-COMPATIBLE ENDPOINT ---
@app.post("/v1/embeddings")
def get_embeddings(req: EmbedRequest):
    # 1. Prepare inputs
    texts = [req.input] if isinstance(req.input, str) else req.input
    
    # 2. Run inference on MLX GPU/NPU
    arr = mlx_embeddings.generate(model, tokenizer, texts)
    
    # 3. Convert MLX Arrays back to Python Lists
    raw_embeddings = getattr(arr, "text_embeds", arr.last_hidden_state).tolist()

    # 4. Map to OpenAI Standard Format (required by the backend)
    data = []
    for idx, emb_vec in enumerate(raw_embeddings):
        data.append({
            "object": "embedding",
            "index": idx,
            "embedding": emb_vec
        })
        
    return {
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
