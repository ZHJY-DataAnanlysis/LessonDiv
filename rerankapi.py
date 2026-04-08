# rerankapi.py
from fastapi import FastAPI, Request
from FlagEmbedding import FlagReranker
import uvicorn

app = FastAPI()
reranker = FlagReranker("/home/lhr/bge-reranker-large", use_fp16=True)

@app.post("/v1/rerank")
async def rerank(request: Request):
    data = await request.json()
    query = data["query"]
    docs = data["documents"]
    scores = reranker.compute_score([[query, doc] for doc in docs], batch_size=8)
    return {"scores": scores}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9001)