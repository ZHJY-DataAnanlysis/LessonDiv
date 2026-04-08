# 在服务器上创建 /home/lhr/model_api.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
import os

app = FastAPI()

# 1. 挂载模型文件为静态文件
app.mount("/BAAI", StaticFiles(directory="/home/lhr/bge-small-zh-v1.5"), name="model")

# 2. 添加编码接口
model = SentenceTransformer('/home/lhr/bge-small-zh-v1.5')

@app.post("/encode")
def encode(text: str):
    return {"embedding": model.encode(text).tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)