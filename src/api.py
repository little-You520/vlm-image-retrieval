import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import faiss
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
import json
from typing import List
from contextlib import asynccontextmanager

# ---------- 全局变量 ----------
model = None
processor = None
index = None
ids = None
items = None

# ---------- 加载函数 ----------
def load_models():
    global model, processor, index, ids, items
    print("🚀 加载微调模型...")
    model = CLIPModel.from_pretrained("checkpoints/clip-finetuned-cifar10")
    processor = CLIPProcessor.from_pretrained("checkpoints/clip-finetuned-cifar10")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    print("📂 加载 FAISS 索引...")
    index = faiss.read_index("data/index_finetuned.faiss")
    ids = np.load("data/index_finetuned_ids.npy", allow_pickle=True)
    
    with open("data/manifest.jsonl", "r") as f:
        items = [json.loads(line) for line in f]
    
    print("✅ 模型和索引加载完成！")

# ---------- 生命周期管理 ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    # 清理资源（可选）

# ---------- FastAPI 应用 ----------
app = FastAPI(title="VLM 跨模态图像检索 API", lifespan=lifespan)

# ---------- 请求/响应模型 ----------
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    image_path: str
    score: float
    label: int

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

# ---------- 检索函数 ----------
def search_images(query: str, top_k: int = 5):
    with torch.no_grad():
        inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        text_feat = model.get_text_features(**inputs)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    
    scores, indices = index.search(text_feat.cpu().numpy().astype("float32"), top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        item = ids[idx]
        results.append({
            "image_path": item["image_path"],
            "score": float(score),
            "label": item["label"]
        })
    return results

# ---------- API 端点 ----------
@app.get("/health")
def health():
    return {"status": "ok", "model": "clip-finetuned-cifar10"}

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="query 不能为空")
    results = search_images(req.query, req.top_k)
    return SearchResponse(query=req.query, results=results)

@app.post("/search_image")
async def search_by_image(file: UploadFile = File(...), top_k: int = 5):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片解析失败: {e}")
    
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        img_feat = model.get_image_features(**inputs)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    
    scores, indices = index.search(img_feat.cpu().numpy().astype("float32"), top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        item = ids[idx]
        results.append({
            "image_path": item["image_path"],
            "score": float(score),
            "label": item["label"]
        })
    return {"top_k": top_k, "results": results}

# ---------- 启动服务 ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)