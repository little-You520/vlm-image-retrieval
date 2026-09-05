import json
import torch
import faiss
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

def build_index():
    print("📂 加载微调模型...")
    model_path = "checkpoints/clip-finetuned-cifar10"
    
    # 加载微调后的模型
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    model = model.to("cuda")
    model.eval()
    
    # 加载数据
    items = []
    with open("data/manifest.jsonl", "r") as f:
        for line in f:
            items.append(json.loads(line.strip()))
    
    print(f"✅ 共加载 {len(items)} 条记录")
    
    # 批量提取特征
    embeddings = []
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            batch_items = items[i:i+batch_size]
            images = [Image.open(item["image_path"]).convert("RGB") for item in batch_items]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy())
            
            if (i + batch_size) % 500 < batch_size or i + batch_size >= len(items):
                print(f"  进度: {min(i+batch_size, len(items))}/{len(items)}")
    
    embeddings = np.concatenate(embeddings, axis=0).astype("float32")
    print(f"✅ 特征提取完成，形状: {embeddings.shape}")
    
    # 构建索引
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    faiss.write_index(index, "data/index_finetuned.faiss")
    np.save("data/index_finetuned_ids.npy", np.array(items))
    
    print(f"✅ 微调模型索引构建完成！")
    print(f"   - 索引文件: data/index_finetuned.faiss (包含 {index.ntotal} 个向量)")

if __name__ == "__main__":
    build_index()