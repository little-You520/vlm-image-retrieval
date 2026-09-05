import json
import torch
import faiss
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def build_index(manifest_path="data/manifest.jsonl",
                index_path="data/index_large.faiss",
                model_name="openai/clip-vit-large-patch14"):  # 关键：换成了 Large 模型
    print(f"📂 正在读取 manifest: {manifest_path}")

    items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line.strip()))
    print(f"✅ 共加载 {len(items)} 条记录")

    print(f"🤖 正在加载大模型 (CLIP Large)，首次下载约 1.5GB，请耐心等待...")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = model.to("cuda")
    model.eval()
    print("✅ 大模型加载成功")

    embeddings = []
    batch_size = 32  # 大模型更吃显存，适当减小批次防止爆显存
    total = len(items)

    print(f"🔄 正在提取 {total} 张图片的特征（大模型）...")

    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch_items = items[i:i + batch_size]
            images = [Image.open(item["image_path"]).convert("RGB") for item in batch_items]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy())

            if (i + batch_size) % 500 < batch_size or i + batch_size >= total:
                print(f"  进度: {min(i + batch_size, total)}/{total}")

    embeddings = np.concatenate(embeddings, axis=0).astype("float32")
    print(f"✅ 特征提取完成，形状: {embeddings.shape}")

    dimension = embeddings.shape[1]  # Large 模型输出是 768 维
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    np.save(index_path.replace(".faiss", "_ids.npy"), np.array(items))

    print(f"✅ 大模型索引构建完成！")
    print(f"   - 索引文件: {index_path} (包含 {index.ntotal} 个向量)")


if __name__ == "__main__":
    build_index()