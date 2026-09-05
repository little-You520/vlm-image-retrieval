import json
import torch
import faiss
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
import ssl

# 临时解决 SSL 问题（确保模型加载顺利）
ssl._create_default_https_context = ssl._create_unverified_context


def build_index(manifest_path="data/manifest.jsonl",
                index_path="data/index.faiss",
                model_name="openai/clip-vit-base-patch32"):
    """
    读取 manifest，提取图像特征，构建 FAISS 索引
    """
    print(f"📂 正在读取 manifest: {manifest_path}")

    # 1. 加载 manifest
    items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line.strip()))
    print(f"✅ 共加载 {len(items)} 条记录")

    # 2. 加载 CLIP 模型（从缓存读取，无需重新下载）
    print("🤖 正在加载 CLIP 模型...")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = model.to("cuda")  # 放到 GPU
    model.eval()
    print("✅ 模型加载成功")

    # 3. 批量提取特征
    embeddings = []
    batch_size = 64  # 4060 显存足够，可以开大一点加快速度
    total = len(items)

    print(f"🔄 正在提取 {total} 张图片的特征...")

    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch_items = items[i:i + batch_size]
            # 加载图片
            images = []
            valid_indices = []
            for idx, item in enumerate(batch_items):
                try:
                    img = Image.open(item["image_path"]).convert("RGB")
                    images.append(img)
                    valid_indices.append(idx)
                except Exception as e:
                    print(f"⚠️ 跳过损坏图片: {item['image_path']} ({e})")

            if not images:
                continue

            # 处理图片
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # 提取特征
            feats = model.get_image_features(**inputs)
            # 归一化（FAISS 内积相似度要求向量是单位向量）
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy())

            if (i + batch_size) % 500 < batch_size or i + batch_size >= total:
                print(f"  进度: {min(i + batch_size, total)}/{total}")

    # 4. 合并所有特征
    embeddings = np.concatenate(embeddings, axis=0).astype("float32")
    print(f"✅ 特征提取完成，形状: {embeddings.shape}")

    # 5. 构建 FAISS 索引（内积相似度，因为特征已经归一化）
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # IP = Inner Product (内积)
    index.add(embeddings)

    # 6. 保存索引和对应的元数据
    faiss.write_index(index, index_path)
    # 保存 items 列表，以便检索时返回图片路径和描述
    np.save(index_path.replace(".faiss", "_ids.npy"), np.array(items))

    print(f"✅ 索引构建完成！")
    print(f"   - 索引文件: {index_path} (包含 {index.ntotal} 个向量)")
    print(f"   - 元数据文件: {index_path.replace('.faiss', '_ids.npy')}")


if __name__ == "__main__":
    build_index()