import json
import torch
import faiss
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import ssl
import argparse

# 临时解决 SSL
ssl._create_default_https_context = ssl._create_unverified_context


def search(query, top_k=5, index_path="data/index.faiss"):
    """
    根据文字描述检索最相似的图片
    """
    # 1. 加载索引和元数据
    print(f"🔍 正在搜索: {query}")
    index = faiss.read_index(index_path)
    items = np.load(index_path.replace(".faiss", "_ids.npy"), allow_pickle=True)

    # 2. 加载 CLIP 模型
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to("cuda")
    model.eval()

    # 3. 将查询文字转为特征向量
    with torch.no_grad():
        inputs = processor(text=[query], return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        text_feat = model.get_text_features(**inputs)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    # 4. 在 FAISS 中搜索
    text_feat_np = text_feat.cpu().numpy().astype("float32")
    scores, indices = index.search(text_feat_np, top_k)

    # 5. 整理结果
    results = []
    for score, idx in zip(scores[0], indices[0]):
        item = items[idx]
        results.append({
            "image_path": item["image_path"],
            "score": float(score),
            "caption": item.get("caption", "unknown"),
            "label": item.get("label", -1)
        })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="搜索描述")
    parser.add_argument("--top_k", type=int, default=5, help="返回结果数")
    args = parser.parse_args()

    results = search(args.query, top_k=args.top_k)

    print(f"\n📸 搜索结果 (共 {len(results)} 条):")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['image_path']} (相似度: {r['score']:.4f})")
        print(f"   标签: {r['label']}, 描述: {r['caption']}")

    # 显示第一张图（如果安装了 PIL 且支持）
    if results and args.top_k >= 1:
        img = Image.open(results[0]["image_path"])
        print(f"\n🖼️ 最相似图片: {results[0]['image_path']}")
        img.show()  # 弹出图片预览窗口（可能不会在所有系统上生效）