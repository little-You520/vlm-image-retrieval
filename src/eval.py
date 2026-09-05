import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import torch
import faiss
from collections import defaultdict
from transformers import CLIPProcessor, CLIPModel
import time

# CIFAR-10 标签对应基础词
BASE_WORDS = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

PREFIXES = ["a", "an", "photo of a", "picture of a", "image of a", "a photo of"]
SUFFIXES = ["", " in the wild", " on the road", " in a city", " on a farm"]


def generate_queries():
    queries = []
    for label, base_word in BASE_WORDS.items():
        for prefix in PREFIXES:
            for suffix in SUFFIXES[:3]:
                if prefix.startswith("an") and base_word[0].lower() in "aeiou":
                    q = f"{prefix} {base_word}{suffix}"
                else:
                    q = f"{prefix} {base_word}{suffix}"
                queries.append((q, label))
        for extra in [f"the {base_word}", f"that {base_word} over there", f"{base_word} animal"]:
            queries.append((extra, label))
    return queries


def evaluate(top_k_list=[1, 5, 10]):
    print("📊 开始评测...")
    start_time = time.time()

    # 1. 加载 manifest 和索引（只加载一次）
    print("📂 加载 manifest 和 FAISS 索引...")
    items = []
    with open("data/manifest.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line.strip()))

    index = faiss.read_index("data/index.faiss")
    ids = np.load("data/index_ids.npy", allow_pickle=True)

    # 2. 加载 CLIP 模型（只加载一次）
    print("🤖 加载 CLIP 模型...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to("cuda")
    model.eval()

    # 3. 生成所有查询
    all_queries = generate_queries()
    print(f"✅ 共生成 {len(all_queries)} 个测试查询")

    # 4. 批量提取所有查询的文本特征
    print("🔄 批量提取查询特征...")
    query_texts = [q[0] for q in all_queries]
    expected_labels = [q[1] for q in all_queries]

    text_embeddings = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(query_texts), batch_size):
            batch = query_texts[i:i + batch_size]
            inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            feats = model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            text_embeddings.append(feats.cpu().numpy())

    text_embeddings = np.concatenate(text_embeddings, axis=0).astype("float32")
    print(f"✅ 文本特征形状: {text_embeddings.shape}")

    # 5. 批量搜索（一次性对所有查询搜索）
    print("🔄 批量检索中...")
    scores_matrix, indices_matrix = index.search(text_embeddings, max(top_k_list))

    # 6. 计算指标
    recalls = {k: [] for k in top_k_list}
    mrr_scores = []

    for i, expected_label in enumerate(expected_labels):
        retrieved_ids = indices_matrix[i]
        retrieved_labels = [ids[idx]["label"] for idx in retrieved_ids]

        for k in top_k_list:
            recalls[k].append(1 if expected_label in retrieved_labels[:k] else 0)

        try:
            rank = retrieved_labels.index(expected_label) + 1
            mrr_scores.append(1.0 / rank)
        except ValueError:
            mrr_scores.append(0.0)

    # 7. 汇总结果
    results = {}
    for k in top_k_list:
        results[f"R@{k}"] = np.mean(recalls[k])
    results["MRR"] = np.mean(mrr_scores)

    elapsed = time.time() - start_time
    print(f"⏱️ 评测耗时: {elapsed:.1f} 秒")

    return results


if __name__ == "__main__":
    metrics = evaluate()

    print("\n" + "=" * 40)
    print("📈 评测结果")
    print("=" * 40)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("=" * 40)

    os.makedirs("results", exist_ok=True)
    with open("results/baseline.json", "w", encoding="utf-8") as f:
        json.dump({"model": "clip-vit-base-patch32", **metrics}, f, indent=2)

    print("\n✅ 结果已保存到 results/baseline.json")
    print(f"\n📋 Baseline 指标：")
    for key, value in metrics.items():
        print(f"   - {key}: {value:.2%}")