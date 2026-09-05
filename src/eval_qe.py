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

# ---------- 1. 定义同义词扩展词典（针对 CIFAR-10）----------
EXPANSION_MAP = {
    "airplane": ["airplane", "plane", "jet", "aircraft"],
    "automobile": ["automobile", "car", "auto", "vehicle", "sedan"],
    "bird": ["bird", "birdie", "fowl"],
    "cat": ["cat", "kitten", "feline", "kitty"],
    "deer": ["deer", "buck", "fawn"],
    "dog": ["dog", "puppy", "canine", "pooch"],
    "frog": ["frog", "toad", "amphibian"],
    "horse": ["horse", "pony", "mare", "stallion"],
    "ship": ["ship", "boat", "vessel", "ocean liner"],
    "truck": ["truck", "lorry", "pickup", "rig"]
}

# 基础词列表（用于生成查询）
BASE_WORDS = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat",
    4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}
PREFIXES = ["a", "an", "photo of a", "picture of a", "a photo of"]
SUFFIXES = ["", " in the wild", " on the road"]

def generate_queries():
    """生成与 eval.py 一致的 210 个测试查询"""
    queries = []
    for label, base_word in BASE_WORDS.items():
        for prefix in PREFIXES:
            for suffix in SUFFIXES[:2]:
                if prefix.startswith("an") and base_word[0].lower() in "aeiou":
                    q = f"{prefix} {base_word}{suffix}"
                else:
                    q = f"{prefix} {base_word}{suffix}"
                queries.append((q, label))
        for extra in [f"the {base_word}", f"that {base_word} over there"]:
            queries.append((extra, label))
    return queries

def expand_query(query):
    """将查询扩展为同义词列表"""
    # 提取关键词（去掉前缀和后缀）
    words = query.lower().split()
    # 尝试匹配基础词
    for base_word, syns in EXPANSION_MAP.items():
        if base_word in words:
            # 保持原始查询格式不变，但替换关键词
            expanded = []
            for syn in syns:
                new_query = query.replace(base_word, syn)
                expanded.append(new_query)
            return expanded
    # 如果没有匹配到，返回原查询
    return [query]

# ---------- 2. 加载模型和索引 ----------
def evaluate_qe():
    print("📊 开始评测 Query Expansion (QE) 对 Baseline 的提升...")
    start_time = time.time()
    
    # 加载 Baseline 索引
    print("📂 加载 Baseline FAISS 索引...")
    index = faiss.read_index("data/index.faiss")
    ids = np.load("data/index_ids.npy", allow_pickle=True)
    
    # 加载 Baseline 模型 (ViT-B/32)
    print("🤖 加载 CLIP 模型...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to("cuda")
    model.eval()
    
    all_queries = generate_queries()  # 210 个
    print(f"✅ 共生成 {len(all_queries)} 个测试查询 (每个查询将扩展为同义词组)")
    
    recalls = {1: [], 5: [], 10: []}
    mrr_scores = []
    
    # 遍历所有查询
    for idx, (query, expected_label) in enumerate(all_queries):
        if idx % 50 == 0:
            print(f"  进度: {idx}/{len(all_queries)}")
        
        # ----- 核心：Query Expansion -----
        expanded_queries = expand_query(query)
        
        # 对每个扩展查询分别检索
        all_results = []  # 存储 (image_path, score, label)
        seen_paths = set()
        
        with torch.no_grad():
            for eq in expanded_queries:
                # 提取文本特征
                inputs = processor(text=[eq], return_tensors="pt", padding=True)
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                text_feat = model.get_text_features(**inputs)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                
                # 搜索 top-10
                scores, indices = index.search(text_feat.cpu().numpy().astype("float32"), 10)
                
                # 收集结果（去重，保留最高分）
                for score, i_idx in zip(scores[0], indices[0]):
                    path = ids[i_idx]["image_path"]
                    if path not in seen_paths:
                        seen_paths.add(path)
                        all_results.append({
                            "path": path,
                            "score": float(score),
                            "label": ids[i_idx]["label"]
                        })
        
        # 按分数降序排序，取 top-10
        all_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = all_results[:10]
        retrieved_labels = [r["label"] for r in top_results]
        
        # 计算指标
        for k in [1, 5, 10]:
            recalls[k].append(1 if expected_label in retrieved_labels[:k] else 0)
        try:
            rank = retrieved_labels.index(expected_label) + 1
            mrr_scores.append(1.0 / rank)
        except ValueError:
            mrr_scores.append(0.0)
    
    # 汇总结果
    results = {
        "R@1": np.mean(recalls[1]),
        "R@5": np.mean(recalls[5]),
        "R@10": np.mean(recalls[10]),
        "MRR": np.mean(mrr_scores)
    }
    
    print(f"⏱️ 评测耗时: {time.time() - start_time:.1f} 秒")
    return results

if __name__ == "__main__":
    metrics = evaluate_qe()
    
    print("\n" + "="*40)
    print("📈 改进点 3 评测结果 (Baseline + Query Expansion)")
    print("="*40)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("="*40)
    
    os.makedirs("results", exist_ok=True)
    with open("results/improve3.json", "w", encoding="utf-8") as f:
        json.dump({"model": "clip-vit-base-patch32 + QE", **metrics}, f, indent=2)
    
    print("\n✅ 结果已保存到 results/improve3.json")
    
    # 与 Baseline 对比
    with open("results/baseline.json", "r") as f:
        base = json.load(f)
    print("\n📊 对比 Baseline (无QE):")
    print(f"   R@1: {base['R@1']:.2%} → {metrics['R@1']:.2%} (提升 {metrics['R@1'] - base['R@1']:.2%})")
    print(f"   MRR: {base['MRR']:.2%} → {metrics['MRR']:.2%} (提升 {metrics['MRR'] - base['MRR']:.2%})")