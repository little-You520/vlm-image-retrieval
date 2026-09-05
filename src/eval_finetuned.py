import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import torch
import faiss
from transformers import CLIPProcessor, CLIPModel
import time

# 与 eval.py 完全相同的查询生成逻辑
BASE_WORDS = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat",
    4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
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

def evaluate():
    print("📊 开始评测微调模型 (Improve 2: Fine-tuned)...")
    start_time = time.time()
    
    # 加载微调模型索引
    index = faiss.read_index("data/index_finetuned.faiss")
    ids = np.load("data/index_finetuned_ids.npy", allow_pickle=True)
    
    # 加载微调模型
    print("🤖 加载微调模型...")
    model = CLIPModel.from_pretrained("checkpoints/clip-finetuned-cifar10")
    processor = CLIPProcessor.from_pretrained("checkpoints/clip-finetuned-cifar10")
    model = model.to("cuda")
    model.eval()
    
    all_queries = generate_queries()
    print(f"✅ 共生成 {len(all_queries)} 个测试查询")
    
    # 批量提取文本特征
    query_texts = [q[0] for q in all_queries]
    expected_labels = [q[1] for q in all_queries]
    
    text_embeddings = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(query_texts), batch_size):
            batch = query_texts[i:i+batch_size]
            inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            feats = model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            text_embeddings.append(feats.cpu().numpy())
    
    text_embeddings = np.concatenate(text_embeddings, axis=0).astype("float32")
    
    # 批量检索
    scores_matrix, indices_matrix = index.search(text_embeddings, 10)
    
    # 计算指标
    recalls = {1: [], 5: [], 10: []}
    mrr_scores = []
    
    for i, expected_label in enumerate(expected_labels):
        retrieved_labels = [ids[idx]["label"] for idx in indices_matrix[i]]
        for k in [1, 5, 10]:
            recalls[k].append(1 if expected_label in retrieved_labels[:k] else 0)
        try:
            rank = retrieved_labels.index(expected_label) + 1
            mrr_scores.append(1.0 / rank)
        except ValueError:
            mrr_scores.append(0.0)
    
    results = {
        "R@1": np.mean(recalls[1]),
        "R@5": np.mean(recalls[5]),
        "R@10": np.mean(recalls[10]),
        "MRR": np.mean(mrr_scores)
    }
    
    print(f"⏱️ 评测耗时: {time.time() - start_time:.1f} 秒")
    return results

if __name__ == "__main__":
    metrics = evaluate()
    
    print("\n" + "="*40)
    print("📈 改进点 2 评测结果 (Fine-tuned on CIFAR-10)")
    print("="*40)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("="*40)
    
    os.makedirs("results", exist_ok=True)
    with open("results/improve2.json", "w", encoding="utf-8") as f:
        json.dump({"model": "clip-vit-base-patch32-finetuned", **metrics}, f, indent=2)
    
    print("\n✅ 结果已保存到 results/improve2.json")