import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import torch
import faiss
from transformers import CLIPProcessor, CLIPModel
from search import search  # 复用搜索逻辑

# 加载数据
print("📂 加载数据...")
items = []
with open("data/manifest.jsonl", "r") as f:
    for line in f:
        items.append(json.loads(line.strip()))

# 加载 FAISS 索引（用于快速批量获取，或者直接用 search 函数）
index = faiss.read_index("data/index.faiss")
ids = np.load("data/index_ids.npy", allow_pickle=True)

# 从 Day 4 的 eval 中提取所有测试查询（210 个）
# 但为了简便，我们直接重新生成 210 个查询，并收集失败的 case
from eval import generate_queries

print("🔍 正在遍历所有查询，找出 R@1 失败案例...")

# 加载 CLIP 模型（只加载一次，加快速度）
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = model.to("cuda")
model.eval()

all_queries = generate_queries()
failures = []
success_count = 0

for query, expected_label in all_queries:
    # 执行搜索
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        text_feat = model.get_text_features(**inputs)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    scores, idx = index.search(text_feat.cpu().numpy().astype("float32"), 1)
    top_label = ids[idx[0][0]]["label"]

    if top_label != expected_label:
        failures.append({
            "query": query,
            "expected": expected_label,
            "got": top_label,
            "score": float(scores[0][0])
        })
    else:
        success_count += 1

print(f"✅ 总查询数: {len(all_queries)}")
print(f"✅ 正确数: {success_count}")
print(f"❌ 失败数: {len(failures)}")
print(f"📊 实际 R@1: {success_count / len(all_queries):.2%}")

# --- 失败分类 ---
# 定义类别名称
LABEL_NAMES = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat",
    4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}

# 分类：按混淆对分组
confusion_pairs = []
for f in failures:
    expected_name = LABEL_NAMES[f["expected"]]
    got_name = LABEL_NAMES[f["got"]]
    confusion_pairs.append(f"{expected_name}→{got_name}")

counter = Counter(confusion_pairs)

print("\n📋 失败分类 (混淆矩阵 Top 5):")
for pair, count in counter.most_common():
    print(f"  - {pair}: {count} 次")

# --- 生成饼图 ---
if len(failures) > 0:
    labels = [f"{k}" for k, v in counter.items()]
    sizes = [v for k, v in counter.items()]

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(f'CIFAR-10 Retrieval Failure Analysis ({len(failures)} errors)')
    plt.axis('equal')

    # 保存图片
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/failure_analysis.png", dpi=300, bbox_inches='tight')
    print("\n✅ 饼图已保存到 results/failure_analysis.png")
    plt.show()
else:
    print("🎉 没有失败案例！R@1 达到 100%！")

# 保存详细失败记录
with open("results/failures_detail.json", "w") as f:
    json.dump(failures, f, indent=2)
print("✅ 详细失败记录已保存到 results/failures_detail.json")