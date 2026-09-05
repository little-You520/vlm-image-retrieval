import json
from pathlib import Path
from PIL import Image
from datasets import load_dataset
import ssl

# 临时解决 SSL 证书问题
ssl._create_default_https_context = ssl._create_unverified_context

# CIFAR-10 标签对应的文本描述（用于检索）
LABEL_TO_TEXT = {
    0: "an airplane",
    1: "an automobile",
    2: "a bird",
    3: "a cat",
    4: "a deer",
    5: "a dog",
    6: "a frog",
    7: "a horse",
    8: "a ship",
    9: "a truck"
}


def build_manifest_from_cifar10(max_count=2000, output_path="data/manifest.jsonl"):
    """
    从 CIFAR-10 数据集构建 manifest 文件
    """
    # 创建数据目录
    Path("data").mkdir(exist_ok=True)
    Path("data/raw").mkdir(exist_ok=True, parents=True)

    # 加载 CIFAR-10 训练集（50000 张图）
    print("正在加载 CIFAR-10 数据集...")
    dataset = load_dataset("cifar10", split="train", streaming=True)

    manifest = []
    count = 0

    print(f"正在生成 manifest，最多 {max_count} 条...")

    for i, sample in enumerate(dataset):
        if count >= max_count:
            break

        # 获取图片和标签
        image = sample['img']  # PIL Image 对象
        label = sample['label']  # int 0-9
        text = LABEL_TO_TEXT[label]

        # 保存图片到本地 data/raw/ 目录
        img_path = f"data/raw/{i:06d}.png"
        image.save(img_path)

        # 添加到 manifest
        manifest.append({
            "image_path": img_path,
            "caption": f"a photo of {text}",
            "label": label
        })

        count += 1
        if count % 500 == 0:
            print(f"  已处理 {count} 张图片...")

    # 写入 manifest.jsonl
    with open(output_path, "w", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 完成！共生成 {len(manifest)} 条记录")
    print(f"   manifest 文件: {output_path}")
    print(f"   图片目录: data/raw/")
    print(f"   标签分布: {dict(zip(*np.unique([m['label'] for m in manifest], return_counts=True)))}")
    return manifest


if __name__ == "__main__":
    import numpy as np  # 用于统计标签分布

    build_manifest_from_cifar10(max_count=2000)