from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# 1. 加载模型（第一次运行会下载 600MB，耐心等1-2分钟）
print("正在加载 CLIP 模型（首次需要下载），请稍候...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. 因为你的显卡是 4060，把模型移到 GPU 上（加速推理）
model = model.to("cuda")

print("模型加载成功！")

# 3. 因为没有 test.jpg，我们直接用代码生成一张临时测试图（256x256 的纯色猫猫色块）
print("生成临时测试图片...")
# 创建一个 256x256 的紫色渐变图（只是为了演示能跑通）
import numpy as np
arr = np.zeros((256, 256, 3), dtype=np.uint8)
arr[:, :, 0] = np.arange(256) // 1  # R通道渐变
arr[:, :, 2] = np.arange(256).reshape(-1, 1) // 1  # B通道渐变
image = Image.fromarray(arr)
image.save("test.jpg")
print("已生成 test.jpg")

# 4. 推理：判断这张图属于哪个文字描述
texts = ["a cat", "a dog", "a car", "a building", "a sunset"]

# 处理输入（注意：要把图片和文字都送到 GPU 上）
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
# 将输入张量也移到 GPU
inputs = {k: v.to("cuda") for k, v in inputs.items()}

# 模型推理
outputs = model(**inputs)

# 计算概率
probs = outputs.logits_per_image.softmax(dim=1)

# 打印结果
print("\n=== 检索结果 ===")
for text, prob in zip(texts, probs[0].tolist()):
    print(f"{text}: {prob:.4f}")

# 输出最高分
top_idx = probs[0].argmax().item()
print(f"\n✅ 模型认为这张图最像: {texts[top_idx]} (置信度: {probs[0][top_idx]:.2%})")