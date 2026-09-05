import torch
from transformers import CLIPModel

print("🔍 检查 Base 模型...")
base = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
print(f"Base 模型图像特征维度: {base.config.projection_dim}")  # 应该是 512

print("\n🔍 检查 Large 模型...")
large = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
print(f"Large 模型图像特征维度: {large.config.projection_dim}")  # 应该是 768