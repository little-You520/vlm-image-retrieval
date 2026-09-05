import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import os
from tqdm import tqdm


# ---------- 1. 数据集 ----------
class CIFAR10TextDataset(Dataset):
    def __init__(self, manifest_path):
        self.items = []
        with open(manifest_path, 'r') as f:
            for line in f:
                self.items.append(json.loads(line))
        self.labels = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        text = f"a photo of a {self.labels[item['label']]}"
        return {"image": image, "text": text}


# ---------- 2. 对比损失 ----------
def contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    logits = (image_embeds @ text_embeds.T) / temperature
    labels = torch.arange(len(logits), device=logits.device)

    loss_i = nn.CrossEntropyLoss()(logits, labels)
    loss_t = nn.CrossEntropyLoss()(logits.T, labels)
    return (loss_i + loss_t) / 2


# ---------- 3. 训练函数（不使用 peft） ----------
def train():
    print("🚀 开始微调 (直接训练视觉编码器最后3层)...")

    # 加载模型
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to("cuda")

    # 冻结文本编码器
    for param in model.text_model.parameters():
        param.requires_grad = False

    # 冻结视觉编码器的前几层，只训练最后3层（减少参数量，显存友好）
    vision_layers = model.vision_model.encoder.layers
    for i, layer in enumerate(vision_layers):
        if i < len(vision_layers) - 3:  # 只训练最后3层
            for param in layer.parameters():
                param.requires_grad = False

    # 统计可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"✅ 可训练参数: {trainable_params:,} / 总参数: {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    model.train()

    # 数据加载
    dataset = CIFAR10TextDataset("data/manifest.jsonl")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: x)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in range(3):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/3")

        for batch in progress_bar:
            images = [item["image"] for item in batch]
            texts = [item["text"] for item in batch]

            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            outputs = model(**inputs)
            loss = contrastive_loss(outputs.image_embeds, outputs.text_embeds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch + 1} 完成, 平均 Loss: {avg_loss:.4f}")

    # 保存模型
    os.makedirs("checkpoints", exist_ok=True)
    model.save_pretrained("checkpoints/clip-finetuned-cifar10")
    processor.save_pretrained("checkpoints/clip-finetuned-cifar10")
    print("✅ 微调完成！模型已保存到 checkpoints/clip-finetuned-cifar10")


if __name__ == "__main__":
    train()