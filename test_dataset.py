import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from datasets import load_dataset

print("正在加载 CIFAR-10 数据集（首次需要下载），请稍候...")
dataset = load_dataset("cifar10", split="train", streaming=True)

print("\n=== 前 5 条数据样例 ===")
for i, sample in enumerate(dataset):
    if i >= 5:
        break
    print(f"\n第 {i+1} 条：")
    print(f"图片: {sample['img']}")
    print(f"标签: {sample['label']}")  # 0=飞机, 1=汽车, 2=鸟, 3=猫, 4=鹿, 5=狗, 6=青蛙, 7=马, 8=船, 9=卡车