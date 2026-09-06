import streamlit as st
import requests
import json
from PIL import Image
import io

# ---------- 页面配置 ----------
st.set_page_config(page_title="VLM 图像检索系统", layout="wide")
st.title("🔍 VLM 跨模态图像检索 Demo")
st.markdown("基于微调 CLIP 的 CIFAR-10 图像检索系统")

# ---------- 侧边栏信息 ----------
st.sidebar.markdown("""
## 📖 使用说明
1. 在输入框中输入文字描述（如 `a cat`）
2. 点击 **“检索”** 按钮
3. 系统会返回最匹配的 5 张图片

## 🧪 示例查询
点击下方按钮快速体验：
""")

# ---------- 示例查询按钮 ----------
col1, col2, col3 = st.sidebar.columns(3)
if col1.button("🐱 Cat"):
    st.session_state.query = "a photo of a cat"
if col2.button("🐶 Dog"):
    st.session_state.query = "a photo of a dog"
if col3.button("🚗 Car"):
    st.session_state.query = "a photo of a automobile"

# ---------- 主界面 ----------
query = st.text_input("输入检索描述", value=st.session_state.get("query", ""), placeholder="例如：a photo of a cat")

# ---------- 检索逻辑 ----------
def search_images(query, top_k=5):
    try:
        response = requests.post(
            "http://localhost:8000/search",
            json={"query": query, "top_k": top_k},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()["results"]
        else:
            st.error(f"API 错误: {response.status_code}")
            return []
    except requests.exceptions.ConnectionError:
        st.error("❌ 无法连接后端服务！请确保已运行 `python src/api.py`")
        return []

if st.button("检索", type="primary"):
    if query:
        with st.spinner("检索中..."):
            results = search_images(query)
        
        if results:
            st.subheader(f"📸 检索结果 (共 {len(results)} 张)")
            
            # 用 5 列显示图片
            cols = st.columns(min(len(results), 5))
            for i, (col, result) in enumerate(zip(cols, results)):
                try:
                    img = Image.open(result["image_path"])
                    col.image(img, caption=f"{result['image_path']}\n相似度: {result['score']:.4f}", use_container_width=True)
                except Exception as e:
                    col.error(f"图片加载失败: {result['image_path']}")
        else:
            st.warning("未找到匹配结果")
    else:
        st.warning("请输入检索描述")

# ---------- 底部 ----------
st.sidebar.markdown("""
---
### 📊 模型信息
- **模型**: CLIP (ViT-B/32) + 微调
- **数据集**: CIFAR-10 (2000 张)
- **R@1**: 100.00%
- **R@5**: 100.00%
""")