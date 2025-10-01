# webui.py
import gradio as gr

import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

# 从 tabs 目录导入各个 Tab 的创建函数
from tabs.agent_tab import agent_tab
from tabs.gemfactory_tab import gemfactory_tab

# --- 全局 CSS 样式 ---
CSS = """
/* 字体与背景 */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
body, .gradio-container { 
    font-family: 'Inter', sans-serif; 
    background-color: #F5F7FA; 
    color: #111827;
}

/* 聊天窗口卡片化 */
#chatbot {
    background-color: #FFFFFF !important;
    border-radius: 16px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
    padding: 12px !important;
}

/* 自定义文件上传提示 */
#my_unique_file_uploader button > div {
    font-size: 0 !important; /* 隐藏原始文字 */
}
#my_unique_file_uploader button > div::after {
    content: "将您的报告拖拽至此 或 点击选择";
    font-size: 1rem;
    color: #666;
}
#my_unique_file_uploader .icon-wrap {
    display: inline-block;
    width: 24px;
    height: 24px;
}

/* 隐藏 Gradio 默认 footer */
footer { visibility: hidden; }

/* 页面标题 */
#app-title h1 {
    font-size: 2.8em;
    color: #2c3e50;
    margin-bottom: 2px;
    font-weight: 600;
}
#app-title .author {
    font-size: 1.1em;
    color: #7f8c8d;
    text-align: left;
}

/* 功能区标题 */
#section-title {
    text-align: center;
    font-size: 2em;
    color: #34495e;
    margin-top: 10px;
    margin-bottom: 0px;
}
"""

# --- 主 UI 构建 ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=CSS, fill_height=True) as demo:
    
    # 顶部标题区
    with gr.Row(show_progress=True):
        gr.Image(
            value="static/logo.png",
            height=150,
            width=150,
            show_download_button=False,
            show_fullscreen_button=False,
            interactive=False,
            show_label=False
        )
        with gr.Column(scale=5, elem_id="app-title"):
            gr.Markdown(
                '<h1>CASPIA — 2025 SJTU-Software</h1>'
                '<p class="author">by <a href="https://github.com/victorzhu30" target="_blank">VictorZhu</a></p>'
            )
    
    # 标签页
    with gr.Tabs():
        with gr.TabItem("🤖 CASPIAgent"):
            agent_tab()
        with gr.TabItem("🧬 GEMFactory"):
            gemfactory_tab()
        with gr.TabItem("🔍 CASPIA-RAG"):
            gr.Markdown("## CASPIA-RAG")

# --- 启动应用 ---
if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", share=True, debug=True)
