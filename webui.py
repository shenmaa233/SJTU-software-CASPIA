# webui.py
import gradio as gr

import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

# 从 tabs 目录导入各个 Tab 的创建函数
from tabs.agent_tab import agent_tab  # 简洁版 - 原生 Gradio ⭐
from tabs.gemfactory_tab import gemfactory_tab
from tabs.tasks_monitor_tab import tasks_monitor_tab

# --- 全局 CSS 样式 ---
CSS = """
/* 高级黑白风格 - 全局样式 */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

body, .gradio-container { 
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; 
    background-color: #ffffff; 
    color: #111827;
    letter-spacing: -0.01em;
}

/* 聊天窗口 */
#chatbot {
    background-color: #ffffff !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    border: 1px solid #e5e7eb !important;
    padding: 12px !important;
}

/* 文件上传提示 */
#my_unique_file_uploader button > div {
    font-size: 0 !important;
}
#my_unique_file_uploader button > div::after {
    content: "拖拽文件或点击选择";
    font-size: 13px;
    color: #6b7280;
}
#my_unique_file_uploader .icon-wrap {
    display: inline-block;
    width: 20px;
    height: 20px;
}

/* 隐藏 footer */
footer { visibility: hidden; }


/* 页面标题 - 简约风格 */
#app-title h1 {
    font-size: 2.5em;
    color: #111827;
    margin-bottom: 4px;
    font-weight: 500;
    letter-spacing: -0.02em;
}
#app-title .author {
    font-size: 0.95em;
    color: #6b7280;
    text-align: left;
    font-weight: 400;
}

#app-title a {
    color: #111827;
    text-decoration: none;
    border-bottom: 1px solid #e5e7eb;
    transition: all 0.15s ease;
}

#app-title a:hover {
    color: #000000;
    border-bottom-color: #000000;
}

/* 功能区标题 */
#section-title {
    text-align: center;
    font-size: 1.8em;
    color: #111827;
    margin-top: 10px;
    margin-bottom: 0px;
    font-weight: 500;
    letter-spacing: -0.01em;
}

/* Tab 标签样式 */
.tabs button {
    background: #ffffff !important;
    color: #6b7280 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-weight: 400 !important;
    font-size: 14px !important;
    transition: all 0.15s ease !important;
}

.tabs button:hover {
    color: #111827 !important;
    background: #fafafa !important;
}

.tabs button.selected {
    color: #000000 !important;
    border-bottom-color: #000000 !important;
    font-weight: 500 !important;
}

/* 全局按钮样式 */
button {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    transition: all 0.15s ease !important;
}

/* 输入框全局样式 */
input, textarea {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    border-color: #d1d5db !important;
    transition: all 0.15s ease !important;
}

input:focus, textarea:focus {
    border-color: #000000 !important;
    box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.05) !important;
}

/* Logo 图片优化 */
img {
    border-radius: 8px;
}

/* 卡片通用样式 */
.card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
}

/* 滚动条全局样式 */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: #f9fafb;
}

::-webkit-scrollbar-thumb {
    background: #d1d5db;
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: #9ca3af;
}
"""

# --- 主 UI 构建 ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="slate"), css=CSS, fill_height=True) as demo:
    
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
                '<p class="author">GitHub <a href="https://github.com/shenmaa233/SJTU-software-CASPIA" target="_blank">SJTU-software-CASPIA</a></p>'
            )
    
    # 标签页
    with gr.Tabs():
        with gr.TabItem("🤖 CASPIAgent"):
            agent_tab()
        with gr.TabItem("🧬 GEMFactory"):
            gemfactory_tab()
        with gr.TabItem("📊 Tasks Monitor"):
            tasks_monitor_tab()
        with gr.TabItem("🔍 CASPIA-RAG"):
            gr.Markdown("## CASPIA-RAG")

# --- 启动应用 ---
if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", share=True, debug=True)
