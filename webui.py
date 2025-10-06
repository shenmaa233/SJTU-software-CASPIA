# webui.py
import gradio as gr

import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

# 从 tabs 目录导入各个 Tab 的创建函数
from tabs.agent_tab import agent_tab  # 简洁版 - 原生 Gradio ⭐
from tabs.gemfactory_tab import gemfactory_tab
from tabs.tasks_monitor_tab import tasks_monitor_tab
from tabs.rag_tab import rag_tab

CSS = """
#header-row {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 20px;
}
#logo {
    display: flex;
    justify-content: center;
}
#logo img {
    border: none !important;
    box-shadow: none !important;
    background: none !important;
}
#app-title {
    text-align: center;
    margin-left: 20px;
}
"""

# --- 主 UI 构建 ---
with gr.Blocks(theme=gr.themes.Monochrome(), fill_height=True, css=CSS) as demo:
    
    with gr.Column(elem_id="header-row"):
        gr.Image(
            value="static/logo.png",
            height=150,
            width=150,
            show_download_button=False,
            show_fullscreen_button=False,
            interactive=False,
            show_label=False,
            container=False,
            elem_id="logo"
        )
        gr.Markdown(
            '<h1>CASPIA — 2025 SJTU-Software</h1>'
            '<p class="author">GitHub <a href="https://github.com/shenmaa233/SJTU-software-CASPIA" target="_blank">SJTU-software-CASPIA</a></p>',
            elem_id="app-title"
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
            rag_tab()

# --- 启动应用 ---
if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", share=True, debug=True)
