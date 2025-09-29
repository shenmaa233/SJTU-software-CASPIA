# webui.py

import gradio as gr

# 从 tabs 目录导入各个 Tab 的创建函数
from tabs.agent_tab import create_agent_tab
from tabs.gemfactory_tab import gemfactory_tab

# --- 从 Agent 标签页提取出的共享 CSS ---
# 将其放在主文件中，以便应用于所有标签页
CSS = """
/* --- 全局字体与背景 --- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
body, .gradio-container { 
    font-family: 'Inter', sans-serif; 
    background-color: #F5F7FA; 
    color: #111827;
}
/* ... (您提供的其余所有 CSS 样式) ... */

/* --- 聊天窗口卡片化 --- */
#chatbot {
    background-color: #FFFFFF !important;
    border-radius: 16px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
    padding: 12px !important;
}
/* ... etc ... */
"""

# --- 主 UI 构建 ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=CSS, fill_height=True) as demo:
    gr.Markdown("# CASPIA")
    
    with gr.Tabs():
        # --- Agent Tab ---
        with gr.TabItem("Agent"):
            create_agent_tab() # 调用函数来构建这个 Tab 的内容

        # --- GEMFactory Tab ---
        with gr.TabItem("GEMFactory"):
            gemfactory_tab() # 调用函数来构建这个 Tab 的内容

        # --- 预留的 Tab ---
        with gr.TabItem("Data Visualization (Reserved)"):
            gr.Markdown("## 此处为数据可视化功能面板")
            gr.Markdown("未来可以集成如 Plotly、Matplotlib 等图表展示功能。")
            
        with gr.TabItem("Model Comparison (Reserved)"):
            gr.Markdown("## 此处为模型比较功能面板")
            gr.Markdown("可以上传多个模型文件（如 ecGEM.json），并对它们的性能指标进行比较。")


# --- 启动应用 ---
if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", share=True, debug=True)