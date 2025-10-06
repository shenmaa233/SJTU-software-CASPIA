import os
import time

import gradio as gr

from src.CASPIA_RAG.agent import Agent
from src.CASPIA_RAG.util import render_markdown

def stream_respond(message, chat_history, mode_selector):
    print("----------- 函数开始 -----------")
    print(f"收到的消息 (message): {message}")
    print(f"收到的历史 (chat_history): {chat_history}")
    print(f"历史的数据类型: {type(chat_history)}")

    chat_history.append((message, "Generating answer..."))
    yield chat_history, ""

    # chat_history[-1][1] = "" TypeError: 'tuple' object does not support item assignment
    chat_history[-1] = (message, "")
    bot_message = render_markdown(agent.process(message, mode_selector))
    print(f"bot_message: {bot_message}")
    
    for char in bot_message:
        chat_history[-1] = (chat_history[-1][0], chat_history[-1][1] + char)
        time.sleep(0.005)
        yield chat_history, ""
    print("----------- 函数结束 -----------")

    return chat_history

agent = Agent()

user_avatar_path = 'static/rag_images/user_avatar.png' if os.path.exists('static/rag_images/user_avatar.png') else None
bot_avatar_path = 'static/rag_images/bot_avatar.png' if os.path.exists('static/rag_images/bot_avatar.png') else None

custom_css = """
.gradio-container { max-width: 95% !important; margin: 0 auto !important; } footer { visibility: hidden; }
"""

change_mode_js_code = """
() => {
    // 获取当前的 URL
    const url = new URL(window.location);

    // 检查 URL 的查询参数中是否包含 '__theme=dark'
    if (url.searchParams.get('__theme') === 'dark') {
        // 如果已经是 dark 模式，则移除该参数，切换回 light 模式
        url.searchParams.delete('__theme');
    } else {
        // 如果是 light 模式，则添加该参数，切换到 dark 模式
        url.searchParams.set('__theme', 'dark');
    }

    // 将浏览器重定向到新的 URL，页面会以新主题重新加载
    window.location.href = url.href;
}
"""
def rag_tab():
    with gr.Blocks(theme=gr.themes.Monochrome(), fill_height=True, css=custom_css) as demo:
        gr.Markdown(
            """
            # 🧬 Cell Design AI Assistant
            Welcome to the assistant. I have integrated a local literature knowledge base, Google search, and general conversational abilities. Please enter your question below.
            **Disclaimer:** The answers provided by this assistant are for research reference only and do not constitute professional advice of any kind.
            """
        )
        
        chatbot = gr.Chatbot(
            label="Chat",
            height=480,
            avatar_images=(user_avatar_path, bot_avatar_path),
            render_markdown=True,
            latex_delimiters=[
                {"left": "$$", "right": "$$", "display": True},
                {"left": "$", "right": "$", "display": False}
            ]
        )

        with gr.Row(equal_height=True):
            textbox = gr.Textbox(lines=1, max_lines=5, placeholder="Please enter your question here... (Shift+Enter or click the Send button to submit)", container=False, scale=7)
                                # label info
            submit_btn = gr.Button("🚀 Send", variant="primary", scale=1)
            clear_btn = gr.Button("🗑️ Clear Histroy", variant="stop", scale=1)
            # UI Theme Toggle: Switch between light/dark appearance modes
            # toggle_btn = gr.Button(value="Change Theme ☀️/🌙", scale=1)

        gr.Examples(
            examples=[
                'Who are you?', 
                'What is GEM?', 
                'Tell me about COBRApy', 
                "In FSEOF, the targets were selected by identifying fluxes that increased upon the application of the enforced objective flux without changing the reaction's direction. How is this mathematically formulated?", 
                'What is iGEM? How has SJTU-Software performed over the years?' 
                ],
            inputs=textbox,
            label="Example Questions (Click to fill)",
            # examples_per_page=3
        )

        mode_selector = gr.Radio(
                            choices=["General Mode", "Expert Mode"], 
                            value="General Mode", 
                            label="Mode Selection"
                        )
            
        submit_action = textbox.submit(
            fn=stream_respond, 
            inputs=[textbox, chatbot, mode_selector], 
            outputs=[chatbot, textbox],
            queue=True 
        )

        submit_btn.click(
            fn=stream_respond, 
            inputs=[textbox, chatbot, mode_selector], 
            outputs=[chatbot, textbox],
            queue=True
        )

        clear_btn.click(
            fn=lambda: ([], ""), 
            inputs=None, 
            outputs=[chatbot, textbox], 
            queue=False
        )

        # 将按钮的 click 事件绑定到上面的 JavaScript 代码
        # fn=None 表示点击按钮时，不调用任何 Python 函数
        # toggle_btn.click(fn=None, js=change_mode_js_code)