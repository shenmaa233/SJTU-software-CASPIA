"""
========================================
CASPIAgent Tab - 简洁版
========================================

使用原生 Gradio 组件
- 极简代码
- 功能完整
- 易于维护

作者: SJTU-Software Team
日期: 2025-10-04
========================================
"""

import gradio as gr
import asyncio
import shutil
from pathlib import Path
from datetime import datetime
from src.CASPIAgent.service import AgentService


# ========================================
# 配置
# ========================================

# Agent 服务
agent_service = AgentService()

# 文件上传目录
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# ========================================
# 文件处理接口
# ========================================

def handle_file_upload(file):
    """
    处理文件上传（接口函数，可自定义）
    
    Args:
        file: Gradio File 对象 (临时路径)
        
    Returns:
        tuple: (保存的文件路径, 显示消息)
    """
    if file is None:
        return None, "未选择文件"
    
    try:
        # 获取原始文件名
        original_name = Path(file.name).name
        
        # 生成时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(original_name).suffix
        new_filename = f"{timestamp}_{original_name}"
        
        # 保存路径
        saved_path = UPLOAD_DIR / new_filename
        
        # 复制文件
        shutil.copy2(file.name, saved_path)
        
        message = f"✅ 文件已上传: {original_name}\n保存位置: {saved_path}"
        
        return str(saved_path), message
        
    except Exception as e:
        return None, f"❌ 上传失败: {str(e)}"


# ========================================
# 聊天处理
# ========================================

async def chat_with_agent(message, history, uploaded_file_path, session_state):
    """
    与 Agent 聊天（流式生成）
    
    Args:
        message: 用户消息
        history: 聊天历史 [[user_msg, bot_msg], ...]
        uploaded_file_path: 上传的文件路径
        session_state: 会话状态
        
    Yields:
        tuple: (历史, 工具调用文本, 会话状态)
    """
    if not message.strip():
        yield history, "", session_state
        return
    
    # 准备文件对象
    file_obj = None
    if uploaded_file_path:
        class FileWrapper:
            def __init__(self, path):
                self.name = path
        file_obj = FileWrapper(uploaded_file_path)
    
    # 转换历史格式
    history_dict = []
    for user_msg, bot_msg in history:
        if user_msg:
            history_dict.append({"role": "user", "content": user_msg})
        if bot_msg:
            history_dict.append({"role": "assistant", "content": bot_msg})
    
    try:
        # 构建 agent
        agent_executor, final_input, chat_history, new_session_state = await agent_service.run(
            message, history_dict, file_obj, session_state
        )
        
        # 添加用户消息
        history.append([message, ""])
        
        # 工具调用记录
        tool_calls_text = ""
        current_response = ""
        
        # 流式执行
        async for event in agent_executor.astream_events(
            {"input": final_input, "chat_history": chat_history},
            version="v1"
        ):
            event_name = event["event"]
            
            # 工具开始
            if event_name == "on_tool_start":
                tool_name = event.get("name", "unknown")
                tool_calls_text += f"🔧 调用工具: {tool_name}\n"
                yield history, tool_calls_text, session_state
            
            # 工具结束
            elif event_name == "on_tool_end":
                tool_name = event.get("name", "unknown")
                output = str(event["data"].get("output", ""))[:200]
                tool_calls_text += f"✅ {tool_name} 完成\n{output}\n\n"
                yield history, tool_calls_text, session_state
            
            # 流式内容
            elif event_name == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and hasattr(chunk, "content"):
                    content = chunk.content
                    if content:
                        current_response += content
                        history[-1][1] = current_response
                        yield history, tool_calls_text, session_state
            
            # 链结束
            elif event_name == "on_chain_end":
                output = event["data"].get("output")
                if output and isinstance(output, dict) and "output" in output:
                    if not current_response:
                        current_response = output["output"]
                        history[-1][1] = current_response
        
        # 确保有响应
        if not history[-1][1]:
            history[-1][1] = "处理完成。"
        
        yield history, tool_calls_text, new_session_state
        
    except Exception as e:
        error_msg = f"❌ 错误: {str(e)}"
        if history and len(history) > 0:
            history[-1][1] = error_msg
        else:
            history.append([message, error_msg])
        yield history, tool_calls_text, session_state


# ========================================
# UI 界面
# ========================================

def agent_tab():
    """创建 Agent Tab"""
    
    # 状态变量
    session_state = gr.State(dict)
    uploaded_file_path = gr.State(None)
    
    with gr.Row():
        with gr.Column(scale=2):
            # 标题
            gr.Markdown("## 🤖 CASPIAgent")
            gr.Markdown("专注于合成生物学和代谢工程的智能助手")
            
            # 聊天窗口
            chatbot = gr.Chatbot(
                label="对话",
                height=500,
                show_copy_button=True,
            )
            
            # 输入区域
            with gr.Row():
                msg_input = gr.Textbox(
                    label="消息",
                    placeholder="输入您的问题...",
                    scale=4,
                    lines=2,
                )
                with gr.Column(scale=1):
                    send_btn = gr.Button("发送", variant="primary", size="lg")
                    clear_btn = gr.Button("清空", size="sm")
        
        with gr.Column(scale=1):
            # 文件上传
            gr.Markdown("### 📎 文件上传")
            file_input = gr.File(
                label="选择文件",
                file_types=None,  # 接受所有文件类型
            )
            upload_status = gr.Textbox(
                label="上传状态",
                interactive=False,
                lines=3,
            )
            
            # 工具调用显示
            gr.Markdown("### 🔧 工具调用")
            tool_display = gr.Textbox(
                label="",
                interactive=False,
                lines=10,
                show_label=False,
            )
            
            # 快捷示例
            gr.Markdown("### 💡 快捷示例")
            example_btn1 = gr.Button("📝 基因组注释", size="sm")
            example_btn2 = gr.Button("🔬 Kcat 预测", size="sm")
            example_btn3 = gr.Button("❓ 功能介绍", size="sm")
    
    # 使用说明
    with gr.Accordion("📖 使用说明", open=False):
        gr.Markdown("""
        ### 功能说明
        
        1. **文件上传**: 点击右侧"选择文件"上传任意类型的文件
        2. **聊天对话**: 在输入框中输入问题，点击发送
        3. **工具调用**: Agent 会自动调用相应工具，右侧显示执行情况
        4. **快捷示例**: 点击快捷按钮快速输入常用问题
        
        ### 可用工具
        
        - 🧬 基因预测 (GeneMarkS)
        - 🔬 Kcat 预测
        - 📊 蛋白质提取
        """)
    
    # ========================================
    # 事件绑定
    # ========================================
    
    # 文件上传
    file_input.change(
        fn=handle_file_upload,
        inputs=[file_input],
        outputs=[uploaded_file_path, upload_status],
    )
    
    # 发送消息
    send_event = send_btn.click(
        fn=chat_with_agent,
        inputs=[msg_input, chatbot, uploaded_file_path, session_state],
        outputs=[chatbot, tool_display, session_state],
    ).then(
        fn=lambda: "",
        outputs=[msg_input],
    )
    
    # 回车发送
    msg_input.submit(
        fn=chat_with_agent,
        inputs=[msg_input, chatbot, uploaded_file_path, session_state],
        outputs=[chatbot, tool_display, session_state],
    ).then(
        fn=lambda: "",
        outputs=[msg_input],
    )
    
    # 清空对话
    clear_btn.click(
        fn=lambda: ([], "", {}),
        outputs=[chatbot, tool_display, session_state],
    )
    
    # 快捷示例
    example_btn1.click(
        fn=lambda: "请对上传的文件进行分析和注释",
        outputs=[msg_input],
    )
    
    example_btn2.click(
        fn=lambda: "请帮我预测酶的 kcat 值",
        outputs=[msg_input],
    )
    
    example_btn3.click(
        fn=lambda: "你有哪些功能？可以帮我做什么？",
        outputs=[msg_input],
    )


# ========================================
# 测试入口
# ========================================

if __name__ == "__main__":
    """独立测试"""
    with gr.Blocks(title="CASPIAgent - Simple") as demo:
        agent_tab()
    
    demo.launch(server_name="0.0.0.0", share=False, debug=True)

