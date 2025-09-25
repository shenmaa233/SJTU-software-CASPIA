# tabs/agent_tab.py

import gradio as gr
from dotenv import load_dotenv
import os
import sys
import json
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import StructuredTool # <--- 1. 导入 StructuredTool
from langchain import hub
from src.tools import multiply, predict_kcat, extract_protein_from_predicted_file, _run_gene_prediction_implementation

def create_agent_tab():
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
    os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
    load_dotenv()

    model = ChatOpenAI(model_name="Qwen/Qwen3-4B-Instruct-2507-FP8", openai_api_base="http://127.0.0.1:8000/v1", openai_api_key="EMPTY", temperature=0.7, max_tokens=2048, streaming=True)
    
    base_tools = [multiply, predict_kcat, extract_protein_from_predicted_file]
    
    prompt = hub.pull("hwchase17/openai-tools-agent")
    
    async def predict_aesthetic(message: str, history: list[dict], tool_history: str, uploaded_file: gr.File | None, session_state: dict):
        current_tools = base_tools[:] 
        final_input_message = message

        if uploaded_file is not None:
            session_state = {} 
            run_prediction_partial = partial(_run_gene_prediction_implementation, genome_file=uploaded_file)
            
            # --- 2. 这里是关键的修改 ---
            # 为了让 StructuredTool 能正确推断无参数的结构，我们定义一个临时的包装函数
            def run_gene_prediction_wrapper():
                """当用户要求对他们刚刚上传的基因组文件进行基因预测时，使用此工具。此工具不需要任何参数，会自动处理已上传的文件。"""
                return run_prediction_partial()

            # 使用 StructuredTool.from_function 来创建工具
            # 它会从包装函数的签名和文档字符串中正确地推断出工具的结构
            file_prediction_tool = StructuredTool.from_function(
                func=run_gene_prediction_wrapper,
                name="run_gene_prediction_real", # 工具名称
            )
            
            current_tools.append(file_prediction_tool)
            
            final_input_message += f"\n\n[系统提示：用户已上传新基因组文件 '{os.path.basename(uploaded_file.name)}'，状态已重置。一个名为 'run_gene_prediction_real' 的工具现在可用。]"

        if session_state:
            context_str = json.dumps(session_state, indent=2)
            final_input_message += f"\n\n[系统提示：当前会话上下文中已有以下信息可供使用：\n{context_str}\n请利用这些信息回应用户请求。例如，如果已有 'protein_faa_path'，你可以使用 'extract_protein_from_predicted_file' 工具从中提取序列。]"

        agent = create_openai_tools_agent(model, current_tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=current_tools, verbose=True, handle_parsing_errors=True)

        langchain_chat_history = []
        for msg in history:
            if msg['role'] == "user": langchain_chat_history.append(HumanMessage(content=msg['content']))
            elif msg['role'] == "assistant": langchain_chat_history.append(AIMessage(content=msg['content']))

        ui_history = list(history)
        ui_history.append({'role': 'user', 'content': message})
        ui_history.append({'role': 'assistant', 'content': ""})

        if tool_history == "*Agent 活动为空*": tool_history_md = ""
        else: tool_history_md = tool_history
            
        yield { chatbot_display: ui_history, tool_history_display: tool_history_md, msg_input: "", file_upload: None, session_state_display: session_state }

        turn_separator = "\n" if tool_history_md else ""

        async for event in agent_executor.astream_events(
            {"input": final_input_message, "chat_history": langchain_chat_history},
            version="v1"
        ):
            event_name = event["event"]
            
            if event_name == "on_tool_start":
                tool_name = event['name']
                tool_log_entry = f'{turn_separator}<div class="tool-tag">{tool_name}</div>'
                tool_history_md += tool_log_entry
                turn_separator = "\n" 
                yield { tool_history_display: tool_history_md }

            elif event_name == "on_tool_end":
                tool_output = event["data"].get("output")
                tool_name = event["name"]
                
                if tool_name == "run_gene_prediction_real":
                    try:
                        # 检查 output 是否为 None
                        if tool_output is None:
                            print("警告: 工具 'run_gene_prediction_real' 返回了 None。")
                            
                            continue

                        if isinstance(tool_output, str):
                            cleaned_output = tool_output.strip().replace("'", '"')
                            output_dict = json.loads(cleaned_output)
                        else:
                            output_dict = tool_output
                        
                        protein_path = output_dict.get("protein_faa_path")
                        if protein_path:
                            session_state["protein_faa_path"] = protein_path
                            print(f"会话状态已更新: {session_state}")
                    except Exception as e:
                        print(f"解析工具输出时发生未知错误: {e}")

                yield { session_state_display: session_state }

            elif event_name == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                if chunk:
                    ui_history[-1]['content'] += chunk
                    yield { chatbot_display: ui_history }

    # UI 布局和事件绑定保持不变
    with gr.Row(elem_id="main-layout"):
        with gr.Column(scale=7):
            chatbot_display = gr.Chatbot(elem_id="chatbot", label="对话", height=650, type="messages")
            with gr.Row(elem_id="upload-area"):
                file_upload = gr.File(label="上传基因组文件 (新文件会重置流程)", file_count="single", file_types=['.fa', '.fna', '.fasta'])
            with gr.Row(elem_id="input-area"):
                msg_input = gr.Textbox(elem_id="chat-input", scale=5, show_label=False, placeholder="与 Agent 对话，或上传文件后让它处理...", container=False)
                submit_btn = gr.Button(value="发送", variant="primary", elem_id="submit-button", scale=1)
        
        with gr.Column(scale=3, elem_id="right-panel"):
            gr.Markdown("### Activity")
            tool_history_display = gr.Markdown(value="*Agent 活动为空*", label="工具历史")
            
            gr.Markdown("### Session State")
            session_state_display = gr.JSON(label="会话状态", scale=1)

    session_state = gr.State(value={})

    submit_btn.click(
        predict_aesthetic, 
        inputs=[msg_input, chatbot_display, tool_history_display, file_upload, session_state], 
        outputs=[chatbot_display, tool_history_display, msg_input, file_upload, session_state_display]
    )
    msg_input.submit(
        predict_aesthetic, 
        inputs=[msg_input, chatbot_display, tool_history_display, file_upload, session_state], 
        outputs=[chatbot_display, tool_history_display, msg_input, file_upload, session_state_display]
    )