# tabs/agent_tab.py
import gradio as gr
from dotenv import load_dotenv
from langchain import hub
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.CASPIAgent.utils import build_agent
from src.CASPIAgent.conversation import format_input_message, convert_history
from src.CASPIAgent.tools import multiply, predict_kcat, extract_protein_from_predicted_file, make_file_prediction_tool

SYSTEM_PROMPT = """You are CASPIAgent, an expert virtual assistant in computational biology and synthetic biology. 
Your role is to help researchers build genome-scale metabolic models (GEMs), complete missing parameters, 
and design optimization strategies through clear, step-by-step reasoning. 
Always explain your thought process in detail, provide interpretable outputs, 
and generate structured results that can be reused in downstream analysis. 
If the user’s request is ambiguous, ask clarifying questions before execution. 
Communicate in a professional yet accessible way, so that even users without programming or bioinformatics backgrounds 
can understand and apply the results."""

def agent_tab():
    load_dotenv()

    # === 初始化模型与 Prompt ===
    model = ChatOpenAI(
        model_name="deepseek-chat",
        openai_api_base="https://api.deepseek.com",
        openai_api_key="sk-e3e38506b27c4ed7bbd9c0e25a7a7de4",
        temperature=0.7,
        max_tokens=2048,
        streaming=True,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}"),
        ("system", "{chat_history}")
    ])

    # === 基础工具 ===
    base_tools = [multiply, predict_kcat, extract_protein_from_predicted_file]

    async def predict(message, history, tool_history, uploaded_file, session_state):
        tools = base_tools[:]
        session_state = session_state or {}

        if uploaded_file:
            tools.append(make_file_prediction_tool(uploaded_file)) # 添加基因注释工具
            session_state = {}  # reset 会话状态

        final_input = format_input_message(message, uploaded_file, session_state)
        agent_executor = build_agent(model, tools, prompt)
        chat_history = convert_history(history)

        ui_history = list(history)
        ui_history.append({'role': 'user', 'content': message})
        ui_history.append({'role': 'assistant', 'content': ""})

        tool_history_md = "" if tool_history == "*Agent 活动为空*" else tool_history
        yield {
            chatbot_display: ui_history,
            tool_history_display: tool_history_md,
            msg_input: "",
            file_upload: None,
            session_state_display: session_state,
        }

        turn_separator = "\n" if tool_history_md else ""

        async for event in agent_executor.astream_events(
            {"input": final_input, "chat_history": chat_history}, version="v1"
        ):
            event_name = event["event"]

            if event_name == "on_tool_start":
                tool_name = event['name']
                tool_log_entry = f'{turn_separator}<div class="tool-tag">{tool_name}</div>'
                tool_history_md += tool_log_entry
                turn_separator = "\n"
                yield {tool_history_display: tool_history_md}

            elif event_name == "on_tool_end":
                tool_output = event["data"].get("output")
                tool_name = event["name"]
                if tool_name == "run_gene_prediction_real":
                    from src.CASPIAgent.utils import safe_parse_output
                    parsed = safe_parse_output(tool_output)
                    if parsed and parsed.get("protein_faa_path"):
                        session_state["protein_faa_path"] = parsed["protein_faa_path"]
                yield {session_state_display: session_state}

            elif event_name == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                if chunk:
                    ui_history[-1]['content'] += chunk
                    yield {chatbot_display: ui_history}

    # === UI 布局 ===
    with gr.Row(elem_id="main-layout"):
        with gr.Column(scale=7):
            chatbot_display = gr.Chatbot(
                elem_id="chatbot", label="对话", height=650, type="messages"
            )
            file_upload = gr.File(
                label="上传基因组文件 (新文件会重置流程)",
                file_count="single",
                file_types=['.fa', '.fna', '.fasta'],
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    elem_id="chat-input", scale=5, show_label=False,
                    placeholder="与 Agent 对话，或上传文件后让它处理...", container=False
                )
                submit_btn = gr.Button(value="发送", variant="primary", elem_id="submit-button", scale=1)

        with gr.Column(scale=3, elem_id="right-panel"):
            tool_history_display = gr.Markdown(value="*Agent 活动为空*", label="工具历史")
            session_state_display = gr.JSON(label="会话状态", scale=1)

    session_state = gr.State(value={})

    submit_btn.click(
        predict,
        inputs=[msg_input, chatbot_display, tool_history_display, file_upload, session_state],
        outputs=[chatbot_display, tool_history_display, msg_input, file_upload, session_state_display],
    )
    msg_input.submit(
        predict,
        inputs=[msg_input, chatbot_display, tool_history_display, file_upload, session_state],
        outputs=[chatbot_display, tool_history_display, msg_input, file_upload, session_state_display],
    )
