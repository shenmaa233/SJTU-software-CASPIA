# frontend/agent_ui.py
import gradio as gr
import websocket
import threading
import json

API_URL = "ws://localhost:8000/ws/chat"

def ws_predict(message, history, tool_history, uploaded_file, session_state):
    ws = websocket.WebSocket()
    ws.connect(API_URL)

    payload = {
        "message": message,
        "history": history,
        "tool_history": tool_history,
        "session_state": session_state,
    }
    ws.send(json.dumps(payload))

    ui_history = list(history)
    ui_history.append({"role": "user", "content": message})
    ui_history.append({"role": "assistant", "content": ""})

    while True:
        response = json.loads(ws.recv())
        if response["type"] == "token":
            ui_history[-1]["content"] += response["content"]
            yield ui_history, tool_history, "", None, session_state
        elif response["type"] == "tool_start":
            tool_history += f"\n<div class='tool-tag'>{response['tool_name']}</div>"
            yield ui_history, tool_history, "", None, session_state
        elif response["type"] == "done":
            session_state = response["session_state"]
            break

    ws.close()

def agent_tab():
    with gr.Row():
        with gr.Column(scale=7):
            chatbot_display = gr.Chatbot(label="对话", height=650, type="messages")
            file_upload = gr.File(label="上传基因组文件", file_types=['.fa', '.fna', '.fasta'])
            msg_input = gr.Textbox(placeholder="与 Agent 对话...", container=False)
            submit_btn = gr.Button("发送", variant="primary")

        with gr.Column(scale=3):
            tool_history_display = gr.Markdown(value="*Agent 活动为空*")
            session_state_display = gr.JSON(label="会话状态")

    session_state = gr.State(value={})

    submit_btn.click(
        ws_predict,
        inputs=[msg_input, chatbot_display, tool_history_display, file_upload, session_state],
        outputs=[chatbot_display, tool_history_display, msg_input, file_upload, session_state_display],
    )
