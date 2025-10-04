# src/CASPIAgent/conversation.py
import json
from langchain.schema import HumanMessage, AIMessage


def format_input_message(message, uploaded_file, session_state):
    """根据用户输入、上传文件和上下文拼接最终输入消息"""
    final_input = message
    if uploaded_file:
        final_input += f"\n\n[系统提示：用户已上传基因组文件 '{uploaded_file.name}'，状态已重置。]"
    if session_state:
        context_str = json.dumps(session_state, indent=2)
        final_input += f"\n\n[系统提示：已有会话上下文：\n{context_str}]"
    return final_input


def convert_history(history):
    """UI历史转 LangChain 格式"""
    lc_history = []
    for msg in history:
        if msg['role'] == "user":
            lc_history.append(HumanMessage(content=msg['content']))
        elif msg['role'] == "assistant":
            lc_history.append(AIMessage(content=msg['content']))
    return lc_history
