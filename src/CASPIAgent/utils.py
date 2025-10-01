# src/CASPIAgent/utils.py
import json
from functools import partial
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import StructuredTool
from src.CASPIAgent.tools import _run_gene_prediction_implementation


def build_agent(model, tools, prompt):
    agent = create_openai_tools_agent(model, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)



def safe_parse_output(tool_output):
    """安全解析工具输出为 dict"""
    if tool_output is None:
        return None
    try:
        if isinstance(tool_output, str):
            cleaned = tool_output.strip().replace("'", '"')
            return json.loads(cleaned)
        elif isinstance(tool_output, dict):
            return tool_output
    except Exception as e:
        print(f"[safe_parse_output] 解析失败: {e}")
    return None
