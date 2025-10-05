# src/CASPIAgent/service.py
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.CASPIAgent.utils import build_agent
from src.CASPIAgent.conversation import format_input_message, convert_history
from src.CASPIAgent.tools import (
    multiply, 
    predict_kcat, 
    submit_gene_annotation,
    submit_gem_build,
    check_task_status,
    make_file_prediction_tool
)

SYSTEM_PROMPT = """You are CASPIAgent, an expert virtual assistant in computational biology and synthetic biology. 
Your role is to help researchers build genome-scale metabolic models (GEMs), complete missing parameters, 
and design optimization strategies through clear, step-by-step reasoning.

CRITICAL RULES FOR TASK SUBMISSION:
1. When you use submit_gene_annotation or submit_gem_build tool:
   - Call the tool ONLY ONCE per user request
   - Once you receive a success response with a task_id, STOP immediately
   - Do NOT call the same tool again - the task is already running in background
   - Return the task_id to the user and tell them to check 'Tasks Monitor' tab

2. NEVER repeat tool calls:
   - If a tool returns success=True, the job is done
   - Do NOT retry or verify by calling again
   - Tasks run asynchronously in the background

Available task submission tools:
- submit_gene_annotation: Submit GeneMarkS annotation job (call ONCE only)
- submit_gem_build: Submit full GEM building pipeline (call ONCE only)
- check_task_status: Check status of a submitted task

For quick operations (like kcat prediction), you can use synchronous tools directly.

Always explain your thought process in detail, provide interpretable outputs, 
and generate structured results that can be reused in downstream analysis. 
If the user's request is ambiguous, ask clarifying questions before execution. 
Communicate in a professional yet accessible way."""

class AgentService:
    def __init__(self):
        load_dotenv()
        self.model = ChatOpenAI(
            model_name="deepseek-chat",
            openai_api_base="https://api.deepseek.com",
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0.7,
            max_tokens=2048,
            streaming=True,
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}"),
            ("system", "{chat_history}")
        ])
        self.base_tools = [
            multiply, 
            predict_kcat, 
            submit_gene_annotation,
            submit_gem_build,
            check_task_status
        ]

    async def run(self, message, history, uploaded_file, session_state):
        tools = self.base_tools[:]
        if uploaded_file:
            tools.append(make_file_prediction_tool(uploaded_file))
            session_state = {}

        final_input = format_input_message(message, uploaded_file, session_state)
        agent_executor = build_agent(self.model, tools, self.prompt)
        chat_history = convert_history(history)
        return agent_executor, final_input, chat_history, session_state
