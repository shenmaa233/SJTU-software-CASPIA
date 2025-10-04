# src/CASPIAgent/service.py
import os
from dotenv import load_dotenv
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
        self.base_tools = [multiply, predict_kcat, extract_protein_from_predicted_file]

    async def run(self, message, history, tool_history, uploaded_file, session_state):
        tools = self.base_tools[:]
        if uploaded_file:
            tools.append(make_file_prediction_tool(uploaded_file))
            session_state = {}

        final_input = format_input_message(message, uploaded_file, session_state)
        agent_executor = build_agent(self.model, tools, self.prompt)
        chat_history = convert_history(history)
        return agent_executor, final_input, chat_history, session_state
