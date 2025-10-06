from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from .util import get_chat_model
from .prompt import TRANSLATE_PROMPT_TPL

def translate_query_to_english(query_chinese):
    """使用 LLM 将中文问题翻译成英文"""
    prompt = PromptTemplate.from_template(TRANSLATE_PROMPT_TPL)
    # 创建一个专门用于翻译的 LLM 实例
    translator_llm = LLMChain(llm = get_chat_model(), prompt = prompt)
    response = translator_llm.invoke({'query_chinese': query_chinese})
    return response['text']

if __name__ == "__main__":
    chinese_question = "你如何评估自己是否是一位专业人员？"
    english_question = translate_query_to_english(chinese_question)
    print(f"翻译后的英文问题: {english_question}")