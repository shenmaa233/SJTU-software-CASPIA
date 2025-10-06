# --- 1. 导入所有需要的库 (放在文件顶部) ---
from dotenv import load_dotenv
from pathlib import Path
import sys
import os
import shutil
import json

from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from transformers import AutoTokenizer

from .util import get_embedding_model 


# --- 2. 设置项目路径 ---
load_dotenv()
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
input_dir = project_root / 'mineru_markdown' / 'revised_output'
markdown_files = list(input_dir.glob('*.md'))

# --- 3. 循环处理所有文件 ---
all_docs = [] 
print(f"找到 {len(markdown_files)} 个 Markdown 文件，开始处理...")

for markdown_file in markdown_files:
    print(f"\n======================\n正在处理文件: {markdown_file.name}\n======================")
    # loader = UnstructuredMarkdownLoader(str(markdown_file))
    loader = TextLoader(str(markdown_file), encoding='utf-8')
    docs_raw = loader.load()

    if not docs_raw:
        print("警告: 文件为空或加载失败，跳过。")
        continue

    # 提取文本内容
    # 通常，UnstructuredMarkdownLoader 会将整个文件的内容加载为列表中的第一个 Document。
    markdown_text = docs_raw[0].page_content

    # --- 阶段一：结构化分割 ---
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=True # 推荐设置为 True，让内容更干净
    )
    md_header_splits = markdown_splitter.split_text(markdown_text)
    print(f"阶段一：结构化分割后，得到 {len(md_header_splits)} 个语义块。")

    if not md_header_splits:
        print("警告: 结构化分割未能生成任何块，跳过此文件。")
        continue

    # --- 阶段二：长度分割 ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE")),  
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP"))
    )
    all_splits_for_file = text_splitter.split_documents(md_header_splits)
    print(f"阶段二：长度分割后，得到 {len(all_splits_for_file)} 个最终片段。")

    # 将当前文件的处理结果累加到总的列表中
    all_docs.extend(all_splits_for_file)

# output_data = []
# for doc in all_docs:
#     output_data.append({
#         'page_content': doc.page_content,
#         'metadata': doc.metadata
#     })
# debug_json_path = './debug_output.json'
# with open(debug_json_path, 'w', encoding='utf-8') as f:
#     # indent=2 让 JSON 文件格式化，更易读
#     json.dump(output_data, f, ensure_ascii=False, indent=2)
# print(f"结构化的分割结果已写入调试文件: {debug_json_path}")

# tokenizer = AutoTokenizer.from_pretrained(os.getenv("EMBEDDING_MODEL")) 
# for doc in all_docs:
#     text_chunk = doc.page_content
#     tokens = tokenizer.encode(text_chunk)
#     print(f"这个文本块的词块数量是: {len(tokens)}") # -> 确保这个数字 <= _

# 找到那个最大的块
# max_len = 0
# max_chunk = None
# tokenizer = AutoTokenizer.from_pretrained(os.getenv("EMBEDDING_MODEL")) 
# for doc in all_docs:
#     token_count = len(tokenizer.encode(doc.page_content))
#     if token_count > max_len:
#         max_len = token_count
#         max_chunk = doc
# print(f"\n\n--- 最大的文本块 ---")
# print(f"Token 数量: {max_len}")
# print(f"字符数量: {len(max_chunk.page_content)}")
# print("元数据 (Metadata):")
# print(max_chunk.metadata)
# print("内容预览 (前 500 字符):")
# print(max_chunk.page_content[:500])
# # 如果想看全部内容，可以写入文件
# with open("max_chunk_content.txt", "w", encoding="utf-8") as f:
#     f.write(max_chunk.page_content)

# --- 4. 将所有处理好的文档存入数据库 ---
if all_docs:
    db_directory = project_root / 'db' 
    # 检查目录是否存在，如果存在就删除它
    if db_directory.exists():
        print(f"发现已存在的数据库目录: {db_directory}，正在删除...")
        shutil.rmtree(db_directory)
        print("旧数据库已删除。")
    db_directory.mkdir()

    print(f"\n--- 所有文件处理完毕，准备将总共 {len(all_docs)} 个文档片段存入数据库 ---")

    vdb = Chroma.from_documents(
        documents=all_docs, # 使用累积了所有文件结果的 all_docs
        embedding=get_embedding_model(),
        persist_directory=str(db_directory) # 转换为字符串
    )
    vdb.persist()
    print(f"--- 成功将文档存入 ChromaDB: {db_directory} ---")
else:
    print("\n--- 所有文件处理完毕，但没有生成任何可用的文档片段，数据库未创建。 ---")