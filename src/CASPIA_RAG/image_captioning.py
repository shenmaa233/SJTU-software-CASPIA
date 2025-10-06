from dotenv import load_dotenv
import base64
import os
from pathlib import Path
import sys
import re

from tqdm import tqdm

from .util import *
from .prompt import *

# 为了能正确导入 'code' 目录下的其他文件，我们将项目根目录添加到 python 路径中
# 项目根目录是 'LLM-RAG'
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

load_dotenv()

def get_image_description(vision_model, image_path:Path):
    """
    使用 Moonshot Vision API 为单个图片生成详细的描述。
    """
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()

        # 使用标准库 base64.b64encode 函数将图片编码成 base64 格式
        image_url = f"data:image/{image_path.suffix[1:]};base64,{base64.b64encode(image_data).decode('utf-8')}"

        completion = vision_model.chat.completions.create(
            model=os.getenv('VISION_MODEL'),
            messages=[
                {
                    "role": "system", 
                    "content": "你是 Kimi。"
                },
                {
                    "role": "user",
                    # 注意这里，content 由原来的 str 类型变更为一个 list，这个 list 中包含多个部分的内容，图片（image_url）是一个部分（part），
                    # 文字（text）是一个部分（part）
                    "content": [
                        {
                            "type": "image_url", # <-- 使用 image_url 类型来上传图片，内容为使用 base64 编码过的图片内容
                            "image_url": {
                                "url": image_url
                            }
                        }, 
                        {
                            "type": "text", 
                            "text": DESCRIBE_IMAGE_PROMPT_TPL # <-- 使用 text 类型来提供文字指令，例如“描述图片内容”
                        },
                    ],
                },
            ],
        )
        return completion.choices[0].message.content
    
    except FileNotFoundError:
        print(f"警告: 图片文件未找到，路径: {image_path}")
        return "[图片未找到]"
    except Exception as e:
        print(f"处理图片 {image_path.name} 时发生错误: {e}")
        return "[生成描述时出错]"

def process_markdown_file(vision_model, md_path: Path, revised_output_dir: Path):
    """
    读取一个 markdown 文件，查找所有图片，生成描述，并保存一个插入了描述的新版本。
    如果修订版文件已存在，则直接跳过。
    """

    # 1.首先确定目标文件的路径
    revised_md_path = revised_output_dir / md_path.name

    # 2.检查目标文件是否已存在，如果存在则直接返回，不再执行后续操作
    if revised_md_path.exists():
        print(f"文件已存在，跳过: {revised_md_path}")
        return  # 提前退出函数

    # --- 如果程序能运行到这里，说明目标文件不存在，可以继续处理 ---
    print(f"正在处理新文件: {md_path.name}")
    content = md_path.read_text(encoding='utf-8')
    
    # 使用正则表达式查找所有 markdown 图片链接，例如 ![...](...)
    # 我们会先找出所有链接，然后再统一处理
    image_links = re.findall(r'(!\[.*?\]\((.*?)\))', content)

    # 确保输出目录存在
    revised_md_path.parent.mkdir(parents=True, exist_ok=True)

    if not image_links:
        # 如果没有图片，直接复制文件
        revised_md_path.write_text(content, encoding='utf-8')
        return

    # 使用 tqdm 创建一个进度条，友好地显示处理进度
    for full_match, image_relative_path in tqdm(image_links, desc=f"正在处理{md_path.name}"):
        # 构建图片的完整绝对路径
        image_path = md_path.parent / image_relative_path
        
        # 检查是否已经为这个链接添加过描述，防止重复处理
        if f"{full_match} [Image Description:" in content:
            continue

        # 从 LLM 获取详细描述
        description = get_image_description(vision_model, image_path)
        
        # 创建要替换的字符串
        replacement_string = f"{full_match} [Image Description: {description}]"
        
        # 用 "图片链接 + 描述" 替换原始的图片链接
        content = content.replace(full_match, replacement_string, 1)

    # 将修改后的内容保存到修订版输出目录
    revised_md_path.write_text(content, encoding='utf-8')
    print(f"成功创建修订版文件: {revised_md_path}")


def main():
    """主函数，运行整个流程。"""
    print("开始图片描述生成流程...")
    
    # 初始化 LLM 客户端
    vision_model = get_vision_model()

    # 定义相对于项目根目录的基础路径
    output_dir = project_root / 'mineru_markdown' / 'output'
    revised_output_dir = project_root / 'mineru_markdown' / 'revised_output'
    
    # 确保修订版输出目录存在
    revised_output_dir.mkdir(exist_ok=True)

    # 查找输出目录中的所有 markdown 文件
    markdown_files = list(output_dir.glob('*.md'))
    
    if not markdown_files:
        print(f"在目录 {output_dir} 中未找到任何 markdown 文件。正在退出。")
        return

    print(f"找到 {len(markdown_files)} 个 markdown 文件待处理。")
    
    for md_path in markdown_files:
        process_markdown_file(vision_model, md_path, revised_output_dir)
        
    print("\n图片描述生成流程成功结束！")


if __name__ == '__main__':
    main()
