# src/tools.py

from langchain.tools import tool
from typing import Tuple, Dict, Any
from src.GeneMarks.GeneMarkSRunner import GeneMarkSRunner
import json
import os
import subprocess

# --- 关键改动：移除了 @tool 装饰器，并重命名 ---
# 这是一个内部实现函数，Agent不应该直接看到它。
def _run_gene_prediction_implementation(genome_file) -> Dict[str, str]:
    """
    内部实现：使用 GeneMarkS 对基因组文件对象进行基因预测。
    这个函数必须接收一个 Gradio 文件对象，而不是文件路径字符串。
    """
    if genome_file is None:
        return {"message": "错误：没有提供基因组文件对象。", "protein_faa_path": ""}
    
    genome_path = genome_file.name
    output_dir = "./genemarks_output"
    runner = GeneMarkSRunner(gms_script_path="/home/shenmaa/gms2_linux_64/gms2.pl")
    try:
        results = runner.run(
            input_fasta=genome_path,
            output_dir=output_dir
        )
        protein_faa = results.get("faa", "")
        msg = f"GeneMarkS 注释完成！蛋白质序列保存在 `{protein_faa}`"
        return {"message": msg, "protein_faa_path": protein_faa}
    except Exception as e:
        return {"message": f"GeneMarkS 执行出错: {e}", "protein_faa_path": ""}


@tool
def extract_protein_from_predicted_file(predicted_protein_path: str, protein_id: str) -> str:
    """
    从基因预测步骤生成的蛋白质 FASTA (.faa) 文件中，根据蛋白质 ID 提取单条氨基酸序列。
    这个工具应该在基因预测成功执行后使用。

    参数:
    - predicted_protein_path (str): GeneMarkS 工具生成的蛋白质 FASTA 文件的路径。
    - protein_id (str): 需要提取的蛋白质的 ID (FASTA 标题 '>' 之后，第一个空格前的部分)。

    返回:
    - str: 找到的蛋白质氨基酸序列。如果找不到，则返回错误信息。
    """
    if not os.path.exists(predicted_protein_path):
        return f"错误: 预测的蛋白质文件未找到: {predicted_protein_path}"
    
    try:
        with open(predicted_protein_path, 'r') as f:
            current_id = None
            sequence = []
            for line in f:
                if line.startswith('>'):
                    if current_id is not None:
                        if current_id == protein_id:
                            return "".join(sequence)
                        sequence = []
                    current_id = line.strip().split()[0][1:]
                elif current_id is not None:
                    sequence.append(line.strip())
            
            if current_id == protein_id:
                return "".join(sequence)

        return f"错误: 在文件 {predicted_protein_path} 中未找到蛋白质 ID '{protein_id}'"
    except Exception as e:
        return f"读取蛋白质文件时发生错误: {e}"


@tool
def predict_kcat(smiles: str, protein_sequence: str, log_transform: bool = True) -> Dict[str, Any]:
    """
    酶催化常数 (kcat) 预测工具...
    """
    # ... 此处代码与之前完全相同 ...
    current_dir = os.path.dirname(os.path.abspath(__file__))
    egnn_dir = os.path.join(current_dir, 'EGNNkcat')
    model_path = os.path.join(egnn_dir, 'model', 'best_model.pth')
    config_path = os.path.join(egnn_dir, 'config.json')
    predict_script = os.path.join(egnn_dir, 'src', 'predict.py')

    if not smiles or not protein_sequence:
        return {'success': False, 'error': 'smiles 和 protein_sequence 为必填参数'}
    
    for path, name in [(model_path, '模型文件'), (config_path, '配置文件'), (predict_script, '预测脚本')]:
        if not os.path.exists(path):
            return {'success': False, 'error': f'{name}不存在: {path}'}
    
    cmd = ['python', predict_script, '--model', model_path, '--config', config_path,
           '--smiles', smiles, '--sequence', protein_sequence]
    if log_transform:
        cmd.append('--log_transform')
    
    original_cwd = os.getcwd()
    egnn_src_dir = os.path.join(egnn_dir, 'src')

    try:
        os.chdir(egnn_src_dir)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            return {'success': False, 'error': f'预测脚本执行失败 (返回码: {result.returncode})',
                    'raw_output': result.stdout.strip(), 'raw_error': result.stderr.strip()}

        predicted_kcat = None
        for line in result.stdout.strip().split('\n'):
            if 'Predicted kcat value:' in line:
                try:
                    predicted_kcat = float(line.split(':')[1].strip().split()[0])
                    break
                except:
                    continue

        if predicted_kcat is None:
            return {'success': False, 'error': '无法从输出解析预测结果',
                    'raw_output': result.stdout.strip(), 'raw_error': result.stderr.strip()}

        return {'success': True,
                'predicted_kcat': predicted_kcat,
                'unit': 's^-1',
                'description': f'预测的酶催化常数为 {predicted_kcat:.4f} s^-1',
                'raw_output': result.stdout.strip()}
    
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': '预测超时 (超过5分钟)'}
    except Exception as e:
        return {'success': False, 'error': f'预测过程中发生错误: {str(e)}'}
    finally:
        os.chdir(original_cwd)


@tool
def multiply(x: int, y: int) -> int:
    """将两个数相乘"""
    return x * y