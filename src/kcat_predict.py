import json
import os
import sys
import torch
import subprocess
from typing import Dict, Any
from qwen_agent.tools.base import BaseTool, register_tool

@register_tool('kcat_predict')
class KcatPredict(BaseTool):
    """
    酶催化常数 (kcat) 预测工具
    使用 EGNN 和 ESM 模型预测酶对特定底物的催化效率
    """
    
    description = """
    酶催化常数 (kcat) 预测工具，基于深度学习模型预测酶对特定底物的催化效率。
    输入酶的氨基酸序列和底物的 SMILES 分子式，输出预测的 kcat 值 (单位: s^-1)。
    该工具结合了分子的 3D 结构信息和蛋白质的序列信息，提供高精度的催化效率预测。
    """
    
    parameters = [
        {
            'name': 'smiles',
            'type': 'string', 
            'description': '底物分子的 SMILES 字符串表示，例如: "CCO" (乙醇)',
            'required': True
        },
        {
            'name': 'protein_sequence',
            'type': 'string',
            'description': '酶的氨基酸序列，使用标准单字母氨基酸代码',
            'required': True
        },
        {
            'name': 'log_transform',
            'type': 'boolean',
            'description': '是否对预测结果进行反对数变换 (默认: true，因为模型使用对数变换训练)',
            'required': False,
            'default': True
        }
    ]
    
    def __init__(self, tool_cfg=None):
        super().__init__(tool_cfg)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.egnn_dir = os.path.join(current_dir, 'EGNNkcat')
        self.model_path = os.path.join(self.egnn_dir, 'model', 'best_model.pth')
        self.config_path = os.path.join(self.egnn_dir, 'config.json')
        self.predict_script = os.path.join(self.egnn_dir, 'src', 'predict.py')
        
    def call(self, params, **kwargs):
        """
        执行 kcat 预测
        
        Args:
            params: 包含 smiles, protein_sequence, log_transform 的参数字典
            
        Returns:
            包含预测结果的字典
        """
        try:
            # 处理参数：支持字符串(JSON)和字典两种格式
            if isinstance(params, str):
                try:
                    params_dict = json.loads(params)
                except json.JSONDecodeError as e:
                    return {
                        'success': False,
                        'error': f'JSON 解析失败: {str(e)}'
                    }
            elif isinstance(params, dict):
                params_dict = params
            else:
                return {
                    'success': False,
                    'error': f'参数类型错误: 期望 str 或 dict，得到 {type(params).__name__}'
                }
            
            # 获取参数
            smiles = params_dict.get('smiles')
            protein_sequence = params_dict.get('protein_sequence')
            log_transform = params_dict.get('log_transform', True)
            
            # 验证必需参数
            if not smiles:
                return {
                    'success': False,
                    'error': '缺少必需参数: smiles'
                }
            
            if not protein_sequence:
                return {
                    'success': False,
                    'error': '缺少必需参数: protein_sequence'
                }
            
            # 验证参数类型
            if not isinstance(smiles, str):
                return {
                    'success': False,
                    'error': f'SMILES 参数类型错误: 期望 str，得到 {type(smiles).__name__}'
                }
            
            if not isinstance(protein_sequence, str):
                return {
                    'success': False,
                    'error': f'蛋白质序列参数类型错误: 期望 str，得到 {type(protein_sequence).__name__}'
                }
            
            # 基本格式验证
            if len(smiles.strip()) == 0:
                return {
                    'success': False,
                    'error': 'SMILES 字符串不能为空'
                }
            
            if len(protein_sequence.strip()) == 0:
                return {
                    'success': False,
                    'error': '蛋白质序列不能为空'
                }
            
            # 检查文件是否存在
            if not os.path.exists(self.model_path):
                return {
                    'success': False,
                    'error': f'模型文件不存在: {self.model_path}'
                }
            
            if not os.path.exists(self.config_path):
                return {
                    'success': False,
                    'error': f'配置文件不存在: {self.config_path}'
                }
            
            if not os.path.exists(self.predict_script):
                return {
                    'success': False,
                    'error': f'预测脚本不存在: {self.predict_script}'
                }
            
            # 构建命令行参数
            cmd = [
                'python',
                self.predict_script,
                '--model', self.model_path,
                '--config', self.config_path,
                '--smiles', smiles,
                '--sequence', protein_sequence
            ]
            
            if log_transform:
                cmd.append('--log_transform')
            
            # 切换到 EGNNkcat/src 目录以确保相对导入正常工作
            original_cwd = os.getcwd()
            egnn_src_dir = os.path.join(self.egnn_dir, 'src')
            
            try:
                os.chdir(egnn_src_dir)
                
                # 执行预测脚本
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=300  # 5分钟超时
                )
                
                if result.returncode == 0:
                    # 解析输出
                    output_lines = result.stdout.strip().split('\n')
                    
                    # 查找预测结果
                    predicted_kcat = None
                    for line in output_lines:
                        if 'Predicted kcat value:' in line:
                            try:
                                # 提取数值，格式类似 "Predicted kcat value: 0.1234 s^(-1)"
                                parts = line.split(':')[1].strip().split()
                                predicted_kcat = float(parts[0])
                                break
                            except (IndexError, ValueError):
                                continue
                    
                    if predicted_kcat is not None:
                        return {
                            'success': True,
                            'predicted_kcat': predicted_kcat,
                            'unit': 's^-1',
                            'smiles': smiles,
                            'protein_sequence_length': len(protein_sequence),
                            'log_transformed': log_transform,
                            'description': f'预测的酶催化常数为 {predicted_kcat:.4f} s^-1',
                            'raw_output': result.stdout.strip()
                        }
                    else:
                        return {
                            'success': False,
                            'error': '无法从输出中解析预测结果',
                            'raw_output': result.stdout.strip(),
                            'raw_error': result.stderr.strip()
                        }
                else:
                    return {
                        'success': False,
                        'error': f'预测脚本执行失败 (返回码: {result.returncode})',
                        'raw_output': result.stdout.strip(),
                        'raw_error': result.stderr.strip()
                    }
            
            finally:
                # 恢复原始工作目录
                os.chdir(original_cwd)
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': '预测超时 (超过5分钟)'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'预测过程中发生错误: {str(e)}'
            }