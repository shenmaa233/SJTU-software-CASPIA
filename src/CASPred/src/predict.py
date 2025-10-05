#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
========================================
Kcat Prediction Script
========================================

命令行预测脚本，用于预测酶催化常数 (kcat)

使用方法:
    python predict.py --model <model_path> --config <config_path> \\
                      --smiles <smiles> --sequence <protein_sequence> \\
                      [--log_transform]

作者: SJTU-Software Team
日期: 2025-10-04
========================================
"""

import argparse
import json
import sys
import os
import torch
import numpy as np
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .model.kcat_model import KcatPredictionModel
from .data.smile_to_3D import smiles_to_3d_conformer
from .data.mol_to_gvp import mol_to_gvp_graph


def load_config(config_path):
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"错误: 无法加载配置文件 {config_path}: {e}")
        sys.exit(1)


def load_model(model_path, config, device='cpu'):
    """
    加载模型
    
    Args:
        model_path: 模型文件路径
        config: 配置字典
        device: 设备 ('cpu' 或 'cuda')
        
    Returns:
        加载的模型
    """
    try:
        # 初始化模型
        model = KcatPredictionModel(
            gvp_params=config['gvp_params'],
            esm_embedding_dim=config['esm_embedding_dim'],
            hidden_dim=config['hidden_dim'],
            dropout=config['dropout']
        )
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=device)
        
        # 处理不同的 checkpoint 格式
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        return model
        
    except Exception as e:
        print(f"错误: 无法加载模型 {model_path}: {e}")
        sys.exit(1)


def get_protein_embedding(protein_sequence, esm_embedding_dim, device='cpu'):
    """
    获取蛋白质嵌入 (使用 ESMC 模型)
    
    Args:
        protein_sequence: 蛋白质氨基酸序列
        esm_embedding_dim: ESM 嵌入维度
        device: 设备
        
    Returns:
        蛋白质嵌入张量 [1, seq_len, esm_embedding_dim]
    """
    try:
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
        
        # 根据配置选择模型
        if esm_embedding_dim == 960:
            model_name = "esmc_300m"
        elif esm_embedding_dim == 1280:
            model_name = "esmc_600m"
        else:
            print(f"警告: 未知的 ESM 嵌入维度 {esm_embedding_dim}，使用默认模型 esmc_300m")
            model_name = "esmc_300m"
        
        # 加载模型
        model = ESMC.from_pretrained(model_name).to(device)
        model.eval()
        
        # 编码蛋白质序列
        protein = ESMProtein(sequence=protein_sequence)
        
        with torch.no_grad():
            protein_tensor = model.encode(protein)
            logits_output = model.logits(
                protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
            )
            # 提取嵌入，shape: (1, seq_len, esm_embedding_dim)
            # 转换为 float32 以匹配模型权重
            embeddings = logits_output.embeddings.float()
        
        return embeddings
        
    except ImportError as e:
        print(f"错误: 未安装 ESM 库 ({e})，使用随机嵌入")
        print("请安装: pip install esm")
        # 使用随机嵌入作为后备方案
        seq_len = len(protein_sequence)
        return torch.randn(1, seq_len, esm_embedding_dim, device=device)
    
    except Exception as e:
        print(f"警告: 获取蛋白质嵌入时出错 ({e})，使用随机嵌入")
        seq_len = len(protein_sequence)
        return torch.randn(1, seq_len, esm_embedding_dim, device=device)


def process_smiles(smiles):
    """
    处理 SMILES 字符串并生成 GVP 图数据
    
    Args:
        smiles: SMILES 字符串
        
    Returns:
        GVP 图数据字典或 None
    """
    try:
        # 转换为 3D 分子
        mol_3d = smiles_to_3d_conformer(smiles)
        if mol_3d is None:
            print(f"错误: 无法为 SMILES '{smiles}' 生成 3D 构象")
            return None
        
        # 转换为 GVP 图
        graph_data = mol_to_gvp_graph(mol_3d)
        if graph_data is None:
            print(f"错误: 无法为分子生成 GVP 图数据")
            return None
        
        return graph_data
        
    except Exception as e:
        print(f"错误: 处理 SMILES 时出错: {e}")
        return None


def predict_kcat(model, graph_data, protein_embedding, device='cpu'):
    """
    执行 kcat 预测
    
    Args:
        model: 加载的模型
        graph_data: GVP 图数据 (tuple 格式: ((node_s, node_v), edge_index, (edge_s, edge_v)))
        protein_embedding: 蛋白质嵌入 [1, seq_len, esm_embedding_dim]
        device: 设备
        
    Returns:
        预测的 kcat 值
    """
    try:
        # 解包图数据 (mol_to_gvp_graph 返回 tuple 格式)
        (node_s, node_v), edge_index, (edge_s, edge_v) = graph_data
        
        # 准备输入 - 注意：node_s 和 node_v 不需要 batch 维度，使用 batch_map 来指示节点归属
        node_s = node_s.to(device)       # [num_nodes, node_s_dim]
        node_v = node_v.to(device)       # [num_nodes, node_v_dim, 3]
        edge_index = edge_index.to(device)       # [2, num_edges]
        edge_s = edge_s.to(device)               # [num_edges, edge_s_dim]
        edge_v = edge_v.to(device)               # [num_edges, edge_v_dim, 3]
        batch_map = torch.zeros(node_s.shape[0], dtype=torch.long, device=device)  # [num_nodes]，所有节点属于 batch 0
        protein_embedding = protein_embedding.to(device)  # [1, seq_len, esm_embedding_dim]
        
        # 预测
        with torch.no_grad():
            prediction = model(node_s, node_v, edge_index, edge_s, edge_v, batch_map, protein_embedding)
        
        return prediction.item()
        
    except Exception as e:
        print(f"错误: 预测过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """主函数"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Kcat Prediction Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model', required=True, help='模型文件路径')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--smiles', required=True, help='底物的 SMILES 字符串')
    parser.add_argument('--sequence', required=True, help='酶的氨基酸序列')
    parser.add_argument('--log_transform', action='store_true', help='对预测结果进行反对数变换')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help='计算设备')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    # 设置设备
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        device = 'cpu'
    
    print(f"使用设备: {device}")
    print(f"SMILES: {args.smiles}")
    print(f"蛋白质序列长度: {len(args.sequence)}")
    print("-" * 50)
    
    # 加载配置
    print("加载配置...")
    config = load_config(args.config)
    
    # 加载模型
    print("加载模型...")
    model = load_model(args.model, config, device)
    
    # 处理 SMILES
    print("处理 SMILES...")
    graph_data = process_smiles(args.smiles)
    if graph_data is None:
        sys.exit(1)
    
    # 获取蛋白质嵌入
    print("生成蛋白质嵌入...")
    protein_embedding = get_protein_embedding(
        args.sequence,
        config['esm_embedding_dim'],
        device
    )
    
    # 预测
    print("执行预测...")
    predicted_value = predict_kcat(model, graph_data, protein_embedding, device)
    
    # 反对数变换（如果需要）
    if args.log_transform:
        predicted_value = np.power(10, predicted_value)
    
    # 输出结果
    print("-" * 50)
    print(f"Predicted kcat value: {predicted_value:.6f} s^(-1)")
    print("-" * 50)
    
    return predicted_value


if __name__ == "__main__":
    main()

