import torch.nn as nn
import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gvp import GVP, GVPConv
from data.mol_to_gvp import mol_to_gvp_graph
from data.smile_to_3D import smiles_to_3d_conformer
from torch_scatter import scatter_mean 


class MoleculeGVP(nn.Module):
    def __init__(self, num_node_s_in, num_node_v_in, num_node_s_hidden, num_node_v_hidden,
                 num_edge_s_in, num_edge_v_in, num_output_s, num_layers=3, drop_rate=0.1):
        """
        用于分子表征学习的 GVP 模型。

        参数:
            num_node_s_in (int): 输入节点标量特征的维度。
            num_node_v_in (int): 输入节点向量特征的数量。
            num_node_s_hidden (int): 隐藏层节点标量特征的维度。
            num_node_v_hidden (int): 隐藏层节点向量特征的数量。
            num_edge_s_in (int): 输入边标量特征的维度。
            num_edge_v_in (int): 输入边向量特征的数量。
            num_output_s (int): 最终输出表征的维度。
            ...
        """
        super().__init__()

        # 输入层: 将原子特征嵌入到隐藏空间
        self.embed_nodes = GVP(
            (num_node_s_in, num_node_v_in),
            (num_node_s_hidden, num_node_v_hidden),
            activations=(None, None)
        )

        # 卷积层列表
        self.conv_layers = nn.ModuleList([
            GVPConv(
                (num_node_s_hidden, num_node_v_hidden), # 节点输入维度
                (num_node_s_hidden, num_node_v_hidden), # 节点输出维度
                (num_edge_s_in, num_edge_v_in),         # 边特征维度
                activations=(nn.ReLU(), None),
                vector_gate=True
            ) for _ in range(num_layers)
        ])
        
        # 输出层，用于最终的表征向量
        self.pool_dim = num_node_s_hidden
        self.out_mlp = nn.Sequential(
            nn.Linear(self.pool_dim, self.pool_dim * 2),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(self.pool_dim * 2, num_output_s)
        )

    def forward(self, node_s, node_v, edge_index, edge_s, edge_v, batch):
        # batch 是一个向量，指明每个节点属于哪个图
        device = next(self.parameters()).device
        node_s, node_v, edge_index, edge_s, edge_v, batch = \
            node_s.to(device), node_v.to(device), edge_index.to(device), \
            edge_s.to(device), edge_v.to(device), batch.to(device)
        
        # 1. 嵌入节点特征
        node_s, node_v = self.embed_nodes((node_s, node_v))

        # 2. 边特征
        edge_attr = (edge_s, edge_v)
        
        # 3. GVP 卷积层
        for conv in self.conv_layers:
            h_s, h_v = conv((node_s, node_v), edge_index, edge_attr)
            node_s = node_s + h_s
            node_v = node_v + h_v

        # 4. 正确的全局平均池化
        # 使用 scatter_mean 根据 batch 向量对节点进行分组池化
        graph_representation = scatter_mean(node_s, batch, dim=0)
        
        # 5. MLP 输出
        final_representation = self.out_mlp(graph_representation)
        
        return final_representation

# --- 模拟运行 ---
if __name__ == "__main__":
    smiles = "O=C(C)Oc1ccccc1C(=O)O"  # 阿司匹林
    mol_3d = smiles_to_3d_conformer(smiles)
    graph_data = mol_to_gvp_graph(mol_3d)
    if 'graph_data' in locals() and graph_data:
        (node_s, node_v), edge_index, (edge_s, edge_v) = graph_data
        
        # 实例化模型，传入正确的特征维度
        model = MoleculeGVP(
            num_node_s_in=node_s.shape[1],
            num_node_v_in=node_v.shape[1],
            num_edge_s_in=edge_s.shape[1],
            num_edge_v_in=edge_v.shape[1],
            num_node_s_hidden=128,  # 隐藏层标量维度
            num_node_v_hidden=16,   # 隐藏层向量维度
            num_output_s=256        # 最终输出表征的维度
        )

        # 执行前向传播
        representation = model(node_s, node_v, edge_index, edge_s, edge_v)

        print("\n--- 模型运行结果 ---")
        print(f"学到的分子表征向量形状: {representation.shape}")
        print(f"最终表征维度与预期相符: {representation.shape[0] == 256}")