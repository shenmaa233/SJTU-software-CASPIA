import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gvp import GVP, GVPConvLayer
from model.gvp_model import MoleculeGVP

class KcatPredictionModel(nn.Module):
    def __init__(self, gvp_params, esm_embedding_dim=320, hidden_dim=512, dropout=0.1):
        """
        Model for predicting kcat values using metabolite GVP and protein ESM embeddings.
        
        Args:
            gvp_params: Dictionary of parameters for the MoleculeGVP model
            esm_embedding_dim: Dimension of ESM protein embeddings (960 for esmc_300m)
            hidden_dim: Hidden dimension for the fusion layers
            dropout: Dropout rate
        """
        super(KcatPredictionModel, self).__init__()
        
        # Initialize the GVP model for metabolite encoding
        self.metabolite_encoder = MoleculeGVP(**gvp_params)
        
        # Protein encoder (ESM embeddings are precomputed, so we just need a projection)
        self.protein_projection = nn.Sequential(
            nn.Linear(esm_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layer to combine metabolite and protein representations
        self.fusion = nn.Sequential(
            nn.Linear(gvp_params['num_output_s'] + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, node_s, node_v, edge_index, edge_s, edge_v, batch_map, protein_embeddings):
        """
        Args:
            node_s, node_v, ...: 来自批处理图的张量
            batch_map: 指示每个节点所属图的向量
            protein_embeddings: 蛋白质嵌入 [batch_size, max_seq_len, embedding_dim]
        """
        # 直接在整个批处理图上运行 GVP 编码器
        metabolite_reprs = self.metabolite_encoder(node_s, node_v, edge_index, edge_s, edge_v, batch_map)
        # metabolite_reprs 的形状现在是 [batch_size, num_output_s]
        
        # 处理蛋白质嵌入 (这部分逻辑不变)
        protein_pooled = protein_embeddings.mean(dim=1)
        protein_reprs = self.protein_projection(protein_pooled)
        
        # 组合表征
        combined_reprs = torch.cat([metabolite_reprs, protein_reprs], dim=1)
        
        # 预测 kcat 值
        kcat_preds = self.fusion(combined_reprs).squeeze(-1)
        
        return kcat_preds