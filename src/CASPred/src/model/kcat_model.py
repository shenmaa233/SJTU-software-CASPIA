import torch
import torch.nn as nn
from ..gvp import GVP, GVPConvLayer
from .gvp_model import MoleculeGVP


class KcatPredictionModel(nn.Module):
    def __init__(self, gvp_params, esm_embedding_dim=320, hidden_dim=512, dropout=0.1, num_heads=4):
        """
        Model for predicting kcat values using metabolite GVP and protein ESM embeddings with cross attention.
        
        Args:
            gvp_params: Dictionary of parameters for the MoleculeGVP model
            esm_embedding_dim: Dimension of ESM protein embeddings (960 for esmc_300m)
            hidden_dim: Hidden dimension for projections
            dropout: Dropout rate
            num_heads: Number of attention heads in cross-attention
        """
        super(KcatPredictionModel, self).__init__()
        
        # Metabolite encoder (GVP)
        self.metabolite_encoder = MoleculeGVP(**gvp_params)
        
        # Protein projection (per residue, not pooled)
        self.protein_projection = nn.Sequential(
            nn.Linear(esm_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Metabolite projection (as query)
        self.metabolite_projection = nn.Sequential(
            nn.Linear(gvp_params['num_output_s'], hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Cross attention: metabolite repr (Q), protein reprs (K,V)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        # Prediction head
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 4),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_s, node_v, edge_index, edge_s, edge_v, batch_map, protein_embeddings):
        """
        Args:
            node_s, node_v, edge_index, edge_s, edge_v, batch_map: Graph inputs
            protein_embeddings: [batch_size, seq_len, esm_embedding_dim]
        """
        # Encode metabolites
        metabolite_reprs = self.metabolite_encoder(node_s, node_v, edge_index, edge_s, edge_v, batch_map)
        # Shape: [batch_size, num_output_s]
        
        # Project metabolite repr to query space
        metabolite_q = self.metabolite_projection(metabolite_reprs).unsqueeze(1)  
        # Shape: [batch_size, 1, hidden_dim]
        
        # Project protein embeddings (residue-level features)
        protein_reprs = self.protein_projection(protein_embeddings)  
        # Shape: [batch_size, seq_len, hidden_dim]
        
        # Cross attention
        attn_output, attn_weights = self.cross_attention(
            query=metabolite_q,      # [B, 1, H]
            key=protein_reprs,       # [B, L, H]
            value=protein_reprs      # [B, L, H]
        )
        # attn_output: [B, 1, H]
        
        fused_repr = attn_output.squeeze(1)  # [B, H]
        
        # Predict kcat
        kcat_preds = self.mlp_head(fused_repr).squeeze(-1)  # [B]
        
        return kcat_preds
