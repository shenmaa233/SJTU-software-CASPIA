import torch
import torch.nn as nn
import json
import os
import argparse
import numpy as np
from rdkit import Chem

from src.data.smile_to_3D import smiles_to_3d_conformer
from src.data.mol_to_gvp import mol_to_gvp_graph
from src.model.kcat_model import KcatPredictionModel
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

def predict_kcat(model_path, config_path, smiles, protein_sequence, log_transform=False):
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load ESM model for protein embeddings
    esm_model_name = config.get('esm_model_name', 'esmc_300m')
    esm_model = ESMC.from_pretrained(esm_model_name).to(device)
    
    # Process metabolite
    mol_3d = smiles_to_3d_conformer(smiles)
    if mol_3d is None:
        print("Failed to generate 3D conformation for the SMILES")
        return None
    
    graph_data = mol_to_gvp_graph(mol_3d)
    if graph_data is None:
        print("Failed to convert molecule to GVP graph")
        return None
        
    node_s, node_v, edge_index, edge_s, edge_v = graph_data
    
    # Process protein
    protein = ESMProtein(sequence=protein_sequence)
    
    with torch.no_grad():
        protein_tensor = esm_model.encode(protein)
        logits_output = esm_model.logits(
           protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )
        protein_embedding = logits_output.embeddings.squeeze(0)
    
    # Load model
    gvp_params = config['gvp_params']
    model = KcatPredictionModel(
        gvp_params=gvp_params,
        esm_embedding_dim=config['esm_embedding_dim'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        node_s = node_s.to(device)
        node_v = node_v.to(device)
        edge_index = edge_index.to(device)
        edge_s = edge_s.to(device)
        edge_v = edge_v.to(device)
        protein_embedding = protein_embedding.unsqueeze(0).to(device)
        
        prediction = model(node_s, node_v, edge_index, edge_s, edge_v, protein_embedding)
        
        # Apply inverse log transform if needed
        if log_transform:
            prediction = 10 ** prediction
    
    return prediction.item()

def main():
    parser = argparse.ArgumentParser(description='Predict kcat value')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--smiles', type=str, required=True, help='SMILES string of the metabolite')
    parser.add_argument('--sequence', type=str, required=True, help='Protein sequence')
    parser.add_argument('--log_transform', action='store_true', help='Whether the model was trained with log-transformed labels')
    
    args = parser.parse_args()
    
    predicted_kcat = predict_kcat(
        model_path=args.model,
        config_path=args.config,
        smiles=args.smiles,
        protein_sequence=args.sequence,
        log_transform=args.log_transform
    )
    
    if predicted_kcat is not None:
        print(f"Predicted kcat value: {predicted_kcat:.4f} s^(-1)")
    else:
        print("Prediction failed.")

if __name__ == '__main__':
    main()