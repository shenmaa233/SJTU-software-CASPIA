import os
import pandas as pd
from tqdm import tqdm
import torch
from esm.models.esmc import ESMC
from src.CASPred.src.data.smile_to_3D import smiles_to_3d_conformer
from src.CASPred.src.data.mol_to_gvp import mol_to_gvp_graph
from esm.sdk.api import ESMProtein, LogitsConfig
import json
from src.CASPred.src.model.kcat_model import KcatPredictionModel
import numpy as np
from typing import List

def prepare_inference_batch(smiles_list: List[str], protein_list: List[str], esm_model) -> dict:
    """
    Prepare molecular graphs and protein embeddings for batched inference.

    Args:
        smiles_list (List[str]): List of SMILES strings representing substrates.
        protein_list (List[str]): List of protein sequences aligned with SMILES list.
        esm_model: Preloaded ESMC model for protein embedding extraction.

    Returns:
        dict: Batched tensors containing node/edge features, protein embeddings, and batch mapping.
    """
    node_s_list, node_v_list = [], []
    edge_index_list, edge_s_list, edge_v_list = [], [], []
    batch_map, protein_embeddings = [], []
    node_offset = 0

    for i, (smiles, seq) in enumerate(zip(smiles_list, protein_list)):
        # === Convert SMILES to 3D molecular graph ===
        mol_3d = smiles_to_3d_conformer(smiles)
        if mol_3d is None:
            raise ValueError(f"Failed to generate 3D conformer for SMILES: {smiles}")

        (node_s, node_v), edge_index, (edge_s, edge_v) = mol_to_gvp_graph(mol_3d)

        node_s_list.append(node_s)
        node_v_list.append(node_v)
        edge_s_list.append(edge_s)
        edge_v_list.append(edge_v)

        num_nodes = node_s.shape[0]
        edge_index_list.append(edge_index + node_offset)
        batch_map.append(torch.full((num_nodes,), i, dtype=torch.long))
        node_offset += num_nodes

        # === Encode protein sequence with ESM ===
        protein = ESMProtein(sequence=seq)
        with torch.no_grad():
            protein_tensor = esm_model.encode(protein)
            logits_output = esm_model.logits(
                protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
            )
            protein_embedding = logits_output.embeddings.squeeze(0)  # Shape: (L, D)
        protein_embeddings.append(protein_embedding)

    # === Pad protein embeddings to the same length ===
    max_length = max(emb.shape[0] for emb in protein_embeddings)
    padded_embeddings = []
    for emb in protein_embeddings:
        pad_len = max_length - emb.shape[0]
        padded_emb = torch.nn.functional.pad(emb, (0, 0, 0, pad_len), value=0)
        padded_embeddings.append(padded_emb)

    return {
        "node_s": torch.cat(node_s_list, dim=0),
        "node_v": torch.cat(node_v_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "edge_s": torch.cat(edge_s_list, dim=0),
        "edge_v": torch.cat(edge_v_list, dim=0),
        "batch_map": torch.cat(batch_map, dim=0),
        "protein_embedding": torch.stack(padded_embeddings),
    }


def load_model(config_path: str, model_path: str, device: torch.device):
    """
    Load a trained KcatPredictionModel from checkpoint.

    Args:
        config_path (str): Path to JSON config file.
        model_path (str): Path to model checkpoint (.pth file).
        device (torch.device): Torch device.

    Returns:
        model: Loaded model ready for inference.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    model = KcatPredictionModel(
        gvp_params=config["gvp_params"],
        esm_embedding_dim=config["esm_embedding_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    return model


def run_in_batches(smiles, proteins, model, esm_model, batch_size, device, log_transform=True):
    """
    Run inference in mini-batches to avoid OOM issues.
    If a SMILES or protein is None, directly assign kcat=10 for that sample.
    """
    results = []

    for start in range(0, len(smiles), batch_size):
        end = min(start + batch_size, len(smiles))
        batch_smiles = smiles[start:end]
        batch_proteins = proteins[start:end]

        valid_indices = [i for i, (s, p) in enumerate(zip(batch_smiles, batch_proteins)) if s is not None and p is not None]
        invalid_indices = [i for i in range(len(batch_smiles)) if i not in valid_indices]

        batch_preds = np.zeros(len(batch_smiles), dtype=np.float32)

        # === 如果有有效样本才跑模型 ===
        if valid_indices:
            valid_smiles = [batch_smiles[i] for i in valid_indices]
            valid_proteins = [batch_proteins[i] for i in valid_indices]

            batch = prepare_inference_batch(valid_smiles, valid_proteins, esm_model)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                preds = model(
                    batch["node_s"],
                    batch["node_v"],
                    batch["edge_index"],
                    batch["edge_s"],
                    batch["edge_v"],
                    batch["batch_map"],
                    batch["protein_embedding"],
                )
            preds = preds.squeeze(-1).cpu().numpy()
            if log_transform:
                preds = np.power(10, preds)

            # 填入有效预测
            for idx, val in zip(valid_indices, preds):
                batch_preds[idx] = val

        # === None 样本直接置为 10 ===
        for idx in invalid_indices:
            batch_preds[idx] = 10.0

        results.append(batch_preds)

    return np.concatenate(results, axis=0)


def ensemble_inference(smiles, proteins, model_paths, config_path, batch_size=32, log_transform=True):
    """
    Perform ensemble inference with multiple models and estimate confidence.
    If a SMILES or protein is None, assign kcat=10 and std/CI=None for that sample.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model = ESMC.from_pretrained("esmc_300m").to(device)

    # 收集每个模型的预测
    all_preds = []
    for model_path in model_paths:
        model = load_model(config_path, model_path, device)
        preds = run_in_batches(smiles, proteins, model, esm_model, batch_size, device, log_transform)
        all_preds.append(preds)

    all_preds = np.stack(all_preds, axis=0)  # (num_models, num_samples)

    mean_preds, std_preds, ci95 = [], [], []
    num_models = len(model_paths)

    for i, (s, p) in enumerate(zip(smiles, proteins)):
        if s is None or p is None:
            mean_preds.append(10.0)
            std_preds.append(None)
            ci95.append((None, None))
        else:
            preds_i = all_preds[:, i]
            mean_val = np.mean(preds_i)
            std_val = np.std(preds_i)
            ci_val = (
                mean_val - 1.96 * std_val / np.sqrt(num_models),
                mean_val + 1.96 * std_val / np.sqrt(num_models),
            )
            mean_preds.append(mean_val)
            std_preds.append(std_val)
            ci95.append(ci_val)

    return {"mean": np.array(mean_preds), "std": std_preds, "95CI": ci95}

def get_kcat_mw(gprdf, result_folder):
    '''
    Make the kcat_mw input file for ecGEM construction.
    '''

    gprdf_rex = gprdf.copy()
    # Remove rows with 'None' in Kcat value or molecular weight
    gprdf_rex = gprdf_rex[gprdf_rex['Kcat value (1/s)'] != 'None']
    gprdf_rex = gprdf_rex[gprdf_rex['mass'].notna()]

    # Convert Kcat value to float
    gprdf_rex['Kcat value (1/s)'] = gprdf_rex['Kcat value (1/s)'].astype(float)

    # Sort by Kcat value and keep only the first occurrence of each reaction
    gprdf_rex = gprdf_rex.sort_values('Kcat value (1/s)', ascending=False).drop_duplicates(subset=['reactions'], keep='first')
    gprdf_rex['kcat_mw'] = gprdf_rex['Kcat value (1/s)'] * 3600 * 1000 / gprdf_rex['mass']

    # Prepare DL_reaction_kact_mw DataFrame
    reaction_kcat_mw = pd.DataFrame()
    reaction_kcat_mw['reactions'] = gprdf_rex['reactions']
    reaction_kcat_mw['data_type'] = 'DLkcat'
    reaction_kcat_mw['kcat'] = gprdf_rex['Kcat value (1/s)']
    reaction_kcat_mw['MW'] = gprdf_rex['mass']
    reaction_kcat_mw['kcat_MW'] = gprdf_rex['kcat_mw']
    reaction_kcat_mw.reset_index(drop=True, inplace=True)
    reaction_kcat_mw_file=f'{result_folder}/reaction_kcat_mw.csv'
    reaction_kcat_mw.to_csv(reaction_kcat_mw_file, index=False)
    print('reaction_kcat_mw generated')
    return reaction_kcat_mw
