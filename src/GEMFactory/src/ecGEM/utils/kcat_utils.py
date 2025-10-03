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
from .metabolite_utils import get_smiles_for_gprdf
from .protein_utils import get_protein_sequences_from_fasta
from .protein_utils import calculate_protein_molecular_weight
from .io_utils import load_model

# === 模型加载函数 ===
def load_models(model_path, config_path, device=None):
    with open(config_path, 'r') as f:
        config = json.load(f)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load ESM model
    esm_model_name = config.get("esm_model_name", "esmc_300m")
    esm_model = ESMC.from_pretrained(esm_model_name).to(device)
    esm_model.eval()

    # Load GVP model
    gvp_params = config["gvp_params"]
    model = KcatPredictionModel(
        gvp_params=gvp_params,
        esm_embedding_dim=config["esm_embedding_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"]
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return esm_model, model, config, device


# === 单条预测函数 ===
def predict_single(smiles, protein_sequence, esm_model, model, device, log_transform=True):
    mol_3d = smiles_to_3d_conformer(smiles)
    if mol_3d is None:
        return None

    graph_data = mol_to_gvp_graph(mol_3d)
    if graph_data is None:
        return None

    (node_s, node_v), edge_index, (edge_s, edge_v) = graph_data

    # 蛋白质 embedding
    protein = ESMProtein(sequence=protein_sequence)
    with torch.no_grad():
        protein_tensor = esm_model.encode(protein)
        logits_output = esm_model.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )
        protein_embedding = logits_output.embeddings.squeeze(0)

    with torch.no_grad():
        node_s, node_v = node_s.to(device), node_v.to(device)
        edge_index, edge_s, edge_v = edge_index.to(device), edge_s.to(device), edge_v.to(device)
        protein_embeddings = protein_embedding.unsqueeze(0).to(device)
        batch_map = torch.zeros(node_s.size(0), dtype=torch.long, device=device)

        pred = model(node_s, node_v, edge_index, edge_s, edge_v, batch_map, protein_embeddings)
        if log_transform:
            pred = 10 ** pred

    return pred.item()


# === caspred: 批量预测 ===
def caspred(gprdf, model_path, config_path, log_transform=True):
    esm_model, model, config, device = load_models(model_path, config_path)
    results = []

    for idx, row in tqdm(gprdf.iterrows(), total=len(gprdf), desc="Predicting kcat"):
        smiles = row.get("SMILES", None)
        seq = row.get("protein_sequence", None)

        if smiles is None or seq is None or str(smiles).strip() == "" or str(seq).strip() == "":
            pred = 10.0  # 默认值
        else:
            try:
                pred = predict_single(smiles, seq, esm_model, model, device, log_transform)
                if pred is None:
                    pred = 10.0
            except Exception as e:
                print(f"⚠️ Failed at index {idx}: {e}")
                pred = 10.0
        results.append(pred)

    gprdf["Kcat value (1/s)"] = results
    return gprdf


# === 主入口函数 ===
def kcat_predict(gprdf, protein_clean_file, model_file, result_folder):
    model = load_model(model_file)
    # 1. 获取 SMILES
    cache_file = "src/GEMFactory/src/ecGEM/utils/smiles_cache.json"
    smiles_list = get_smiles_for_gprdf(gprdf, model, cache_file)
    gprdf["SMILES"] = smiles_list

    # 2. 获取蛋白质序列
    protein_sequences = get_protein_sequences_from_fasta(protein_clean_file)

    sequences, molecular_weights = [], []
    for idx, row in gprdf.iterrows():
        gene_id = row["genes"]
        if gene_id in protein_sequences:
            seq = protein_sequences[gene_id]
            sequences.append(seq)
            mw = calculate_protein_molecular_weight(seq)
            molecular_weights.append(mw)
        else:
            sequences.append(None)
            molecular_weights.append(None)

    gprdf["protein_sequence"] = sequences
    gprdf["mass"] = molecular_weights

    # 3. 预测 kcat
    gprdf = caspred(
        gprdf,
        model_path="src/CASPred/model/EnzyExtractData/model_1.pth",
        config_path="src/CASPred/config.json",
        log_transform=True
    )

    # 4. 保存结果
    os.makedirs(result_folder, exist_ok=True)
    out_path = os.path.join(result_folder, "full_metabolites_reactions.csv")
    gprdf.to_csv(out_path, index=False)
    return gprdf

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
