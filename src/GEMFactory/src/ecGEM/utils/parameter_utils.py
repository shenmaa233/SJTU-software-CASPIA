from .metabolite_utils import get_smiles_for_gprdf
from .protein_utils import get_protein_sequences_from_fasta
from .protein_utils import calculate_protein_molecular_weight
from .io_utils import load_model
import os
from .kcat_utils import ensemble_inference
from .topt_utils import topt_predict_batch
from math import exp


# === 主入口函数 ===
def parameter_predict(gprdf, protein_clean_file, model_file, result_folder, is_etc=False, T=37.0):
    if is_etc and T is None:
        print("Optimal temperature is required when building enzyme-temperature-constrained GEM. Exiting.")
        exit(1)

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
    protein_sequences = gprdf["protein_sequence"]
    smiles = gprdf["SMILES"]
    smiles = list(smiles)
    protein_sequences = list(protein_sequences)

    # find model files in src/CASPred/model/kcatmodels and add them into model_path
    kcat_model_path = []
    for model_file in os.listdir("src/CASPred/model/kcat_models"):
        if model_file.endswith(".pth"):
            kcat_model_path.append(os.path.join("src/CASPred/model/kcat_models", model_file))

    kcat_result = ensemble_inference(smiles, protein_sequences, kcat_model_path, "src/CASPred/config.json", batch_size=64, log_transform=True)
    gprdf["kcat"] = kcat_result["mean"]
    gprdf["kcat_std"] = kcat_result["std"]
    gprdf["kcat_95CI"] = kcat_result["95CI"]


    
    # 4. 预测 Topt (if etc)
    if is_etc:
        topt_model_path = os.path.join("src/CASPred/model/HEATMAPData/model_1.pt")
        topt = topt_predict_batch(protein_sequences, topt_model_path)
        for i in range(len(gprdf)):
            gprdf.at[i, "kcat"] = gprdf.at[i, "kcat"] * exp(-(topt[i] - T) ** 2)
            if gprdf.at[i, "kcat_std"] is not None:
                gprdf.at[i, "kcat_std"] = gprdf.at[i, "kcat_std"] * exp(-(topt[i] - T) ** 2)
            if gprdf.at[i, "kcat_95CI"] is not None:
                gprdf.at[i, "kcat_95CI"] = gprdf.at[i, "kcat_95CI"] * exp(-(topt[i] - T) ** 2)

    # 5. 保存结果
    os.makedirs(result_folder, exist_ok=True)
    out_path = os.path.join(result_folder, "full_metabolites_reactions.csv")
    gprdf.to_csv(out_path, index=False)
    return gprdf