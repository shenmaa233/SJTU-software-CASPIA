from .metabolite_utils import get_smiles_for_gprdf
from .protein_utils import get_protein_sequences_from_fasta
from .protein_utils import calculate_protein_molecular_weight
from .io_utils import load_model
import os
from .kcat_utils import ensemble_inference
from .topt_utils import topt_predict_batch
from math import exp
import ast
import os
import pandas as pd
from math import exp
import re

def safe_parse_ci(x):
    if not isinstance(x, str):
        return None
    # 去掉 np.float64() 包装
    cleaned = re.sub(r"np\.float64\((.*?)\)", r"\1", x)
    try:
        return ast.literal_eval(cleaned)
    except Exception:
        return None

# === 主入口函数 ===
def parameter_predict(
    gprdf,
    protein_clean_file,
    model_file,
    result_folder,
    is_etc=False,
    T=37.0
):
    if is_etc and T is None:
        raise ValueError("Optimal temperature is required when building enzyme-temperature-constrained GEM")

    os.makedirs(result_folder, exist_ok=True)

    # === Step 0: 保存原始 gprdf ===
    step0_path = os.path.join(result_folder, "step0_raw_gprdf.csv")
    if not os.path.exists(step0_path):
        gprdf.to_csv(step0_path, index=False)
    else:
        gprdf = pd.read_csv(step0_path)

    # === Step 1: 获取 SMILES ===
    step1_path = os.path.join(result_folder, "step1_with_smiles.csv")
    if os.path.exists(step1_path):
        gprdf = pd.read_csv(step1_path)
    else:
        model = load_model(model_file)
        cache_file = "src/GEMFactory/src/ecGEM/utils/smiles_cache.json"
        smiles_list = get_smiles_for_gprdf(gprdf, model, cache_file)
        gprdf["SMILES"] = smiles_list
        gprdf.to_csv(step1_path, index=False)

    # === Step 2: 获取蛋白质序列 & 分子量 ===
    step2_path = os.path.join(result_folder, "step2_with_proteins.csv")
    if os.path.exists(step2_path):
        gprdf = pd.read_csv(step2_path)
    else:
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
        gprdf.to_csv(step2_path, index=False)

    # === Step 3: 预测 kcat ===
    step3_path = os.path.join(result_folder, "step3_with_kcat.csv")
    if os.path.exists(step3_path):
        gprdf = pd.read_csv(step3_path)
        # 注意：kcat_95CI 原来是 tuple，需要恢复成 tuple
        gprdf["kcat_95CI"] = gprdf["kcat_95CI"].apply(safe_parse_ci)
    else:
        protein_sequences = list(gprdf["protein_sequence"])
        smiles = list(gprdf["SMILES"])
        kcat_model_path = [
            os.path.join("src/CASPred/model/kcat_models", f)
            for f in os.listdir("src/CASPred/model/kcat_models")
            if f.endswith(".pth")
        ]
        kcat_result = ensemble_inference(
            smiles, protein_sequences,
            kcat_model_path, "src/CASPred/config.json",
            batch_size=64, log_transform=True
        )
        gprdf["kcat"] = kcat_result["mean"]
        gprdf["kcat_std"] = kcat_result["std"]
        gprdf["kcat_95CI"] = kcat_result["95CI"]
        gprdf.to_csv(step3_path, index=False)

    # === Step 4: 预测 Topt & 调整 kcat ===
    if is_etc:
        step4_path = os.path.join(result_folder, "step4_with_topt.csv")
        if os.path.exists(step4_path):
            gprdf = pd.read_csv(step4_path)
            gprdf["kcat_95CI"] = gprdf["kcat_95CI"].apply(safe_parse_ci)
        else:
            topt_model_path = os.path.join("src/CASPred/model/HEATMAPData/model_1.pt")
            seqs = []
            for seq in gprdf["protein_sequence"]:
                if isinstance(seq, str) and seq.strip():
                    seqs.append(seq)
                else:
                    seqs.append(None)
            topt = topt_predict_batch(seqs, topt_model_path)
            for i in range(len(gprdf)):
                factor = exp(-(topt[i] - T) ** 2)
                gprdf.at[i, "kcat"] = gprdf.at[i, "kcat"] * factor
                if gprdf.at[i, "kcat_std"] is not None:
                    gprdf.at[i, "kcat_std"] = gprdf.at[i, "kcat_std"] * factor
                if gprdf.at[i, "kcat_95CI"] is not None:
                    ci_low, ci_high = gprdf.at[i, "kcat_95CI"]
                    if ci_low is not None and ci_high is not None:
                        ci_low, ci_high = ci_low * factor, ci_high * factor
                        gprdf.at[i, "kcat_95CI"] = (ci_low, ci_high)
            gprdf.to_csv(step4_path, index=False)

    # === Step 5: 保存最终结果 ===
    out_path = os.path.join(result_folder, "full_metabolites_reactions.csv")
    gprdf.to_csv(out_path, index=False)
    return gprdf
