import os
import pandas as pd
import torch
import sys
from tqdm import tqdm
import json

# Add the CASPred src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../CASPred/src'))

from hyena.tokenizer import CharacterTokenizer
from hyena.model import HyenaDNAModel
from .protein_utils import get_protein_sequences_from_fasta
from .io_utils import load_model

# === 模型加载函数 ===
def load_topt_model(model_path, device=None):
    """
    Load the HyenaDNA model for optimal temperature prediction.
    
    Args:
        model_path (str): Path to the model checkpoint file
        device (torch.device, optional): Device to load the model on
        
    Returns:
        tuple: (model, tokenizer, device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # List of standard amino acids
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    
    # Model parameters (based on topt_model.py)
    d_model = 256
    n_layer = 4
    hidden_size = 128
    max_length = 1000
    
    # Create the CharacterTokenizer for protein sequences
    tokenizer = CharacterTokenizer(characters=amino_acids, model_max_length=max_length)
    
    # Initialize the model
    model = HyenaDNAModel(
        d_model=d_model, 
        n_layer=n_layer, 
        d_inner=hidden_size, 
        vocab_size=len(amino_acids) + 1, 
        use_head=True, 
        n_classes=1
    )
    model.to(device)
    
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded Topt model from {model_path}")
    return model, tokenizer, device

# === 单条预测函数 ===
def predict_single_topt(protein_sequence, model, tokenizer, device, max_length=1000):
    """
    Predict the optimal temperature for a single protein sequence.
    
    Args:
        protein_sequence (str): The protein sequence to predict for
        model (HyenaDNAModel): The trained model
        tokenizer (CharacterTokenizer): The tokenizer used to encode the sequence
        device (torch.device): The device to run the prediction on
        max_length (int): Maximum length of the sequence
        
    Returns:
        float: Predicted optimal temperature, or None if prediction fails
    """
    if not protein_sequence or str(protein_sequence).strip() == "":
        return None
        
    try:
        model.eval()
        with torch.no_grad():
            # Encode the sequence with padding/truncation to max_length
            encoded_sequence = tokenizer.encode(protein_sequence)[:max_length]
            # Pad or truncate the sequence to the specified max_length
            if len(encoded_sequence) < max_length:
                encoded_sequence += [0] * (max_length - len(encoded_sequence))

            input_tensor = torch.tensor([encoded_sequence], dtype=torch.long).to(device)
            prediction = model(input_tensor)
            return prediction.item()
    except Exception as e:
        print(f"Error predicting Topt for sequence: {e}")
        return None

# === 批量预测函数 ===
def topt_predict_batch(gprdf, model_path, max_length=1000):
    """
    Predict optimal temperatures for multiple protein sequences in a DataFrame.
    
    Args:
        gprdf (pd.DataFrame): DataFrame containing protein sequences
        model_path (str): Path to the model checkpoint file
        max_length (int): Maximum length of the sequence
        
    Returns:
        pd.DataFrame: DataFrame with added 'Topt' column
    """
    model, tokenizer, device = load_topt_model(model_path)
    results = []
    
    for idx, row in tqdm(gprdf.iterrows(), total=len(gprdf), desc="Predicting Topt"):
        protein_sequence = row.get("protein_sequence", None)
        
        if protein_sequence is None or str(protein_sequence).strip() == "":
            pred = 37.0  # Default temperature (human body temperature)
        else:
            try:
                pred = predict_single_topt(protein_sequence, model, tokenizer, device, max_length)
                if pred is None:
                    pred = 37.0
            except Exception as e:
                print(f"⚠️ Failed at index {idx}: {e}")
                pred = 37.0
        results.append(pred)
    
    gprdf["Topt"] = results
    return gprdf

# === 主入口函数 ===
def topt_predict(gprdf, protein_clean_file, model_file, result_folder):
    """
    Main function to predict optimal temperatures for proteins in the GPR DataFrame.
    
    Args:
        gprdf (pd.DataFrame): GPR DataFrame containing gene information
        protein_clean_file (str): Path to the cleaned protein FASTA file
        model_file (str): Path to the model file (not used for Topt, but kept for consistency)
        result_folder (str): Folder to save results
        
    Returns:
        pd.DataFrame: Updated GPR DataFrame with Topt predictions
    """
    # 1. 获取蛋白质序列
    protein_sequences = get_protein_sequences_from_fasta(protein_clean_file)
    
    sequences = []
    for idx, row in gprdf.iterrows():
        gene_id = row["genes"]
        if gene_id in protein_sequences:
            seq = protein_sequences[gene_id]
            sequences.append(seq)
        else:
            sequences.append(None)
    
    gprdf["protein_sequence"] = sequences
    
    # 2. 预测 Topt
    gprdf = topt_predict_batch(
        gprdf,
        model_path="src/CASPred/model/HEATMAPData/model_1.pt"
    )
    
    # 3. 保存结果
    os.makedirs(result_folder, exist_ok=True)
    out_path = os.path.join(result_folder, "full_metabolites_reactions_with_topt.csv")
    gprdf.to_csv(out_path, index=False)
    return gprdf

def get_topt_data(gprdf, result_folder):
    """
    Extract Topt data for ecGEM construction.
    
    Args:
        gprdf (pd.DataFrame): DataFrame with Topt predictions
        result_folder (str): Folder to save results
        
    Returns:
        pd.DataFrame: DataFrame with Topt data for reactions
    """
    gprdf_topt = gprdf.copy()
    
    # Remove rows with 'None' in Topt value
    gprdf_topt = gprdf_topt[gprdf_topt['Topt'].notna()]
    
    # Convert Topt value to float
    gprdf_topt['Topt'] = gprdf_topt['Topt'].astype(float)
    
    # Sort by Topt value and keep only the first occurrence of each reaction
    gprdf_topt = gprdf_topt.sort_values('Topt', ascending=True).drop_duplicates(subset=['reactions'], keep='first')
    
    # Prepare Topt DataFrame
    reaction_topt = pd.DataFrame()
    reaction_topt['reactions'] = gprdf_topt['reactions']
    reaction_topt['data_type'] = 'Topt'
    reaction_topt['Topt'] = gprdf_topt['Topt']
    reaction_topt.reset_index(drop=True, inplace=True)
    
    reaction_topt_file = f'{result_folder}/reaction_topt.csv'
    reaction_topt.to_csv(reaction_topt_file, index=False)
    print('reaction_topt generated')
    return reaction_topt
