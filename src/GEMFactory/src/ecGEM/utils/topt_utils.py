import os
import pandas as pd
import torch
from tqdm import tqdm
from src.CASPred.src.hyena.tokenizer import CharacterTokenizer
from src.CASPred.src.hyena.model import HyenaDNAModel
from .protein_utils import get_protein_sequences_from_fasta

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

# === Batch prediction function ===
def topt_predict_batch(protein_sequences, model_path, max_length=1000):
    """
    Batch predict optimal temperatures for multiple protein sequences.

    Args:
        protein_sequences (List[str]): List of protein sequences
        model_path (str): Path to the model checkpoint file
        max_length (int): Maximum length of the sequence

    Returns:
        List[float]: List of predicted Topt values for each protein sequence
    """
    model, tokenizer, device = load_topt_model(model_path)
    results = []

    for idx, protein_sequence in enumerate(tqdm(protein_sequences, total=len(protein_sequences), desc="Predicting Topt")):
        if protein_sequence is None or str(protein_sequence).strip() == "" or len(protein_sequence) > max_length:
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

    return results