import argparse
import pandas as pd
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hyena.tokenizer import CharacterTokenizer
from hyena.model import HyenaDNAModel

# Function to load the model checkpoint 
def load_checkpoint(model, checkpoint_path, optimizer=None):
    """
    Loads the model checkpoint from the specified path.

    Args:
        model (torch.nn.Module): The model to load the checkpoint into.
        checkpoint_path (str): Path to the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into, if provided.

    Returns:
        model (torch.nn.Module): Model with loaded weights.
        optimizer (torch.optim.Optimizer, optional): Optimizer with loaded state, if provided.
        epoch (int): The epoch number at which the checkpoint was saved.
        loss (float): The loss value when the checkpoint was saved.
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Loaded checkpoint from epoch {epoch} with loss {loss}")
    return model, optimizer, epoch, loss

# Function to predict the optimal temperature for a given protein sequence
def predict_optimal_temperature(model, tokenizer, sequence, device, max_length):
    """
    Predicts the optimal temperature for a given protein sequence.

    Args:
        model (torch.nn.Module): The trained model.
        tokenizer (CharacterTokenizer): The tokenizer used to encode the sequence.
        sequence (str): The protein sequence to predict for.
        device (str): The device to run the prediction on ('cpu' or 'cuda').
        max_length (int): Maximum length of the sequence.

    Returns:
        float: Predicted optimal temperature.
    """
    model.eval()
    with torch.no_grad():
        # Encode the sequence with padding/truncation to max_length
        encoded_sequence = tokenizer.encode(sequence)[:max_length]
        # Pad or truncate the sequence to the specified max_length
        if len(encoded_sequence) < max_length:
            encoded_sequence += [0] * (max_length - len(encoded_sequence))

        input_tensor = torch.tensor([encoded_sequence], dtype=torch.float).to(device)
        prediction = model(input_tensor)
        return prediction.item()

# Main function to handle argument parsing and prediction
def main():
    parser = argparse.ArgumentParser(description="Predict optimal temperature for a protein sequence using a trained model.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint file')
    parser.add_argument('--sequence', type=str, required=True, help='Protein sequence to predict the optimal temperature for')
    parser.add_argument('--d_model', type=int, default=256, help='Dimension size of the model')
    parser.add_argument('--n_layer', type=int, default=4, help='Number of layers in the model')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--max_length', type=int, default=1000, help='Maximum length of the input sequence')
    args = parser.parse_args()

    # List of standard amino acids
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

    # Set up the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # Create the CharacterTokenizer for protein sequences
    protein_tokenizer = CharacterTokenizer(characters=amino_acids, model_max_length=args.max_length)

    # Initialize the model
    model = HyenaDNAModel(
        d_model=args.d_model, 
        n_layer=args.n_layer, 
        d_inner=args.hidden_size, 
        vocab_size=len(amino_acids) + 1, 
        use_head=True, 
        n_classes=1
    )
    model.to(device)

    # Load the model checkpoint
    load_checkpoint(model, args.checkpoint)

    # Predict optimal temperature for the provided sequence
    predicted_temperature = predict_optimal_temperature(model, protein_tokenizer, args.sequence, device, args.max_length)
    print(f"Predicted optimal temperature: {predicted_temperature:.2f}")

if __name__ == "__main__":
    main()

