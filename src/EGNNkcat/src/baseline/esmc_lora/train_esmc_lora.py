import json
import logging
import os
import sys
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

# Add the project root to the Python path for absolute imports.
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from src.baseline.esmc_lora.esmc_lora_model import ESMCLoRAModel

# A type alias for a batch to improve code readability.
Batch = List[Tuple[torch.Tensor, torch.Tensor]]


def setup_logger(log_dir: str) -> logging.Logger:
    """Configures a logger to save to a file and print to the console."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_handler = logging.FileHandler(os.path.join(log_dir, f"train_{timestamp}.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def log_model_parameters(model: nn.Module, logger: logging.Logger):
    """Calculates and logs the number of total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percentage = 100 * trainable_params / total_params

    logger.info("--- Model Parameter Analysis ---")
    logger.info(f"Total parameters: {total_params / 1e6:.2f}M")
    logger.info(f"Trainable LoRA parameters: {trainable_params / 1e6:.2f}M")
    logger.info(f"Trainable parameters are {percentage:.4f}% of the total.")
    logger.info("---------------------------------")


def collate_fn(batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collates sequences, handling variable lengths and filtering."""
    MAX_PROTEIN_LENGTH = 2048
    filtered_batch = [item for item in batch if item[0].shape[0] <= MAX_PROTEIN_LENGTH]

    if not filtered_batch:
        embedding_dim = batch[0][0].shape[1] if batch else 960
        return (torch.empty(0, 0, embedding_dim), torch.empty(0))

    embeddings, labels = zip(*filtered_batch)
    padded_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
    stacked_labels = torch.stack(labels)
    return padded_embeddings, stacked_labels


class ProteinSequenceDataset(Dataset):
    """PyTorch Dataset for loading protein sequences and k-cat values."""
    def __init__(self, csv_path: str, esm_model: ESMC):
        self.data = pd.read_csv(csv_path)
        self.esm_model = esm_model
        self.logits_config = LogitsConfig(sequence=True, return_embeddings=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.data.iloc[idx]['aaseq']
        label = float(self.data.iloc[idx]['label'])
        protein = ESMProtein(sequence=sequence)

        with torch.no_grad():
            protein_tensor = self.esm_model.encode(protein)
            logits_output = self.esm_model.logits(protein_tensor, self.logits_config)
            embedding_tensor = logits_output.embeddings.squeeze(0)

        return embedding_tensor, torch.tensor(label, dtype=torch.float32)


def train_esmc_lora():
    """Main function to train and evaluate the ESMC LoRA model."""
    # --- 1. Configuration and Setup ---
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))

    log_dir = os.path.join(script_dir, "log")
    result_dir = os.path.join(script_dir, "result")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    logger = setup_logger(log_dir)

    config_path = os.path.join(script_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Configuration loaded: {config}")

    # --- 2. Load Models and Data ---
    logger.info("Loading base ESMC model for data processing...")
    esm_model = ESMC.from_pretrained("esmc_300m").to(device)

    logger.info("Loading datasets...")
    data_dir = os.path.join(project_root, "data", "data_without_zero")
    train_dataset = ProteinSequenceDataset(os.path.join(data_dir, "train.csv"), esm_model)
    val_dataset = ProteinSequenceDataset(os.path.join(data_dir, "val.csv"), esm_model)
    test_dataset = ProteinSequenceDataset(os.path.join(data_dir, "test.csv"), esm_model)

    batch_size = config.get("batch_size", 64)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    logger.info("Initializing ESMCLoRAModel...")
    model = ESMCLoRAModel(config_path).to(device)
    log_model_parameters(model, logger)

    # --- 3. Optimizer and Learning Rate Scheduler ---
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.get("learning_rate", 0.0001)
    )

    num_epochs = config.get("num_epochs", 100)
    warmup_epochs = config.get("warmup_epochs", 5)
    warmup_steps = warmup_epochs * len(train_loader)
    total_steps = num_epochs * len(train_loader)

    warmup_scheduler = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps])

    # --- 4. Training Loop ---
    best_spearman = -1.0  # Higher is better for Spearman correlation
    patience = config.get("patience", 20)
    patience_counter = 0

    logger.info("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", file=sys.stdout)
        for inputs, targets in train_iterator:
            if inputs.numel() == 0:
                continue

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            train_iterator.set_postfix(loss=loss.item())

        # --- 5. Validation Loop ---
        model.eval()
        total_val_loss = 0
        all_preds, all_targets = [], []

        val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", file=sys.stdout)
        with torch.no_grad():
            for inputs, targets in val_iterator:
                if inputs.numel() == 0:
                    continue
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                total_val_loss += loss.item()

                all_preds.append(outputs.squeeze().cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                val_iterator.set_postfix(loss=loss.item())

        # Calculate metrics for the entire validation set
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        spearman_corr, _ = spearmanr(all_preds, all_targets)
        if np.isnan(spearman_corr):  # Handle case with no variance
            spearman_corr = 0.0

        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Spearman: {spearman_corr:.4f}"
        )

        # --- 6. Early Stopping and Model Checkpointing (based on Spearman) ---
        if spearman_corr > best_spearman:
            best_spearman = spearman_corr
            patience_counter = 0
            model_save_path = os.path.join(result_dir, "esmc_lora_best_model.pth")
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Validation Spearman improved to {best_spearman:.4f}. Saved best model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} as Spearman did not improve.")
                break

    # --- 7. Final Evaluation on Test Set ---
    logger.info("Loading best model for final testing...")
    model_load_path = os.path.join(result_dir, "esmc_lora_best_model.pth")
    if os.path.exists(model_load_path):
        model.load_state_dict(torch.load(model_load_path))
    else:
        logger.warning("No best model found. Skipping final test.")
        return

    model.eval()
    total_test_loss = 0
    all_preds_test, all_targets_test = [], []
    
    test_iterator = tqdm(test_loader, desc="Final Testing", file=sys.stdout)
    with torch.no_grad():
        for inputs, targets in test_iterator:
            if inputs.numel() == 0:
                continue
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            total_test_loss += loss.item()
            
            all_preds_test.append(outputs.squeeze().cpu().numpy())
            all_targets_test.append(targets.cpu().numpy())

    avg_test_loss = total_test_loss / len(test_loader)
    all_preds_test = np.concatenate(all_preds_test)
    all_targets_test = np.concatenate(all_targets_test)
    test_spearman, _ = spearmanr(all_preds_test, all_targets_test)
    if np.isnan(test_spearman):
        test_spearman = 0.0

    logger.info(f"Final Test Loss: {avg_test_loss:.4f}, Final Test Spearman: {test_spearman:.4f}")

    final_model_path = os.path.join(result_dir, "esmc_lora_final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training completed. Final model saved to {final_model_path}")


if __name__ == "__main__":
    train_esmc_lora()