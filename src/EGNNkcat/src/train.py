import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import torch.multiprocessing as mp
import logging

# Set multiprocessing start method to 'spawn' to avoid CUDA issues
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from src.data.dataset import KcatDataset
from src.model.kcat_model import KcatPredictionModel

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    
    # --- 图数据批处理 ---
    node_s_list, node_v_list, edge_index_list, edge_s_list, edge_v_list = [], [], [], [], []
    batch_map = [] # 用于记录每个节点属于哪个图
    node_offset = 0

    for i, item in enumerate(batch):
        # 收集节点和边特征
        node_s_list.append(item['node_s'])
        node_v_list.append(item['node_v'])
        edge_s_list.append(item['edge_s'])
        edge_v_list.append(item['edge_v'])
        
        # 调整边索引并创建 batch 映射
        num_nodes = item['node_s'].shape[0]
        edge_index_list.append(item['edge_index'] + node_offset)
        batch_map.append(torch.full((num_nodes,), i, dtype=torch.long))
        node_offset += num_nodes

    # --- 蛋白质嵌入填充 ---
    protein_embeddings = [item['protein_embedding'] for item in batch]
    max_length = max(embedding.shape[0] for embedding in protein_embeddings)
    padded_embeddings = []
    for embedding in protein_embeddings:
        pad_length = max_length - embedding.shape[0]
        padded_embedding = torch.nn.functional.pad(embedding, (0, 0, 0, pad_length), mode='constant', value=0)
        padded_embeddings.append(padded_embedding)
    
    return {
        'node_s': torch.cat(node_s_list, dim=0),
        'node_v': torch.cat(node_v_list, dim=0),
        'edge_index': torch.cat(edge_index_list, dim=1),
        'edge_s': torch.cat(edge_s_list, dim=0),
        'edge_v': torch.cat(edge_v_list, dim=0),
        'batch_map': torch.cat(batch_map, dim=0), # 添加 batch 映射
        'protein_embedding': torch.stack(padded_embeddings),
        'kcat': torch.stack([item['kcat'] for item in batch])
    }

def train_model(config_path, output_dir, cache_dir=None, log_transform=True, checkpoint_path=None):
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    if cache_dir is not None:
        # Use cache folders for train and validation
        train_cache_dir = os.path.join(cache_dir, 'train_cache')
        val_cache_dir = os.path.join(cache_dir, 'val_cache')
        
        train_dataset = KcatDataset(config['data_path'], cache_dir=train_cache_dir, log_transform=log_transform)
        val_dataset = KcatDataset(config['data_path'], cache_dir=val_cache_dir, log_transform=log_transform)
    else:
        # Use single dataset for both train and validation (original behavior)
        train_dataset = KcatDataset(config['data_path'], cache_dir=cache_dir, log_transform=log_transform)
        val_dataset = KcatDataset(config['data_path'], cache_dir=cache_dir, log_transform=log_transform)
    
    # Print kcat statistics for debugging
    if log_transform:
        logger.info("Using log-transformed kcat values")
    else:
        logger.info("Using raw kcat values")
    
    # Create data loaders (filter out None items)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Model parameters
    gvp_params = config['gvp_params']
    
    # Initialize model
    model = KcatPredictionModel(
        gvp_params=gvp_params,
        esm_embedding_dim=config['esm_embedding_dim'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    ).to(device)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model size: {total_params} total parameters ({trainable_params} trainable)")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Cosine annealing with restarts learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config['num_epochs']//10, T_mult=1, eta_min=1e-6)
    
    # Load checkpoint if provided
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        logger.info(f"Resuming training from epoch {start_epoch + 1}")
    
    # Early stopping parameters
    best_loss = float('inf')
    patience = config.get('patience', 10)
    patience_counter = 0
    
    # Initialize lists to store losses for plotting
    train_losses = []
    val_losses = []
    learning_rates = []
    
    # Training loop
    for epoch in range(start_epoch, config['num_epochs']):
        model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')
        for batch in progress_bar:
            if batch is None:
                continue
                
            # 直接从批处理字典中获取数据
            node_s = batch['node_s']
            node_v = batch['node_v']
            edge_index = batch['edge_index']
            edge_s = batch['edge_s']
            edge_v = batch['edge_v']
            batch_map = batch['batch_map'] # 获取 batch_map
            protein_embeddings = batch['protein_embedding'].to(device)
            kcats = batch['kcat'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # 更新模型调用
            predictions = model(node_s, node_v, edge_index, edge_s, edge_v, batch_map, protein_embeddings)

            # Compute loss
            # When using log transformation, targets are already log-transformed in the dataset
            # So we only need to ensure predictions are positive
            loss = criterion(predictions, kcats.squeeze(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': loss.item()})
        
        # Compute average training loss for the epoch
        avg_train_loss = total_loss / num_batches
        train_losses.append(avg_train_loss)
        logger.info(f"Epoch [{epoch+1}/{config['num_epochs']}], Average Training Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_total_loss = 0
        val_num_batches = 0
        
        with torch.no_grad():
            val_progress_bar = tqdm(val_dataloader, desc=f'Validation Epoch {epoch+1}/{config["num_epochs"]}')
            for batch in val_progress_bar:
                if batch is None:
                    continue
                    
                # 直接从批处理字典中获取数据
                node_s = batch['node_s']
                node_v = batch['node_v']
                edge_index = batch['edge_index']
                edge_s = batch['edge_s']
                edge_v = batch['edge_v']
                batch_map = batch['batch_map'] # 获取 batch_map
                protein_embeddings = batch['protein_embedding'].to(device)
                kcats = batch['kcat'].to(device)
                
                # 更新模型调用
                predictions = model(node_s, node_v, edge_index, edge_s, edge_v, batch_map, protein_embeddings)

                # Compute validation loss
                val_loss = criterion(predictions, kcats.squeeze(-1))
                
                val_total_loss += val_loss.item()
                val_num_batches += 1
                
                # Update validation progress bar
                val_progress_bar.set_postfix({'Val Loss': val_loss.item()})
        
        # Compute average validation loss for the epoch
        avg_val_loss = val_total_loss / val_num_batches if val_num_batches > 0 else float('inf')
        val_losses.append(avg_val_loss)
        logger.info(f"Epoch [{epoch+1}/{config['num_epochs']}], Average Validation Loss: {avg_val_loss:.4f}")
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)
        logger.info(f"Current learning rate: {current_lr:.6f}")
        
        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f'model_checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved at epoch {epoch+1}")
        
        # Save best model and check early stopping based on validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            logger.info(f"Best model saved with validation loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Save training logs
    log_data = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates
    }
    log_file_path = os.path.join(output_dir, 'training_logs.json')
    with open(log_file_path, 'w') as f:
        json.dump(log_data, f)
    logger.info(f"Training logs saved to {log_file_path}")
    
    logger.info("Training completed!")

def main():
    parser = argparse.ArgumentParser(description='Train kcat prediction model')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--output', type=str, required=True, help='Output directory for model checkpoints')
    parser.add_argument('--cache_dir', type=str, default=None, help='Directory containing precomputed embeddings')
    parser.add_argument('--log_transform', action='store_true', help='Whether to apply log10 transformation to kcat values')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file to resume training')
    
    args = parser.parse_args()
    
    train_model(args.config, args.output, args.cache_dir, args.log_transform, args.checkpoint)

if __name__ == '__main__':
    main()