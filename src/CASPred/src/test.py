import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import argparse
import logging
import sys

# Add the parent directory to the path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def evaluate_model(model_path, config_path, test_data_path, cache_dir=None, log_transform=True):
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create test dataset
    if cache_dir is not None:
        # Use cache folder for test
        test_cache_dir = os.path.join(cache_dir, 'test_cache')
        test_dataset = KcatDataset(cache_dir=test_cache_dir, log_transform=log_transform)
    else:
        # Use single dataset for test
        test_dataset = KcatDataset(test_data_path, log_transform=log_transform)
    
    # Print kcat statistics for debugging
    if log_transform:
        logger.info("Using log-transformed kcat values")
    else:
        logger.info("Using raw kcat values")
    
    # Create data loader (filter out None items)
    test_dataloader = DataLoader(
        test_dataset, 
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
    
    # Load trained model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    logger.info("Model loaded successfully")
    
    # Evaluation
    all_predictions = []
    all_targets = []
    total_loss = 0
    num_batches = 0
    
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch in test_dataloader:
            if batch is None:
                continue
                
            # Get data from batch
            node_s = batch['node_s']
            node_v = batch['node_v']
            edge_index = batch['edge_index']
            edge_s = batch['edge_s']
            edge_v = batch['edge_v']
            batch_map = batch['batch_map']
            protein_embeddings = batch['protein_embedding'].to(device)
            kcats = batch['kcat'].to(device)
            
            # Forward pass
            predictions = model(node_s, node_v, edge_index, edge_s, edge_v, batch_map, protein_embeddings)
            
            # Compute loss
            loss = criterion(predictions, kcats.squeeze(-1))
            
            total_loss += loss.item()
            num_batches += 1
            
            # Store predictions and targets for metrics calculation
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(kcats.squeeze(-1).cpu().numpy())
    
    # Calculate average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Apply inverse log transform if needed
    if log_transform:
        all_predictions = 10 ** all_predictions
        all_targets = 10 ** all_targets
    
    # Calculate metrics
    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_predictions)
    
    # Calculate Pearson correlation coefficient
    pearson_correlation = np.corrcoef(all_targets, all_predictions)[0, 1]
    
    # <<< NEW: Calculate Spearman correlation coefficient >>>
    spearman_correlation, _ = spearmanr(all_targets, all_predictions)
    # Handle potential NaN result if there is no variance in the data
    if np.isnan(spearman_correlation):
        spearman_correlation = 0.0
    
    # Print results
    logger.info("Test Results:")
    logger.info(f"Average Loss: {avg_loss:.4f}")
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"R²: {r2:.4f}")
    logger.info(f"Pearson Correlation: {pearson_correlation:.4f}") # Renamed for clarity
    logger.info(f"Spearman Correlation: {spearman_correlation:.4f}") # <<< NEW
    
    return {
        'avg_loss': avg_loss,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson_correlation': pearson_correlation, # Renamed for clarity
        'spearman_correlation': spearman_correlation, # <<< NEW
        'predictions': all_predictions,
        'targets': all_targets
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate kcat prediction model on test set')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--test_data', type=str, help='Path to test data CSV file')
    parser.add_argument('--cache_dir', type=str, help='Directory containing precomputed embeddings')
    parser.add_argument('--log_transform', action='store_true', 
                        help='Whether the model was trained with log-transformed labels')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.cache_dir is None and args.test_data is None:
        parser.error("Either --test_data or --cache_dir must be provided")
    
    results = evaluate_model(
        model_path=args.model,
        config_path=args.config,
        test_data_path=args.test_data,
        cache_dir=args.cache_dir,
        log_transform=args.log_transform
    )
    
    # Save results
    output_dir = os.path.dirname(args.model) if os.path.dirname(args.model) else '.'
    results_file = os.path.join(output_dir, 'test_results.json')
    
    # Convert numpy arrays to lists for JSON serialization
    results_for_json = results.copy()
    results_for_json['predictions'] = results_for_json['predictions'].tolist()
    results_for_json['targets'] = results_for_json['targets'].tolist()
    
    with open(results_file, 'w') as f:
        json.dump(results_for_json, f, indent=2)
    
    print(f"Test results saved to {results_file}")

if __name__ == '__main__':
    main()