import pandas as pd
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from .smile_to_3D import smiles_to_3d_conformer
from .mol_to_gvp import mol_to_gvp_graph
import os
import pickle

class KcatDataset(Dataset):
    def __init__(self, data_path=None, esm_model_name="esmc_300m", cache_dir=None, log_transform=False):
        """
        Dataset for kcat prediction.
        
        Args:
            data_path: Path to the JSON file containing kcat data (optional if using cache_dir)
            esm_model_name: Name of the ESM model to use for protein embeddings
            cache_dir: Directory containing precomputed embeddings (optional)
            log_transform: Whether to apply log10 transformation to kcat values
        """
        self.cache_dir = cache_dir
        self.log_transform = log_transform
        
        # Load data from CSV or cache
        if cache_dir is not None:
            # When using cache, we only need to know the number of samples
            import glob
            cache_files = glob.glob(os.path.join(cache_dir, 'sample_*.pt'))
            self.num_samples = len(cache_files)
            self.esm_model = None
            print(f"Using precomputed embeddings from: {cache_dir}, found {self.num_samples} samples")
        elif data_path is not None:
            # Load data from CSV
            df = pd.read_csv(data_path)
            self.data = df.to_dict('records')
            # Load ESM model for protein embeddings
            self.esm_model = ESMC.from_pretrained(esm_model_name).to("cuda") # or "cpu"
            print(f"Loading dataset from: {data_path}, found {len(self.data)} samples")
        else:
            raise ValueError("Either data_path or cache_dir must be provided")
    
    def __len__(self):
        if self.cache_dir is not None:
            return self.num_samples
        else:
            return len(self.data)
    
    def __getitem__(self, idx):
        # If using cache, load precomputed data
        if self.cache_dir is not None:
            cache_file = os.path.join(self.cache_dir, f"sample_{idx}.pt")
            if os.path.exists(cache_file):
                cached_data = torch.load(cache_file)
                
                # Return None if cached data is None
                if cached_data is None:
                    return None
                
                # Extract data from cache
                return cached_data
            else:
                # If cache file doesn't exist, return None
                return None
        
        # Otherwise, compute embeddings on-the-fly
        item = self.data[idx]
        
        # Get metabolite graph data
        smiles = item['Smiles']
        mol_3d = smiles_to_3d_conformer(smiles)
        if mol_3d is None:
            # Return None to be filtered out later
            return None
        
        graph_data = mol_to_gvp_graph(mol_3d)
        if graph_data is None:
            # Return None to be filtered out later
            return None
            
        (node_s, node_v), edge_index, (edge_s, edge_v) = graph_data
        
        # Get protein embedding using ESM
        sequence = item['Sequence']
        protein = ESMProtein(sequence=sequence)
        
        with torch.no_grad():
            protein_tensor = self.esm_model.encode(protein)
            logits_output = self.esm_model.logits(
               protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
            )
            protein_embedding = logits_output.embeddings.squeeze(0)
        
        # Debug prints to check dimensions
        # print(f"Protein embedding shape: {protein_embedding.shape}")
        # print(f"Protein embedding dtype: {protein_embedding.dtype}")
        
        # Get kcat value
        kcat = float(item['Value'])
        
        # Apply log10 transformation if requested
        if self.log_transform:
            import numpy as np
            kcat = np.log10(kcat) if kcat > 0 else -10  # Handle zero values
        if idx == 0:
            print(f"[DEBUG] Sample 0: Original kcat = {np.exp(kcat)}, Transformed kcat = {kcat}")
        return {
            'node_s': node_s,
            'node_v': node_v,
            'edge_index': edge_index,
            'edge_s': edge_s,
            'edge_v': edge_v,
            'protein_embedding': protein_embedding,
            'kcat': torch.tensor([kcat], dtype=torch.float32)  # Ensure consistent dimension with model output
        }