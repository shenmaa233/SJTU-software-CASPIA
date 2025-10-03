import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Dataset class for protein sequences
class ProteinDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length, use_padding=True, split='train', test_size=0.2, random_state=42):
        """
        Initializes the ProteinDataset.

        Args:
            file_path (str): Path to the CSV file containing protein sequences and labels.
            tokenizer (CharacterTokenizer): Tokenizer used to encode protein sequences.
            max_length (int): Maximum length of the encoded sequences.
            use_padding (bool): Whether to use padding for sequences (default is True).
            split (str): Indicates whether to use 'train' or 'test' data (default is 'train').
            test_size (float): Proportion of the dataset to include in the test split (default is 0.2).
            random_state (int): Random seed for reproducibility (default is 42).
        """
        # Load data from CSV file into a pandas DataFrame
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_padding = use_padding
        
        # Split the data into training and testing sets
        train_data, test_data = train_test_split(self.data, test_size=test_size, random_state=random_state)
        # Select the appropriate split based on the input argument
        self.data = train_data if split == 'train' else test_data

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the encoded protein sequence (as a tensor) and the corresponding label (as a tensor).
        """
        # Get the row at the specified index
        row = self.data.iloc[idx]
        # Extract the protein sequence and label from the row
        sequence = row[0]  # Assuming the first column contains the sequence
        label = row[1]     # Assuming the second column contains the label
        
        # Encode the protein sequence using the tokenizer
        encoded_sequence = self.tokenizer.encode(sequence)
        
        # Return the encoded sequence and label as tensors
        return torch.tensor(encoded_sequence), torch.tensor([label], dtype=torch.float)
