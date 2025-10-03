# CharacterTokenizer for protein sequences
class CharacterTokenizer:
    def __init__(self, characters, model_max_length):
        """
        Initializes the tokenizer with a set of characters and the maximum sequence length.

        Args:
            characters (list): A list of characters that represent the vocabulary (e.g., amino acids).
            model_max_length (int): The maximum length of sequences the model expects.
        """
        self.characters = characters
        self.model_max_length = model_max_length
        
        # Create a dictionary that maps each character to a unique index, starting from 1
        # 'PAD' is indexed as 0, used for padding sequences
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(characters)}  
        self.char_to_idx['PAD'] = 0  # Add a padding token with index 0

    def encode(self, sequence, add_special_tokens=False):
        """
        Encodes a protein sequence into a list of numerical indices.

        Args:
            sequence (str): The protein sequence to encode.
            add_special_tokens (bool): Whether to add special tokens (not used in this context).

        Returns:
            list: A list of integers representing the encoded sequence, padded or truncated to `model_max_length`.
        """
        # Convert each character in the sequence to its corresponding index,
        # skipping characters that are not in the `char_to_idx` dictionary
        encoded = [self.char_to_idx[char] for char in sequence if char in self.char_to_idx]

        # Pad the encoded sequence with zeros (representing 'PAD') if it's shorter than `model_max_length`
        if len(encoded) < self.model_max_length:
            encoded = [0] * (self.model_max_length - len(encoded)) + encoded
        else:
            # Truncate the sequence to `model_max_length` if it's longer
            encoded = encoded[:self.model_max_length]
            
        return encoded
