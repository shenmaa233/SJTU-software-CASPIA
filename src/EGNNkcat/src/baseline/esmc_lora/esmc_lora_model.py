import json
import os
import torch
import torch.nn as nn
from esm.models.esmc import ESMC
from peft import LoraConfig, TaskType, get_peft_model


class ESMCLoRAModel(nn.Module):
    """
    An ESMC model fine-tuned with Low-Rank Adaptation (LoRA) for a
    regression/classification task.
    """
    def __init__(self, config_path="config.json"):
        super().__init__()

        # --- Load Configuration ---
        if not os.path.isabs(config_path):
            # Attempt to locate the config file relative to the script's directory
            script_dir = os.path.dirname(__file__)
            absolute_config_path = os.path.abspath(os.path.join(script_dir, config_path))
            if os.path.exists(absolute_config_path):
                config_path = absolute_config_path

        with open(config_path, 'r') as f:
            config = json.load(f)

        # --- Load and Configure the Base ESMC Model ---
        esm_model_name = config.get('esm_model_name', 'esmc_300m')
        self.esm_model = ESMC.from_pretrained(esm_model_name)

        # --- Configure and Apply LoRA ---
        # A more focused LoRA configuration targeting only the QKV layers.
        lora_config = LoraConfig(
            r=config.get("lora_rank", 8),
            lora_alpha=config.get("lora_alpha", 16),
            target_modules=["layernorm_qkv.1"], # Target only the attention QKV layers
            lora_dropout=config.get("lora_dropout", 0.1),
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )

        # The `get_peft_model` function automatically freezes the base model
        # and makes only the LoRA parameters trainable.
        self.esm_model = get_peft_model(self.esm_model, lora_config)

        # --- Define the Task-Specific Head ---
        # This head takes the pooled protein embedding and maps it to a single output.
        embedding_dim = config.get('esm_embedding_dim', 960)
        self.regression_head = nn.Linear(embedding_dim, 1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            embeddings: A tensor of pre-computed protein embeddings with shape
                        [batch_size, seq_len, embedding_dim].

        Returns:
            A tensor of logits/predictions with shape [batch_size, 1].
        """
        # The training pipeline provides embeddings directly.
        # Use simple average pooling across the sequence length dimension.
        # This creates a single fixed-size representation for each protein.
        pooled_embeddings = embeddings.mean(dim=1)

        # Pass the pooled representation through the final regression head.
        output = self.regression_head(pooled_embeddings)
        return output


def load_esmc_lora_model(config_path="config.json") -> ESMCLoRAModel:
    """
    A helper function to instantiate the ESMCLoRAModel.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        An instance of the ESMCLoRAModel.
    """

    model = ESMCLoRAModel(config_path)
    return model