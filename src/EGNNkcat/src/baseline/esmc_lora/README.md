# ESM3 LoRA Fine-tuning

This directory contains code for fine-tuning the ESM3 model using Parameter-Efficient Fine-Tuning (PEFT) with LoRA (Low-Rank Adaptation).

## Overview

The ESM3 model is a protein language model that can be fine-tuned for various downstream tasks. This implementation uses LoRA to reduce the number of trainable parameters while maintaining performance.

## Files

- `esm3_lora_model.py`: Contains the ESM3 model with LoRA adapters
- `train_esm3_lora.py`: Training script using the provided protein sequence dataset

## Usage

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the training script with the provided dataset:
   ```
   python src/baseline/esm3_lora/train_esm3_lora.py
   ```

## Implementation Details

The implementation follows these steps:

1. Load the ESM3 model using the configuration from `config.json`
2. Apply LoRA adapters to the model using the PEFT library
3. Freeze all parameters except the LoRA adapters
4. Add a task-specific head for the target task (in this example, a simple linear layer)
5. Train only the LoRA parameters and the task-specific head

## Configuration

The LoRA configuration can be adjusted in `esm3_lora_model.py`:

- `r`: The rank of the LoRA adapters (default: 8)
- `lora_alpha`: Scaling factor for the LoRA adapters (default: 32)
- `target_modules`: Which modules to apply LoRA to (default: ["query", "value"])
- `lora_dropout`: Dropout rate for the LoRA adapters (default: 0.1)

## Dataset

The training script uses the dataset in `data/data_without_zero/` which contains:
- `train.csv`: Training data with protein sequences and normalized kcat values
- `val.csv`: Validation data
- `test.csv`: Test data

Each CSV file has two columns:
- `aaseq`: Amino acid sequence of the protein
- `label`: Normalized kcat value in the range [-1, 1]

## Notes

- This is a basic implementation for demonstration purposes
- For actual use, you would need to adapt the model and training script to your specific task
- The current implementation uses a simple average pooling of the sequence embeddings, which may not be optimal for all tasks