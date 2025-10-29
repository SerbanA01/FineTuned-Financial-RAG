# pipelines/10_finetune_model.py

"""
FINE-TUNE A CAUSAL LANGUAGE MODEL USING LoRA

This script orchestrates the fine-tuning of a base language model for one of
three specific downstream tasks: intent classification, metadata extraction, or
final answer generation. It leverages Parameter-Efficient Fine-Tuning (PEFT)
with LoRA and 4-bit quantization (QLoRA) to make the process memory-efficient
and suitable for execution in environments with limited GPU resources, such as
Google Colab.

The script is driven by a command-line argument (`--task`) that configures the
entire pipeline, including which dataset to use, how to format the prompts, and
where to save the resulting trained adapter weights.
"""

import os
import sys
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

# Add the project root directory to the Python path.
# This allows the script to import modules from the 'src' directory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import our custom prompt formatting functions, which are crucial for
# structuring the input to the model in a consistent, expected format.
from src.finetuning.prompt_formatter import (
    format_intent_training_example,
    format_metadata_extraction_example,
    format_final_answer_example
)

def main():
    """
    Main function to drive the model fine-tuning process.

    Parses command-line arguments to determine the specific task, then loads
    the appropriate dataset and configuration. It handles model and tokenizer
    setup, including 4-bit quantization for memory efficiency. The script then
    initializes and runs the `SFTTrainer` to perform the fine-tuning and
    finally saves the trained LoRA adapter.
    """
    parser = argparse.ArgumentParser(description="Fine-tune a model for a specific task.")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=['intent', 'metadata', 'final_answer'],
        help="The fine-tuning task to perform, which determines the dataset and prompt format."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        required=True,
        help="Hugging Face API token for downloading gated models."
    )
    args = parser.parse_args()

    # --- 1. Configuration ---
    # The base model to be fine-tuned. All task-specific models will be derived from this.
    base_model_id = "meta-llama/Llama-3.1-8B-Instruct"
    
    # A configuration dictionary to dynamically select settings based on the task.
    # This design pattern avoids complex if/else blocks and makes the script easily extensible.
    task_configs = {
        'intent': {
            'dataset_file': 'data/finetuning/intent_dataset_sp500.jsonl',
            'adapter_dir': 'models/adapters/intent_adapter',
            'formatting_func': format_intent_training_example
        },
        'metadata': {
            'dataset_file': 'data/finetuning/metadata_dataset_sp500_final.jsonl',
            'adapter_dir': 'models/adapters/metadata_adapter',
            'formatting_func': format_metadata_extraction_example
        },
        'final_answer': {
            'dataset_file': 'data/finetuning/final_answer_dataset_sp500.jsonl',
            'adapter_dir': 'models/adapters/final_answer_adapter',
            'formatting_func': format_final_answer_example
        }
    }
    
    config = task_configs[args.task]
    dataset_file = config['dataset_file']
    adapter_output_dir = config['adapter_dir']
    formatting_func = config['formatting_func']
    
    os.makedirs(adapter_output_dir, exist_ok=True)

    print(f"--- Starting Fine-Tuning Pipeline for Task: '{args.task}' ---")

    # --- 2. Hugging Face Login ---
    # Authentication is required to download gated models like Llama 3.
    print("Logging into Hugging Face Hub...")
    login(token=args.hf_token)

    # --- 3. Load Model and Tokenizer ---
    print(f"Loading base model: {base_model_id}")
    # Configure BitsAndBytes for 4-bit quantization (QLoRA).
    # This dramatically reduces the model's memory footprint.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # A highly effective quantization type.
        bnb_4bit_compute_dtype=torch.float16, # The compute dtype during the forward pass.
        bnb_4bit_use_double_quant=True, # Further memory optimization.
    )

    # Load the base model with the specified quantization config.
    # `device_map="auto"` intelligently distributes the model layers across available hardware (GPUs/CPU).
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    # The pad token is set to the end-of-sequence token, a common practice for causal LMs.
    tokenizer.pad_token = tokenizer.eos_token
    # Padding on the right side is important for decoder-only models.
    tokenizer.padding_side = "right"

    # --- 4. Load Dataset ---
    print(f"Loading dataset from: {dataset_file}")
    dataset = load_dataset("json", data_files=dataset_file, split="train")

    # --- 5. Configure Training ---
    # LoRA (Low-Rank Adaptation) configuration.
    # We are only training a small number of adapter weights, not the full model.
    peft_config = LoraConfig(
        lora_alpha=16,   # A scaling factor for the LoRA weights.
        lora_dropout=0.1,# Dropout for regularization.
        r=64,            # The rank of the update matrices, a key hyperparameter.
        bias="none",
        task_type="CAUSAL_LM", # Specifies the model type for PEFT.
    )

    # SFTConfig contains all the training arguments for the SFTTrainer.
    training_arguments = SFTConfig(
        output_dir=os.path.join(adapter_output_dir, "logs"), # Directory for logs and checkpoints.
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit", # Paged optimizer is memory-efficient for QLoRA.
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False, # fp16 and bf16 are mutually exclusive.
        bf16=True,  # Use bfloat16 for mixed-precision training on modern GPUs.
        max_grad_norm=0.3, # Gradient clipping to prevent exploding gradients.
        max_steps=-1, # If > 0, overrides num_train_epochs.
        warmup_ratio=0.03, # A small warmup period for the learning rate.
        group_by_length=True, # Batches samples of similar length together to minimize padding and speed up training.
        lr_scheduler_type="constant",
        max_length=2048, # Maximum sequence length for the model.
        packing=False # If True, would pack multiple short sequences into one, not used here for simplicity.
    )

    # --- 6. Initialize and Run Trainer ---
    # The SFTTrainer from TRL is specialized for supervised fine-tuning.
    # It seamlessly integrates with PEFT and handles the data formatting.
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_func, # Our custom function to format each training example.
        args=training_arguments
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")

    # --- 7. Save Adapter ---
    # Only the trained LoRA adapter weights are saved, not the entire base model.
    # This makes the output small and portable.
    print(f"Saving adapter to {adapter_output_dir}")
    trainer.save_model(adapter_output_dir)

    print(f"\n--- Pipeline Finished Successfully for Task: '{args.task}' ---")
    print(f"Adapter saved in: {adapter_output_dir}")

if __name__ == "__main__":
    main()