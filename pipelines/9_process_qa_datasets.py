# pipelines/9_process_qa_datasets.py

import os
import sys
import argparse

# Add the project root directory to the Python path.
# This allows the script to import modules from the 'src' directory,
# treating the project as a package.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import all necessary processing functions from our modularized library.
# This modular approach keeps the pipeline script clean and focused on orchestration,
# while the complex logic resides in dedicated, reusable modules.
from src.finetuning.data_generation.dataset_splitter import (
    extract_intent, 
    extract_metadata, 
    extract_final_answer
)
from src.finetuning.data_generation.dataset_cleaner import process_jsonl_file
from src.finetuning.data_generation.dataset_utils import shuffle_jsonl, validate_year_field
from src.finetuning.data_generation.dataset_standardizer import standardize_fincap_data

def main():
    """
    Orchestrates the end-to-end processing of raw, LLM-generated QA data.

    This script acts as a pipeline that takes the single, multi-part output file
    from the generation step and transforms it into three distinct, clean, and
    standardized datasets. Each final dataset is tailored for a specific
    fine-tuning task: intent detection, metadata extraction, and final answer
    generation. The process involves splitting the raw data, followed by a
    multi-step cleaning, validation, and standardization workflow, particularly
    for the metadata dataset.
    """
    parser = argparse.ArgumentParser(
        description="Split, clean, and standardize the generated QA dataset."
    )
    parser.add_argument(
        "--market",
        type=str,
        default="sp500",
        help="The market index suffix for the input/output files (e.g., 'sp500')."
    )
    args = parser.parse_args()

    print("--- Starting Step 9: Processing QA Datasets ---")

    # --- 1. Configuration ---
    # Define the directory structure and file paths for this pipeline stage.
    input_dir = "data/finetuning"
    output_dir = "data/finetuning"
    os.makedirs(output_dir, exist_ok=True)
    
    # The primary input is the raw output from the LLM generation step.
    raw_generated_dataset = os.path.join(input_dir, f"raw_generated_qa_{args.market}.jsonl")

    # Define the final, clean output files for each fine-tuning task.
    intent_dataset_path = os.path.join(output_dir, f"intent_dataset_{args.market}.jsonl")
    final_answer_dataset_path = os.path.join(output_dir, f"final_answer_dataset_{args.market}.jsonl")
    final_metadata_dataset_path = os.path.join(output_dir, f"metadata_dataset_{args.market}_final.jsonl")

    # Define paths for intermediate files created during the metadata processing workflow.
    # Using temporary files helps break down the process into logical, debuggable steps.
    metadata_dataset_raw = os.path.join(output_dir, f"metadata_dataset_{args.market}_raw.jsonl")
    metadata_cleaned_temp = os.path.join(output_dir, f"metadata_dataset_{args.market}_cleaned_temp.jsonl")
    metadata_shuffled_temp = os.path.join(output_dir, f"metadata_dataset_{args.market}_shuffled_temp.jsonl")
    
    # --- 2. Dataset Splitting ---
    # The raw LLM output contains all components (intent, metadata, answer) in one structure.
    # The first step is to split this into three separate files, each tailored for its specific purpose.
    print(f"\n--- Step 9.1: Splitting '{os.path.basename(raw_generated_dataset)}' ---")
    if not os.path.exists(raw_generated_dataset):
        print(f"Error: Raw generated data not found at '{raw_generated_dataset}'.")
        print("Please run the generation pipeline (step 8) first.")
        return
        
    # Extracts the user query and the corresponding 'intent' JSON.
    extract_intent(raw_generated_dataset, intent_dataset_path)
    print(f"  ✓ Created intent dataset: {os.path.basename(intent_dataset_path)}")
    
    # Extracts the user query and the final, context-based answer.
    extract_final_answer(raw_generated_dataset, final_answer_dataset_path)
    print(f"  ✓ Created final answer dataset: {os.path.basename(final_answer_dataset_path)}")
    
    # Extracts the user query and the 'metadata_tools' JSON.
    extract_metadata(raw_generated_dataset, metadata_dataset_raw)
    print(f"  ✓ Created raw metadata dataset: {os.path.basename(metadata_dataset_raw)}")

    # --- 3. Metadata Dataset Processing Workflow ---
    # The metadata dataset requires the most intensive processing to ensure it's a high-quality
    # fine-tuning resource. This workflow cleans, validates, and standardizes it.
    print("\n--- Step 9.2: Cleaning and Standardizing the Metadata Dataset ---")
    
    # Step a: Perform initial cleaning. This typically involves fixing common JSON errors,
    # removing malformed entries, and ensuring a consistent basic structure.
    print("\n  Cleaning metadata records...")
    process_jsonl_file(metadata_dataset_raw, metadata_cleaned_temp)

    # Step b: Shuffle the dataset. This is a crucial step to prevent the model from
    # learning any unintentional order-based patterns from the generation process.
    print("  Shuffling cleaned data...")
    shuffle_jsonl(metadata_cleaned_temp, metadata_shuffled_temp)

    # Step c: Perform a sanity check on a critical data field. Validating the 'year'
    # ensures that a key piece of metadata is present and plausible.
    print("  Validating 'year' field in shuffled data...")
    validate_year_field(metadata_shuffled_temp)

    # Step d: Apply final standardizations. This involves casting data to correct types
    # (e.g., ensuring 'year' is an integer) to create the final, model-ready dataset.
    print("  Standardizing data types for final output...")
    standardize_fincap_data(metadata_shuffled_temp, final_metadata_dataset_path)
    
    # --- 4. Cleanup ---
    # Remove the intermediate files to keep the data directory clean and avoid confusion.
    print("\n  Cleaning up intermediate files...")
    os.remove(metadata_dataset_raw)
    os.remove(metadata_cleaned_temp)
    os.remove(metadata_shuffled_temp)
    print("  ✓ Intermediate files removed.")

    print("\n--- Pipeline Step 9 Finished Successfully ---")
    print("Final datasets are available:")
    print(f"  - Intent:     {intent_dataset_path}")
    print(f"  - Metadata:   {final_metadata_dataset_path}")
    print(f"  - Final Answer: {final_answer_dataset_path}")


if __name__ == "__main__":
    # Standard entry point for executing the script from the command line.
    main()