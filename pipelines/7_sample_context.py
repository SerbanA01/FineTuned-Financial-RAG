# pipelines/7_sample_contexts.py

import os
import sys
import json
import argparse

# Add the project root directory to the Python path.
# This allows the script to import modules from the 'src' directory,
# treating the project as a package.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.finetuning.data_generation.sampling_utils import get_random_non_overlapping_groups

def main():
    """
    Samples groups of context chunks from processed financial documents.

    This script serves as a preparatory step for generating synthetic fine-tuning data.
    It reads the chunked .jsonl files for a specific market (e.g., S&P 500),
    randomly samples a specified number of non-overlapping groups of chunks from
    each file, and saves these sampled groups into a single JSON file. This output
    file is then used by the next pipeline step to generate question-answer pairs
    based on the provided contexts.
    """
    parser = argparse.ArgumentParser(
        description="Sample context groups from curated data chunks."
    )
    parser.add_argument(
        "--market",
        type=str,
        default="sp500",
        help="The market index to sample from (e.g., 'sp500'). This determines the input directory."
    )
    args = parser.parse_args()
    
    print("--- Starting Step 7: Sampling Context Groups ---")

    # --- 1. Configuration ---
    # Define the source and destination paths for the data.
    input_chunks_dir = f"data/processed/chunks_{args.market}"
    output_dir = "data/finetuning"
    output_file_path = os.path.join(output_dir, f"sampled_contexts_{args.market}.json")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the parameters for the sampling process.
    # N_CHUNKS_PER_GROUP: How many consecutive chunks form a single context group.
    # N_GROUPS_PER_FILE: The maximum number of context groups to sample from each source document.
    N_CHUNKS_PER_GROUP = 5
    N_GROUPS_PER_FILE = 250

    # --- 2. Data Sampling ---
    print(f"Scanning for chunk files in '{input_chunks_dir}'...")
    
    # This dictionary will store the results, mapping a source filename to a list of sampled groups.
    sampled_groups_by_file = {}
    
    # Pre-flight check to ensure the source data directory exists.
    if not os.path.exists(input_chunks_dir):
        print(f"Error: Input directory not found: {input_chunks_dir}.")
        print("Please ensure you have run the chunking pipeline (step 5) first.")
        return

    # Find all relevant .jsonl files in the input directory.
    chunk_files = [f for f in os.listdir(input_chunks_dir) if f.endswith('.jsonl')]
    if not chunk_files:
        print(f"Warning: No .jsonl files found in '{input_chunks_dir}'.")
        return
        
    for doc in chunk_files:
        file_path = os.path.join(input_chunks_dir, doc)
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read all lines from the chunk file into memory, filtering out empty lines.
            rows = [line.strip() for line in f if line.strip()]
            
            # Delegate the core sampling logic to the utility function. This ensures
            # that we get distinct, non-overlapping sets of chunks from the document.
            groups_for_file = get_random_non_overlapping_groups(
                rows, N_CHUNKS_PER_GROUP, N_GROUPS_PER_FILE
            )
            # Only add an entry to the results if groups were actually sampled.
            if groups_for_file:
                sampled_groups_by_file[doc] = groups_for_file
    
    total_groups = sum(len(v) for v in sampled_groups_by_file.values())
    if total_groups == 0:
        print("Warning: No context groups were sampled. The input files might be too small or empty.")
        return
        
    print(f"Successfully sampled a total of {total_groups} groups from {len(sampled_groups_by_file)} files.")

    # --- 3. Save Output ---
    # The sampled data is persisted to a JSON file, which will be the input
    # for the next step in the fine-tuning data generation pipeline.
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_groups_by_file, f, indent=2)

    print(f"--- Pipeline Step 7 Finished ---")
    print(f"Sampled contexts saved to: {output_file_path}")

if __name__ == "__main__":
    # Standard entry point for executing the script from the command line.
    main()