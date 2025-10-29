# pipelines/8_generate_synthetic_qa.py

import os
import sys
import json
import argparse

# Add the project root directory to the Python path.
# This allows the script to import modules from the 'src' directory,
# treating the project as a package.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.finetuning.data_generation.llm_generator import generate_dataset_safely

def main():
    """
    Orchestrates the generation of a synthetic question-answer dataset.

    This script takes the groups of context chunks sampled by the previous pipeline
    step (7_sample_contexts.py) and uses a large language model (LLM) to generate
    question-answer pairs based on that context. The output is a raw, unprocessed
    .jsonl file containing the synthetic data, ready for further cleaning and
    formatting.
    """
    parser = argparse.ArgumentParser(
        description="Generate a synthetic QA dataset from sampled context groups using an LLM."
    )
    # Allows specifying which market's data to process, enabling parallel or targeted data creation.
    parser.add_argument(
        "--market",
        type=str,
        default="sp500",
        help="The market index suffix for the input/output files (e.g., 'sp500')."
    )
    # A practical feature for development and debugging to run the pipeline on a small subset of data.
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="Limit the generation to a small number of groups for a quick test run."
    )
    args = parser.parse_args()

    print("--- Starting Step 8: Generating Synthetic QA Pairs ---")

    # --- 1. Configuration ---
    # Define input and output paths based on the specified market.
    input_dir = "data/finetuning"
    input_file_path = os.path.join(input_dir, f"sampled_contexts_{args.market}.json")
    
    output_dir = "data/finetuning"
    output_file_path = os.path.join(output_dir, f"raw_generated_qa_{args.market}.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    
    # Specify the LLM to be used for generation. This is a critical parameter that
    # determines the quality and style of the generated questions and answers.
    LLM_MODEL = "meta-llama/Llama-3-70b-chat-hf"

    # --- 2. Load Sampled Data ---
    print(f"Loading sampled contexts from: {input_file_path}")
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            sampled_contexts = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file_path}")
        print("Please run the sampling pipeline (step 7) first.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file_path}. The file may be corrupt.")
        return

    # --- 3. Generate Synthetic Data ---
    print(f"Starting generation with model: {LLM_MODEL}")
    
    # To prevent appending to a previous run's results, the script ensures a clean slate
    # by deleting the output file if it already exists.
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
        print(f"Removed existing output file to start fresh: {output_file_path}")
        
    # Delegate the complex task of interacting with the LLM to a specialized function.
    # This function is designed to handle API calls, prompting, retries, and error logging gracefully.
    generate_dataset_safely(
        results_data=sampled_contexts,
        model=LLM_MODEL,
        output_file=output_file_path,
        test_run_limit=args.test_limit
    )

    print(f"\n--- Pipeline Step 8 Finished ---")
    # A final check to confirm that the output was actually created.
    if os.path.exists(output_file_path):
        print(f"Raw generated QA dataset saved to: {output_file_path}")
    else:
        print("Warning: Output file was not created. The generation process may have failed or produced no data.")


if __name__ == "__main__":
    # Standard entry point for executing the script from the command line.
    main()