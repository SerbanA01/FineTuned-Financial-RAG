# pipelines/5_create_filtered_chunks.py

import os
import glob
import json
import argparse
import sys

# Add the project root to the Python path.
# This is a common pattern to ensure that the script can import modules
# from the 'src' directory, treating the project root as a package source.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data_processing.chunking import (
    process_sec_file,
    process_earnings_file,
    sec_text_splitter,
    earnings_text_splitter,
)
from src.utils.ticker_utils import (
    get_sp500_tickers,
    get_nasdaq_tickers,
    get_other_tickers,
    get_global_tickers,
    get_all_tickers
)

# --- Configuration ---
# Define the source directories for the partitioned financial documents.
# These paths are relative to the project root.
# Note: Multiple 10-K directories are supported to accommodate different data batches.
INPUT_DIRS_10K = [
    "data/raw/10-K_sectioned",
    "data/raw/10-K-sectioned_newer"
]
INPUT_DIR_10Q = "data/raw/10-Q_sectioned"
INPUT_DIR_EARNINGS = "data/raw/processed_transcripts_json"

# Define the destination for the final chunked output.
OUTPUT_DIR = "data/processed/chunks"
# Ensure the output directory exists before writing files.
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    """
    Filters and chunks financial documents based on a specified market index.

    This script serves as a key step in preparing data for a RAG system. It first
    identifies a target set of company tickers based on a command-line argument
    (e.g., 'sp500', 'nasdaq'). It then iterates through pre-processed 10-K, 10-Q,
    and earnings transcript files, processing and chunking only those that belong
    to the target tickers. The output is saved in JSONL format, which is well-suited
    for streaming large datasets.
    """
    # Set up a command-line interface to allow users to specify which market's
    # documents they want to process.
    parser = argparse.ArgumentParser(
        description="Filter and chunk financial documents for a specific market index."
    )
    parser.add_argument(
        "--market",
        type=str,
        default="sp500",
        choices=['sp500', 'nasdaq', 'nyse', 'global', 'all'],
        help="The market index to filter by (e.g., 'sp500', 'nasdaq', 'nyse', 'global', 'all')."
    )
    args = parser.parse_args()

    print(f"--- Starting filtered chunking process for market: {args.market} ---")

    # 1. Map the string-based market argument to its corresponding ticker-fetching function.
    # This design pattern avoids a large if/elif/else block and makes it easy to add new markets.
    market_functions = {
        'sp500': get_sp500_tickers,
        'nasdaq': get_nasdaq_tickers,
        'nyse': get_other_tickers, # 'nyse' maps to 'other' tickers in this context.
        'global': get_global_tickers,
        'all': get_all_tickers
    }

    selected_market = args.market.lower()
    ticker_function = market_functions.get(selected_market)

    if not ticker_function:
        print(f"Error: Market '{args.market}' is not supported.")
        return

    # 2. Load the list of tickers for the selected market.
    # This list will be used as the filter for which documents to process.
    try:
        print(f"Attempting to load tickers for market: {selected_market}...")
        # Converting the list of tickers to a set provides O(1) average time complexity
        # for lookups, which is much more efficient than list lookups (O(n)) inside a loop.
        target_tickers = set(ticker_function())
        if not target_tickers:
            print(f"Warning: No tickers were loaded for market '{selected_market}'. The function may have returned an empty list. Aborting.")
            return
        print(f"Successfully loaded {len(target_tickers)} tickers for {selected_market}.")
    except Exception as e:
        print(f"FATAL: An error occurred while loading tickers for {selected_market}: {e}")
        return

    # Define dynamic output file paths based on the selected market.
    output_10k_jsonl = os.path.join(OUTPUT_DIR, f"processed_10k_{args.market}.jsonl")
    output_10q_jsonl = os.path.join(OUTPUT_DIR, f"processed_10q_{args.market}.jsonl")
    output_earnings_jsonl = os.path.join(OUTPUT_DIR, f"processed_earnings_{args.market}.jsonl")

    # --- Process 10-K Filings ---
    print("\n--- Processing 10-K Filings ---")
    with open(output_10k_jsonl, "w", encoding="utf-8") as outfile_10k:
        for input_dir in INPUT_DIRS_10K:
            if not os.path.isdir(input_dir):
                print(f"Warning: 10-K input directory not found: {input_dir}")
                continue
            # Recursively search for all .json files within the directory.
            for filepath in glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # The core filtering logic: check if the document's ticker is in our target set.
                    ticker = data.get("metadata", {}).get("ticker")
                    if ticker and ticker in target_tickers:
                        filename_only = os.path.basename(filepath)
                        # If it's a match, delegate to the chunking and processing function.
                        process_sec_file(filepath, filename_only, sec_text_splitter, outfile_10k, "10k")
                except Exception as e:
                    # Log errors per-file to prevent a single corrupt file from halting the entire pipeline.
                    print(f"Could not process file {filepath}: {e}")

    # --- Process 10-Q Filings ---
    print("\n--- Processing 10-Q Filings ---")
    with open(output_10q_jsonl, "w", encoding="utf-8") as outfile_10q:
        if INPUT_DIR_10Q and os.path.isdir(INPUT_DIR_10Q):
            for filepath in glob.glob(os.path.join(INPUT_DIR_10Q, "**", "*.json"), recursive=True):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    ticker = data.get("metadata", {}).get("ticker")
                    if ticker and ticker in target_tickers:
                        filename_only = os.path.basename(filepath)
                        process_sec_file(filepath, filename_only, sec_text_splitter, outfile_10q, "10q")
                except Exception as e:
                    print(f"Could not process file {filepath}: {e}")
        else:
            print(f"Warning: 10-Q input directory not found: {INPUT_DIR_10Q}")

    # --- Process Earnings Transcripts ---
    print("\n--- Processing Earnings Transcripts ---")
    with open(output_earnings_jsonl, "w", encoding="utf-8") as outfile_earnings:
        if INPUT_DIR_EARNINGS and os.path.isdir(INPUT_DIR_EARNINGS):
            for filepath in glob.glob(os.path.join(INPUT_DIR_EARNINGS, "**", "*.json"), recursive=True):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # Note: The metadata structure for earnings transcripts is slightly different.
                    ticker = data.get("document_metadata", {}).get("ticker")
                    if ticker and ticker in target_tickers:
                        filename_only = os.path.basename(filepath)
                        process_earnings_file(filepath, filename_only, earnings_text_splitter, outfile_earnings)
                except Exception as e:
                    print(f"Could not process file {filepath}: {e}")
        else:
            print(f"Warning: Earnings input directory not found: {INPUT_DIR_EARNINGS}")

    print("\n--- Filtered Chunking Process Complete ---")
    print(f"10-K chunks saved to {output_10k_jsonl}")
    print(f"10-Q chunks saved to {output_10q_jsonl}")
    print(f"Earnings chunks saved to {output_earnings_jsonl}")

if __name__ == "__main__":
    # This standard Python construct ensures that the main function is called
    # only when the script is executed directly.
    main()