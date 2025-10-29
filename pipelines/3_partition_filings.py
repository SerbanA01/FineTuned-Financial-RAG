import json
import pathlib
import sys
import os
from tqdm import tqdm

# This boilerplate allows the script to be run from the 'scripts' directory
# and still import modules from the 'src' directory as if it were a package.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_partitioning.filing_partitioner import (
    split_sections_10k,
    process_sections_10k,
    extract_sections_from_text_10q
)

# Define key directory paths for data processing.
# Using pathlib.Path ensures cross-platform compatibility.
CLEAN_FILINGS_DIR = pathlib.Path("data/processed/sec_filings_clean")
PARTITIONED_10K_DIR = pathlib.Path("data/processed/10-K_sectioned")
PARTITIONED_10Q_DIR = pathlib.Path("data/processed/10-Q_sectioned")


def run_10k_partitioning():
    """
    Identifies, parses, and partitions cleaned 10-K filings into their standard sections.

    This function scans the directory of cleaned filings, processes each 10-K document
    to extract key sections (e.g., "Item 1A. Risk Factors"), and saves the structured,
    sectioned data into a new directory, organized by company ticker.
    """
    print("--- Starting Step 1: Partitioning Cleaned 10-K Filings ---")
    PARTITIONED_10K_DIR.mkdir(exist_ok=True, parents=True)

    # Recursively find all cleaned 10-K JSON files.
    clean_10k_files = list(CLEAN_FILINGS_DIR.glob("**/10-K/*.json"))
    if not clean_10k_files:
        print(f"Warning: No cleaned 10-K files found in '{CLEAN_FILINGS_DIR}'. Skipping 10-K partitioning.")
        return

    # Use tqdm to show a progress bar, which is helpful for long-running tasks.
    for file_path in tqdm(clean_10k_files, desc="Partitioning 10-K files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Accommodate two possible input JSON structures for metadata.
            # This makes the script more robust to variations in the upstream cleaning process.
            if data.get("metadata") is None:
                ticker = data.get("ticker", "UNKNOWN_TICKER")
                date = data.get("date", None)
                cik = data.get("cik")
                year = data.get("year", "Unknown_Year")
                filing_type = data.get("filing_type", "10-K")
            else:
                metadata_in = data["metadata"]
                ticker = metadata_in.get("ticker", "UNKNOWN_TICKER")
                cik = metadata_in.get("cik")
                year = metadata_in.get("year", "Unknown_Year")
                date = metadata_in.get("date", None)
                filing_type = metadata_in.get("filing_type", "10-K")

            # Standardize the output metadata structure.
            metadata_out = {
                "ticker": ticker,
                "date": date,
                "cik": cik,
                "year": year,
                "filing_type": filing_type
            }

            # The core logic: split the raw text into sections and process them.
            text_to_partition = data.get("cleaned_text", "")
            sections = split_sections_10k(text_to_partition)
            processed_sections = process_sections_10k(sections)

            # If no sections were successfully extracted, skip creating an empty file.
            if not processed_sections:
                continue

            # Organize output files into directories named by ticker for easy access.
            output_ticker_directory = PARTITIONED_10K_DIR / (ticker if ticker else "UNKNOWN_TICKER")
            output_ticker_directory.mkdir(exist_ok=True, parents=True)

            # Create a consistent filename based on the filing's year and date.
            filename = f"{year}_{date}.json" if date else f"{year}.json"
            output_file_path = output_ticker_directory / filename

            # Combine metadata and sectioned content into a single output object.
            output_data = {"metadata": metadata_out, "sections": processed_sections}
            with open(output_file_path, 'w', encoding='utf-8') as out_f:
                json.dump(output_data, out_f, ensure_ascii=False, indent=4)

        except Exception as e:
            # Catching exceptions on a per-file basis prevents one bad file from
            # halting the entire batch processing job.
            print(f"Error partitioning 10-K file {file_path}: {e}")

    print("10-K partitioning finished.")


def run_10q_partitioning():
    """
    Identifies, parses, and partitions cleaned 10-Q filings into their standard sections.

    Similar to the 10-K function, but uses a different partitioning logic
    (`extract_sections_from_text_10q`) tailored to the structure of quarterly reports.
    """
    print("\n--- Starting Step 2: Partitioning Cleaned 10-Q Filings ---")
    PARTITIONED_10Q_DIR.mkdir(exist_ok=True, parents=True)
    
    clean_10q_files = list(CLEAN_FILINGS_DIR.glob("**/10-Q/*.json"))
    if not clean_10q_files:
        print(f"Warning: No cleaned 10-Q files found in '{CLEAN_FILINGS_DIR}'. Skipping 10-Q partitioning.")
        return

    for file_path in tqdm(clean_10q_files, desc="Partitioning 10-Q files"):
        try:
            data = json.loads(file_path.read_text(encoding='utf-8'))
            
            # The core partitioning logic for 10-Q filings.
            sections = extract_sections_from_text_10q(data.get('cleaned_text', ''))
            
            # Only proceed if the partitioning logic successfully found sections.
            if sections:
                # Extract metadata, providing default values for robustness.
                ticker = data.get('ticker', 'UNKNOWN')
                year = data.get('year', 'UNKNOWN_YEAR')
                cik = data.get('cik', 'UNKNOWN_CIK')
                date = data.get('date', 'UNKNOWN_DATE')
                
                metadata = {
                    'ticker': ticker, 'year': year, 'cik': cik,
                    'date': date, 'filing_type': '10-Q'
                }
                
                output_data = {'metadata': metadata, 'sections': sections}

                # Define and create the output directory structure.
                out_dir_path = PARTITIONED_10Q_DIR / ticker
                out_dir_path.mkdir(parents=True, exist_ok=True)
                out_file = out_dir_path / f"{year}_{date}.json"
                
                # Write the final partitioned data to a new JSON file.
                with open(out_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error partitioning 10-Q file {file_path}: {e}")

    print("10-Q partitioning finished.")


if __name__ == "__main__":
    # This script is designed to be run as a standalone process.
    # It executes the partitioning steps sequentially for annual (10-K)
    # and quarterly (10-Q) reports.
    run_10k_partitioning()
    run_10q_partitioning()
    print("\n✅ All partitioning steps are complete.")