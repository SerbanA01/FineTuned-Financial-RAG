import glob
import sys
import os
from pathlib import Path
from tqdm import tqdm

# This boilerplate allows the script to be run from the 'scripts' directory
# and still import modules from the 'src' directory as if it were a package.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing.cleaners import process_file as process_general_file
from src.data_processing.apple_cleaner import process_file_apple_v2

# Define the directory paths for raw and cleaned data.
# Using pathlib.Path makes the code OS-agnostic.
RAW_FILINGS_DIR = Path("data/raw/sec_filings")
CLEAN_FILINGS_DIR = Path("data/processed/sec_filings_clean")

def run_general_cleaning():
    """
    Processes all downloaded SEC filings using a generic cleaning function.

    This function iterates through all raw .json filings, applies a standard
    cleaning and text extraction process, and saves the cleaned output.
    It explicitly skips Apple (AAPL) filings, which are handled by a specialized
    function due to their unique formatting.
    """
    print("--- Starting Step 1: General Filing Cleaning ---")
    
    # Ensure the destination directory for cleaned files exists before processing.
    CLEAN_FILINGS_DIR.mkdir(parents=True, exist_ok=True)

    # Recursively find all .json files in the raw data directory.
    files_to_clean = glob.glob(str(RAW_FILINGS_DIR / "**/*.json"), recursive=True)
    
    # If no files are found, inform the user and exit gracefully to prevent errors.
    if not files_to_clean:
        print(f"Warning: No raw filings found in '{RAW_FILINGS_DIR}'. Skipping general cleaning.")
        return

    errors = 0
    # Use tqdm to create a progress bar for better user experience during the long-running process.
    for fp in tqdm(files_to_clean, desc="Cleaning general filings"):
        try:
            # Apple's filings have a distinct structure and require a dedicated cleaner.
            # We skip them here to be processed in the next step.
            if "AAPL" in str(fp):
                continue
            process_general_file(Path(fp), CLEAN_FILINGS_DIR)
        except Exception as e:
            # Basic error handling to prevent the entire script from crashing on a single bad file.
            # Errors are logged to a file for later review.
            errors += 1
            with open("clean_errors.log", "a", encoding="utf-8") as log:
                log.write(f"{fp}\t{e}\n")

    print(f"General cleaning finished. {len(files_to_clean) - errors} cleaned, {errors} errors (see clean_errors.log).")


def run_apple_specific_cleaning():
    """
    Processes raw SEC filings for Apple (AAPL) using a specialized cleaner.

    Apple's filings often contain unique table formats and section headers
    that the general-purpose cleaner may not handle optimally. This function
    isolates and processes them with a custom-built solution.
    """
    print("\n--- Starting Step 2: Apple-Specific Filing Cleaning ---")
    
    apple_raw_dir = RAW_FILINGS_DIR / "AAPL"
    apple_clean_dir = CLEAN_FILINGS_DIR
    
    # Check if the source directory for Apple filings exists before proceeding.
    if not apple_raw_dir.exists():
        print(f"Info: Apple-specific raw data directory not found at '{apple_raw_dir}'. Skipping.")
        return

    apple_files_to_clean = list(apple_raw_dir.glob("**/*.json"))
    
    # Also check if there are any actual files to process.
    if not apple_files_to_clean:
        print(f"Info: No raw Apple filings found in '{apple_raw_dir}'. Skipping.")
        return

    for file_path in tqdm(apple_files_to_clean, desc="Cleaning Apple filings"):
        if file_path.is_file():
            # Invoke the specialized cleaner for Apple filings.
            process_file_apple_v2(file_path, apple_clean_dir, apple_raw_dir)
            
    print("Apple-specific cleaning finished.")


if __name__ == "__main__":
    # This block defines the main execution flow of the script.
    # It's structured as a pipeline: first, clean the general documents,
    # then handle the specific case of Apple documents.
    run_general_cleaning()
    run_apple_specific_cleaning()
    print("\n✅ All cleaning steps are complete.")