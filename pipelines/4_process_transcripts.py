import os
import re
import sys
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# This boilerplate allows the script to be run from the 'scripts' directory
# and still import modules from the 'src' directory as if it were a package.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_ingestion.transcript_parser import process_transcript_to_json_speaker_turns

# Configure basic logging to provide progress and error information during execution.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_transcript_processing():
    """
    Processes a collection of raw earnings call transcripts from a pandas DataFrame.

    This function reads a pickle file containing transcript data, iterates through each
    transcript, extracts relevant metadata (ticker, date, quarter), and uses a
    specialized parser to convert the raw transcript text into a structured JSON format
    organized by speaker turns. The resulting JSON files are saved in a nested
    directory structure: `output_dir/TICKER/YEAR_QUARTER.json`.
    """
    # Define the source data file and the target directory for processed outputs.
    pickle_file_path = "data/raw/earnings_transcripts/motley-fool-data.pkl"
    output_dir = "data/processed/processed_transcripts_json"

    # Define constants for the expected column names in the source DataFrame.
    # This improves readability and makes future column name changes easier to manage.
    transcript_col = 'transcript'
    ticker_col = 'ticker'
    date_col = 'date'
    quarter_info_col = 'q'
    exchange_col = 'exchange'

    # Ensure the output directory exists, creating it if necessary.
    os.makedirs(output_dir, exist_ok=True)

    # Halt execution if the source data file is not found.
    if not os.path.exists(pickle_file_path):
        logging.error(f"Pickle file not found: {pickle_file_path}")
        return

    try:
        df = pd.read_pickle(pickle_file_path)
        logging.info(f"Successfully loaded DataFrame from {pickle_file_path}. Shape: {df.shape}")
    except Exception as e:
        logging.error(f"Error loading pickle file {pickle_file_path}: {e}")
        return

    processed_count = 0
    error_count = 0
    # Iterate over each row in the DataFrame with a progress bar for user feedback.
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing transcripts"):
        try:
            transcript_text = row.get(transcript_col)
            # Basic validation: Skip rows that lack valid transcript text.
            if not isinstance(transcript_text, str) or not transcript_text.strip():
                logging.warning(f"Skipping row {index} (Ticker: {row.get(ticker_col, 'N/A')}) due to empty transcript.")
                continue

            # --- Metadata Extraction and Normalization ---
            base_metadata = {"original_df_index": index}
            ticker_val = row.get(ticker_col)
            base_metadata["ticker"] = str(ticker_val).upper() if pd.notna(ticker_val) else "UNKNOWN_TICKER"
            
            # Initialize year and quarter with placeholder values.
            base_metadata["year"] = "YYYY"
            base_metadata["quarter"] = "QX"
            
            # Primary method: Attempt to parse year and quarter from the dedicated 'q' column.
            q_val = row.get(quarter_info_col)
            if pd.notna(q_val) and isinstance(q_val, str):
                q_match = re.match(r"(\d{4})-(Q[1-4])", str(q_val))
                if q_match:
                    base_metadata["year"] = q_match.group(1)
                    base_metadata["quarter"] = q_match.group(2)

            # Fallback method: If 'q' column parsing failed, try to infer from the 'date' column.
            if base_metadata["year"] == "YYYY" or base_metadata["quarter"] == "QX":
                date_val = row.get(date_col)
                if pd.notna(date_val):
                    try:
                        dt_object = pd.to_datetime(date_val, errors='coerce')
                        if pd.notna(dt_object):
                            if base_metadata["year"] == "YYYY":
                                base_metadata["year"] = str(dt_object.year)
                            if base_metadata["quarter"] == "QX":
                                # Calculate the quarter from the month number.
                                quarter_month = (dt_object.month - 1) // 3 + 1
                                base_metadata["quarter"] = f"Q{quarter_month}"
                            base_metadata["full_date_parsed_from_df"] = dt_object.strftime('%Y-%m-%d')
                    except Exception as e:
                        logging.warning(f"Error parsing date for row {index}: {e}")
            
            # Store original date values for traceability and debugging.
            base_metadata["original_date_col_value"] = str(row.get(date_col, ''))
            base_metadata["original_q_col_value"] = str(row.get(quarter_info_col, ''))

            # Extract exchange information if available.
            if exchange_col in df.columns and pd.notna(row.get(exchange_col)):
                base_metadata["exchange"] = str(row.get(exchange_col)).split(":")[0].strip().upper()  

            # --- File Path and Name Generation ---
            # Sanitize the ticker symbol to create a valid directory name.
            fn_ticker = re.sub(r'[^\w\.-]', '', base_metadata["ticker"])
            fn_year = base_metadata.get("year", "YYYY")
            fn_quarter = base_metadata.get("quarter", "QX")
            
            # Organize processed files into subdirectories by ticker.
            ticker_dir = os.path.join(output_dir, fn_ticker)
            os.makedirs(ticker_dir, exist_ok=True)

            output_json_filename = os.path.join(ticker_dir, f"{fn_year}_{fn_quarter}.json")
            base_metadata["conceptual_filename"] = os.path.basename(output_json_filename).replace(".json", "")

            # Delegate the core parsing logic to the specialized function.
            process_transcript_to_json_speaker_turns(
                transcript_content=transcript_text,
                output_filename=output_json_filename,
                base_doc_metadata=base_metadata
            )
            processed_count +=1
        except Exception as e:
            # A broad exception handler ensures that an error in one row does not stop the entire process.
            error_count +=1
            logging.error(f"Critical error on row {index}: {e}", exc_info=True)

    logging.info(f"Finished. Total rows: {len(df)}. Processed: {processed_count}. Errors: {error_count}.")


if __name__ == "__main__":
    # This block executes the main function when the script is run directly.
    run_transcript_processing()
    print("\n✅ Transcript processing complete.")