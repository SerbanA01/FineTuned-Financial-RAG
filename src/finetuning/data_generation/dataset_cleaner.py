import json
import re
import sys

# This script provides functions to clean and standardize records from a JSONL file,
# specifically targeting the structure of synthetic fine-tuning data. The primary goal
# is to consolidate inconsistent field names and standardize values within the
# 'metadata.focuses' array of each record.

def clean_record(record):
    """
    Cleans and standardizes a single JSON record from the dataset.

    This function performs several key transformations on the `metadata.focuses`
    list within a record:
    1. Consolidates 'quarter' and 'raw_quarter' fields into a single 'quarter' field.
    2. Standardizes the 'quarter' value to a consistent "QX" format (e.g., "Q1").
    3. Consolidates multiple possible document type fields into a single 'doc_type'.
    4. Removes several other redundant or inconsistent keys to create a cleaner,
       more uniform data structure.

    @param record: A dictionary representing one line of the JSONL file.
    @return: The cleaned dictionary, or None if the record is invalid and should be skipped.
    """
    # A record is considered invalid if it lacks a user query.
    user_query = record.get("user_query", "").strip()
    if not user_query:
        return None

    # If the expected metadata structure doesn't exist, return the record as-is to avoid errors.
    if "metadata" not in record or "focuses" not in record.get("metadata", {}):
        return record

    cleaned_focuses = []
    for focus in record["metadata"]["focuses"]:
        cleaned_focus = focus.copy()

        # --- Quarter Field Consolidation ---
        # The raw data might contain 'quarter' or 'raw_quarter'. This logic
        # safely extracts both, ensuring 'raw_quarter' is always removed,
        # and then prioritizes the value from 'quarter' if it exists.
        quarter_from_q = cleaned_focus.pop("quarter", None)
        quarter_from_raw = cleaned_focus.pop("raw_quarter", None)

        # Use the value from 'quarter' if available, otherwise fall back to 'raw_quarter'.
        quarter_val = quarter_from_q or quarter_from_raw
        
        # --- Quarter Value Standardization ---
        # This block ensures the final quarter value is in the format 'QX'.
        if quarter_val:
            # Handle cases where the value might be a list.
            if isinstance(quarter_val, list) and quarter_val:
                quarter_val = quarter_val[0]
            
            # Use regex to find the digit, regardless of "q" prefix or data type.
            quarter_match = re.search(r'[qQ]?(\d)', str(quarter_val))
            if quarter_match:
                cleaned_focus["quarter"] = f"Q{quarter_match.group(1)}"
            else:
                cleaned_focus["quarter"] = None # Nullify if the format is unrecognizable.
        else:
            cleaned_focus["quarter"] = None

        # --- Document Type Consolidation ---
        # The raw data has several possible keys for the document type. This
        # consolidates them into a single, standardized 'doc_type' field.
        doc_type = (cleaned_focus.pop("normalized_doc_type", None) or
                    cleaned_focus.pop("source_type", None) or
                    cleaned_focus.pop("filing_type", None))
        if doc_type:
            cleaned_focus["doc_type"] = str(doc_type).upper().replace("_", " ").strip()
        else:
            cleaned_focus["doc_type"] = None

        # Remove other keys that are inconsistent or not needed in the final dataset.
        for key in ["company_name", "exchange", "date", "time", "original_file_name"]:
            cleaned_focus.pop(key, None)

        cleaned_focuses.append(cleaned_focus)

    record["metadata"]["focuses"] = cleaned_focuses
    return record

def process_jsonl_file(input_path, output_path):
    """
    Reads a JSONL file, applies cleaning logic to each line, and writes the
    results to a new JSONL file.

    This function serves as the main orchestrator, handling file I/O and
    line-by-line processing with robust error handling for malformed JSON.

    @param input_path: The path to the source JSONL file.
    @param output_path: The path where the cleaned JSONL file will be saved.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:

            print(f"Starting processing of '{input_path}'...")
            processed_count = 0
            written_count = 0

            # Process the file line by line to handle large files efficiently.
            for line in f_in:
                processed_count += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    original_record = json.loads(line)
                    cleaned_record = clean_record(original_record)
                    # Only write the record if the cleaning function returns a valid object.
                    if cleaned_record:
                        f_out.write(json.dumps(cleaned_record) + '\n')
                        written_count += 1
                except json.JSONDecodeError:
                    # If a line is not valid JSON, log a warning and continue to the next.
                    # This prevents the entire script from failing on a single corrupt line.
                    print(f"Warning: Skipping malformed JSON line {processed_count}: {line}", file=sys.stderr)
                    continue

            print("\nProcessing complete.")
            print(f"Total lines read: {processed_count}")
            print(f"Records written to '{output_path}': {written_count}")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)