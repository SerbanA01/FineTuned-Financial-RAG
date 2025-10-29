import json

# This script is designed to perform the final standardization step on a dataset,
# ensuring that all records adhere to a consistent schema and that values have the
# correct data types. This is a crucial pre-processing step before using the
# data for model fine-tuning, as models often expect strictly typed and consistent inputs.

def cast_value(key, value):
    """
    Casts a value to a specific data type based on its associated key.

    This function acts as a type-enforcement utility. It handles common cases like
    converting year values to integers and ensures that other specific fields are
    strings. It also gracefully handles null or empty values.

    @param key: The key of the value, which determines the target data type.
    @param value: The value to be cast.
    @return: The value cast to its appropriate type, or None if the value is missing or invalid.
    """
    # Treat empty strings and None as null.
    if value is None or value == "":
        return None

    # Enforce integer type for specific keys.
    if key in ['year', 'source_id']:
        try:
            return int(value)
        except (ValueError, TypeError):
            # If casting fails (e.g., trying to cast "N/A" to int), return None.
            return None
            
    # Enforce string type for a set of known textual fields.
    if key in ['ticker', 'quarter', 'raw_quarter', 'normalized_doc_type', 
               'exchange', 'date', 'time', 'company_name']:
        return str(value)
        
    # If the key is not in any of the specific casting rules, return the value as-is.
    return value

def standardize_fincap_data(input_file_path, output_file_path):
    """
    Reads a JSONL file, standardizes each record, and writes to a new JSONL file.

    The standardization process involves:
    1. Ensuring every record has a consistent top-level structure.
    2. Handling variations in the `metadata.focuses` field (can be a list or a single dict).
    3. Forcing each `focus` object to have a complete set of possible keys, with null
       values for any missing keys.
    4. Applying explicit data type casting to key fields (e.g., 'year' becomes an int).

    @param input_file_path: The path to the source JSONL file.
    @param output_file_path: The path where the standardized JSONL file will be saved.
    """
    print(f"Standardizing data types in '{input_file_path}'...")
    
    # This list defines the complete, canonical set of keys that every 'focus' object should have.
    all_possible_keys = [
        'ticker', 'year', 'quarter', 'raw_quarter', 'normalized_doc_type',
        'source_id', 'exchange', 'date', 'time', 'company_name'
    ]
    
    # Load the entire dataset into memory. This is suitable for moderately sized files.
    original_data = []
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            original_data.append(json.loads(line))

    standardized_data = []
    for record in original_data:
        # Skip any records that are not dictionaries or lack the required 'metadata' key.
        if not isinstance(record, dict) or "metadata" not in record:
            continue

        # Start building a new, clean record with a guaranteed structure.
        new_record = {
            "user_query": record.get("user_query"),
            "final_answer": record.get("final_answer"),
            "metadata": {"focuses": []}
        }

        metadata = record["metadata"]
        focus_list = []

        # The 'focuses' field can sometimes be a single object instead of a list.
        # This logic handles both cases gracefully, ensuring `focus_list` is always a list.
        if "focuses" in metadata:
            if isinstance(metadata["focuses"], list):
                focus_list = metadata["focuses"]
            elif isinstance(metadata["focuses"], dict):
                focus_list = [metadata["focuses"]]
        else:
            # Handle older formats where the metadata object itself was the focus item.
            focus_list = [metadata]

        for focus_item in focus_list:
            if not isinstance(focus_item, dict):
                continue
            
            # Create a new dictionary for the focus item, ensuring all possible keys are present.
            standardized_focus = {}
            for key in all_possible_keys:
                original_value = focus_item.get(key)
                # Use the cast_value function to enforce the correct data type.
                standardized_focus[key] = cast_value(key, original_value)
            
            new_record["metadata"]["focuses"].append(standardized_focus)

        standardized_data.append(new_record)

    # Write the list of standardized records back to a new JSONL file.
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in standardized_data:
            f.write(json.dumps(item) + '\n')
            
    print(f"Successfully processed {len(standardized_data)} records with type casting.")
    print(f"Type-consistent data saved to: {output_file_path}")