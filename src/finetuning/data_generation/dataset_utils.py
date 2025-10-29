import json
import random

# This module provides a suite of utility functions for cleaning, validating,
# and manipulating JSONL (JSON Lines) files, which are a common format for storing
# structured data, especially in machine learning pipelines.

def find_and_parse_json_chunks(text: str):
    """
    Finds and parses all valid JSON objects or arrays embedded within a larger string.

    This generator function is designed to be robust against malformed text files
    where valid JSON might be mixed with logging output, notes, or other non-JSON
    text. It carefully balances braces and brackets to identify the boundaries of
    potential JSON structures and attempts to parse them.

    @param text: A string that may contain one or more JSON objects/arrays.
    @yield: A parsed Python dictionary or list for each valid JSON structure found.
    """
    pos = 0
    while pos < len(text):
        # Find the next opening brace or bracket.
        start_brace = text.find('{', pos)
        start_bracket = text.find('[', pos)

        # If neither is found, we're done.
        if start_brace == -1 and start_bracket == -1:
            break

        # Determine which comes first to set the start of our potential JSON object.
        if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
            start_index = start_brace
            open_char, close_char = '{', '}'
        else:
            start_index = start_bracket
            open_char, close_char = '[', ']'

        # Scan forward, counting nested levels to find the matching closing character.
        level = 1
        end_index = -1
        for i in range(start_index + 1, len(text)):
            char = text[i]
            if char == open_char:
                level += 1
            elif char == close_char:
                level -= 1
                if level == 0:
                    end_index = i
                    break
        
        # If a complete, balanced structure was found...
        if end_index != -1:
            chunk_str = text[start_index : end_index + 1]
            try:
                # ...attempt to parse it as JSON and yield the result.
                yield json.loads(chunk_str)
            except json.JSONDecodeError:
                # If parsing fails, silently ignore the chunk and continue scanning.
                pass
            # Continue scanning from after the end of this chunk.
            pos = end_index + 1
        else:
            # If no closing character was found, advance past the opening one to avoid an infinite loop.
            pos = start_index + 1

def convert_to_jsonl(input_file_path, output_file_path):
    """
    Converts a text file containing noisy, mixed JSON into a clean JSONL file.

    This function reads an entire file, uses `find_and_parse_json_chunks` to
    extract all valid JSON, and writes each resulting object as a separate line
    in the output file, adhering to the JSONL format.

    @param input_file_path: Path to the messy input text file.
    @param output_file_path: Path where the clean JSONL file will be saved.
    """
    print(f"Starting robust conversion from '{input_file_path}' to '{output_file_path}'...")
    
    record_count = 0
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f_in:
            content = f_in.read()

        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            # Iterate through the parsed JSON structures from the input file.
            for parsed_data in find_and_parse_json_chunks(content):
                # If the parsed data is a list, iterate through its items.
                if isinstance(parsed_data, list):
                    for item in parsed_data:
                        if isinstance(item, dict):
                            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                            record_count += 1
                # If it's a single dictionary, write it directly.
                elif isinstance(parsed_data, dict):
                    f_out.write(json.dumps(parsed_data, ensure_ascii=False) + '\n')
                    record_count += 1
        
        print(f"Successfully created '{output_file_path}' with {record_count} records.")

    except FileNotFoundError:
        print(f"Error: The input file was not found at '{input_file_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def shuffle_jsonl(input_file_path, output_file_path):
    """
    Randomly shuffles the lines of a JSONL file.

    This is a crucial step in preparing data for machine learning to prevent the model
    from learning any spurious patterns related to the original order of the data.

    @param input_file_path: Path to the JSONL file to be shuffled.
    @param output_file_path: Path to save the shuffled JSONL file.
    """
    print(f"Shuffling lines from '{input_file_path}'...")
    # This reads the entire file into memory, so it's best for moderately sized files.
    with open(input_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    random.shuffle(lines)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"Successfully shuffled and saved to '{output_file_path}'.")


def validate_year_field(file_path):
    """
    Scans a JSONL file to perform a specific data quality check: finding records
    where the 'year' field is a string instead of an integer.

    This is a diagnostic tool to identify data type inconsistencies that might
    cause issues in downstream processing or model training.

    @param file_path: The path to the JSONL file to scan.
    """
    problematic_records = []
    print(f"Scanning file for string-based 'year' fields: {file_path}\n")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    
                    # This logic handles two possible schemas for where 'focuses' might be located.
                    if 'metadata_extraction' in data:
                        focuses = data.get('metadata_extraction', {}).get('focuses')
                    elif 'metadata' in data:
                        focuses = data.get('metadata', {}).get('focuses')
                    else:
                        focuses = None

                    if isinstance(focuses, list):
                        for focus_item in focuses:
                            if 'year' in focus_item:
                                year_value = focus_item.get('year')
                                # The core validation check.
                                if isinstance(year_value, str):
                                    problematic_records.append({
                                        "line_number": i,
                                        "year_value": year_value,
                                        "record_content": line.strip()
                                    })
                                    break # Move to the next line once a problem is found in this record.
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON on line {i}. Skipping.")
                except Exception as e:
                    print(f"An unexpected error occurred on line {i}: {e}")
        
        if problematic_records:
            print(f"SUCCESS: Found {len(problematic_records)} records where 'year' is a string.")
            for record in problematic_records:
                print(f"  - Problem on Line {record['line_number']}: 'year' is \"{record['year_value']}\"")
        else:
            print("Scan complete. No records found where 'year' is a string.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")

def validate_jsonl_file(file_path):
    """
    Checks if a file is a valid JSONL file, meaning each line is a valid JSON object.

    This is a fundamental sanity check to ensure a data file is well-formed before
    attempting to process it.

    @param file_path: The path to the file to validate.
    """
    print(f"Validating format of '{file_path}'...")
    problematic_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                # The test is simply trying to load each line as JSON.
                json.loads(line)
            except json.JSONDecodeError as e:
                # If it fails, record the line number, error, and content.
                problematic_lines.append((i, str(e), line.strip()))

    if problematic_lines:
        print(f"Found {len(problematic_lines)} problematic lines:")
        for line_no, error, content in problematic_lines:
            print(f"Line {line_no}: {error}\nContent: {content}\n")
    else:
        print(f"'{file_path}' is a valid JSONL file.")