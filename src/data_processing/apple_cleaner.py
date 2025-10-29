import re
import json
import logging
from pathlib import Path

import ftfy
from dateutil import parser

# --- Configuration ---
# Set up basic logging to monitor the script's execution.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Regular Expressions for Cleaning ---
# A list of compiled regex patterns to identify and remove common headers and footers
# found in Apple's filings, such as "TABLE OF CONTENTS" or page number lines.
# These are often OCR artifacts or navigational aids not part of the narrative content.
HEADER_FOOTER_PATTERNS_APPLE = [
    re.compile(r'^\s*TABLE OF CONTENTS\s*$', re.IGNORECASE),
    re.compile(r'^\s*INDEX TO FINANCIAL STATEMENTS\s*$', re.IGNORECASE),
    re.compile(r'^\s*Apple Inc\.\s*\|\s*(Q\d\s+\d{4}|FY\d{4})?\s*Form (10-K|10-Q)\s*\|\s*\d+\s*$', re.IGNORECASE),
    # Matches lines that are just a page number (e.g., "F-15") or a simple integer.
    re.compile(r'^\s*F-\d+\s*$|^\s*\d+\s*$', re.IGNORECASE),
]

# Regex to find the *last* occurrence of "PART I Item 1.". This is crucial for 10-K filings,
# as earlier occurrences might be in the table of contents. The actual content starts later.
START_PATTERN_10K_LAST = re.compile(
    r"""
    \bPART\s+I\s*[,]?\s*Item\s+1\s*[\.\u2022] # Matches "PART I Item 1" with flexible punctuation.
    """,
    re.IGNORECASE | re.VERBOSE # VERBOSE allows for comments and cleaner layout.
)

# Regex to find the *first* occurrence of "Item 1.". This is suitable for 10-Q filings
# and serves as a fallback for 10-K if the "PART I" pattern isn't found.
START_PATTERN_10Q_FIRST = re.compile(
    r"""
    (?: \bPART\s+I\s*[,]?\s*)? # Optionally matches "PART I".
    Item\s+1\s*[\.\u2022]
    """,
    re.IGNORECASE | re.VERBOSE
)

# --- Text Processing Functions ---

def strip_specific_headers_footers(text: str) -> str:
    """
    Removes lines from the text that match the defined header/footer patterns.

    @param text: The input text as a single string.
    @return: The text with header and footer lines removed.
    """
    lines = []
    for ln in text.splitlines():
        # A line is kept only if it does not match any of the patterns.
        if any(pat.match(ln) for pat in HEADER_FOOTER_PATTERNS_APPLE):
            continue
        lines.append(ln)
    return "\n".join(lines)

def clean_narrative_text_apple(text: str) -> str:
    """
    Performs fine-grained cleaning on the narrative text of an Apple filing.

    This function is designed to handle common OCR and text extraction artifacts, such as
    mangled punctuation, misplaced capitalization, and excessive whitespace.

    @param text: The narrative text block.
    @return: The cleaned text.
    """
    if not text:
        return ""
    # Use ftfy to fix text encoding issues and replace common problematic characters.
    text = ftfy.fix_text(text.replace('�', "'").replace('’', "'"))
    # Add a space between a lowercase letter and an uppercase letter (CamelCase splitting).
    # This fixes words that were erroneously joined during text extraction.
    text = re.sub(r'(?<=[a-z])(?=[A-Z][a-z])', ' ', text)
    # Normalize all whitespace (newlines, tabs, etc.) into single spaces.
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text

def find_last_match_start(pattern, text):
    """
    Finds the starting index of the *last* match of a regex pattern in a string.

    This is necessary because `re.search` only finds the first match. Iterating
    through all matches is required to locate the final one.

    @param pattern: A compiled regular expression object.
    @param text: The string to search within.
    @return: The starting index of the last match, or -1 if no match is found.
    """
    last_match = None
    for match in pattern.finditer(text):
        last_match = match
    return last_match.start() if last_match else -1

def process_file_apple_v2(path: Path, dest_root: Path, raw_root: Path):
    """
    Main processing function for a single raw Apple filing JSON.

    This function orchestrates the entire cleaning pipeline for a given file:
    1. Reads and parses the raw JSON.
    2. Joins text chunks into a single document.
    3. Intelligently finds the true start of the narrative content.
    4. Applies several layers of cleaning and normalization.
    5. Extracts and standardizes metadata.
    6. Saves the cleaned data to a new JSON file in the destination directory.

    @param path: The `pathlib.Path` to the input JSON file.
    @param dest_root: The root directory for the processed output files.
    @param raw_root: The root directory of the raw input files, used for preserving relative structure.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {path}: {e}")
        return
    except Exception as e:
        logging.error(f"Error reading file {path}: {e}")
        return

    # Extract key fields from the raw JSON.
    chunks = data.get("text_chunks", [])
    filing_type = data.get("filing_type")
    ticker = data.get("ticker")

    # --- Initial Validation ---
    if not chunks:
        logging.warning(f"No text_chunks found in {path.name}")
        return
    if not filing_type:
        logging.warning(f"No filing_type found in {path.name}, skipping.")
        return
    # This cleaner is specifically tailored for Apple, so we skip other tickers.
    if ticker != "AAPL":
        logging.info(f"Skipping non-AAPL file: {path.name} (ticker: {ticker})")
        return

    # In 10-K filings, the first chunk is often a cover page or TOC; skipping it can improve quality.
    start_index = 1 if filing_type == "10-K" else 0
    
    full_doc_text_from_chunks = ' '.join(chunks[start_index:])
    if not full_doc_text_from_chunks.strip():
        logging.warning(f"No text after initial chunk joining for {path.name}")
        return

    # --- Find the Start of the Main Content ---
    content_start_index = -1
    if filing_type == "10-K":
        content_start_index = find_last_match_start(START_PATTERN_10K_LAST, full_doc_text_from_chunks)
        # If the preferred 10-K start pattern fails, try the more general 10-Q pattern as a fallback.
        if content_start_index == -1:
            logging.warning(f"10-K: 'PART I Item 1.' not found in {path.name}. Trying 'Item 1.'")
            match_item1 = START_PATTERN_10Q_FIRST.search(full_doc_text_from_chunks)
            if match_item1:
                content_start_index = match_item1.start()
    elif filing_type == "10-Q":
        match = START_PATTERN_10Q_FIRST.search(full_doc_text_from_chunks)
        if match:
            content_start_index = match.start()
    
    # Isolate the main content text based on the found start index.
    if content_start_index == -1:
        logging.warning(f"Could not find suitable start pattern for {path.name} ({filing_type}). Processing from beginning of joined chunks (after initial 10-K skip).")
        main_content_text = full_doc_text_from_chunks
    else:
        main_content_text = full_doc_text_from_chunks[content_start_index:]
        logging.info(f"Extracted content from index {content_start_index} for {path.name}")

    if not main_content_text.strip():
        logging.warning(f"No text after attempting to find start of content for {path.name}")
        return

    # --- Apply Cleaning Functions ---
    cleaned_text = ftfy.fix_text(main_content_text)
    cleaned_text = strip_specific_headers_footers(cleaned_text)
    cleaned_text = clean_narrative_text_apple(cleaned_text)

    # --- Final Quality Check ---
    MIN_WORDS = 50
    word_count = len(cleaned_text.split())

    if word_count < MIN_WORDS:
        logging.warning(f"Dropped {path.name} ({filing_type}) as it has too few words ({word_count}) after cleaning. Original chunks had text.")
        return

    # --- Metadata Extraction and Standardization ---
    date_str = data.get("date")
    try:
        # Parse the filing date to extract just the year.
        year_of_filing = parser.parse(date_str).year if date_str else None
    except Exception as e:
        logging.warning(f"Could not parse year from {date_str} for {path.name}: {e}")
        year_of_filing = None

    # Extract the CIK from the source URL for consistent metadata.
    source_url = data.get("url")
    cik_match = re.search(r"/data/(\d+)/", source_url or "")
    extracted_cik = cik_match.group(1).zfill(10) if cik_match else None

    # --- Assemble and Save Output ---
    output_data = {
        "cleaned_text": cleaned_text,
        "ticker": ticker,
        "date": date_str,
        "year": year_of_filing,
        "cik": extracted_cik,
        "filing_type": filing_type,
        "source_filename": path.name,
    }

    # Construct the output path, preserving the original directory structure (e.g., /AAPL/10-K/).
    out_path = dest_root / path.relative_to(raw_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info(f"Successfully processed and saved: {out_path.name} (words: {word_count})")
    except Exception as e:
        logging.error(f"Error writing output file {out_path}: {e}")