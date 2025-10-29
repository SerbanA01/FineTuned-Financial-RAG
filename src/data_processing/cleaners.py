import re
import json
import logging
from pathlib import Path

import ftfy
from nltk.tokenize import sent_tokenize
from dateutil import parser

# A list of compiled regular expression patterns to identify and remove common
# boilerplate lines found in SEC filings. This includes tables of contents,
# page numbers, and document headers that are not part of the core narrative.
HEADER_FOOTER_PATTERNS = [
    re.compile(r'^\s*TABLE OF CONTENTS\s*$',                 re.I),
    re.compile(r'^\s*INDEX TO FINANCIAL STATEMENTS\s*$',     re.I),
    # Matches lines that are just a page number (e.g., "F-15") or a simple integer.
    re.compile(r'^\s*F-\d+\s*$|^\s*\d+\s*$',                 re.I),
    # Matches document type headers, like "10-K".
    re.compile(r'^\s*(10-K|10-Q)\b.*$',                      re.I),
    # Matches lines that appear to be table of contents entries.
    re.compile(r'^\s*Item\s+\d+[A-Z]?\.?.*\s+\d+\s*$',       re.I),
    # Example of a company-specific header to be removed.
    re.compile(r'^\s*THE ORIGINAL BARK COMPANY.*$',          re.I),
]

def strip_headers_footers(text: str) -> str:
    """
    Removes lines from the text that match any of the predefined header/footer patterns.

    @param text: The input text as a single string.
    @return: The text with header and footer lines removed.
    """
    lines = []
    for ln in text.splitlines():
        # A line is kept only if it does not match any of the patterns.
        if any(pat.match(ln) for pat in HEADER_FOOTER_PATTERNS):
            continue
        lines.append(ln)
    return "\n".join(lines)

def clean_text(text: str) -> str:
    """
    Performs comprehensive cleaning on a block of text.

    This function applies multiple cleaning steps:
    1. Fixes common text encoding issues using `ftfy`.
    2. Normalizes whitespace and punctuation.
    3. Splits the text into sentences.
    4. Deduplicates sentences to remove redundant content.
    5. Joins the cleaned, unique sentences back into a single text block.

    @param text: The raw text block to clean.
    @return: The cleaned and deduplicated text.
    """
    # Use ftfy to fix text encoding issues (e.g., Mojibake) and replace common problematic characters.
    text = ftfy.fix_text(text.replace('�', "'").replace('’', "'"))
    # Add a space between a lowercase letter and an uppercase letter (CamelCase splitting)
    # to fix words that were erroneously joined during text extraction.
    text = re.sub(r'(?<=[a-z])(?=[A-Z][a-z])', ' ', text)
    # Normalize all whitespace (newlines, tabs, etc.) into single spaces.
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    
    # Segment the cleaned text into sentences. Use NLTK if available, otherwise a regex fallback.
    sents = sent_tokenize(text) if sent_tokenize else re.split(r'(?<=[.!?])\s+(?=[A-Z(])', text)

    # Deduplicate sentences to remove repetitive boilerplate that might have survived previous cleaning steps.
    dedup, seen = [], set()
    for s in sents:
        # Normalize the sentence for case-insensitive comparison.
        norm = re.sub(r'\s+', ' ', s).strip().lower()
        if norm and norm not in seen:
            dedup.append(s.strip())
            seen.add(norm)

    return ' '.join(dedup)

def process_file(path: Path, dest_root: Path):
    """
    Orchestrates the cleaning process for a single raw SEC filing JSON.

    This function reads a JSON file containing text chunks from an SEC filing,
    joins them, intelligently trims the content to the start of "Item 1.",
    applies multiple cleaning functions, extracts key metadata, and saves the
    processed result to a new JSON file.

    @param path: The `pathlib.Path` to the input JSON file.
    @param dest_root: The root directory for the processed output files.
    """
    # Load the raw data from the input JSON file.
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = data.get("text_chunks", [])
    
    # For 10-K filings, the first chunk is often a cover page or TOC; skipping it
    # can improve the quality of the joined text.
    if data.get("filing_type") == "10-K":
        chunks = chunks[1:]

    # Join the raw text chunks into a single document, deduplicating chunks as we go.
    # Some extraction methods can produce many identical chunks (e.g., from headers/footers).
    joined, seen = [], set()
    for ch in chunks:
        # Ignore empty chunks or those containing common XML schema identifiers.
        if not ch or "us-gaap" in ch.lower():
            continue
        # Normalize for case-insensitive duplicate checking.
        norm = re.sub(r'\s+', ' ', ch).strip().lower()
        if norm in seen:
            continue
        seen.add(norm)
        joined.append(ch.strip())

    full_text = ' '.join(joined)

    # Find the start of the main content, which is reliably marked by "Item 1.".
    # This helps trim any preceding boilerplate or tables of contents.
    m = re.search(
        r'''(?is)      # (?i) for case-insensitivity, (?s) for dotall
            Item        # Literal "Item"
            \s*1        # Whitespace and the number 1
            \s*[\.\u2022] # Whitespace and a period or bullet
            (?!\d)      # Negative lookahead to ensure it's not "Item 10", "Item 11", etc.
        ''',
        full_text,
        re.VERBOSE  # Allows for comments and clean layout in the regex pattern.
        )
    if m:
        full_text = full_text[m.start():]
    else:
        # This warning is important for debugging parsing issues with specific files.
        print(f"Warning: 'Item 1.' not found in {path.name}")

    # Apply the main cleaning functions to the isolated content.
    full_text = strip_headers_footers(full_text)
    full_text = clean_text(full_text)

    # A final quality filter: if the cleaned text is very short, it's likely not useful.
    MIN_WORDS = 50
    if len(full_text.split()) < MIN_WORDS:
        logging.info("Dropped %s (only %d words)", path, len(full_text.split()))
        return

    # --- Metadata Extraction and Standardization ---
    date_str = data.get("date")
    try:
        # Parse the filing date string to extract just the year.
        year_of_filing = parser.parse(date_str).year if date_str else None
    except Exception as e:
        print(f"Could not parse year from {date_str}: {e}")
        year_of_filing = None

    # Extract the CIK (Central Index Key) from the source URL for consistent identification.
    source_url   = data.get("url")
    cik_match    = re.search(r"/data/(\d+)/", source_url or "")
    extracted_cik = cik_match.group(1).zfill(10) if cik_match else None

    # Assemble the final, cleaned data object.
    out = {
        "cleaned_text": full_text,
        "ticker":      data.get("ticker"),
        "date":        date_str,
        "year":        year_of_filing,
        "cik":         extracted_cik,
        "filing_type": data.get("filing_type"),
    }
    
    # This line assumes a global variable `RAW_ROOT` is defined elsewhere,
    # which is necessary to compute the correct relative output path.
    # As per the directive, the original logic is preserved.
    out_path = dest_root / path.relative_to(RAW_ROOT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")