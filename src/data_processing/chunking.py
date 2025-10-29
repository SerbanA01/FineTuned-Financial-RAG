import os
import glob
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CHUNKING CONFIGURATION ---

# Define the target size for each text chunk. This is a crucial parameter for RAG systems,
# as it determines the amount of context provided to the LLM for answering questions.
# A size of 1000 characters is a common starting point.
CHUNK_SIZE = 1000

# Define the overlap between consecutive chunks. An overlap ensures that semantic context
# is not lost at the boundary of a split, helping to maintain coherence.
CHUNK_OVERLAP = 150

# Instantiate a text splitter specifically for SEC filings.
# The RecursiveCharacterTextSplitter is robust as it tries to split on a sequence of
# separators (like "\n\n", "\n", " ") to keep related text together.
sec_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = CHUNK_SIZE,
    chunk_overlap = CHUNK_OVERLAP,
    length_function = len # Use the standard character length function.
)

# Instantiate a separate text splitter for earnings call transcripts.
# Transcripts may have different structural properties than SEC filings (e.g., shorter
# paragraphs), so using a slightly different configuration (smaller chunk size)
# can be beneficial for preserving the conversational flow.
earnings_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    length_function=len,
)

# A global counter to assign a unique, sequential ID to every chunk created
# across all files. This is useful for downstream processes that may require a
# stable and unique identifier for each piece of text.
global_chunk_id_counter = 0

def process_sec_file(filepath, filename_only, text_splitter, output_file_handle, filing_category):
    """
    Reads a partitioned SEC filing JSON, splits its sections into chunks, and writes them to a JSONL file.

    Each chunk is enriched with extensive metadata from the source file, such as the
    ticker, year, filing type, and the specific section (e.g., "Item 1A. Risk Factors")
    from which the chunk originated.

    @param filepath: The full path to the input JSON file.
    @param filename_only: The base name of the input file.
    @param text_splitter: An initialized text splitter instance.
    @param output_file_handle: An open file handle for writing the output JSONL data.
    @param filing_category: The general category of the filing ('10k' or '10q').
    """
    # We need to modify the global counter from within this function.
    global global_chunk_id_counter
    print(f"Processing SEC ({filing_category}) file: {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract document-level metadata. This applies to all chunks from this file.
        global_metadata = data.get("metadata", {})
        ticker = global_metadata.get("ticker", "UNKNOWN_TICKER")
        year = global_metadata.get("year", "UNKNOWN_YEAR")
        cik = global_metadata.get("cik", "UNKNOWN_CIK")
        date = global_metadata.get("date", "UNKNOWN_DATE")
        filing_type_from_meta = global_metadata.get("filing_type", filing_category.upper())

        # Iterate through each pre-partitioned section of the filing.
        for section_data in data.get("sections", []):
            section_text = section_data.get("text", "")
            if not section_text:
                continue

            # Extract section-specific metadata.
            item = section_data.get("item", "UNKNOWN_ITEM")
            section_name_from_json = section_data.get("section", "UNKNOWN_SECTION")
            
            # Split the text of the current section into smaller chunks.
            chunks = text_splitter.split_text(section_text)

            # Create a JSON object for each chunk, combining global and section metadata.
            for i, chunk_content in enumerate(chunks):
                chunk_metadata = {
                    "source_type": "sec_filing",
                    "filing_category": filing_category,
                    "original_file_name": filename_only,
                    "ticker": ticker,
                    "year": year,
                    "cik": cik,
                    "date": date,
                    "filing_type": filing_type_from_meta,
                    "item": item,
                    "section_name": section_name_from_json,
                    "chunk_sequence_in_section": i, # Tracks the order of chunks within a section.
                    "global_chunk_id": f"sec_{global_chunk_id_counter}"
                }
                global_chunk_id_counter += 1
                
                # The final object contains the text and its rich metadata.
                chunk_json_object = {"text": chunk_content, "metadata": chunk_metadata}
                
                # Write the object as a single line in the output file (JSONL format).
                # JSONL is efficient for streaming and processing large datasets.
                output_file_handle.write(json.dumps(chunk_json_object) + "\n")

    except json.JSONDecodeError:
        print(f"Error decoding JSON from SEC file: {filepath}")
    except Exception as e:
        print(f"An error occurred while processing SEC file {filepath}: {e}")


def process_earnings_file(filepath, filename_only, text_splitter, output_file_handle):
    """
    Reads a processed earnings transcript, splits speaker turns into chunks, and writes them to a JSONL file.

    This function is tailored to the structure of the transcript data, chunking the text
    within each speaker's turn and enriching the chunks with metadata like the speaker's
    name, the section of the call (e.g., Q&A), and company information.

    @param filepath: The full path to the input JSON file.
    @param filename_only: The base name of the input file.
    @param text_splitter: An initialized text splitter instance.
    @param output_file_handle: An open file handle for writing the output JSONL data.
    """
    global global_chunk_id_counter
    print(f"Processing Earning Transcript file: {filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            earnings_data = json.load(f)

        # Extract document-level metadata from the transcript file.
        doc_meta = earnings_data.get("document_metadata", {})
        ticker = doc_meta.get("ticker", "UNKNOWN_TICKER")
        year = doc_meta.get("year", "UNKNOWN_YEAR")
        quarter = doc_meta.get("quarter", "UNKNOWN_QUARTER")
        original_date = doc_meta.get("original_date_col_value", "UNKNOWN_DATE")
        company_name = doc_meta.get("company_name", "UNKNOWN_COMPANY")
        exchange = doc_meta.get("exchange", "UNKNOWN_EXCHANGE")

        # The primary unit of content in a transcript is a "speaker turn".
        speaker_turns = earnings_data.get("speaker_turns", [])
        if not speaker_turns:
            print(f"  No speaker turns found in {filename_only}. Skipping.")
            return

        # Iterate through each speaker's turn in the transcript.
        for turn_index, turn_data in enumerate(speaker_turns):
            turn_text = turn_data.get("text", "")
            if not turn_text:
                continue

            # Extract metadata specific to this turn.
            turn_id = turn_data.get("turn_id", f"turn_{turn_index}")
            speaker_full = turn_data.get("speaker_full", "UNKNOWN_SPEAKER")
            speaker_simple = turn_data.get("speaker_simple_name", "UNKNOWN_SPEAKER")
            turn_section = turn_data.get("section", "UNKNOWN_SECTION")

            # Split the text of the current speaker's turn into chunks.
            chunks = text_splitter.split_text(turn_text)

            for chunk_index, chunk_content in enumerate(chunks):
                chunk_metadata = {
                    "source_type": "earnings_transcript",
                    "original_file_name": filename_only,
                    "ticker": ticker,
                    "year": year,
                    "quarter": quarter,
                    "date": original_date,
                    "company_name": company_name,
                    "exchange": exchange,
                    "turn_id": turn_id, # Link back to the specific turn.
                    "turn_speaker_full": speaker_full,
                    "turn_speaker_simple": speaker_simple,
                    "turn_section": turn_section, # e.g., "Prepared Remarks" or "Q&A"
                    "chunk_sequence_in_turn": chunk_index,
                    "global_chunk_id": f"earn_{global_chunk_id_counter}"
                }
                global_chunk_id_counter += 1

                chunk_json_object = {"text": chunk_content, "metadata": chunk_metadata}
                output_file_handle.write(json.dumps(chunk_json_object) + "\n")

    except json.JSONDecodeError:
        print(f"Error decoding JSON from earnings file: {filepath}")
    except FileNotFoundError:
        print(f"Error: Earnings file not found: {filepath}")
    except Exception as e:
        print(f"An error occurred while processing earnings file {filepath}: {e}")