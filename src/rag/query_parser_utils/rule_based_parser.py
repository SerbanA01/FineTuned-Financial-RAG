import re
from typing import List, Tuple, Set, Dict, Any, Optional

from .schemas import DocType, EntityType, QueryFocus
from .constants import (
    COMPANY_TO_TICKER_HINTS,
    TICKER_TO_COMPANY_HINTS,
    TICKER_REGEX,
    YEAR_REGEX,
    QUARTER_REGEX,
    QUARTER_WORD_TO_NUM,
    DOC_KEYWORD_TO_NORMALIZED_MAP,
    DOC_TYPE_KEYWORDS_REGEX,
    ENTITY_PRIORITIES,
)


def _extract_entities_from_segment_text(segment_text: str) -> List[Dict[str, Any]]:
    """
    Scans a given text segment and extracts all potential financial entities.

    This is a low-level helper function that uses regex and dictionary lookups
    to find mentions of tickers, company names, years, quarters, and document
    type keywords. Each found entity is returned as a dictionary containing its
    type, value, position, and priority.

    @param segment_text: The string to be scanned for entities.
    @return: A list of dictionaries, each representing a potential entity.
    """
    potential_entities: List[Dict[str, Any]] = []
    # --- Tickers, Companies, Years, Quarters ---

    # 1. Extract Tickers: Use a regex to find capitalized, ticker-like strings.
    for match in TICKER_REGEX.finditer(segment_text):
        ticker_candidate = match.group(1)
        # Validate against our known list to reduce false positives.
        if ticker_candidate in TICKER_TO_COMPANY_HINTS:
            potential_entities.append({
                "text": ticker_candidate, "value": ticker_candidate, "type": EntityType.TICKER,
                "start": match.start(1), "end": match.end(1), "priority": ENTITY_PRIORITIES[EntityType.TICKER]
            })

    # 2. Extract Company Names: Iterate through known company names.
    # Sorting by length ensures we match longer names first (e.g., "johnson & johnson" before "johnson").
    for cn_key in sorted(COMPANY_TO_TICKER_HINTS.keys(), key=len, reverse=True):
        ticker = COMPANY_TO_TICKER_HINTS[cn_key]
        pattern = r'\b' + re.escape(cn_key) + r'\b' # Match whole words only.
        for cmatch in re.finditer(pattern, segment_text, re.IGNORECASE):
            potential_entities.append({
                "text": cmatch.group(0), "value": ticker, "type": EntityType.COMPANY,
                "start": cmatch.start(0), "end": cmatch.end(0), "priority": ENTITY_PRIORITIES[EntityType.COMPANY]
            })

    # 3. Extract Years: Use regex to find 4-digit numbers in a plausible range.
    for ymatch in YEAR_REGEX.finditer(segment_text):
        year_str = ymatch.group(1)
        potential_entities.append({
            "text": year_str, "value": int(year_str), "type": EntityType.YEAR,
            "start": ymatch.start(1), "end": ymatch.end(1), "priority": ENTITY_PRIORITIES[EntityType.YEAR]
        })

    # 4. Extract Quarters: Use regex to find various quarter formats (e.g., Q1, 2nd Quarter).
    for qmatch in QUARTER_REGEX.finditer(segment_text):
        # The regex has multiple capture groups for different formats.
        q_val_str = qmatch.group(1) or qmatch.group(2) or qmatch.group(3) or qmatch.group(4)
        q_num = None
        if q_val_str:
            # Normalize to a numeric value (1-4).
            q_num = int(q_val_str) if q_val_str.isdigit() else QUARTER_WORD_TO_NUM.get(q_val_str.lower())
        if q_num:
            potential_entities.append({
                "text": qmatch.group(0), "value": q_num, "type": EntityType.QUARTER,
                "start": qmatch.start(0), "end": qmatch.end(0), "priority": ENTITY_PRIORITIES[EntityType.QUARTER]
            })

    # 5. Extract Document Type Keywords: Use a combined regex for all known document keywords.
    for dmatch in DOC_TYPE_KEYWORDS_REGEX.finditer(segment_text):
        keyword_found = dmatch.group(0).lower()
        normalized_doc_type = DOC_KEYWORD_TO_NORMALIZED_MAP.get(keyword_found)
        if normalized_doc_type: # Should always be true due to regex construction.
            potential_entities.append({
                "text": dmatch.group(0), "value": normalized_doc_type,
                "type": EntityType.DOC_TYPE_KEYWORD,
                "start": dmatch.start(0), "end": dmatch.end(0), "priority": ENTITY_PRIORITIES[EntityType.DOC_TYPE_KEYWORD]
            })
    return potential_entities

# Regex for splitting the query into sentences.
SENTENCE_DELIMITERS_REGEX = re.compile(r'[.?!]')
# Regex for splitting sentences into smaller, context-rich clauses.
CLAUSE_DELIMITERS_REGEX = re.compile(r'\s+(?:and|or|but)\s+|,', re.IGNORECASE)

def extract_structured_metadata(query: str) -> Tuple[List[QueryFocus], str]:
    """
    Performs rule-based extraction on a user query to identify distinct financial data requests.

    The strategy involves several steps:
    1.  A global pass extracts all entities to establish "defaults" (e.g., if only one
        year is mentioned, it's assumed for all requests lacking a specific year).
    2.  The query is segmented into sentences and clauses to isolate contexts.
    3.  Each segment is processed to find stocks and associate them with the nearest
        year, quarter, and document type found within that same segment.
    4.  Global defaults are applied if a segment lacks a specific time component.
    5.  A "cleaned" version of the query is produced by removing the text of all
        extracted entities, which can be useful for downstream keyword analysis.

    @param query: The raw user query string.
    @return: A tuple containing:
             - A list of structured QueryFocus objects, representing each identified request.
             - A modified query string with all entity text removed.
    """
    all_query_focuses_set: Set[QueryFocus] = set()

    # --- Global Analysis: Find all entities and establish default context ---
    glob_potential_entities = _extract_entities_from_segment_text(query)
    # Sort by start position, then priority, then by length (longest match first) for tie-breaking.
    glob_potential_entities.sort(key=lambda x: (x["start"], x["priority"], -(x["end"] - x["start"])))

    # Filter out overlapping entities, keeping the one with the highest priority.
    # This prevents "MSFT" (company) and "MSFT" (ticker) from both being selected for the same text.
    all_extracted_entities_globally: List[Dict[str, Any]] = []
    _last_covered_idx = -1
    for entity in glob_potential_entities:
        if entity["start"] >= _last_covered_idx:
            all_extracted_entities_globally.append(entity)
            _last_covered_idx = entity["end"]

    # If only one unique year or quarter is mentioned anywhere, it becomes the default.
    global_years_map: Dict[int, int] = {e["value"]: global_years_map.get(e["value"], 0) + 1 for e in all_extracted_entities_globally if e["type"] == EntityType.YEAR}
    global_quarters_map: Dict[int, int] = {e["value"]: global_quarters_map.get(e["value"], 0) + 1 for e in all_extracted_entities_globally if e["type"] == EntityType.QUARTER}

    default_year = list(global_years_map.keys())[0] if len(global_years_map) == 1 else None
    default_raw_quarter = list(global_quarters_map.keys())[0] if len(global_quarters_map) == 1 else None

    # --- Segmentation: Break the query into smaller, manageable parts ---
    query_segments = []
    initial_sentences = SENTENCE_DELIMITERS_REGEX.split(query)
    for sentence_text in initial_sentences:
        if not sentence_text.strip(): continue
        clauses_text = []
        last_clause_split_end = 0
        for match in CLAUSE_DELIMITERS_REGEX.finditer(sentence_text):
            clauses_text.append(sentence_text[last_clause_split_end:match.start()].strip())
            last_clause_split_end = match.end()
        clauses_text.append(sentence_text[last_clause_split_end:].strip())
        query_segments.extend([c for c in clauses_text if c])
    if not query_segments and query.strip(): query_segments.append(query.strip())


    # --- Segment-level Analysis: Process each clause to associate entities ---
    for segment_text in query_segments:
        if not segment_text.strip(): continue

        # Re-extract and de-conflict entities, but only for the current segment.
        segment_potential_entities = _extract_entities_from_segment_text(segment_text)
        segment_potential_entities.sort(key=lambda x: (x["start"], x["priority"], -(x["end"] - x["start"])))

        segment_selected_entities: List[Dict[str, Any]] = []
        _last_seg_idx = -1
        for entity in segment_potential_entities:
            if entity["start"] >= _last_seg_idx:
                segment_selected_entities.append(entity)
                _last_seg_idx = entity["end"]

        seg_stocks = [e for e in segment_selected_entities if e["type"] in (EntityType.TICKER, EntityType.COMPANY)]
        seg_years_entities = sorted([e for e in segment_selected_entities if e["type"] == EntityType.YEAR], key=lambda y: y["start"])
        seg_quarters_entities = sorted([e for e in segment_selected_entities if e["type"] == EntityType.QUARTER], key=lambda q: q["start"])
        seg_doc_type_keyword_entities = sorted([e for e in segment_selected_entities if e["type"] == EntityType.DOC_TYPE_KEYWORD], key=lambda d: d["start"])

        # If no company/ticker is found in this segment, there's nothing to process.
        if not seg_stocks: continue

        # Determine the document type for this segment based on a priority system.
        segment_normalized_doc_type: Optional[str] = None
        doc_types_found_in_segment = {dt_entity['value'] for dt_entity in seg_doc_type_keyword_entities}
        if DocType.EARNINGS_TRANSCRIPT in doc_types_found_in_segment:
            segment_normalized_doc_type = DocType.EARNINGS_TRANSCRIPT
        elif DocType.K10 in doc_types_found_in_segment:
            segment_normalized_doc_type = DocType.K10
        elif DocType.Q10 in doc_types_found_in_segment:
            segment_normalized_doc_type = DocType.Q10

        # For each stock found, associate it with the closest time entities in its segment.
        for stock_entity in seg_stocks:
            current_ticker = stock_entity["value"]
            assigned_year: Optional[int] = None
            assigned_raw_quarter: Optional[int] = None

            # The core association logic: find the entity whose start index is closest.
            if seg_years_entities:
                assigned_year = min(seg_years_entities, key=lambda y: abs(y["start"] - stock_entity["start"]))["value"]
            if seg_quarters_entities:
                assigned_raw_quarter = min(seg_quarters_entities, key=lambda q: abs(q["start"] - stock_entity["start"]))["value"]

            # Fall back to global defaults if no specific time entity was in the segment.
            if assigned_year is None and default_year is not None:
                 assigned_year = default_year
            if assigned_raw_quarter is None and default_raw_quarter is not None:
                assigned_raw_quarter = default_raw_quarter

            # Add the completed focus to a set to automatically handle duplicates.
            all_query_focuses_set.add(QueryFocus(
                ticker=current_ticker,
                year=assigned_year,
                quarter=assigned_raw_quarter,
                doc_type=segment_normalized_doc_type
            ))

    # --- Finalization: Clean the query and return results ---
    modified_query_parts = []
    current_pos = 0
    # Reconstruct the query string, omitting the text of every extracted entity.
    for entity in all_extracted_entities_globally:
        if entity["start"] > current_pos:
            modified_query_parts.append(query[current_pos:entity["start"]])
        end_skip = entity["end"]
        # Special handling for possessives like "Apple's" to remove the "'s" as well.
        if entity["type"] in (EntityType.TICKER, EntityType.COMPANY) and \
           not entity["text"].lower().endswith("'s") and \
           entity["end"] + 1 < len(query) and query[entity["end"]] == "'" and query[entity["end"]+1].lower() == "s":
            end_skip = entity["end"] + 2
        current_pos = end_skip
    if current_pos < len(query): modified_query_parts.append(query[current_pos:])
    modified_query = ' '.join("".join(modified_query_parts).split()).strip()

    # Return the sorted list of focuses and the cleaned query.
    return sorted(list(all_query_focuses_set), key=lambda qf: (
        qf.ticker, qf.year or -1, qf.quarter or -1, qf.doc_type or ""
    )), modified_query