import re

# --- 10-K Partitioning Logic ---

# This regular expression is designed to identify the start of standard sections in a 10-K filing.
# It looks for the pattern "Item X." (e.g., "Item 1A.", "Item 7.").
# The `num` capture group extracts the item number (e.g., "1A", "7").
# An optional "PART" prefix is also handled (e.g., "PART II Item 5.").
HEADER_RE_10K = re.compile(
    r"\b"
    r"(?:PART\s(?:I{1,3}|IV)\s+)?"
    r"Item\s"
    r"(?P<num>(?:[1-9]|1[0-6])[A-Z]?)"
    r"\."
)

def split_sections_10k(text: str) -> list[str]:
    """
    Splits the full text of a 10-K filing into a list of its constituent sections.

    This function uses the HEADER_RE_10K regex to find all "Item" headers and
    slices the text into chunks based on the start and end positions of these headers.
    Any text preceding the first "Item" is treated as an introductory section.

    @param text: The raw string content of the 10-K filing.
    @return: A list of strings, where each string is the content of a single section.
    """
    matches = list(HEADER_RE_10K.finditer(text))
    # If no standard "Item" headers are found, return the entire text as a single section.
    if not matches:
        return [text.strip()] if text.strip() else []

    sections = []
    # Capture any introductory text that appears before the first matched "Item".
    intro = text[:matches[0].start()].strip()
    if intro:
        sections.append(intro)

    # Iterate through the matches to create slices of the text for each section.
    for i, m in enumerate(matches):
        start = m.start()
        # The section ends where the next section begins, or at the end of the text.
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(text[start:end].strip())

    return sections

def process_sections_10k(sections: list[str], *, as_mapping: bool = False):
    """
    Processes a list of raw 10-K sections into a structured format.

    This function takes the output of `split_sections_10k` and enriches it by
    identifying the item number for each section and mapping it to a human-readable
    section name. It filters out sections with minimal content to reduce noise.

    @param sections: A list of raw section strings.
    @param as_mapping: If True, returns a dictionary mapping item numbers to text.
                       If False, returns a list of dictionaries with detailed info.
    @return: A list of structured section dictionaries or a single dictionary map.
    """
    # A mapping from the SEC's item numbers to their official, human-readable names.
    section_names_dict = {
        "1": "Business",
        "1A": "Risk Factors",
        "1B": "Unresolved Staff Comments",
        "2": "Properties",
        "3": "Legal Proceedings",
        "4": "Mine Safety Disclosures",
        "5": "Market for Registrant’s Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities",
        "6": "[Reserved]",
        "7": "Management’s Discussion and Analysis of Financial Condition and Results of Operations (MD&A)",
        "7A": "Quantitative and Qualitative Disclosures About Market Risk",
        "8": "Financial Statements and Supplementary Data",
        "9": "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
        "9A": "Controls and Procedures",
        "9B": "Other Information",
        "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
        "10": "Directors, Executive Officers and Corporate Governance",
        "11": "Executive Compensation",
        "12": "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters",
        "13": "Certain Relationships and Related Transactions, and Director Independence",
        "14": "Principal Accountant Fees and Services",
        "15": "Exhibits, Financial Statement Schedules",
        "16": "Form 10-K Summary (Optional)"
    }

    processed_list = []
    processed_map  = {}

    for sec in sections:
        header_match = HEADER_RE_10K.match(sec)
        if header_match:
            # Extract the item number and look up its friendly name.
            label = header_match.group("num")
            section_name = section_names_dict.get(label, f"Item {label}")
            body  = sec[header_match.end():].lstrip()
        else:
            # If no header matches, it's likely the preface/introductory text.
            label = "PREFACE"
            section_name = "Preface or Miscellaneous"
            body  = sec

        if as_mapping:
            processed_map[label] = body
        else:
            # A simple heuristic to filter out empty or non-substantive sections.
            if len(body) > 500:
                processed_list.append({
                    "item": label,
                    "section": section_name,
                    "text": body
                })

    return processed_map if as_mapping else processed_list

def cut_text_10K(text: str):
    """
    Truncates the 10-K text to start from the last occurrence of "Item 1.".

    This is a pre-processing step to handle cases where filings might contain
    multiple tables of contents or amended sections, ensuring that parsing begins
    at the start of the primary document content.

    @param text: The full text of the 10-K filing.
    @return: The truncated text, or the original text if the search string is not found.
    """
    search_string = "Item 1."
    # Find the last occurrence, as earlier ones might be in a table of contents.
    last_occurrence_index = text.rfind(search_string)

    if last_occurrence_index != -1:
        return text[last_occurrence_index:]
    else:
        # This warning is important for debugging parsing issues.
        print("Warning: 'Item 1.' not found in the text. Returning the original text.")
        return text

# --- 10-Q Partitioning Logic ---

# Defines the specific, high-value sections to extract from a 10-Q filing.
# The tuple keys `(item_key, keyword)` provide a robust way to identify sections,
# requiring both the correct item number and a keyword in the section's title.
TARGET_SECTIONS_10Q = {
    ('ITEM 1A', 'RISK')    : 'Risk Factors',
    ('ITEM 2',  'MANAGEMENT') : "MD&A (Management’s Discussion & Analysis)",
    ('ITEM 3',  'MARKET')  : 'Quant & Qualitative Market Risk'
}

# A simpler regex for 10-Q filings, as their structure is less complex than 10-K.
HEADER_RE_10Q = re.compile(r'ITEM\s+(\d+[A-Z]?)\.', re.IGNORECASE)

def extract_sections_from_text_10q(text: str,
                               targets: dict = TARGET_SECTIONS_10Q,
                               min_chars: int = 500,
                               max_digit_ratio: float = 0.40):
    """
    Extracts a predefined set of target sections from the text of a 10-Q filing.

    Unlike the 10-K parser which extracts all sections, this function is selective,
    only capturing sections deemed most valuable for analysis (e.g., MD&A, Risk Factors).
    It includes quality filters to discard sparse or table-heavy content.

    @param text: The raw string content of the 10-Q filing.
    @param targets: A dictionary defining which sections to extract.
    @param min_chars: The minimum character length for a section to be included.
    @param max_digit_ratio: The maximum ratio of digits to total characters, used
                            to filter out sections that are primarily numerical tables.
    @return: A list of dictionaries, where each dictionary represents a captured section.
    """
    matches = list(HEADER_RE_10Q.finditer(text))
    sections = []

    for i, m in enumerate(matches):
        item_num = m.group(1).upper()
        # Look ahead a short distance from the header to find keywords confirming the section type.
        lookahead = text[m.end(): m.end() + 120].upper()

        for (item_key, keyword), friendly in targets.items():
            # A section is a match if its item number and a keyword are found.
            if item_key == f'ITEM {item_num}' and keyword in lookahead:
                start = m.end()
                end   = matches[i+1].start() if i+1 < len(matches) else len(text)
                chunk = text[start:end].strip()

                # Apply quality filters.
                if len(chunk) >= min_chars:
                    # Calculate the ratio of digits to filter out dense tables.
                    digit_ratio = sum(c.isdigit() for c in chunk) / len(chunk)
                    if digit_ratio < max_digit_ratio:
                        hit = {
                            'item': item_num,
                            'section': friendly,
                            'text': chunk
                        }
                        sections.append(hit)
                break # Move to the next header match once a target is found.
    return sections