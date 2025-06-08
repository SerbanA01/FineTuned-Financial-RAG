#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sentence_transformers import SentenceTransformer

# Load the same model used for indexing
# Make sure this matches what generated your .npy files
embedding_model_name = "BAAI/bge-base-en-v1.5"
try:
    query_model = SentenceTransformer(embedding_model_name)
    print(f"Embedding model '{embedding_model_name}' loaded successfully.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    # Handle error: model might not be downloaded, or path is wrong
    query_model = None


def get_query_embedding(query_text: str):
    if not query_model:
        raise ValueError("Embedding model not loaded.")
    # The model expects a list of texts, even if it's just one
    #q_emb = enc.encode(question, normalize_embeddings=True).tolist()
    q_emb = query_model.encode(query_text, normalize_embeddings=True).tolist()
    return q_emb



# In[11]:




# In[12]:


from dotenv import load_dotenv
import os

load_dotenv()  # Now it should work


# In[1]:


import re
from typing import List, Tuple, Set, Dict, Any, Optional
import json
import os

# --- Mappings ---
COMPANY_TO_TICKER_HINTS = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "nvidia": "NVDA", "meta": "META", "tesla": "TSLA",
    "meta platforms": "META", "facebook": "META",
    "berkshire hathaway": "BRK.B", "berkshire hathaway a": "BRK.A",
    "bank of america": "BAC", "boa": "BAC",
    "boeing": "BA", "coca cola": "KO", "coca-cola": "KO", "cola": "KO", "coke": "KO",
    "johnson & johnson": "JNJ", "j&j": "JNJ", "johnson and johnson": "JNJ",
    "procter & gamble": "PG", "p&g": "PG", "procter and gamble": "PG",
    "walmart": "WMT", "wal-mart": "WMT", "wally world": "WMT",
    "united parcel service": "UPS", "ups": "UPS", "parcel service": "UPS",
    "general electric": "GE", "ge": "GE",
    "ibm": "IBM", "international business machines": "IBM",
    "american express": "AXP", "amex": "AXP",
    "home depot": "HD", "hd": "HD",
    "mcdonald's": "MCD", "mcdonalds": "MCD", "mcd": "MCD", "mickey d's": "MCD",
    "jpmorgan": "JPM", "jp morgan": "JPM", "jpm": "JPM",
    "wells fargo": "WFC", "citigroup": "C", "citi": "C",
    "goldman sachs": "GS", "morgan stanley": "MS",
    "charles schwab": "SCHW", "schwab": "SCHW",
    "blackrock": "BLK", "black rock": "BLK",
    "s&p global": "SPGI", "sandp global": "SPGI",
    "moody's": "MCO", "moodys": "MCO",
    "intercontinental exchange": "ICE",
    "walt disney": "DIS", "disney": "DIS", "disney co": "DIS",
    "comcast": "CMCSA", "netflix": "NFLX",
    "verizon": "VZ", "at&t": "T", "att": "T",
    "t-mobile": "TMUS", "tmobile": "TMUS",
    "charter": "CHTR", "fox": "FOXA", "news corp": "NWSA",
    "honeywell": "HON", "union pacific": "UNP",
    "3m": "MMM", "caterpillar": "CAT", "cat": "CAT",
    "lockheed martin": "LMT", "raytheon": "RTX",
    "northrop grumman": "NOC", "northrop": "NOC",
    "illinois tool works": "ITW", "deere": "DE", "john deere": "DE",
    "fedex": "FDX", "fed ex": "FDX",
    "pepsico": "PEP", "pepsi": "PEP",
    "costco": "COST", "mondelez": "MDLZ", "colgate-palmolive": "CL", "colgate": "CL",
    "kimberly-clark": "KMB", "general mills": "GIS", "kraft heinz": "KHC",
    "altria": "MO", "philip morris": "PM", "pm": "PM",
    "exxon mobil": "XOM", "exxon": "XOM", "mobil": "XOM",
    "chevron": "CVX", "conocophillips": "COP", "conoco": "COP",
    "schlumberger": "SLB", "halliburton": "HAL",
    "eog resources": "EOG", "marathon petroleum": "MPC", "marathon": "MPC",
    "phillips 66": "PSX", "valero": "VLO",
    "kinder morgan": "KMI", "williams companies": "WMB",
    "devon energy": "DVN",
    "nextera energy": "NEE", "duke energy": "DUK",
    "southern company": "SO", "dominion energy": "D",
    "american electric power": "AEP", "exelon": "EXC",
    "sempra": "SRE", "xcel energy": "XEL",
    "public service enterprise group": "PEG",
    "consolidated edison": "ED", "entergy": "ETR", "firstenergy": "FE",
    "american tower": "AMT", "prologis": "PLD",
    "crown castle": "CCI", "equinix": "EQIX", "public storage": "PSA",
    "simon property": "SPG", "digital realty": "DLR",
    "welltower": "WELL", "realty income": "O",
    "alexandria real estate": "ARE", "avalonbay": "AVB",
    "equity residential": "EQR",
    "linde": "LIN", "air products": "APD",
    "sherwin-williams": "SHW", "sherwin williams": "SHW",
    "dow": "DOW", "dupont": "DD", "newmont": "NEM",
    "freeport-mcmoran": "FCX", "freeport": "FCX",
    "international paper": "IP", "ball": "BALL", "albemarle": "ALB",
    "vision 2020": "V2020", "vision2020": "V2020", "v2020": "V2020",
}

TICKER_TO_COMPANY_HINTS = {}
for name, ticker in COMPANY_TO_TICKER_HINTS.items():
    if ticker not in TICKER_TO_COMPANY_HINTS:
        TICKER_TO_COMPANY_HINTS[ticker] = name.title()
if "BRK.A" not in TICKER_TO_COMPANY_HINTS and "berkshire hathaway a" in COMPANY_TO_TICKER_HINTS:
     TICKER_TO_COMPANY_HINTS["BRK.A"] = COMPANY_TO_TICKER_HINTS["berkshire hathaway a"].title()
if "V2020" not in TICKER_TO_COMPANY_HINTS and "vision 2020" in COMPANY_TO_TICKER_HINTS:
     TICKER_TO_COMPANY_HINTS["V2020"] = COMPANY_TO_TICKER_HINTS["vision 2020"].title()


TICKER_REGEX = re.compile(r'\b([A-Z]{1,5}(\.[A-Z])?)\b')
YEAR_REGEX = re.compile(r'\b(19[7-9]\d|20\d{2})\b')
QUARTER_REGEX = re.compile(
    r'\b(?:Q([1-4])|'
    r'(?:Quarter\s*([1-4]))|'
    r'(1st|2nd|3rd|4th)\s*Quarter|'
    r'(first|second|third|fourth)\s*Quarter)\b',
    re.IGNORECASE
)
QUARTER_WORD_TO_NUM = {
    "1st": 1, "first": 1, "2nd": 2, "second": 2,
    "3rd": 3, "third": 3, "4th": 4, "fourth": 4,
}

class DocType: # Normalized document types
    K10 = "10-K"
    Q10 = "10-Q"
    EARNINGS_TRANSCRIPT = "Earnings Transcript"

# Maps keywords to normalized DocType constants
DOC_KEYWORD_TO_NORMALIZED_MAP = {
    "10-k": DocType.K10, "10k": DocType.K10,
    "10-K": DocType.K10, "10-K filing": DocType.K10,
    "annual report": DocType.K10, "annual filing": DocType.K10,
    "10-q": DocType.Q10, "10q": DocType.Q10,
    "10-Q": DocType.Q10, "10-Q filing": DocType.Q10,
    "quarterly report": DocType.Q10, "quarterly filing": DocType.Q10,
    "earnings transcript": DocType.EARNINGS_TRANSCRIPT,
    "earnings call": DocType.EARNINGS_TRANSCRIPT,
    "financial call": DocType.EARNINGS_TRANSCRIPT,
    "investor call": DocType.EARNINGS_TRANSCRIPT,
    "conference call": DocType.EARNINGS_TRANSCRIPT,
    "transcript": DocType.EARNINGS_TRANSCRIPT,
    "earnings report": DocType.EARNINGS_TRANSCRIPT, # Commonly used term
}
# Regex for finding any of these keywords
DOC_TYPE_KEYWORDS_REGEX = re.compile(
    r'\b(?:' + '|'.join(re.escape(k) for k in DOC_KEYWORD_TO_NORMALIZED_MAP.keys()) + r')\b',
    re.IGNORECASE
)

class EntityType:
    TICKER = "ticker"; COMPANY = "company"; YEAR = "year"; QUARTER = "quarter"
    DOC_TYPE_KEYWORD = "doc_type_keyword" # Stores the actual keyword found

ENTITY_PRIORITIES = {
    EntityType.TICKER: 0, EntityType.YEAR: 1, EntityType.QUARTER: 2,
    EntityType.DOC_TYPE_KEYWORD: 3, EntityType.COMPANY: 4
}

class QueryFocus:
    def __init__(self, ticker: str, year: Optional[int] = None, quarter: Optional[int] = None, doc_type: Optional[str] = None):
        self.ticker = ticker
        self.year = year
        self.quarter = quarter # This will be the RAW quarter initially, then finalized
        self.doc_type = doc_type # This will be the NORMALIZED doc_type (or None)
    def __repr__(self):
        return f"QueryFocus(ticker='{self.ticker}', year={self.year}, quarter={self.quarter}, doc_type='{self.doc_type}')"
    def __eq__(self, other):
        return isinstance(other, QueryFocus) and self.ticker == other.ticker and \
               self.year == other.year and self.quarter == other.quarter and self.doc_type == other.doc_type
    def __hash__(self):
        return hash((self.ticker, self.year, self.quarter, self.doc_type))

def _extract_entities_from_segment_text(segment_text: str) -> List[Dict[str, Any]]:
    potential_entities: List[Dict[str, Any]] = []
    # Tickers, Companies, Years, Quarters (same as before)
    for match in TICKER_REGEX.finditer(segment_text):
        ticker_candidate = match.group(1)
        if ticker_candidate in TICKER_TO_COMPANY_HINTS:
            potential_entities.append({
                "text": ticker_candidate, "value": ticker_candidate, "type": EntityType.TICKER,
                "start": match.start(1), "end": match.end(1), "priority": ENTITY_PRIORITIES[EntityType.TICKER]
            })
    for cn_key in sorted(COMPANY_TO_TICKER_HINTS.keys(), key=len, reverse=True):
        ticker = COMPANY_TO_TICKER_HINTS[cn_key]
        pattern = r'\b' + re.escape(cn_key) + r'\b'
        for cmatch in re.finditer(pattern, segment_text, re.IGNORECASE):
            potential_entities.append({
                "text": cmatch.group(0), "value": ticker, "type": EntityType.COMPANY,
                "start": cmatch.start(0), "end": cmatch.end(0), "priority": ENTITY_PRIORITIES[EntityType.COMPANY]
            })
    for ymatch in YEAR_REGEX.finditer(segment_text):
        year_str = ymatch.group(1)
        potential_entities.append({
            "text": year_str, "value": int(year_str), "type": EntityType.YEAR,
            "start": ymatch.start(1), "end": ymatch.end(1), "priority": ENTITY_PRIORITIES[EntityType.YEAR]
        })
    for qmatch in QUARTER_REGEX.finditer(segment_text):
        q_val_str = qmatch.group(1) or qmatch.group(2) or qmatch.group(3) or qmatch.group(4)
        q_num = None
        if q_val_str: q_num = int(q_val_str) if q_val_str.isdigit() else QUARTER_WORD_TO_NUM.get(q_val_str.lower())
        if q_num:
            potential_entities.append({
                "text": qmatch.group(0), "value": q_num, "type": EntityType.QUARTER, # Raw quarter 1-4
                "start": qmatch.start(0), "end": qmatch.end(0), "priority": ENTITY_PRIORITIES[EntityType.QUARTER]
            })
    # Document Type Keywords
    for dmatch in DOC_TYPE_KEYWORDS_REGEX.finditer(segment_text):
        keyword_found = dmatch.group(0).lower()
        normalized_doc_type = DOC_KEYWORD_TO_NORMALIZED_MAP.get(keyword_found)
        if normalized_doc_type: # Should always be true due to regex construction
            potential_entities.append({
                "text": dmatch.group(0), "value": normalized_doc_type, # Store normalized type
                "type": EntityType.DOC_TYPE_KEYWORD, # Mark as keyword entity
                "start": dmatch.start(0), "end": dmatch.end(0), "priority": ENTITY_PRIORITIES[EntityType.DOC_TYPE_KEYWORD]
            })
    return potential_entities

SENTENCE_DELIMITERS_REGEX = re.compile(r'[.?!]')
CLAUSE_DELIMITERS_REGEX = re.compile(r'\s+(?:and|or|but)\s+|,', re.IGNORECASE)

def extract_structured_metadata(query: str) -> Tuple[List[QueryFocus], str]:
    """
    Rule-based extraction. Creates provisional QueryFocus objects.
    - `quarter` will be the raw quarter (1-4) if found.
    - `doc_type` will be the normalized DocType (e.g., DocType.K10) if a keyword
      for it is found in the segment, otherwise None.
    Final quarter/doc_type rules are NOT applied here.
    """
    all_query_focuses_set: Set[QueryFocus] = set()
    
    glob_potential_entities = _extract_entities_from_segment_text(query)
    glob_potential_entities.sort(key=lambda x: (x["start"], x["priority"], -(x["end"] - x["start"])))
    
    all_extracted_entities_globally: List[Dict[str, Any]] = []
    _last_covered_idx = -1
    for entity in glob_potential_entities:
        if entity["start"] >= _last_covered_idx:
            all_extracted_entities_globally.append(entity)
            _last_covered_idx = entity["end"]
    
    # Create global maps for years and quarters
    global_years_map: Dict[int, int] = {}
    global_quarters_map: Dict[int, int] = {}
    
    global_years_map = {e["value"]: global_years_map.get(e["value"], 0) + 1 for e in all_extracted_entities_globally if e["type"] == EntityType.YEAR}
    global_quarters_map = {e["value"]: global_quarters_map.get(e["value"], 0) + 1 for e in all_extracted_entities_globally if e["type"] == EntityType.QUARTER}
            
    default_year = list(global_years_map.keys())[0] if len(global_years_map) == 1 else None
    default_raw_quarter = list(global_quarters_map.keys())[0] if len(global_quarters_map) == 1 else None # Raw default
    
    query_segments = []
    # ... (segmentation logic remains the same)
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


    for segment_text in query_segments:
        if not segment_text.strip(): continue

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
        seg_quarters_entities = sorted([e for e in segment_selected_entities if e["type"] == EntityType.QUARTER], key=lambda q: q["start"]) # Raw quarters
        seg_doc_type_keyword_entities = sorted([e for e in segment_selected_entities if e["type"] == EntityType.DOC_TYPE_KEYWORD], key=lambda d: d["start"])


        if not seg_stocks: continue

        # Determine normalized doc_type for the segment (if any keyword present)
        # Priority: ET > K10 > Q10 if multiple types of keywords in same segment
        segment_normalized_doc_type: Optional[str] = None
        doc_types_found_in_segment = {dt_entity['value'] for dt_entity in seg_doc_type_keyword_entities}
        
        if DocType.EARNINGS_TRANSCRIPT in doc_types_found_in_segment:
            segment_normalized_doc_type = DocType.EARNINGS_TRANSCRIPT
        elif DocType.K10 in doc_types_found_in_segment:
            segment_normalized_doc_type = DocType.K10
        elif DocType.Q10 in doc_types_found_in_segment:
            segment_normalized_doc_type = DocType.Q10

        for stock_entity in seg_stocks:
            current_ticker = stock_entity["value"]
            assigned_year: Optional[int] = None
            assigned_raw_quarter: Optional[int] = None # Store raw quarter (1-4)
            
            if seg_years_entities:
                assigned_year = min(seg_years_entities, key=lambda y: abs(y["start"] - stock_entity["start"]))["value"]
            if seg_quarters_entities: # these are raw quarters
                assigned_raw_quarter = min(seg_quarters_entities, key=lambda q: abs(q["start"] - stock_entity["start"]))["value"]
            
            # Apply defaults
            if assigned_year is None and default_year is not None:
                 assigned_year = default_year
            if assigned_raw_quarter is None and default_raw_quarter is not None:
                assigned_raw_quarter = default_raw_quarter

            all_query_focuses_set.add(QueryFocus(
                ticker=current_ticker, 
                year=assigned_year, 
                quarter=assigned_raw_quarter, # Store raw quarter
                doc_type=segment_normalized_doc_type # Store determined normalized doc type for segment
            ))

    # --- Final Query Modification (remains mostly the same) ---
    modified_query_parts = []
    current_pos = 0
    # Use all_extracted_entities_globally for query cleaning
    for entity in all_extracted_entities_globally:
        if entity["start"] > current_pos:
            modified_query_parts.append(query[current_pos:entity["start"]])
        end_skip = entity["end"]
        if entity["type"] in (EntityType.TICKER, EntityType.COMPANY) and \
           not entity["text"].lower().endswith("'s") and \
           entity["end"] + 1 < len(query) and query[entity["end"]] == "'" and query[entity["end"]+1].lower() == "s":
            end_skip = entity["end"] + 2
        current_pos = end_skip
    if current_pos < len(query): modified_query_parts.append(query[current_pos:])
    modified_query = ' '.join("".join(modified_query_parts).split()).strip()

    return sorted(list(all_query_focuses_set), key=lambda qf: (
        qf.ticker, qf.year or -1, qf.quarter or -1, qf.doc_type or ""
    )), modified_query


# --- Groq LLM Integration ---
from groq import Groq

try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY environment variable is not set.")
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    groq_client = None

LLM_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

def format_provisional_output_for_prompt(focuses: List[QueryFocus]) -> str:
    if not focuses: return "Rule-based extraction found no provisional focuses."
    formatted_str = "Provisional Rule-Based Extraction (raw quarter, normalized doc_type if keyword found):\n"
    for i, focus in enumerate(focuses):
        year_str = str(focus.year) if focus.year is not None else "Not specified"
        # Displaying raw quarter as found by rules
        quarter_str = "Q" + str(focus.quarter) if focus.quarter is not None else "Not specified"
        doc_type_str = focus.doc_type if focus.doc_type is not None else "Not specified (no keyword)"
        formatted_str += f"- Focus {i+1}: Ticker={focus.ticker}, Year={year_str}, Raw Quarter={quarter_str}, Normalized DocType (from keywords)={doc_type_str}\n"
    return formatted_str.strip()

def refine_with_llm(original_query: str, provisional_rule_based_focuses: List[QueryFocus]) -> List[QueryFocus]:

    if not groq_client:
        print("Groq client not initialized. Returning provisional rule-based results.")
        return provisional_rule_based_focuses

    provisional_output_str = format_provisional_output_for_prompt(provisional_rule_based_focuses)

    # --- UPDATED SYSTEM PROMPT ---
    system_prompt = f"""
You are an expert financial data analyst assistant. Your task is to re-evaluate a user's query to produce a definitive, corrected list of financial data focuses.
You are provided with a provisional, rule-based extraction; treat it as a HINT, but the user's original query is the absolute source of truth. You MUST override the provisional extraction if it is incorrect or incomplete based on a full reading of the query.

Instructions:
1.  **Analyze the User's Intent:** Carefully read the ENTIRE user query to understand each distinct request. The query is the ultimate source of truth.
2.  **Extract Entities for Each Request:**
    a.  **Ticker**: Identify the company ticker.
    b.  **Year**: Identify the associated year.
    c.  **Raw_Quarter**: Identify the quarter (1, 2, 3, or 4).
    d.  **Normalized_DocumentType**: This is a critical step. Search for document type keywords within the same phrase or clause as the company/ticker.
        - Keywords for "{DocType.K10}": "10-K", "10K", "annual report", "annual filing".
        - Keywords for "{DocType.Q10}": "10-Q", "10Q", "quarterly report", "quarterly filing".
        - Keywords for "{DocType.EARNINGS_TRANSCRIPT}": "earnings transcript", "earnings call", etc.
3.  **Handle Pronouns:** Pay close attention to pronouns like "its", "their", "the company's". Resolve them to the most recent preceding company mentioned. For example, in "...Apple... see its 10-Q...", "its" refers to Apple.
4.  **Strict Association:** You MUST be diligent in linking the document type to the correct company. If a query says "...Apple's 2023 Q2 10-Q...", the `normalized_doc_type` for the Apple focus MUST be "{DocType.Q10}".
5.  **Handle Ambiguity:** If a company is mentioned conversationally (e.g., "I ate an apple"), only create a financial data request if it is clearly linked to financial terms like "10-Q", "annual report", etc.

Output Format:
Return ONLY a valid JSON object with a "focuses" key. "focuses" should be an array of objects.
Each object MUST have: "ticker" (string), "year" (integer or null), "raw_quarter" (integer 1-4 or null), and "normalized_doc_type" (string: "{DocType.K10}", "{DocType.Q10}", "{DocType.EARNINGS_TRANSCRIPT}", or null).

Example 1:
Query: "Apple Q1 2022 earnings transcript and Google Q3 2023 10-K."
Expected JSON:
{{"focuses": [
  {{"ticker": "AAPL", "year": 2022, "raw_quarter": 1, "normalized_doc_type": "{DocType.EARNINGS_TRANSCRIPT}"}},
  {{"ticker": "GOOGL", "year": 2023, "raw_quarter": 3, "normalized_doc_type": "{DocType.K10}"}}
]}}

Example 2 (Coreference Resolution):
Query: "I ate an apple, but I want to see its 2023 Q2 10-Q and also Microsoft's 2022 annual report."
Expected JSON:
{{"focuses": [
  {{"ticker": "AAPL", "year": 2023, "raw_quarter": 2, "normalized_doc_type": "{DocType.Q10}"}},
  {{"ticker": "MSFT", "year": 2022, "raw_quarter": null, "normalized_doc_type": "{DocType.K10}"}}
]}}

Example 3:
Query: "Who is the CEO of Apple?
Expected JSON:
{{"focuses": [
  {{"ticker": "AAPL", "year": null, "raw_quarter": null, "normalized_doc_type": null}}
]}}
"""

    user_content = f"""
Original Query:
"{original_query}"

Provisional Rule-Based Extraction (this is just a HINT, the Original Query is the source of truth):
{provisional_output_str}

Task:
Provide the "Corrected List of Focuses" as a JSON object based on the instructions.
Extract ticker, year, raw_quarter (1-4 or null), and normalized_doc_type ("{DocType.K10}", "{DocType.Q10}", "{DocType.EARNINGS_TRANSCRIPT}", or null).
The JSON must have a single "focuses" key which contains a list of objects.
"""
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            model=LLM_MODEL_NAME, temperature=0.0, max_tokens=1500,
            response_format={"type": "json_object"}
        )
        response_content = chat_completion.choices[0].message.content
        if response_content.startswith("```json"): response_content = response_content[7:-3] if response_content.endswith("```") else response_content[7:]
        response_content = response_content.strip()

        llm_data = json.loads(response_content)
        refined_provisional_focuses = []
        if isinstance(llm_data, dict) and "focuses" in llm_data:
            focuses_list = llm_data["focuses"]
            if isinstance(focuses_list, list):
                for item in focuses_list:
                    if isinstance(item, dict) and "ticker" in item:
                        refined_provisional_focuses.append(QueryFocus(
                            ticker=item.get("ticker"),
                            year=item.get("year"),
                            quarter=item.get("raw_quarter"),
                            doc_type=item.get("normalized_doc_type")
                        ))
                    else: print(f"Warning: LLM returned an invalid item: {item}")
            elif focuses_list is not None:
                print(f"Warning: LLM 'focuses' field not a list. Response: {llm_data}")
                return provisional_rule_based_focuses
        else:
            print(f"Warning: LLM did not return expected format. Response: {llm_data}")
            return provisional_rule_based_focuses
        return refined_provisional_focuses
    except json.JSONDecodeError as e:
        print(f"Error decoding LLM JSON: {e}\nRaw Response: {response_content}")
        return provisional_rule_based_focuses
    except Exception as e:
        print(f"Error during LLM refinement: {e}")
        return provisional_rule_based_focuses

def apply_final_rules_to_focuses(provisional_focuses: List[QueryFocus]) -> List[QueryFocus]:
    """
    Applies the strict document type vs. quarter rules.
    Modifies the 'quarter' field in each QueryFocus object in place.
    """
    finalized_focuses = []
    for focus in provisional_focuses:
        final_focus = QueryFocus(focus.ticker, focus.year, focus.quarter, focus.doc_type) # Create a copy

        if final_focus.doc_type == DocType.K10:
            final_focus.quarter = None  # 10-K never has a quarter
        elif final_focus.doc_type == DocType.Q10:
            if final_focus.quarter not in [1, 2, 3]: # Includes Q4 or None
                final_focus.quarter = None # 10-Q only for Q1, Q2, Q3
        # If DocType.EARNINGS_TRANSCRIPT or doc_type is None, quarter (1-4 or None) is kept as is.
        finalized_focuses.append(final_focus)
    return finalized_focuses

def process_query_with_llm_refinement(query: str) -> Tuple[List[QueryFocus], str]:
    # 1. Rule-based provisional extraction
    provisional_focuses_rules, modified_query_by_rules = extract_structured_metadata(query)
    
    # 2. LLM refinement of provisional focuses
    if groq_client:
        print(f"\n--- Calling LLM for query: \"{query}\" ---")
        print(f"Provisional rule-based output (pre-LLM, pre-final-rules): {provisional_focuses_rules}")
        provisional_focuses_llm = refine_with_llm(query, provisional_focuses_rules)
        print(f"Provisional LLM output (pre-final-rules): {provisional_focuses_llm}")
        # Use LLM output if available, otherwise fallback to rule-based
        current_provisional_focuses = provisional_focuses_llm
    else:
        print("LLM client not available, using rule-based provisional output.")
        current_provisional_focuses = provisional_focuses_rules
    
    # 3. Apply final heuristic rules
    final_focuses = apply_final_rules_to_focuses(current_provisional_focuses)
    print(f"Final focuses after applying heuristic rules: {final_focuses}")
    
    return final_focuses, modified_query_by_rules





# In[2]:
#distinguish between price searches and data based searches


from qdrant_client import QdrantClient, models
from dateutil.parser import parse as date_parse
from collections import OrderedDict
from typing import List, Optional

# Assume all your previous code (parsing, QueryFocus, DocType, etc.) is here.
# ...

# --- Helper Function to Infer Quarter from Date ---

def get_quarter_from_date(date_str: Optional[str]) -> Optional[int]:
    """
    Infers the calendar quarter (1, 2, 3, or 4) from a date string.
    Returns None if the date string is invalid or missing.
    """
    if not date_str:
        return None
    try:
        # dateutil.parser is robust and can handle various date formats
        dt = date_parse(date_str)
        return (dt.month - 1) // 3 + 1
    except (ValueError, TypeError):
        # Handles cases where date is None, not a string, or un-parseable
        return None

# --- Step 1: Building the Qdrant Filter (Broad Search) ---

def build_single_focus_filter(focus: QueryFocus) -> models.Filter:
    """
    Builds a robust Qdrant filter for a SINGLE QueryFocus object.
    This filter is specifically designed to handle the observed data discrepancies.
    """
    must_conditions = []

    # Condition: Ticker (Always reliable)
    must_conditions.append(
        models.FieldCondition(key="ticker", match=models.MatchValue(value=focus.ticker))
    )

    # Condition: Year (Handles string vs. integer discrepancy)
    if focus.year:
        # As seen in payloads, 'year' can be '2020' (str) or 2024 (int).
        # This 'should' clause ensures we match either type.
        must_conditions.append(
            models.Filter(
                should=[
                    models.FieldCondition(key="year", match=models.MatchValue(value=focus.year)),
                    models.FieldCondition(key="year", match=models.MatchValue(value=str(focus.year)))
                ]
            )
        )

    # Condition: Document Type & Quarter (The most nuanced part)
    if focus.doc_type == DocType.K10:
        # Payloads confirm 10-K is identified by these two fields.
        must_conditions.append(models.FieldCondition(key="source_type", match=models.MatchValue(value="sec_filing")))
        must_conditions.append(models.FieldCondition(key="filing_category", match=models.MatchValue(value="10k")))
        # We do NOT filter on date/quarter, as 'date' is None for 10-Ks.

    elif focus.doc_type == DocType.EARNINGS_TRANSCRIPT:
        # Payloads confirm this source_type and a reliable 'quarter' field like "Q1".
        must_conditions.append(models.FieldCondition(key="source_type", match=models.MatchValue(value="earnings_transcript")))
        if focus.quarter:
            # The 'quarter' field is reliable for transcripts, so we filter directly.
            must_conditions.append(
                models.FieldCondition(key="quarter", match=models.MatchValue(value=f"Q{focus.quarter}"))
            )

    elif focus.doc_type == DocType.Q10:
        # Payloads confirm 10-Q is identified by these two fields.
        must_conditions.append(models.FieldCondition(key="source_type", match=models.MatchValue(value="sec_filing")))
        must_conditions.append(models.FieldCondition(key="filing_category", match=models.MatchValue(value="10q")))
        # CRITICAL: We DO NOT filter by quarter here. 10-Q payloads lack a 'quarter'
        # field but have a 'date' field. We will use the date for post-filtering.

    return models.Filter(must=must_conditions)


def search_qdrant_per_focus(
    query_vector,
    focuses: List[QueryFocus],
    client: QdrantClient,
    collection_name: str,
    k: int = 10
) -> OrderedDict[QueryFocus, list[models.ScoredPoint]]:
    """
    Runs one similarity search for each QueryFocus using the broad filter.
    This returns initial candidate results that will be refined later.
    """
    vec = query_vector.tolist() if hasattr(query_vector, "tolist") else query_vector
    initial_results = OrderedDict()

    for focus in focuses:
        query_filter = build_single_focus_filter(focus)
        
        try:
            res = client.search(
                collection_name=collection_name,
                query_vector=vec,
                limit=k,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
                timeout=10
            )
            initial_results[focus] = res
        except Exception as e:
            print(f"Error searching Qdrant for focus {focus}: {e}")
            initial_results[focus] = []
            
    return initial_results


# --- Step 2: Applying the Post-Filter (Precise Filtering) ---

def post_filter_results(
    initial_results_dict: OrderedDict[QueryFocus, list[models.ScoredPoint]]
) -> OrderedDict[QueryFocus, list[models.ScoredPoint]]:
    """
    Applies precise filtering rules AFTER the Qdrant search, specifically for 10-Q quarters.
    """
    final_results_dict = OrderedDict()

    for focus, points in initial_results_dict.items():
        # The only case requiring post-filtering is a 10-Q with a specific quarter.
        if not (focus.doc_type == DocType.Q10 and focus.quarter is not None):
            final_results_dict[focus] = points # No post-filtering needed, pass through.
            continue

        # This is the key logic for 10-Q quarter filtering.
        filtered_points = []
        for point in points:
            # Payloads confirm 'date' is the field to use.
            payload_date = point.payload.get("date")
            inferred_quarter = get_quarter_from_date(payload_date)
            
            # Match the inferred quarter with the requested quarter.
            if inferred_quarter == focus.quarter:
                filtered_points.append(point)
        
        final_results_dict[focus] = filtered_points

    return final_results_dict


# --- Main Orchestration Function ---

# ==============================================================================
# FINAL CODE WITH RELAXATION STRATEGY
# ==============================================================================
from qdrant_client import QdrantClient, models
from dateutil.parser import parse as date_parse
from collections import OrderedDict
from typing import List, Optional, Tuple, Dict
import copy # Needed to safely modify focuses

# --- Keep all your existing functions ---
# get_quarter_from_date, build_single_focus_filter, post_filter_results, etc.
# They are the building blocks and do not need to change.
# ...

# --- NEW Orchestration Function with Relaxation ---
# ==============================================================================
# ADD THIS ENTIRE BLOCK TO YOUR CODE
# It replaces the old process_and_retrieve_with_relaxation function
# ==============================================================================
import copy
from typing import List, Tuple, Dict

# This is a new class to hold tiered results
class SearchResult:
    """A container for search results to add metadata like tier."""
    def __init__(self, point, tier: str):
        self.point = point
        self.tier = tier # e.g., "Exact Match", "Augmented: Other Years"

    def __repr__(self):
        return f"SearchResult(Tier: '{self.tier}', Score: {self.point.score:.4f})"

# This is a new helper function to simplify the main loop
def search_and_filter(
    query_vector,
    focus: "QueryFocus",
    client: "QdrantClient",
    collection_name: str,
    k: int
) -> List["models.ScoredPoint"]:
    """Helper function to run a single search-and-filter operation."""
    initial_search_res = search_qdrant_per_focus(
        query_vector=query_vector,
        focuses=[focus],
        client=client,
        collection_name=collection_name,
        k=k,
    )
    final_filtered_res = post_filter_results(initial_search_res)
    return final_filtered_res.get(focus, [])


# This is the NEW main function that handles both relaxation and augmentation
def process_and_retrieve_with_augmentation(
    query: str,
    client: "QdrantClient",
    collection_name: str,
    query_model,
    min_results_k: int = 5,
) -> Tuple[Dict["QueryFocus", List[SearchResult]], str]:
    """
    Full pipeline with a two-phase approach:
    1. Full Relaxation: Find *any* relevant document if the exact query fails.
    2. Result Augmentation: If a query succeeds but returns < k results,
       pad the results with slightly relaxed searches.
    """
    original_focuses, modified_query = process_query_with_llm_refinement(query)
    
    if not original_focuses:
        return OrderedDict(), modified_query

    print("\n--- Original Parsed Focuses ---")
    for f in original_focuses: print(f)

    query_vector = query_model.encode(modified_query, normalize_embeddings=True)
    final_results_for_query = OrderedDict()

    for original_focus in original_focuses:
        print(f"\n--- Processing Focus with Augmentation: {original_focus} ---")
        
        # --- PHASE 1: Find a base set of results using full relaxation ---
        base_results = []
        best_focus_found = None
        
        relaxation_sequence = []
        relaxation_sequence.append((original_focus, "Exact Match"))
        if original_focus.quarter:
            focus_no_q = copy.deepcopy(original_focus); focus_no_q.quarter = None
            relaxation_sequence.append((focus_no_q, f"Relaxed: All quarters for {original_focus.year}"))
        if original_focus.year:
            focus_no_y = copy.deepcopy(original_focus); focus_no_y.year = None; focus_no_y.quarter = None
            relaxation_sequence.append((focus_no_y, "Relaxed: Most relevant year"))
        if original_focus.doc_type:
             focus_no_dt = copy.deepcopy(original_focus); focus_no_dt.doc_type = None; focus_no_dt.year = None; focus_no_dt.quarter = None
             relaxation_sequence.append((focus_no_dt, "Relaxed: Any document type"))

        print("  -> Phase 1: Finding best possible results via relaxation...")
        for focus_to_try, message in relaxation_sequence:
            print(f"    - Attempting search with: {focus_to_try}")
            points = search_and_filter(query_vector, focus_to_try, client, collection_name, k=min_results_k)
            if points:
                print(f"    - SUCCESS: Found {len(points)} results for '{message}'. This is our base.")
                base_results = [SearchResult(p, tier=message) for p in points]
                best_focus_found = focus_to_try
                break
        
        if not base_results:
            print("  -> Phase 1 FAILED: No results found even after full relaxation.")
            final_results_for_query[original_focus] = []
            continue

        # --- PHASE 2: Augment results if we have fewer than k ---
        if len(base_results) < min_results_k:
            print(f"  -> Phase 2: Augmenting results (found {len(base_results)} of {min_results_k})...")
            
            existing_ids = {res.point.id for res in base_results}
            
            augmentation_sequence = []
            if best_focus_found and best_focus_found.quarter:
                focus_aug = copy.deepcopy(best_focus_found); focus_aug.quarter = None
                augmentation_sequence.append((focus_aug, f"Augmented: Other quarters from {best_focus_found.year}"))
            if best_focus_found and best_focus_found.year:
                focus_aug = copy.deepcopy(best_focus_found); focus_aug.year = None; focus_aug.quarter = None
                augmentation_sequence.append((focus_aug, "Augmented: Other relevant years"))
            if best_focus_found and best_focus_found.doc_type:
                 focus_aug = copy.deepcopy(best_focus_found); focus_aug.doc_type = None
                 augmentation_sequence.append((focus_aug, "Augmented: Other document types"))

            for focus_to_try, message in augmentation_sequence:
                if len(base_results) >= min_results_k: break

                needed = min_results_k - len(base_results)
                print(f"    - Attempting to find {needed} more results with: {focus_to_try}")
                
                # Fetch more than needed to account for duplicates
                points = search_and_filter(query_vector, focus_to_try, client, collection_name, k=min_results_k * 2)
                
                new_points_added = 0
                for p in points:
                    if len(base_results) >= min_results_k: break
                    if p.id not in existing_ids:
                        base_results.append(SearchResult(p, tier=message))
                        existing_ids.add(p.id)
                        new_points_added += 1
                
                if new_points_added > 0:
                    print(f"    - SUCCESS: Added {new_points_added} new results.")

        final_results_for_query[original_focus] = base_results

    # 5. Display final results
    print("\n\n--- FINAL TIERED RESULTS ---")
    for focus, results in final_results_for_query.items():
        print(f"\nResults for Original Request: {focus}")
        if results:
            for res in results:
                payload = res.point.payload
                print(f"  - Tier: '{res.tier}' | "
                      f"{payload.get('filing_category') or payload.get('source_type')} "
                      f"({payload.get('year')}, Date: {payload.get('date')}) | "
                      f"Score: {res.point.score:.4f}")
        else:
            print("  No relevant documents could be found.")
            
    return final_results_for_query, modified_query

#reranking
def _get_tier_priority(tier_string: str) -> int:
    """Assigns a numerical priority (lower is better) to each tier string."""
    if not tier_string:
        return 99
    if tier_string == "Exact Match":
        return 0
    if tier_string.startswith("Relaxed:"):
        return 1
    if tier_string.startswith("Augmented:"):
        return 2
    return 99

from sentence_transformers import CrossEncoder

def finalize_and_rerank_results(
    query: str,
    candidate_results: List["SearchResult"],
    reranker: "CrossEncoder",
    final_k: int
) -> List["SearchResult"]:

    if not candidate_results:
        print("  -> Reranker received no candidates. Returning empty list.")
        return []

    print(f"  -> Reranking {len(candidate_results)} candidates to select top {final_k}...")

    # 1. Create pairs of [query, document_text] for the model
    sentence_pairs = [[query, res.point.payload.get('chunk_text', '')] for res in candidate_results]

    # 2. Predict new relevance scores for all candidates
    try:
        rerank_scores = reranker.predict(sentence_pairs, show_progress_bar=False)
    except Exception as e:
        print(f"  -> ERROR during reranking: {e}. Returning original top-k without reranking.")
        return candidate_results[:final_k]

    # 3. Attach the new score and a tier priority to each result object
    for res, score in zip(candidate_results, rerank_scores):
        res.rerank_score = score
        res.tier_priority = _get_tier_priority(res.tier)

    # 4. Perform the crucial multi-level sort
    #    - Primary sort: by tier_priority (Exact > Relaxed > Augmented)
    #    - Secondary sort: by rerank_score (highest score first)
    sorted_results = sorted(
        candidate_results,
        key=lambda x: (x.tier_priority, -x.rerank_score)
    )

    # 5. Select the final top-k results from the perfectly sorted list
    final_top_k = sorted_results[:final_k]
    print(f"  -> Reranking complete. Final list has {len(final_top_k)} items.")
    
    return final_top_k


# In[3]:


#generation
from typing import List, Dict, OrderedDict

# Assume your SearchResult class is defined and has attributes like:
# .point, .tier, .rerank_score

def format_context_for_llm(
    retrieved_results: Dict["QueryFocus", List["SearchResult"]]
) -> str:
    """
    Formats the tiered and reranked search results into a single string of 
    context for the LLM, with clear source identifiers for citations.
    """
    if not retrieved_results:
        return "No relevant documents were found."

    context_str = "CONTEXT:\n"
    context_str += "The following document chunks were retrieved as relevant to the user's query:\n\n"
    
    chunk_index = 1
    for focus, results in retrieved_results.items():
        if not results:
            continue
        
        # Add a header for the results related to each focus
        context_str += f"--- Documents related to: {focus} ---\n"
        for res in results:
            payload = res.point.payload
            
            # Create a clear source description
            source_info = f"{payload.get('source_type', '')}"
            if payload.get('filing_category'):
                source_info += f" ({payload.get('filing_category').upper()})"
            if payload.get('year'):
                source_info += f", Year: {payload.get('year')}"
            if payload.get('quarter'):
                 source_info += f", Quarter: {payload.get('quarter')}"
            
            # This [CHUNK X] is the key for citations
            context_str += f"[CHUNK {chunk_index}] - Source: {source_info}\n"
            context_str += f"Content: \"\"\"\n{payload.get('chunk_text', '')}\n\"\"\"\n\n"
            chunk_index += 1
    
    return context_str.strip()


# In[4]:


#yahoo finance price retrieval

#words that trigger price retrieval
PRICE_KEYWORDS = [
    "price", "current price", "stock price", "share price", "trading price",
    "closing price", "last price", "latest price", "market price",
    "current value", "stock value", "share value", "valuation", "pe ratio", "p/e", "dividend yield",
    "market cap", "eps multiple", "target price"
]

def needs_price_retrieval(query: str) -> bool:
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in PRICE_KEYWORDS)

import yfinance as yf
from datetime import datetime, timedelta
def fetch_price_chunk(ticker: str,
                      asof: datetime | None = None) -> dict | None:
    """
    Returns a dict with the exact fields your LLM formatter needs,
    or None if the fetch fails.
    """
    try:
        stock = yf.Ticker(ticker)
        if asof is None:
            asof = datetime.utcnow()
        # Get last close before `asof`
        hist = stock.history(start=asof.date(), end=(asof.date() + timedelta(days=1)))
        last_close = float(hist['Close'][-1])  # will raise IndexError if empty
        dividend = float(stock.info.get("dividendRate") or 0.0)
        return {
            "source_type": "market_data",
            "meta_source": "YahooFinance",
            "ticker": ticker.upper(),
            "date": asof.strftime("%Y-%m-%d"),
            "chunk_text":
                f"Close price on {asof.date()}: ${last_close:,.2f}. "
                f"Dividend per share (TTM): ${dividend:,.2f}."
        }
    except Exception as exc:
        print(f"[market-data] fetch failed for {ticker}: {exc}")
        return None
    
#wrap answer in dummy searchresul
from qdrant_client.models import ScoredPoint  # only for type compatibility
import uuid

def make_price_search_result(price_payload: dict) -> SearchResult:
    dummy_point = ScoredPoint(
        id=str(uuid.uuid4()),
        score=1.0,          # irrelevant, we skip rerank
        payload=price_payload,
        vector=None,
        version=0
    )
    return SearchResult(dummy_point, tier="Market Data")


# In[5]:


from groq import Groq # Or your preferred LLM client

# Assume your other functions (format_context_for_llm, etc.) and imports are the same

# In your generate_final_answer function, replace the system_prompt with this new one.

def generate_final_answer(
    original_query: str,
    retrieved_results: Dict["QueryFocus", List["SearchResult"]],
    llm_client: "Groq",
    llm_model_name: str = "llama3-8b-8192"
) -> str:
    """
    Takes the retrieved context, formats it, and uses an LLM to generate 
    a final, citable answer. This version uses the "Balanced Analyst" prompt.
    """
    formatted_context = format_context_for_llm(retrieved_results)

    if "No relevant documents" in formatted_context:
        return "I could not find any relevant documents to answer your query."

    # --- THE BALANCED ANALYST PROMPT ---
    system_prompt = """
You are a sophisticated Financial Analyst assistant. Your goal is to provide insightful answers based **strictly** on the provided context, acting as a bridge between raw data and human understanding.

**Core Directives**

1. **Fact First**  
   • Answer with facts that appear verbatim in the provided [CHUNK X] sections.  
   • Cite the source chunk for every fact, e.g. [CHUNK 1].  

2. **Permitted Inference (Linking Ideas)**  
   • After the facts, you may highlight logical connections supported by the text.  
   • Always introduce this step with phrases such as:  
     – “The documents suggest a potential link between …”  
     – “Based on the context, an implication is that …”  
     – “Connecting the information from these chunks suggests that …”  
   • This sign-posting keeps hard facts separate from analysis.

3. **Strict Boundaries (What NOT to do)**  
   • **DO NOT** use external knowledge; rely only on the context.  
   • **DO NOT** extrapolate beyond the dates in the text.  
   • **DO NOT** state an inference as a hard fact.  
   • If the information is missing, reply: “The provided documents do not contain this information.”  
   • Chunks whose `source_type` is **"market_data"** (e.g., Yahoo Finance snapshots) may be used **only** for share-price, dividend, or volume statistics, and must be cited like any other chunk.

4. **Hypothetical Outlook / Non-binding Analysis** (optional)  
   • Include this section **only if the user explicitly asks** for forward-looking views, price implications, valuation multiples, or strategic recommendations.  
   • Begin with:  
     ⚠️ This is a hypothetical scenario for educational purposes; it is not investment advice.  
   • Build simple ratios or projections **derived solely from the provided chunks**.  
   • If no market-data chunk is present, write:  
     “No price data provided; unable to compute valuation.”  
   • Prefix every assumption with **“Assumption:”** and every conclusion with **“Implication:”**.  
   • Use probabilistic language (“could”, “may”); never express certainty (“will”).

**Example of Good Output**

* **Fact:** “The company’s revenue was $50 billion [CHUNK 2].”  
* **Fact:** “Apple’s closing share price on 2024-03-29 was $185.56 [CHUNK 3].”  
* **Fact:** “The company also launched three new products in the same quarter [CHUNK 7].”  
* **Permitted Inference:** “The documents suggest a potential link between the three new products [CHUNK 7] and the reported revenue of $50 billion [CHUNK 2].”

**Hypothetical Outlook / Non-binding Analysis** (include only if the user explicitly requests forward-looking views)

⚠️ This is a hypothetical scenario for educational purposes; it is not investment advice.  
Assumption: Services revenue maintains a mid-teens growth rate.  
Assumption: Operating expenses grow no faster than 5 % YoY.  
Implication: If both assumptions hold, operating margin could expand by ~0.5 percentage points, which historically corresponded to a 4 – 6 % uplift in valuation multiples.  
No price data provided; unable to compute valuation.

"""

    # The user_prompt and rest of the function remain the same
    user_prompt = f"""{formatted_context}

---
Based **only** on the context provided above in the [CHUNK X] sections, provide a detailed, synthesized answer to the following user query. Follow all rules, especially the distinction between facts and permitted inferences.

**User's Query:** "{original_query}"
"""

    print("\n--- Sending final context to LLM for synthesis ---")
    
    try:
        chat_completion = llm_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=llm_model_name,
            temperature=0.2, # A slight increase to allow for more nuanced language
            max_tokens=1400,
            top_p=0.9 # Use top-p to allow for more diverse responses
        )
        print(f"LLM model loaded successfully ", llm_model_name)
        raw_answer = chat_completion.choices[0].message.content
        return raw_answer # We still return the raw answer with citations
    except Exception as e:
        print(f"An error occurred during final answer generation: {e}")
        return "Sorry, an error occurred while trying to generate the answer."

# Remember to keep using your post_process_answer function in the main script!
# final_answer = post_process_answer(raw_answer)


# In[ ]:


# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient, models
# import numpy as np, pprint, textwrap

# MODEL_NAME = "BAAI/bge-base-en-v1.5"
# COLLECTION  = "financial_sp500_local_final_v2"
# TOP_K       = 70

# # 1. Connect and load encoder
# enc = SentenceTransformer(MODEL_NAME, device="cpu")   # or "cuda"
# cli = QdrantClient(url="http://localhost:6333")

# # 2. Ad-hoc query
# query = "Tell me about Apple's Q2 2024? And can you give me some strategic and price recommendations? "

# # 3. Search with broad filter
# #use the functions from above
# results, modified_query = process_and_retrieve_with_augmentation(
#     query=query,
#     client=cli,
#     collection_name=COLLECTION,
#     query_model=enc,
#     min_results_k=TOP_K
# )

# #4. rerank the results
# final_results_for_generation = OrderedDict()
# RERANKER_PATH = "ms-marco-MiniLM-L-6-v2"

# reranker_model = CrossEncoder(
#     model_name_or_path=RERANKER_PATH,
#     device="cpu"  # or "cuda" if you have a GPU
# )
# for focus, candidates in results.items():
#     print(f"\nProcessing candidates for focus: {focus}")
    
#     # Call your new standalone reranking function
#     final_top_k_list = finalize_and_rerank_results(
#         query= query,
#         candidate_results=candidates,
#         reranker=reranker_model, # Your loaded model
#         final_k=7
#     )
#     final_results_for_generation[focus] = final_top_k_list

# #inject price if needed
# if needs_price_retrieval(query):
#     print("[market-data] User query triggers price lookup.")
#     for focus, doc_list in final_results_for_generation.items():
#         # We assume every QueryFocus has a .ticker attribute
#         ticker = focus.ticker
#         #fetch the date from the qeury                       f"({payload.get('year')}, Date: {payload.get('date')}) | "
#         #if payload has date use date for price retrieval
        
#         price_payload = fetch_price_chunk(ticker)
#         if price_payload:
#             price_res = make_price_search_result(price_payload)
#             # Push to front so it becomes CHUNK 1 or close to it
#             doc_list.insert(0, price_res)
#         else:
#             print(f"[market-data] No price data fetched for {ticker}.")
# else:
#     print("[market-data] Query contains no valuation keywords; skipping price fetch.")

# if final_results_for_generation:
#     final_answer = generate_final_answer(
#         original_query=query, # The user's original, full query
#         retrieved_results=final_results_for_generation,
#         llm_client=groq_client, # Your initialized Groq client
#         llm_model_name='llama-3.3-70b-versatile'
#     )
    
#     print("\n" + "="*50)
#     print("✅ FINAL GENERATED ANSWER")
#     print("="*50)
#     print(final_answer)
# else:
#     print("\n" + "="*50)
#     print("❌ No answer could be generated as no relevant documents were found.")
#     print("="*50)


# In[8]:


from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from collections import OrderedDict

# Load once
MODEL_NAME = "BAAI/bge-base-en-v1.5"
RERANKER_PATH = "D:/Licenta/ms-marco-MiniLM-L-6-v2"
COLLECTION  = "financial_sp500_local_final_v2"
TOP_K = 70

enc = SentenceTransformer(MODEL_NAME, device="cpu")
reranker_model = CrossEncoder(RERANKER_PATH, device="cuda") 
cli = QdrantClient(url="http://localhost:6333")


def remove_chunk_references(text: str) -> str:
    # Match anything like [CHUNK 1], [CHUNK 2-8], [CHUNK 1, CHUNK 4], etc.
    cleaned_text = re.sub(r"\[CHUNK[^\]]*\]", "", text)
    return cleaned_text.strip()

#classify intent

from enum import Enum, auto

class QueryIntent(Enum):
    # Needs to search Qdrant for document context.
    DOCUMENT_SEARCH = auto() 
    # Only needs to call yfinance for price/market data.
    MARKET_DATA_ONLY = auto()
    # Needs both documents and market data.
    HYBRID = auto()
    # A conversational or unanswerable query.
    GENERAL_KNOWLEDGE = auto()

def classify_query_intent(query: str, client: "Groq") -> QueryIntent:
    """
    Uses an LLM to classify the user's query into one of the defined intents.
    """
   # Enhanced document search keywords
    doc_search_keywords = [
        # Existing keywords
        "risk", "strategy", "revenue", "segment", "guidance", "outlook", "competition", 
        "management discussion", "MD&A", "10-K", "10-Q", "earnings call", "transcript",
        
        # Financial statement keywords
        "balance sheet", "income statement", "cash flow", "statement of operations",
        "comprehensive income", "shareholders equity", "retained earnings",
        
        # Business performance keywords
        "operating margin", "gross margin", "net income", "EBITDA", "operating expenses",
        "cost of goods sold", "research and development", "R&D", "sales and marketing",
        "general and administrative", "G&A", "working capital", "free cash flow",
        "business performance", "operational performance", 
        
        # Strategic and operational keywords
        "business model", "competitive advantage", "market share", "customer acquisition",
        "product development", "innovation", "digital transformation", "sustainability",
        "ESG", "corporate governance", "regulatory", "compliance", "litigation",
        
        # Risk and forward-looking keywords
        "risk factors", "headwinds", "tailwinds", "challenges", "opportunities",
        "future prospects", "expansion plans", "acquisitions", "divestitures",
        "restructuring", "cost reduction", "efficiency", "synergies",
        
        # Industry-specific keywords
        "supply chain", "manufacturing", "distribution", "logistics", "inventory",
        "backlog", "bookings", "pipeline", "funnel", "recurring revenue", "subscription",
        
        # Management and leadership
        "CEO commentary", "management team", "leadership changes", "succession planning",
        "board of directors", "executive compensation",
        
        # Document-specific terms
        "filing", "SEC", "annual report", "quarterly report", "proxy statement",
        "8-K", "press release", "investor presentation", "conference call transcript"
    ]
    
    # Enhanced market data keywords
    market_data_keywords = [
        # Price-related terms
        "current price", "stock price", "share price", "closing price", "opening price",
        "high", "low", "52-week high", "52-week low", "all-time high", "all-time low",
        "intraday", "after hours", "pre-market", "bid", "ask", "spread",
        
        # Valuation metrics
        "P/E", "price to earnings", "PEG ratio", "price to book", "P/B",
        "price to sales", "P/S", "EV/EBITDA", "enterprise value", "book value",
        "tangible book value", "price to cash flow", "price to free cash flow",
        
        # Financial ratios and metrics
        "return on equity", "ROE", "return on assets", "ROA", "return on investment", "ROI",
        "debt to equity", "current ratio", "quick ratio", "asset turnover",
        "inventory turnover", "receivables turnover",
        
        # Dividend and yield information
        "dividend yield", "dividend rate", "dividend per share", "DPS", "payout ratio",
        "dividend growth", "ex-dividend", "dividend history", "special dividend",
        
        # Trading and technical analysis
        "trading volume", "average volume", "market value", "float", "short interest",
        "institutional ownership", "insider ownership", "beta", "volatility",
        "moving average", "support", "resistance", "momentum",
        
        # Market-specific performance (when context is clearly price-based)
        "stock performance", "price performance", "market performance", "trading performance",
        "financial performance",
    ]
    
    # Keywords that specifically indicate HYBRID queries
    hybrid_keywords = [
        # MOVE GENERAL "PERFORMANCE" HERE
        "performance", "quarterly performance", "annual performance", "total return", 
        "price appreciation", "YTD", "year to date",
        
        # Price reaction to events
        "stock reaction", "price reaction", "market response", "investor reaction",
        "stock performance after", "price impact", "market impact",
        
        # Valuation in context
        "justified valuation", "fair value", "target price", "price target",
        "overvalued", "undervalued", "expensive", "cheap", "attractive valuation",
        
        # Investment thesis
        "investment case", "investment thesis", "buy recommendation", "sell recommendation",
        "hold recommendation", "rating", "analyst opinion", "price objective",
        
        # Event-driven analysis
        "earnings impact", "guidance impact", "announcement impact", "news impact",
        "catalyst", "event driven", "earnings surprise", "beat expectations",
        "miss expectations",
        
        # Comparison terms (these often need both data types)
        "vs peers", "vs index", "vs S&P 500", "vs sector", "relative performance",
        "outperform", "underperform", "benchmark",
        
        # Analysis that typically combines both
        "returns", "valuation", "market cap", "shares", "trading"
    ]
    
    # General knowledge/conversational keywords


    system_prompt = f"""
You are an expert query router for a financial analysis system. Your task is to analyze a user's query and classify it into one of three categories. Your goal is to determine what kind of information (documents, market data, or both) is needed to provide a complete answer.

First, consider the query in light of these keyword lists.

- **Document Search Keywords:** These suggest a need for information from official filings (10-K, 10-Q) or transcripts. Examples: {', '.join(doc_search_keywords[:20])}...
- **Market Data Keywords:** These suggest a need for stock prices, trading data, or valuation metrics from a real-time data feed. Examples: {', '.join(market_data_keywords[:20])}...
- **Hybrid Keywords:** These strongly suggest a need to connect document context with market data. Examples: {', '.join(hybrid_keywords[:15])}...

After your analysis, you MUST classify the query into one of the following three categories. Respond with ONLY the category name and nothing else.

**Categories:**

1.  **HYBRID**: The query's primary goal is to connect business fundamentals (from documents) with market performance (from data). It explicitly uses Hybrid keywords, or it clearly contains a mix of keywords from both the Document Search and Market Data lists.
    - Example: "Was Apple's valuation justified by its Q2 earnings report?"
    - Example: "How did the market react to Microsoft's new strategy announcement?"

2.  **DOCUMENT_SEARCH**: The query asks about a company's business, strategy, financial results, risks, or management commentary. It predominantly uses Document Search keywords and does NOT ask for specific stock prices or real-time trading data.
    - Example: "What were the main risk factors mentioned in Google's latest 10-K?"
    - Example: "Summarize the CEO's outlook from the last earnings call."

3.  **MARKET_DATA_ONLY**: The query ONLY asks for quantitative market data like stock prices (current or historical), trading volume, or standard valuation multiples (like P/E ratio). It does not ask "why" or seek qualitative context from a document.
    - Example: "What was the closing price of NVDA on May 1st, 2023?"
    - Example: "Show me the 52-week high and low for Tesla."
    - Example: "Who is the CEO of Microsoft?"
    - Example: "What industry is Amazon in?"

Analyze the user's query and return the single most appropriate category name.
"""
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct", # Use a fast model for this
            temperature=0.0
        )
        response = chat_completion.choices[0].message.content.strip()

        # Convert string response to Enum member
        if response == "DOCUMENT_SEARCH":
            return QueryIntent.DOCUMENT_SEARCH
        elif response == "MARKET_DATA_ONLY":
            return QueryIntent.MARKET_DATA_ONLY
        elif response == "HYBRID":
            return QueryIntent.HYBRID
        else:
            return QueryIntent.GENERAL_KNOWLEDGE

    except Exception as e:
        print(f"Error during intent classification: {e}. Defaulting to HYBRID.")
        # Default to the most comprehensive path on error
        return QueryIntent.HYBRID


def extract_date_from_query(query: str) -> str:
    """
    Extract a date from the query using dateutil for better parsing.
    Returns the first date found in YYYY-MM-DD format or empty string.
    """
    import re
    from dateutil import parser as date_parser
    from datetime import datetime
    
    # Common date patterns to look for
    date_patterns = [
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",  # YYYY-MM-DD or YYYY-M-D
        r"\b\d{4}/\d{1,2}/\d{1,2}\b",  # YYYY/MM/DD or YYYY/M/D
        r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b",  # MM-DD-YYYY or MM/DD/YYYY
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",  # Month DD, YYYY
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b",  # DD Month YYYY
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b",  # Month YYYY
    ]
    
    # Search for potential date strings
    for pattern in date_patterns:
        matches = re.finditer(pattern, query, re.IGNORECASE)
        for match in matches:
            date_text = match.group(0)
            try:
                # Use dateutil to parse the date
                parsed_date = date_parser.parse(date_text, fuzzy=False)
                return parsed_date.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                continue
    return ""
def extract_ticker_fast(query: str) -> List[str]:
    """Extract tickers using your existing comprehensive system."""
    focuses, _ = process_query_with_llm_refinement(query)
    return [focus.ticker for focus in focuses]






# def get_rag_response(query: str) -> str:

#     intent = classify_query_intent(query, groq_client)
#     if intent == QueryIntent.GENERAL_KNOWLEDGE:
#         return "This query is too general or conversational. Please ask a specific question about a company or financial topic."
#     elif intent == QueryIntent.MARKET_DATA_ONLY:
        
#         pass
#     elif intent == QueryIntent.DOCUMENT_SEARCH or intent == QueryIntent.HYBRID:
#         print(f"Query classified as {intent.name}. Proceeding with document search and retrieval.")

#     results, modified_query = process_and_retrieve_with_augmentation(
#         query=query,
#         client=cli,
#         collection_name=COLLECTION,
#         query_model=enc,
#         min_results_k=TOP_K
#     )

#     final_results_for_generation = OrderedDict()
#     for focus, candidates in results.items():
#         top_k = finalize_and_rerank_results(
#             query=query,
#             candidate_results=candidates,
#             reranker=reranker_model,
#             final_k=7
#         )
#         final_results_for_generation[focus] = top_k

#     if needs_price_retrieval(query):
#         for focus, doc_list in final_results_for_generation.items():
#             ticker = focus.ticker
#             #get date from the query or focus
#             # If the focus has a date, use it; otherwise, use today's date
#             if hasattr(focus, 'date') and focus.date:
#                 asof_date = focus.date
#             elif hasattr(focus, 'year') and focus.year:
#                 asof_date = f"{focus.year}-12-31"
#             else:
#                 asof_date = datetime.utcnow().strftime("%Y-%m-%d")
#             asof_date = datetime.strptime(asof_date, "%Y-%m-%d")

#             price_payload = fetch_price_chunk(ticker, asof=asof_date)
#             if price_payload:
#                 price_res = make_price_search_result(price_payload)
#                 doc_list.insert(0, price_res)

#     if final_results_for_generation:
#         response = generate_final_answer(
#             original_query=query,
#             retrieved_results=final_results_for_generation,
#             llm_client=groq_client,
#             llm_model_name='llama-3.3-70b-versatile'
#         )
#         response = remove_chunk_references(response)
#         return  response
#     else:
#         return "❌ No relevant information found."


# In[9]:

def process_market_data_only(query: str) -> str:
    """
    Handle queries that only need market data (no document search).
    Can handle multiple tickers, multiple dates, and various data requests.
    """
    import yfinance as yf
    from datetime import datetime, timedelta
    
    # Extract tickers and dates from query
    tickers = extract_ticker_fast(query)
    date_str = extract_date_from_query(query)
    
    if not tickers:
        return "I couldn't identify any stock tickers in your query. Please specify a company or ticker symbol."
    
    # Parse date if provided
    target_date = None
    if date_str:
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            pass
    
    # If no date specified, use current date
    if not target_date:
        target_date = datetime.now()
    
    results = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            
            # Determine what type of data to fetch based on query content
            query_lower = query.lower()
            
            if any(keyword in query_lower for keyword in ["ceo", "chief executive", "president"]):
                # Leadership information
                info = stock.info
                ceo = info.get('fullTimeEmployees', 'N/A')  # yfinance doesn't always have CEO
                results.append(f"**{ticker}**: CEO information not available via market data API. This requires document search.")
                
            elif any(keyword in query_lower for keyword in ["industry", "sector", "business"]):
                # Industry/sector information
                info = stock.info
                sector = info.get('sector', 'N/A')
                industry = info.get('industry', 'N/A')
                results.append(f"**{ticker}**: Sector: {sector}, Industry: {industry}")
                
            elif any(keyword in query_lower for keyword in ["price", "stock price", "closing", "current"]):
                # Price information
                if "historical" in query_lower or date_str:
                    # Historical price
                    hist = stock.history(start=target_date.date(), end=(target_date + timedelta(days=1)).date())
                    if not hist.empty:
                        close_price = hist['Close'].iloc[-1]
                        results.append(f"**{ticker}** on {target_date.strftime('%Y-%m-%d')}: ${close_price:.2f}")
                    else:
                        results.append(f"**{ticker}**: No price data available for {target_date.strftime('%Y-%m-%d')}")
                else:
                    # Current price
                    info = stock.info
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                    if current_price:
                        results.append(f"**{ticker}** current price: ${current_price:.2f}")
                    else:
                        results.append(f"**{ticker}**: Current price not available")
                        
            elif any(keyword in query_lower for keyword in ["dividend", "yield"]):
                # Dividend information
                info = stock.info
                div_yield = info.get('dividendYield', 0)
                div_rate = info.get('dividendRate', 0)
                if div_yield:
                    results.append(f"**{ticker}**: Dividend Yield: {div_yield*100:.2f}%, Annual Dividend: ${div_rate:.2f}")
                else:
                    results.append(f"**{ticker}**: No dividend information available")
                    
            elif any(keyword in query_lower for keyword in ["p/e", "pe ratio", "price to earnings"]):
                # P/E ratio
                info = stock.info
                pe_ratio = info.get('trailingPE')
                if pe_ratio:
                    results.append(f"**{ticker}**: P/E Ratio: {pe_ratio:.2f}")
                else:
                    results.append(f"**{ticker}**: P/E ratio not available")
                    
            elif any(keyword in query_lower for keyword in ["market cap", "market capitalization"]):
                # Market cap
                info = stock.info
                market_cap = info.get('marketCap')
                if market_cap:
                    results.append(f"**{ticker}**: Market Cap: ${market_cap:,.0f}")
                else:
                    results.append(f"**{ticker}**: Market cap not available")
                    
            elif any(keyword in query_lower for keyword in ["52 week", "52-week", "high", "low"]):
                # 52-week high/low
                info = stock.info
                week_52_high = info.get('fiftyTwoWeekHigh')
                week_52_low = info.get('fiftyTwoWeekLow')
                if week_52_high and week_52_low:
                    results.append(f"**{ticker}**: 52-week High: ${week_52_high:.2f}, Low: ${week_52_low:.2f}")
                else:
                    results.append(f"**{ticker}**: 52-week range not available")
            
            else:
                # General market data summary
                info = stock.info
                current_price = info.get('currentPrice') or info.get('regularMarketPrice', 'N/A')
                pe_ratio = info.get('trailingPE', 'N/A')
                market_cap = info.get('marketCap', 'N/A')
                
                summary_parts = [f"Price: ${current_price:.2f}" if isinstance(current_price, (int, float)) else f"Price: {current_price}"]
                if isinstance(pe_ratio, (int, float)):
                    summary_parts.append(f"P/E: {pe_ratio:.2f}")
                if isinstance(market_cap, (int, float)):
                    summary_parts.append(f"Market Cap: ${market_cap:,.0f}")
                    
                results.append(f"**{ticker}**: {', '.join(summary_parts)}")
                
        except Exception as e:
            results.append(f"**{ticker}**: Error fetching data - {str(e)}")
    
    if results:
        return "\n".join(results)
    else:
        return "Unable to fetch the requested market data."


def get_rag_response(query: str) -> str:
    """Updated main response function with better intent handling."""
    
    intent = classify_query_intent(query, groq_client)
    print(f"Query classified as: {intent.name}")
    
    if intent == QueryIntent.GENERAL_KNOWLEDGE:
        return "This query is too general or conversational. Please ask a specific question about a company or financial topic."
        
    elif intent == QueryIntent.MARKET_DATA_ONLY:
        # Handle pure market data queries
        return process_market_data_only(query)
        
    elif intent == QueryIntent.DOCUMENT_SEARCH:
        # Handle pure document search queries
        return process_document_search_only(query)
        
    elif intent == QueryIntent.HYBRID:
        # Handle queries that need both document and market data
        return process_hybrid_query(query)
    
    else:
        return "I couldn't determine how to process your query. Please try rephrasing."


def process_document_search_only(query: str) -> str:
    """Handle queries that only need document search."""
    results, modified_query = process_and_retrieve_with_augmentation(
        query=query,
        client=cli,
        collection_name=COLLECTION,
        query_model=enc,
        min_results_k=TOP_K
    )

    final_results_for_generation = OrderedDict()
    for focus, candidates in results.items():
        top_k = finalize_and_rerank_results(
            query=query,
            candidate_results=candidates,
            reranker=reranker_model,
            final_k=7
        )
        final_results_for_generation[focus] = top_k

    if final_results_for_generation:
        response = generate_final_answer(
            original_query=query,
            retrieved_results=final_results_for_generation,
            llm_client=groq_client,
            llm_model_name='llama-3.3-70b-versatile'
        )
        return remove_chunk_references(response)
    else:
        return "❌ No relevant documents found."


def process_hybrid_query(query: str) -> str:
    """Handle queries that need both document search and market data."""
    # First get document results
    results, modified_query = process_and_retrieve_with_augmentation(
        query=query,
        client=cli,
        collection_name=COLLECTION,
        query_model=enc,
        min_results_k=TOP_K
    )

    final_results_for_generation = OrderedDict()
    for focus, candidates in results.items():
        top_k = finalize_and_rerank_results(
            query=query,
            candidate_results=candidates,
            reranker=reranker_model,
            final_k=7
        )
        final_results_for_generation[focus] = top_k

    # Add market data if needed
    if needs_price_retrieval(query):
        for focus, doc_list in final_results_for_generation.items():
            ticker = focus.ticker
            
            # Determine date for price retrieval
            if hasattr(focus, 'date') and focus.date:
                asof_date = focus.date
            elif hasattr(focus, 'year') and focus.year:
                asof_date = f"{focus.year}-12-31"
            else:
                asof_date = datetime.utcnow().strftime("%Y-%m-%d")
            asof_date = datetime.strptime(asof_date, "%Y-%m-%d")

            price_payload = fetch_price_chunk(ticker, asof=asof_date)
            if price_payload:
                price_res = make_price_search_result(price_payload)
                doc_list.insert(0, price_res)

    if final_results_for_generation:
        response = generate_final_answer(
            original_query=query,
            retrieved_results=final_results_for_generation,
            llm_client=groq_client,
            llm_model_name='llama-3.3-70b-versatile'
        )
        return remove_chunk_references(response)
    else:
        return "❌ No relevant information found."

# query = "Who is the CEO of Microsoft?"

# response = get_rag_response(query)
# print(response)


# In[14]:


# # Example: Get some points that should be AAPL 2020 10-K
# sample_filter_fixed = models.Filter(
#     must=[
#         models.FieldCondition(key="ticker", match=models.MatchValue(value="AAPL")),
#         models.FieldCondition(key="year", match=models.MatchValue(value=2024)),
#         models.FieldCondition(key="source_type", match=models.MatchValue(value="sec_filing")),
#         models.FieldCondition(key="filing_category", match=models.MatchValue(value="10q"))

#     ]
# )
# # Then use this sample_filter_fixed with cli.scroll
# # ... (rest of your scroll code)
# try:
#     scroll_response = cli.scroll(
#         collection_name=COLLECTION,
#         scroll_filter=sample_filter_fixed,
#         limit=30,
#         with_payload=True
#     )
#     if scroll_response[0]: # scroll_response is a tuple (points, next_offset)
#         print("Sample matching points from Qdrant:")
#         for point in scroll_response[0]:
#             print(f"ID: {point.id}, Payload: {point.payload}")
#     else:
#         print("No points found in Qdrant matching the sample filter directly via scroll.")
# except Exception as e:
#     print(f"Error scrolling Qdrant: {e}")








