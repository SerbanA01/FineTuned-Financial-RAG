import re
from .schemas import DocType, EntityType

# --- Mappings and Regular Expressions for Entity Extraction ---

# Maps common company names, variations, and colloquialisms to their official stock ticker.
# This allows for flexible user input, such as "apple", "google", or "mickey d's",
# and resolves them to a standardized identifier (e.g., AAPL, GOOGL, MCD).
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
    # Special case for a non-standard or internal ticker.
    "vision 2020": "V2020", "vision2020": "V2020", "v2020": "V2020",
}

# Invert the company-to-ticker map for reverse lookups.
# This is useful for displaying a canonical company name when only a ticker is known.
# The first name encountered for a given ticker becomes the canonical name.
TICKER_TO_COMPANY_HINTS = {}
for name, ticker in COMPANY_TO_TICKER_HINTS.items():
    if ticker not in TICKER_TO_COMPANY_HINTS:
        TICKER_TO_COMPANY_HINTS[ticker] = name.title()

# Manual overrides to ensure specific tickers map to their most appropriate canonical name,
# especially in cases where multiple names point to the same ticker in the original map.
if "BRK.A" not in TICKER_TO_COMPANY_HINTS and "berkshire hathaway a" in COMPANY_TO_TICKER_HINTS:
     TICKER_TO_COMPANY_HINTS["BRK.A"] = COMPANY_TO_TICKER_HINTS["berkshire hathaway a"].title()
if "V2020" not in TICKER_TO_COMPANY_HINTS and "vision 2020" in COMPANY_TO_TICKER_HINTS:
     TICKER_TO_COMPANY_HINTS["V2020"] = COMPANY_TO_TICKER_HINTS["vision 2020"].title()


# Regex to identify potential stock tickers. It looks for 1-5 uppercase letters,
# optionally followed by a dot and another letter (e.g., 'BRK.B').
# The `\b` word boundaries prevent matching parts of other words.
TICKER_REGEX = re.compile(r'\b([A-Z]{1,5}(\.[A-Z])?)\b')

# Regex to identify a four-digit year, constrained to a reasonable range for financial data (1970-2099).
YEAR_REGEX = re.compile(r'\b(19[7-9]\d|20\d{2})\b')

# Regex to identify a financial quarter, accepting various formats like:
# - "Q1", "Q2", etc.
# - "Quarter 1", "Quarter 2", etc.
# - "1st Quarter", "2nd Quarter", etc.
# - "first Quarter", "second Quarter", etc.
QUARTER_REGEX = re.compile(
    r'\b(?:Q([1-4])|'
    r'(?:Quarter\s*([1-4]))|'
    r'(1st|2nd|3rd|4th)\s*Quarter|'
    r'(first|second|third|fourth)\s*Quarter)\b',
    re.IGNORECASE
)

# Helper map to normalize written-out quarter identifiers to their numeric equivalent.
QUARTER_WORD_TO_NUM = {
    "1st": 1, "first": 1, "2nd": 2, "second": 2,
    "3rd": 3, "third": 3, "4th": 4, "fourth": 4,
}

# Maps various keywords and phrases for financial documents to a standardized DocType enum.
# This allows the system to understand different user queries for the same document type.
DOC_KEYWORD_TO_NORMALIZED_MAP = {
    # 10-K (Annual Report) variations
    "10-k": DocType.K10, "10k": DocType.K10,
    "10-K": DocType.K10, "10-K filing": DocType.K10,
    "annual report": DocType.K10, "annual filing": DocType.K10,
    # 10-Q (Quarterly Report) variations
    "10-q": DocType.Q10, "10q": DocType.Q10,
    "10-Q": DocType.Q10, "10-Q filing": DocType.Q10,
    "quarterly report": DocType.Q10, "quarterly filing": DocType.Q10,
    # Earnings Transcript variations
    "earnings transcript": DocType.EARNINGS_TRANSCRIPT,
    "earnings call": DocType.EARNINGS_TRANSCRIPT,
    "financial call": DocType.EARNINGS_TRANSCRIPT,
    "investor call": DocType.EARNINGS_TRANSCRIPT,
    "conference call": DocType.EARNINGS_TRANSCRIPT,
    "transcript": DocType.EARNINGS_TRANSCRIPT,
    "earnings report": DocType.EARNINGS_TRANSCRIPT, # "Earnings report" is often used to mean the transcript or call.
}

# A single, efficient regex compiled from all the keys in the document keyword map.
# This is used to quickly scan text for any mention of a supported document type.
DOC_TYPE_KEYWORDS_REGEX = re.compile(
    r'\b(?:' + '|'.join(re.escape(k) for k in DOC_KEYWORD_TO_NORMALIZED_MAP.keys()) + r')\b',
    re.IGNORECASE
)

# Defines the processing priority for different entity types. A lower number indicates a higher priority.
# This is used to resolve ambiguities during parsing. For example, a Ticker is a more
# specific and higher-priority entity than a general Company name.
ENTITY_PRIORITIES = {
    EntityType.TICKER: 0,
    EntityType.YEAR: 1,
    EntityType.QUARTER: 2,
    EntityType.DOC_TYPE_KEYWORD: 3,
    EntityType.COMPANY: 4}