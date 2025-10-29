from typing import Optional

class DocType:
    """
    Defines standardized string constants for different types of financial documents.
    This ensures consistency throughout the system when referring to a specific document.
    """
    K10 = "10-K"
    Q10 = "10-Q"
    EARNINGS_TRANSCRIPT = "Earnings Transcript"

class EntityType:
    """
    Defines string constants for the various types of entities that can be
    extracted from a user's query during the parsing process.
    """
    TICKER = "ticker"
    COMPANY = "company"
    YEAR = "year"
    QUARTER = "quarter"
    DOC_TYPE_KEYWORD = "doc_type_keyword" # The specific keyword found, e.g., "annual report"

class QueryFocus:
    """
    Represents a single, structured request for a specific financial document.

    This class encapsulates all the key pieces of information (ticker, year, quarter,
    and document type) needed to retrieve a document. It serves as the primary data
    structure for passing parsed query information between different parts of the system.

    Attributes:
        ticker (str): The stock ticker symbol for the company.
        year (Optional[int]): The financial year of the document.
        quarter (Optional[int]): The financial quarter (1-4) of the document.
        doc_type (Optional[str]): The normalized document type, matching a value
                                  from the DocType class.
    """
    def __init__(self, ticker: str, year: Optional[int] = None, quarter: Optional[int] = None, doc_type: Optional[str] = None):
        self.ticker = ticker
        self.year = year
        self.quarter = quarter
        self.doc_type = doc_type # This holds the NORMALIZED doc_type (e.g., "10-K").

    def __repr__(self) -> str:
        """Provides a developer-friendly string representation for debugging."""
        return f"QueryFocus(ticker='{self.ticker}', year={self.year}, quarter={self.quarter}, doc_type='{self.doc_type}')"

    def __eq__(self, other) -> bool:
        """Enables value-based comparison between two QueryFocus objects."""
        return isinstance(other, QueryFocus) and self.ticker == other.ticker and \
               self.year == other.year and self.quarter == other.quarter and self.doc_type == other.doc_type

    def __hash__(self) -> int:
        """Allows QueryFocus objects to be used in hash-based collections like sets and dictionary keys."""
        return hash((self.ticker, self.year, self.quarter, self.doc_type))