import yfinance as yf
from datetime import datetime, timedelta
from qdrant_client.models import ScoredPoint
import uuid

from .process import SearchResult

# A list of keywords used to determine if a user's query is asking for financial market data,
# such as stock prices or valuation metrics. This helps route the query appropriately.
PRICE_KEYWORDS = [
    "price", "current price", "stock price", "share price", "trading price",
    "closing price", "last price", "latest price", "market price",
    "current value", "stock value", "share value", "valuation", "pe ratio", "p/e", "dividend yield",
    "market cap", "eps multiple", "target price"
]

def needs_price_retrieval(query: str) -> bool:
    """
    Checks if the user's query contains any of the predefined price-related keywords.

    @param query: The raw user query string.
    @return: True if a price-related keyword is found, False otherwise.
    """
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in PRICE_KEYWORDS)

def fetch_price_chunk(ticker: str,
                      asof: datetime | None = None) -> dict | None:
    """
    Fetches the last closing price and dividend for a given ticker from Yahoo Finance.

    This function acts as an adapter. It retrieves external market data and formats it
    into a dictionary that mimics the structure of a document chunk retrieved from
    the vector database. This allows market data to be seamlessly integrated into the
    same context-building pipeline as document text for the final LLM answer generation.

    @param ticker: The stock ticker symbol to look up (e.g., "AAPL").
    @param asof: The date for which to fetch the data. Defaults to the current time.
    @return: A dictionary structured like a document payload, or None if the fetch fails.
    """
    try:
        stock = yf.Ticker(ticker)
        if asof is None:
            asof = datetime.utcnow()
        # Fetch a minimal history to get the last available closing price up to the 'asof' date.
        hist = stock.history(start=asof.date(), end=(asof.date() + timedelta(days=1)))
        last_close = float(hist['Close'][-1])  # Raises IndexError if no data is returned.
        dividend = float(stock.info.get("dividendRate") or 0.0)

        # The payload is structured with keys like 'source_type' and 'chunk_text'
        # to match the schema of documents stored in the vector DB. This consistency
        # is key for the downstream processing steps.
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
        # Gracefully handle API failures or cases where the ticker is invalid.
        print(f"[market-data] fetch failed for {ticker}: {exc}")
        return None

def make_price_search_result(price_payload: dict) -> SearchResult:
    """
    Wraps a price data payload dictionary into a standard SearchResult object.

    This function ensures that market data can be treated identically to document
    search results by the rest of the application. It creates a "dummy" ScoredPoint
    since the data doesn't come from a vector search and thus has no inherent score.

    @param price_payload: The dictionary returned by `fetch_price_chunk`.
    @return: A SearchResult object containing the price data.
    """
    # Create a placeholder ScoredPoint. The ID is random, and the score is irrelevant
    # because this result will bypass the reranking stage.
    dummy_point = ScoredPoint(
        id=str(uuid.uuid4()),
        score=1.0,
        payload=price_payload,
        vector=None,
        version=0
    )
    return SearchResult(dummy_point, tier="Market Data")