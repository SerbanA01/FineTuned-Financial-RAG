import asyncio
import sys
import os

# This boilerplate allows the script to be run from the 'scripts' directory
# and still import modules from the 'src' directory as if it were a package.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_ingestion.sec_downloader import run_pipeline
from src.utils.ticker_utils import get_sp500_tickers, get_nasdaq_tickers, get_other_tickers


async def main():
    """
    Entry point for the SEC filing download pipeline.

    This function orchestrates the data ingestion process by:
    1. Compiling a list of target company tickers from various sources.
    2. Invoking the main pipeline function to download the filings for these tickers.
    """
    # Aggregate tickers from major indices (S&P 500, NASDAQ) and a curated list
    # of other relevant companies to form a comprehensive download list.
    tickers = get_sp500_tickers() + get_nasdaq_tickers() + get_other_tickers()
    
    # Ensure each ticker is unique to avoid redundant downloads.
    TICKERS_TO_DOWNLOAD = list(set(tickers))
    
    print("--- Starting SEC Filing Download Pipeline ---")
    print(f"Target tickers: {', '.join(TICKERS_TO_DOWNLOAD)}")
    
    # Execute the core data ingestion pipeline.
    # The configuration specifies the number of historical filings to retrieve
    # for each type (10-K for annual, 10-Q for quarterly).
    await run_pipeline(
        tickers=TICKERS_TO_DOWNLOAD,
        max_10k=5,   # Download the 5 most recent annual reports.
        max_10q=15   # Download the 15 most recent quarterly reports.
    )

if __name__ == "__main__":
    # This block ensures the script's logic is executed only when the script
    # is run directly (not when imported as a module).
    try:
        # `asyncio.run()` is the standard way to start and manage an async event loop.
        asyncio.run(main())
    except KeyboardInterrupt:
        # Provides a clean exit path if the user manually stops the script (e.g., with Ctrl+C).
        print("Pipeline manually interrupted.")