import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO
import investpy

def get_sp500_tickers():
    """
    Fetches the list of S&P 500 tickers by scraping Wikipedia.

    This function sends a request to the Wikipedia page for the List of S&P 500
    companies, parses the HTML to find the constituents table, and extracts
    the ticker symbols.

    @return: A list of S&P 500 stock ticker symbols. Returns an empty list on failure.
    @rtype: list
    """
    # The URL for the Wikipedia page containing the S&P 500 company list.
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # The table containing the ticker symbols has a specific ID 'constituents'.
    table = soup.find('table', {'id': 'constituents'})
    # pandas.read_html can directly parse an HTML table into a DataFrame.
    df = pd.read_html(StringIO(str(table)))[0]
    return df['Symbol'].tolist()

def get_nasdaq_tickers():
    """
    Fetches a list of all NASDAQ-listed common stocks from the official source.

    This function downloads a pipe-delimited text file from the NASDAQ Trader
    website, which contains a directory of all listed securities. It then filters
    this list to exclude non-common-stock assets like ETFs and test issues.

    @return: A list of NASDAQ stock ticker symbols. Returns an empty list on failure.
    @rtype: list
    """
    # This URL points to a raw text file provided by NASDAQ for all listed securities.
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    columns = [
        'Symbol', 'Security Name', 'Market Category', 'Test Issue',
        'Financial Status', 'Round Lot Size', 'ETF', 'NextShares'
    ]
    df = pd.read_csv(url, sep="|", names=columns)
    df = df[:-1]  # The last row is a summary/footer row that needs to be removed.
    # We filter for securities that are not test issues and not ETFs to get a cleaner list of common stocks.
    df = df[(df['Test Issue'] == 'N') & (df['ETF'] == 'N')]
    return df['Symbol'].tolist()

def get_other_tickers():
    """
    Fetches a list of NYSE, AMEX, and other exchange-listed stocks.

    Similar to the NASDAQ function, this retrieves a file from the NASDAQ Trader
    website that lists securities from other major exchanges. It applies the
    same filtering to remove ETFs and test issues.

    @return: A list of NYSE/AMEX stock ticker symbols. Returns an empty list on failure.
    @rtype: list
    """
    # This URL points to the directory for non-NASDAQ listed securities.
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    columns = [
        'ACT Symbol', 'Security Name', 'Exchange', 'CQS Symbol',
        'ETF', 'Round Lot Size', 'Test Issue', 'NASDAQ Symbol'
    ]
    df = pd.read_csv(url, sep="|", names=columns)
    df = df[:-1]  # Remove the summary footer row.
    # Filter out non-common-stock assets.
    df = df[(df['Test Issue'] == 'N') & (df['ETF'] == 'N')]
    return df['ACT Symbol'].tolist()

def get_global_tickers():
    """
    Fetches a list of global stock tickers using the 'investpy' library.

    This function provides a broad list of international stock tickers. It includes
    error handling to manage cases where the `investpy` library is not installed
    or when the external data source is unavailable.

    @return: A list of global stock ticker symbols. Returns an empty list if
             `investpy` is not installed or if an error occurs.
    @rtype: list
    """
    try:
        stocks_df = investpy.stocks.get_stocks()
        # Further refine the list to only include assets explicitly typed as 'Stock'.
        if 'type' in stocks_df.columns:
            stocks_df = stocks_df[stocks_df['type'] == 'Stock']
        return stocks_df['symbol'].unique().tolist()
    except ImportError:
        # Gracefully fail if the optional dependency isn't present.
        print("Global tickers disabled: install investpy (pip install investpy)")
        return []
    except Exception as e:
        # Catch other potential issues like network errors or API changes.
        print(f"Error fetching global tickers: {str(e)}")
        return []

def get_all_tickers():
    """
    Aggregates tickers from all available sources into a single, clean list.

    This function acts as the main entry point, calling each specialized ticker-fetching
    function. It handles errors from any individual source gracefully, ensuring that
    a failure in one does not prevent others from succeeding. The final combined
    list is then cleaned of any duplicates and sorted.

    @return: A sorted list of unique stock ticker symbols from all sources.
    @rtype: list
    """
    tickers = []

    # Each fetch operation is wrapped in a try/except block to make the process resilient.
    try:
        tickers += get_sp500_tickers()
        print(f"Found {len(tickers)} S&P 500 tickers")
    except Exception as e:
        print(f"Error fetching S&P 500: {str(e)}")

    try:
        nasdaq = get_nasdaq_tickers()
        tickers += nasdaq
        print(f"Added {len(nasdaq)} NASDAQ tickers")
    except Exception as e:
        print(f"Error fetching NASDAQ: {str(e)}")

    try:
        other = get_other_tickers()
        tickers += other
        print(f"Added {len(other)} NYSE/AMEX tickers")
    except Exception as e:
        print(f"Error fetching NYSE/AMEX: {str(e)}")

    try:
        global_tickers = get_global_tickers()
        tickers += global_tickers
        print(f"Added {len(global_tickers)} global tickers")
    except Exception as e:
        print(f"Error fetching global tickers: {str(e)}")

    # Final cleanup: ensure all tickers are stripped of whitespace, uppercased for consistency,
    # and that the final list contains only unique entries.
    clean_tickers = list(set([t.strip().upper() for t in tickers if isinstance(t, str)]))
    return sorted(clean_tickers)