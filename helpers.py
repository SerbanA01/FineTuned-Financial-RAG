#maybe consider FTSE100
#don t forget about cryptos and forex



import requests
import pandas as pd
from bs4 import BeautifulSoup # For parsing HTML (used in get_sp500_tickers)
from io import StringIO     # For reading string as file (used in get_sp500_tickers)
import investpy             # For fetching global stock data (used in get_global_tickers)





def get_sp500_tickers():
    """Get S&P 500 tickers with more reliable table parsing"""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    df = pd.read_html(StringIO(str(table)))[0]
    return df['Symbol'].tolist()

def get_nasdaq_tickers():
    """Get NASDAQ-listed common stocks with proper column handling"""
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    columns = [
        'Symbol', 'Security Name', 'Market Category', 'Test Issue',
        'Financial Status', 'Round Lot Size', 'ETF', 'NextShares'
    ]
    df = pd.read_csv(url, sep="|", names=columns)
    df = df[:-1]  # Remove summary row
    # Filter out ETFs and test issues
    df = df[(df['Test Issue'] == 'N') & (df['ETF'] == 'N')]
    return df['Symbol'].tolist()

def get_other_tickers():
    """Get NYSE/AMEX stocks with proper column handling"""
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    columns = [
        'ACT Symbol', 'Security Name', 'Exchange', 'CQS Symbol',
        'ETF', 'Round Lot Size', 'Test Issue', 'NASDAQ Symbol'
    ]
    df = pd.read_csv(url, sep="|", names=columns)
    df = df[:-1]  # Remove summary row
    # Filter out ETFs and test issues
    df = df[(df['Test Issue'] == 'N') & (df['ETF'] == 'N')]
    return df['ACT Symbol'].tolist()

def get_global_tickers():
    """Get global tickers with error handling"""
    try:
        stocks_df = investpy.stocks.get_stocks()
        # Filter for common stock types (adjust based on available data)
        if 'type' in stocks_df.columns:
            stocks_df = stocks_df[stocks_df['type'] == 'Stock']
        return stocks_df['symbol'].unique().tolist()
    except ImportError:
        print("Global tickers disabled: install investpy (pip install investpy)")
        return []
    except Exception as e:
        print(f"Error fetching global tickers: {str(e)}")
        return []

def get_all_tickers():
    """Combine all sources with error handling"""
    tickers = []

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

    # Clean and deduplicate
    clean_tickers = list(set([t.strip().upper() for t in tickers if isinstance(t, str)]))
    return sorted(clean_tickers)



#