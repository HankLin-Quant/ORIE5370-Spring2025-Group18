
import yfinance as yf
import pandas as pd
import os


def get_sp500_adjusted_close(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieve daily adjusted closing prices for all S&P 500 constituents 
    between the given start and end dates using yfinance.
    Data is cached locally in /data folder for efficiency.

    Parameters:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame with dates as index and tickers as columns, 
                      containing adjusted closing prices.
    """
    # Define cache path
    os.makedirs("data", exist_ok=True)
    cache_path = f"data/sp500_adjclose_{start_date}_to_{end_date}.csv"

    # If cache exists, load and return
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    print("Cache not found. Downloading data from yfinance...")

    # Get the latest list of S&P 500 constituents from Wikipedia
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    sp500_table = table[0]
    tickers = sp500_table['Symbol'].tolist()
    
    # Replace dot with dash for yfinance compatibility (e.g., BRK.B -> BRK-B)
    tickers = [ticker.replace('.', '-') for ticker in tickers]

    # Download data
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=False,
        group_by='ticker',
        auto_adjust=False,
        threads=True
    )

    # Extract only adjusted close prices
    adj_close_df = pd.DataFrame({
        ticker: data[ticker]['Adj Close']
        for ticker in tickers
        if (ticker in data.columns.get_level_values(0)) and ('Adj Close' in data[ticker])
    })

    # Save to cache
    adj_close_df.to_csv(cache_path)
    print(f"Data saved to {cache_path}")

    return adj_close_df

def feature_compare(result_df, feature):
    ''' 
    This function converts the long data format into a wide format, using the specified feature as columns, 
    other settings combined as the row index, and strategy_variance as the values.
    This allows us to observe how changes in the target feature affect strategy_variance.
    
    Parameters:
        result_df (pd.DataFrame): The input DataFrame containing 'strategy_variance' and various features.
        feature (str): The column name of the feature to compare.
    
    Returns:
        pd.DataFrame: A pivoted DataFrame with the target feature as columns.
    '''
    # Extract all columns except 'strategy_variance'
    all_cols = list(result_df.columns)
    features = [col for col in all_cols if col != 'strategy_variance']

    # Set the index to all features except the one to compare
    index_cols = [col for col in features if col != feature]

    # Pivot the table to wide format
    pivot_df = result_df.pivot_table(
        index=index_cols,
        columns=feature,
        values='strategy_variance'
    )

    wins_count=pivot_df.idxmin(axis=1).value_counts()

    result={
        'pivot_df':pivot_df,
        'wins_count':wins_count
    }

    return result