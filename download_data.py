import pandas as pd
from tqdm import tqdm

from src.finam_download import get

def get_tickers_sp500():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    sp500_df = table[0]

    # Extract the tickers column
    tickers = sp500_df['Symbol'].tolist()
    return tickers
    
def get_all_sp500_data():
    sp500_tickers = get_tickers_sp500()
    failed_list = []
    for ticker in tqdm(sp500_tickers):
        try:
            get(ticker, 'M1', f'data/ticker_data/{ticker}.csv')

        except:
            print(f'Error witn ticker {ticker}')
            failed_list.append(ticker)

    print(len(failed_list), failed_list)

if __name__ == '__main__':
    get_all_sp500_data()