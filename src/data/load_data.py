from typing import Dict
import pandas as pd
import yaml

from src.data.preprocessing import read_data

def load_config(conf_path: str) -> Dict[str, str]:
    with open(conf_path, 'r') as f:
        best_stocks = yaml.load(f, Loader=yaml.FullLoader)
    return best_stocks

def data_loading(
    ticker_data_path: str, 
    best_stocks_path: str, 
    filter_best=True
) -> pd.DataFrame:
    df = read_data(ticker_data_path)
    if filter_best:
        best_stocks = load_config(best_stocks_path)
        df = df.query("Stock in @best_stocks")
    return df

def general_preprocessing(
    df, 
    agg_freq: str='', 
    X_col_agg_finctions={'Close': 'last'},
) -> pd.DataFrame:

    if agg_freq:
        df = df.set_index('Datetime').groupby(
            ['Stock', pd.Grouper(freq=agg_freq)],
        ).agg(X_col_agg_finctions)

    return df

def pipeline_data(
    ticker_data_path: str = 'data/all_tickers.csv', 
    best_stocks_path: str = 'configs/best_stocks_nans_rate.yaml', 
    filter_best: bool = True,
    agg_freq: str = 'h',
    col_agg_finctions: Dict[str, str] = {'Close': 'last'},
):
    df = data_loading(ticker_data_path, best_stocks_path, filter_best)
    data = general_preprocessing(
        df,
        agg_freq, 
        col_agg_finctions,
    )

    return data