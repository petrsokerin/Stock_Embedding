import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import pandas as pd
import numpy as np
import yaml

from src.data.preprocessing import read_data, preprocessing, data_to_np_tensor


@hydra.main(config_path='configs', config_name='preprocess_config', version_base=None)
def main(cfg: DictConfig):
    
    with open('configs/best_stocks_nans_rate.yaml') as f:
        best_stocks = yaml.load(f, Loader=yaml.FullLoader)

    best_stocks = list(best_stocks.keys())
    
    df = read_data('data/all_tickers.csv')
    
    if cfg['best_stocks']:
        tmp = df.copy(deep=True)
        df = df.query("Stock in @best_stocks")
    
    df_agg = df.set_index('Datetime').groupby(
    ['Stock', pd.Grouper( freq=cfg['frequency'])],
).agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': "sum",
}).reset_index()
    
    train_start = cfg['train_start'] if cfg['train_start'] else data
    train_end = cfg['train_end'] if cfg['train_start'] else data
    
    test_start = cfg['test_start'] if cfg['train_start'] else data
    test_end =  cfg['test_end'] if cfg['train_start'] else data
    
    train_data = preprocessing(
    df_agg, 
    cfg['features'],         
    start_date = train_start,
    end_date = train_end,
    tickers_save = best_stocks
)
    
    test_data = preprocessing(
    df_agg, 
    cfg['features'],          
    start_date = test_start,
    end_date = test_end,
    tickers_save = best_stocks
)