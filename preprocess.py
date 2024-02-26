import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import pandas as pd
import yaml

from src.data.preprocessing import read_data, preprocessing, data_to_np_tensor


@hydra.main(config_path='configs', config_name='preprocess_config', version_base=None)
def main(cfg: DictConfig):
    
    df = read_data('data/all_tickers.csv')
    
    if cfg['stocks']=='best':
        with open('configs/best_stocks_nans_rate.yaml') as f:
            stocks = yaml.load(f, Loader=yaml.FullLoader)
        stocks = list(stocks.keys())
    
    elif cfg['stocks'] != None:
        stocks = cfg['stocks']
    
    else:
        stocks = df['Stock'].unique()
    
    df = df.query("Stock in @stocks")
    
    df_agg = df.set_index('Datetime').groupby(
    ['Stock', pd.Grouper( freq=cfg['frequency'])],
).agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': "sum",
}).reset_index()
    
    train_start = cfg['train_start'] if cfg['train_start'] else df['Datetime'].sort_values().unique()[0]
    train_end = cfg['train_end'] if cfg['train_start'] else cfg['test_start']
    
    test_start = cfg['test_start'] if cfg['train_start'] else cfg['train_end']
    test_end =  cfg['test_end'] if cfg['train_start'] else df['Datetime'].sort_values().unique()[-1]

    if test_start == None and train_end == None:
        dates = pd.date_range(train_start, test_end)
        test_start = train_end = dates[ round( cfg['split'] * len(dates) ) ]
    
    train_data = preprocessing(
    df_agg, 
    cfg['features'],         
    start_date = train_start,
    end_date = train_end,
    tickers_save = stocks
)
    
    test_data = preprocessing(
    df_agg, 
    cfg['features'],          
    start_date = test_start,
    end_date = test_end,
    tickers_save = stocks
)
    print(test_data)
    return test_data, train_data

if __name__ == "__main__":
    main()