import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import pandas as pd
import numpy as np
import yaml

from src.data.preprocessing import read_data, preprocessing, data_to_np_tensor, save_config

CONFIG_NAME = 'preprocess_config'


@hydra.main(config_path='configs', config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):

    df = read_data(cfg['data_path'])

    if cfg['stocks'] == 'best':
        with open('configs/best_stocks_nans_rate.yaml') as f:
            stocks = yaml.load(f, Loader=yaml.FullLoader)

        stocks = list(stocks.keys())

    elif cfg['stocks'] != None:
        stocks = cfg['stocks']

    else:
        stocks = df['Stock'].unique()

    df = df.query("Stock in @stocks")

    df_agg = df.set_index('Datetime').groupby(['Stock', pd.Grouper(freq=cfg['frequency'])]).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': "sum",
    })

    df_agg = df_agg.groupby('Stock').pct_change(
    ).reset_index() if cfg['pct_change'] else df.reset_index()

    train_start = cfg['train_start'] if cfg['train_start'] else df['Datetime'].min()
    train_end = cfg['train_end'] if cfg['train_start'] else cfg['test_start']

    test_start = cfg['test_start'] if cfg['train_start'] else cfg['train_end']
    test_end = cfg['test_end'] if cfg['train_start'] else df['Datetime'].max()

    if test_start == None and train_end == None:
        dates = pd.date_range(train_start, test_end)
        test_start = train_end = dates[round(cfg['split'] * len(dates))]

    train_data = preprocessing(
        df_agg,
        cfg['features'],
        start_date=train_start,
        end_date=train_end,
        tickers_save=stocks
    )

    test_data = preprocessing(
        df_agg,
        cfg['features'],
        start_date=test_start,
        end_date=test_end,
        tickers_save=stocks
    )

    if cfg['save']:
        df['Stock'].unique().tofile('data_prep/stocks.csv', sep=';')
        data_to_np_tensor(train_data).tofile('data_prep/train.csv', sep=';')
        data_to_np_tensor(test_data).tofile('data_prep/test.csv', sep=';')
        save_config(CONFIG_NAME)

    return train_data, test_data


if __name__ == "__main__":
    main()
