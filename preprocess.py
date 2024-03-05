import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import pandas as pd
import numpy as np
import yaml

from src.data.preprocessing import read_data, preprocessing, data_to_np_tensor, save_config, stocks_df, train_test_split

CONFIG_NAME = 'preprocess_config'


@hydra.main(config_path='configs', config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):
    
    df = stocks_df(cfg['data_path'], cfg['stocks'])

    df_agg = df.set_index('Datetime').groupby(['Stock', pd.Grouper(freq=cfg['frequency'])],).agg( dict( cfg['features'] ) )
    df_agg = df_agg.groupby('Stock').pct_change().reset_index() if cfg['pct_change'] else df.reset_index()

    train_start, train_end, test_start, test_end = train_test_split(df_agg, cfg['train_start'], cfg['train_end'], cfg['test_start'], cfg['test_end'], cfg['split'])

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
        df['Stock'].unique().tofile('data/stocks.csv', sep=';')
        data_to_np_tensor(train_data).tofile('data/train.csv', sep=';')
        data_to_np_tensor(test_data).tofile('data/test.csv', sep=';')
        save_config(CONFIG_NAME)

    return train_data, test_data


if __name__ == "__main__":
    main()
