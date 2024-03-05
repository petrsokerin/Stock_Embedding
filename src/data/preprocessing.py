import os
import shutil
import yaml

from typing import NoReturn, List, Dict

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

from src.models.ts2vec_src.ts2vec import TS2Vec


COLUMNS_FINAM = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
TYPES_FINAM = ['str'] * 2 + ['float64'] * 4 + ['int']


def drop_csv_empty_files(path: str = 'data/ticker_data') -> NoReturn:

    all_stock_files = sorted(os.listdir(path))              
    for file in tqdm(all_stock_files):
        try:
            dataset = pd.read_csv(f'{path}/{file}')
        except:
            os.remove(f'{path}/{file}')


def collect_data_in_one_file(
        tickers_path: str = 'data/ticker_data', 
        save_path: str = 'data/all_tickers.csv',
) -> NoReturn:
    df = pd.DataFrame()
    all_stock_files = sorted(os.listdir(tickers_path)) 
    dtype = dict(zip(range(len(TYPES_FINAM)), TYPES_FINAM))

    for file in tqdm(all_stock_files):
        try:
            dataset = pd.read_csv(f'{tickers_path}/{file}', header=None, dtype=dtype)
            dataset.columns = COLUMNS_FINAM

            dataset['Datetime'] = dataset['Date'].astype('str') + '-' + dataset['Time'].astype('str')
            dataset['Datetime'] = pd.to_datetime(dataset['Datetime'], format='%Y%m%d-%H%M%S')
            ticker = file.replace('.csv', '')
            dataset['Stock'] = ticker

            df = pd.concat([df, dataset])
        except:
            print(f"Can't read file {file}")

    df.to_csv(save_path)


def read_data(path: str, clean: bool=True) -> pd.DataFrame:
    df = pd.read_csv(path, index_col = 0)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    df['Date'] = df['Datetime'].dt.date
    df['Day_week'] = df['Datetime'].dt.day_name()
    df['Time'] = df['Datetime'].dt.time

    if clean:
        df = df[df['Day_week'] != 'Saturday']
        df = df[
            (df['Time'] >= pd.to_datetime('17:30:00').time()) & \
            (df['Time'] < pd.to_datetime('22:59:00').time())
        ]
    return df


def feature_preprocessing(
        data: pd.DataFrame,
        value_col: str = 'Close',
        nan_thrs: float = 0.3,
        tickers_save: List[str] = [],
) -> pd.DataFrame:

    df_tickers = data.pivot(index='Datetime', columns='Stock', values=value_col)

    if len(tickers_save) == 0:
        nan_df = df_tickers.isna().astype(int).mean().to_frame(name='nan_ratio')
        tickers_save = nan_df[nan_df['nan_ratio'] < nan_thrs].index 
        print('N_saved_stocks ', len(tickers_save))

    df_tickers = df_tickers[tickers_save]

    df_tickers = df_tickers.interpolate(method='linear', limit_direction='both')
    print('NAN in data ', df_tickers.isna().sum().sum())

    return df_tickers


def preprocessing(
        data: pd.DataFrame,
        value_columns: List[str]= 'Close',
        start_date: str = '2023-12-20',
        end_date: str = '2023-12-31',
        nan_thrs: float = 0.3,
        tickers_save: List[str] = [],
) -> Dict[str, pd.DataFrame]:
    
    mask_lower = data['Datetime'].dt.date >= pd.Timestamp(start_date).date()
    mask_upper = data['Datetime'].dt.date < pd.Timestamp(end_date).date()
    mask_tickers = data['Stock'].isin(tickers_save)
    data = data[(mask_lower) & (mask_upper) & (mask_tickers)]
    
    res = dict()

    for value_col in value_columns:
        res[value_col] = feature_preprocessing(data, value_col, nan_thrs, tickers_save)

    return res


def data_to_np_tensor(data: Dict[str, pd.DataFrame]) -> np.ndarray:
    features_names = list(data.keys())
    features_one = features_names[0]

    n_objects = data[features_one].shape[1]
    seq_len = data[features_one].shape[0]
    n_features = len(data)

    res = np.zeros((n_objects, seq_len, n_features))
    for i, features in enumerate(data.values()):
        res[:, :, i] = features.values.T

    return res


def train_test_split(
        df: pd.DataFrame,
        train_start_date: str, 
        train_end_date: str, 
        test_start_date: str, 
        test_end_date: str,
        split:float,
    ):
    
    train_start = train_start_date if train_start_date else df['Datetime'].min()
    train_end = train_end_date if train_end_date else test_start_date

    test_start = train_start_date if train_start_date else train_end_date
    test_end = test_end_date if test_end_date else df['Datetime'].max()

    if test_start == None and train_end == None:
        dates = pd.date_range(train_start, test_end)
        test_start = train_end = dates[round(split * len(dates))]
    
    return train_start, train_end, test_start, test_end


def save_config(config_name) -> None:
    shutil.copyfile(f'configs/{config_name}.yaml', 'data/'+config_name+'.yaml')


def stocks_df(path, stocks) -> pd.DataFrame:
    df = read_data(path)

    if stocks == 'best':
        with open('configs/best_stocks_nans_rate.yaml') as f:
            stocks = yaml.load(f, Loader=yaml.FullLoader)

        stocks = list(stocks.keys())

    elif stocks != None:
        stocks = stocks

    else:
        stocks = df['Stock'].unique()

    return df.query("Stock in @stocks")

    
    








