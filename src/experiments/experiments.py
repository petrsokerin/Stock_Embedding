import os
from abc import ABC, abstractmethod
import datetime

import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE

from src.data.preprocessing import data_to_np_tensor, preprocess_split

class AbcExperiment(ABC):
    def __init__(
        self, 
        train_start, 
        train_end, 
        test_start, 
        test_end,
        label_name: str = 'Close',
        use_pct_changes_data: bool = False,
        use_pct_changes_labels: bool = False,
    ):
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.label_name = label_name
        self.use_pct_changes_data = use_pct_changes_data
        self.use_pct_changes_labels = use_pct_changes_labels

    @staticmethod
    def stock_pct_change(data):

        def custom_pct_change(data):
            return data.pct_change().iloc[1:]

        res = data \
            .groupby('Stock') \
            .apply(custom_pct_change) \
            .reset_index(level=0, drop=True)
        return res

    @abstractmethod
    def prepare_data(self):
        pass
    
    @abstractmethod
    def fit_model(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    def get_y_start_test(self, y):
        date_array = y.index.get_level_values('Datetime').map(datetime.datetime.date)
        y_train = y[date_array < pd.Timestamp(self.train_end).date()]
        last_train_date = y_train \
            .reset_index() \
            .groupby(['Stock'])['Datetime']\
            .last() \
            .reset_index()
        
        self.y_start_test = y \
            .reset_index() \
            .merge(last_train_date, how='inner', on=['Stock', 'Datetime']) \
            .set_index(['Stock', 'Datetime'])
        
    
    def train_test_split_dt(self, df):
        date_array = df.index.get_level_values('Datetime').map(datetime.datetime.date)
        df_train = df[(date_array >= pd.Timestamp(self.train_start).date()) & 
                    (date_array < pd.Timestamp(self.train_end).date())]

        df_test = df[(date_array >= pd.Timestamp(self.test_start).date()) & 
                    (date_array < pd.Timestamp(self.test_end).date())]
        
        return df_train, df_test
    
    def data_labels_split(self, df):
        X = df.drop(self.label_name, axis=1)
        y = df[self.label_name]
        return X, y

    def estimate_results(
        self,
        y_test, 
        y_pred, 
        metric_func=MAPE,         
    ):
        if not self.use_pct_changes_labels:
            return metric_func(y_test, y_pred)
        
        df_preds = y_test.copy().to_frame(name='True').reset_index()
        df_preds['True'] = df_preds['True'] + 1
        df_preds['Preds'] = y_pred + 1
        
        starts = self.y_start_test.sort_values('Stock')['Close'].values
        orig_close = df_preds.pivot(columns=['Stock'], index='Datetime', values='True').cumprod() * starts
        preds_base = orig_close.shift(1)
        preds_base.iloc[0] = starts
        preds_changes = df_preds.pivot(columns=['Stock'], index='Datetime', values='Preds')
        pred_close = preds_changes * preds_base

        pred_close = pred_close.reset_index().melt(id_vars=['Datetime'], value_name='Pred')
        orig_close = orig_close.reset_index().melt(id_vars=['Datetime'], value_name='True')

        metric_df = pd.merge(pred_close, orig_close, how='inner', on=['Stock', 'Datetime'])
        return metric_func(metric_df['True'], metric_df['Pred'])

    def pipeline(self, df, metric_func=MAPE):
        X_train, X_test, y_train, y_test = self.prepare_data(df)

        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert 'Datetime' in X_train.index.names and 'Stock' in X_train.index.names
        assert 'Datetime' in X_test.index.names and 'Stock' in X_test.index.names
        assert 'Datetime' in y_train.index.names and 'Stock' in y_train.index.names
        assert 'Datetime' in y_test.index.names and 'Stock' in y_test.index.names

        assert (self.use_pct_changes_labels and \
                y_train.mean() <= 2 and y_test.mean() <= 2) or \
                (not self.use_pct_changes_labels and \
                y_train.mean() > 2 and y_test.mean() > 2)

        self.fit_model(X_train, y_train)
        preds = self.predict(X_test)
        results = self.estimate_results(y_test, preds, metric_func)
        return results, preds


class LagModelExperint(AbcExperiment):
    def __init__(self, model, window_size=20, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.window_size = window_size

    def add_shifts(self, data_for_shifts):
        y = data_for_shifts.copy()

        if self.use_pct_changes_labels:
            self.get_y_start_test(y)
            y = self.stock_pct_change(y)

        X = data_for_shifts.copy()

        if self.use_pct_changes_data:
            X = self.stock_pct_change(X)

        for i in range(1, self.window_size + 1):
            X[f'shift_{i}'] = X.groupby(by=['Stock']).shift(i)[self.label_name]

        X = X.dropna().drop(self.label_name, axis=1)
        Xy = X.join(y, how='inner')
        X, y = self.data_labels_split(Xy)
        return X, y

    def prepare_data(self, df):
        data_for_shifts = df[[self.label_name]]
        X, y = self.add_shifts(data_for_shifts)
        X_train, X_test = self.train_test_split_dt(X)
        y_train, y_test = self.train_test_split_dt(y)
        return X_train, X_test, y_train, y_test
    
    def fit_model(self, X_train, y_train, model=None):
        if model:
            self.model = model
        X, y = X_train.reset_index(drop=True), y_train.reset_index(drop=True)
        self.model.fit(X, y)

    def predict(self, X_test):
        X_test.reset_index(drop=True)
        return self.model.predict(X_test)
    

class SelfSupervisedExperint(AbcExperiment):
    def __init__(self, model, emb_model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.emb_model = emb_model

    def prepare_data(self, df):
        X = df.copy() # DATA LEAK

        if self.use_pct_changes_data:
            X = self.stock_pct_change(X)

        y = df[[self.label_name]]

        if self.use_pct_changes_labels:
            self.get_y_start_test(y)
            y = self.stock_pct_change(y)
    
        Xy = X.drop(self.label_name, axis=1).join(y, how='inner')
        X, y = self.data_labels_split(Xy)

        X_train = preprocess_split(
            X.reset_index(), 
            X.columns, 
            self.train_start,
            self.train_end, 
            tickers_save=X.index.get_level_values('Stock').unique()
        )

        X_test = preprocess_split(
            X.reset_index(),
            X.columns, 
            self.test_start,
            self.test_end, 
            tickers_save=X.index.get_level_values('Stock').unique()
        )

        y_train, y_test = self.train_test_split_dt(y)

        return X_train, X_test, y_train, y_test
    
    def load_emb_model(self, path):
        self.emb_model.load(path)

    def save_emb_model(self, path):
        folder_path = '/'.join(path.split('/')[:-1])
        if os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.emb_model.save(path)

    def fit_emb_model(self, X):
        X_transformed = data_to_np_tensor(X)
        self.emb_model.fit(X_transformed)

    def transform_emb_model(self, X):
        X_transformed = data_to_np_tensor(X)
        X_emb = self.emb_model.encode(X_transformed)
        emb_size = X_emb.shape[-1]
        X_emb = pd.DataFrame(
            data = X_emb.reshape(-1, emb_size), 
            index = X.index, 
            columns=['emb_' + str(i) for i in range(emb_size)]
        )
        return X_emb
    
    def fit_model(
        self, 
        X_train, 
        y_train, 
        emb_model_path='',
        train_emb_model: bool = True,
        model=None, 
        emb_model=None, 
    ):
        if model:
            self.model = model

        if emb_model:
            self.emb_model = emb_model

        if len(emb_model_path) > 0:
            self.load_emb_model(emb_model_path)
        elif train_emb_model:
            self.fit_emb_model(X_train)

        X_emb = self.transform_emb_model(X_train)
        X_train, y_train = X_emb.reset_index(drop=True), y_train.reset_index(drop=True)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        X_emb = self.transform_emb_model(X_test)
        X_test = X_emb.reset_index(drop=True)
        return self.model.predict(X_test)


class ConstPredExperiment(AbcExperiment):
    def __init__(self, method='last', **kwargs):
        super().__init__(**kwargs)
        self.method = method

    def prepare_data(self, df):
        X = df[[self.label_name]].copy()
        y = df[self.label_name]

        if self.use_pct_changes_data:
            X = self.stock_pct_change(X)

        if self.use_pct_changes_labels:
            self.get_y_start_test(y)
            y = self.stock_pct_change(y)

        X_test = X \
            .reset_index() \
            .pivot(index='Datetime', columns='Stock') \
            .reset_index()
        X_test.columns = X_test.columns.droplevel()
        X_test.columns = ['Datetime'] + X_test.columns.tolist()[1:]

        _, y_test = self.train_test_split_dt(y)
    
        return X_test, y_test
    
    def fit_model(*args, **kwargs):
        pass

    def predict(self, X_test):
        date_array = X_test['Datetime'].map(datetime.datetime.date)
        test_indexes = X_test[
            (date_array >= pd.Timestamp(self.test_start).date()) & 
            (date_array < pd.Timestamp(self.test_end).date())
        ].index
        
        X_test = X_test.drop(columns=['Datetime'])

        seq_len_test = len(test_indexes)
        n_stocks = X_test.shape[1]
        y_pred_all = np.zeros((seq_len_test, n_stocks))

        for i, test_idx in enumerate(test_indexes):
            y_pred_all[i] = X_test.iloc[test_idx - 1].values
        
        return y_pred_all.T.reshape(-1, 1)
    
    def pipeline(self, df, metric_func=MAPE):
        X_test, y_test = self.prepare_data(df)
        preds = self.predict(X_test)
        results = self.estimate_results(y_test, preds, metric_func)
        return results, preds    


class FoundationZeroShort(ConstPredExperiment):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def predict(self, X_test):
        date_array = X_test['Datetime'].map(datetime.datetime.date)
        test_indexes = X_test[
            (date_array >= pd.Timestamp(self.test_start).date()) & 
            (date_array < pd.Timestamp(self.test_end).date())
        ].index
        
        X_chron_stocks = X_test.drop(columns=['Datetime'])

        seq_len_test = len(test_indexes)
        n_stocks = X_chron_stocks.shape[1]
        y_pred_all = np.zeros((seq_len_test, n_stocks))

        for i, test_idx in enumerate(tqdm(test_indexes)):
            X_stock_test = X_chron_stocks.iloc[:test_idx-1]
            chron_input = torch.tensor(X_stock_test.values.T)
            
            forecast = self.model.predict(
                chron_input,
                prediction_length=1,
                num_samples=50,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
            ) 

            pred = np.median(forecast.numpy(), axis=1).flatten()
            y_pred_all[i] = pred
        
        return y_pred_all.T.reshape(-1, 1)
