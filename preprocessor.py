import numpy
import pandas as pd
import os
from stockstats import StockDataFrame


def load_and_preprocess(path:str) -> pd.DataFrame:
    stock_data = pd.read_csv(path, index_col=0)
    # print(stock_data.info())
    stock_data = stock_data[['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol']]
    stock_data = add_technical_indicator(stock_data)
    return stock_data

def add_technical_indicator(stock_data:pd.DataFrame) -> pd.DataFrame:
    stock_data_indi = stock_data.copy()

    # 分离20只股票，分别计算每只股票的MACD, KDJ(K线和D线), RSI指标
    stock_data_dict = dict(tuple(stock_data_indi.groupby('ts_code')))

    macd = pd.Series()
    kdjk = pd.Series()
    kdjd = pd.Series()
    rsi  = pd.Series()

    for k in stock_data_dict.keys():
        one_stock = stock_data_dict[k]
        one_stock.rename({'trade_date': 'date', 'vol': 'volume'}, axis=1) # 适配stockstats库做一些列命名调整
        one_stock = StockDataFrame.retype(one_stock)

        one_stock_macd = one_stock['macd']
        macd = macd.append(one_stock_macd)

        one_stock_kdjk = one_stock['kdjk']
        kdjk = kdjk.append(one_stock_kdjk)

        one_stock_kdjd = one_stock['kdjd']
        kdjd = kdjd.append(one_stock_kdjd)

        one_stock_rsi = one_stock['rsi_14']
        rsi = rsi.append(one_stock_rsi)

    stock_data_indi['macd'] = macd
    stock_data_indi['kdjk'] = kdjk
    stock_data_indi['kdjd'] = kdjd
    stock_data_indi['rsi'] = rsi
    stock_data_indi.fillna(method='bfill', inplace=True) # 填充每只股票第一天交易数据缺少的技术指标

    return stock_data_indi