from typing import Dict, List
import numpy
import numpy as np
import pandas as pd
import os
from stockstats import StockDataFrame


def load_and_preprocess(path: str) -> pd.DataFrame:
    """加载原始股价数据csv文件，并添加技术指标数据。

    :param path: 数据文件路径
    :return: 添加技术指标列后的DataFrame
    """
    stock_data = pd.read_csv(path, index_col=0)
    # print(stock_data.info())
    stock_data = stock_data[['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol']]
    stock_data = _add_technical_indicator(stock_data)
    return stock_data


def _add_technical_indicator(stock_data: pd.DataFrame) -> pd.DataFrame:
    stock_data_indi = stock_data.copy()

    # 分离20只股票，分别计算每只股票的MACD, KDJ(K线和D线), RSI指标
    stock_data_dict = dict(tuple(stock_data_indi.groupby('ts_code')))

    macd = pd.Series()
    kdjk = pd.Series()
    kdjd = pd.Series()
    rsi = pd.Series()

    for k in stock_data_dict.keys():
        one_stock = stock_data_dict[k]
        one_stock.rename({'trade_date': 'date', 'vol': 'volume'}, axis=1)  # 适配stockstats库做一些列命名调整
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
    stock_data_indi.fillna(method='bfill', inplace=True)  # 填充每只股票第一天交易数据缺少的技术指标
    return stock_data_indi


def subdata_by_range(data: pd.DataFrame, start_date: int, end_date: int) -> pd.DataFrame:
    """通过开始日期和结束日期截取数据集的子集

    :param data: 原始数据
    :param start_date: 开始日期
    :param end_date: 结束日期
    """
    sub_data = data[(data['trade_date'] >= start_date) & (data['trade_date'] <= end_date)].reset_index(drop=True)
    return sub_data


def subdata_by_ndays(data: pd.DataFrame, n_days: int, start_date: int = 0) -> pd.DataFrame:
    """通过开始日期和交易天数截取数据集的子集。如果不声明开始日期，则默认从原数据开头开始截取。请确保开始日期存在于原始数据中

    :param data: 原始数据
    :param n_days: 截取交易天数
    :param start_date: 开始日期
    """
    dates = list(data['trade_date'].unique())
    index = 0 if (start_date == 0) else dates.index(start_date)
    target_dates = dates[index : index+n_days]
    # print('DATES')
    # print(target_dates)
    sub_data = data[data['trade_date'].isin(target_dates)].reset_index(drop=True)
    return sub_data


def to_daily_data(data: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """将所有股票所有日期的DataFrame分割为日期为键，当日所有股票数据DataFrame为值的字典

    :param data: 原始数据
    """
    daily_data = dict(tuple(data.groupby('trade_date')))
    for v in daily_data.values():
        v.reset_index(inplace=True)
    return daily_data

def get_stock_codes(data: pd.DataFrame) -> List[str]:
    """获取原始数据中所有股票代码的list

    :param data: 原始数据
    """
    return list(data['ts_code'].unique())


def remove_anomaly(data: pd.DataFrame) -> pd.DataFrame:
    """茅台600519.SH，片仔癀600436.SH和山西汾酒600809.SH三只股票近10年涨幅过高，此函数从原数据中去除这三只股票的数据。

    :param data: 原始数据
    """
    return data[(data['ts_code'] != '600519.SH')
               & (data['ts_code'] != '600436.SH')
               & (data['ts_code'] != '600809.SH')]
