from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from preprocessor import *


def plot_daily(x: list, y: list, path: str):
    """绘制每日数据的辅助函数。

    :param x: 横轴数据，应为日期的列表
    :param y: 纵轴数据，任意绘制数据的列表
    :param path: 图像保存路径
    """
    # plt.style.use('seaborn')
    plt.figure(figsize=(30, 10))
    plt.rc('font', size=18)
    plt.margins(x=0.02)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    interval = np.clip(len(x) // 10, 1, 120) # 调整横轴日期间距，避免过挤
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    plt.plot(x, y, linewidth=3, color='blue')
    plt.gcf().autofmt_xdate()
    plt.savefig(path)
    plt.close()

def plot_daily_compare(x: list, y1: list, y2: list, diff_y_scale: bool, path:str, label_y1: str, label_y2: str = 'baseline'):
    if len(y2) == 0:
        data = subdata_by_range(read_data(), 20180101, 20211231)
        y2 = calc_daily_mean(data)[:-1]

    # plt.style.use('seaborn')
    plt.figure(figsize=(30, 10))
    plt.rc('font', size=18)
    plt.rc('legend', fontsize=20, handlelength=3, edgecolor='black')
    plt.margins(x=0.02)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    interval = np.clip(len(x) // 10, 1, 120)  # 调整横轴日期间距，避免过挤
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))

    if diff_y_scale:
        ax1 = plt.gca()
        ax1.set_ylabel(label_y1, color='red', fontsize=16)
        ax1.plot(x, y1, linewidth=3, color='red')

        ax2 = ax1.twinx()
        ax2.set_ylabel(label_y2, color='blue', fontsize=16)
        ax2.plot(x, y2, linewidth=3, color='blue')
    else:
        plt.plot(x, y1, linewidth=3, color='red', label=label_y1)
        plt.plot(x, y2, linewidth=3, color='blue', label=label_y2)
        plt.legend(loc='upper left')

    plt.gcf().autofmt_xdate()
    plt.savefig(path)
    plt.close()


def read_data():
    """仅测试用"""
    import configparser
    conf = configparser.ConfigParser()
    conf.read('./config/config.ini', encoding='utf-8')
    STOCK_DATA_PATH = conf.get('path', 'preprocessed_stock_data')
    stock_data = pd.read_csv(STOCK_DATA_PATH, index_col=0)
    stock_data = remove_anomaly(stock_data)
    return stock_data


def draw_stock_price(stock_data: pd.DataFrame):
    # 横轴为交易日期（所有股票统一）
    x = [datetime.strptime(str(d), '%Y%m%d').date() for d in stock_data['trade_date'].unique()]

    # stock_data_dict = dict(tuple(stock_data.groupby('ts_code')))
    # for k, v in stock_data_dict.items():
    #     plt.figure(figsize=(18, 6))
    #     plt.margins(x=0.02)
    #     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    #     plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=120))
    #     plt.plot(x, v['close'])
    #     plt.gcf().autofmt_xdate()
    #     plt.savefig('./figs/stock_price/{}.png'.format(k))
    #     plt.close()

    y = []
    stock_data_dict = to_daily_data(stock_data[(stock_data['ts_code'] != '600519.SH')
                                                            & (stock_data['ts_code'] != '600436.SH')
                                                            & (stock_data['ts_code'] != '600809.SH')])

    for k, v in stock_data_dict.items():
        y.append(v['close'].mean())
    plt.figure(figsize=(18, 6))
    plt.margins(x=0.02)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=120))
    plt.plot(x, y)
    plt.gcf().autofmt_xdate()
    plt.savefig('./figs/stock_price/avg20_noanomaly.png')
    plt.close()


def calc_daily_mean(data: pd.DataFrame) -> list:
    # daily_data = to_daily_data(data)
    # mean_data = pd.DataFrame(columns=['mean'])
    # for k, v in daily_data.items():
    #     mean_data.loc[k] = v['close'].mean()
    # return mean_data
    mean_data = []
    daily_data = to_daily_data(data)
    for v in daily_data.values():
        mean_data.append(v['close'].mean())
    return mean_data