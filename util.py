from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from preprocessor import *

plt.style.use('seaborn')

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


def calc_daily_mean(data: pd.DataFrame) -> pd.DataFrame:
    daily_data = to_daily_data(data)
    mean_data = pd.DataFrame(columns=['mean'])
    for k, v in daily_data.items():
        mean_data.loc[k] = v['close'].mean()
    return mean_data

if __name__ == '__main__':
    import configparser
    conf = configparser.ConfigParser()
    conf.read('./config/config.ini', encoding='utf-8')
    STOCK_DATA_PATH = conf.get('path', 'stock_data')
    stock_data = pd.read_csv(STOCK_DATA_PATH, index_col=0)
    stock_data = remove_anomaly(stock_data)
    #draw_stock_price(remove_anomaly(stock_data))
    mean_data = calc_daily_mean(stock_data)

    data1, data2 = mean_data.loc[20190110]['mean'], mean_data.loc[20211231]['mean']
    print('20180102 {:.2f} 20211231 {:.2f} rate {:.2f}%'.format(data1, data2, (data2-data1) / data1 * 100))