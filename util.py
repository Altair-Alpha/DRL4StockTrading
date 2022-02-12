from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.style.use('seaborn')

def draw_stock_price(stock_data:pd.DataFrame):
    # 横轴为交易日期（所有股票统一）
    x = [datetime.strptime(str(d), '%Y%m%d').date() for d in stock_data['trade_date'].unique()]

    stock_data_dict = dict(tuple(stock_data.groupby('ts_code')))
    for k, v in stock_data_dict.items():
        plt.figure(figsize=(18, 6))
        plt.margins(x=0.02)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=120))
        plt.plot(x, v['close'])
        plt.gcf().autofmt_xdate()
        plt.savefig('./figs/stock_price/{}.png'.format(k))


if __name__ == '__main__':
    import configparser
    conf = configparser.ConfigParser()
    conf.read('./config/config.ini', encoding='utf-8')
    stock_data_path = conf.get('path', 'stock_data')
    stock_data = pd.read_csv(stock_data_path, index_col=0)
    draw_stock_price(stock_data)