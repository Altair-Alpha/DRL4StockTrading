from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy

from preprocessor import *


def get_quarter_dates(start_date: int, end_date: int) -> list:
    """
    获取两个日期间的以每个季度（92天）为间隔的日期列表

    :param start_date: 8位整数表示的开始日期
    :param end_date: 8位整数表示的结束日期
    :return: 8位整数的日期列表（包括开始和结束日期）
    """
    start_date = datetime.strptime(str(start_date), '%Y%m%d')
    end_date = datetime.strptime(str(end_date), '%Y%m%d')
    cur_date = start_date
    date_lst = []
    while True:
        date_lst.append(int(datetime.strftime(cur_date, '%Y%m%d')))
        cur_date = cur_date + relativedelta(days=92)
        if cur_date > end_date:
            break
    date_lst.append(int(datetime.strftime(end_date, '%Y%m%d')))
    return date_lst


def get_year_diff(start_date: int, end_date: int):
    """
    获取两个日期间相差的年数。注意该函数不会计算相差天数，20000101-20011231也会被算为1年。

    :param start_date: 8位整数表示的开始日期
    :param end_date: 8位整数表示的结束日期
    :return: 相差年数（绝对值）
    """
    start_date = datetime.strptime(str(start_date), '%Y%m%d')
    end_date = datetime.strptime(str(end_date), '%Y%m%d')
    diff = relativedelta(start_date, end_date)
    return abs(diff.years)


def plot_daily(x: list, y: list, path: str = None):
    """
    绘制每日数据的辅助函数。

    :param x: 横轴数据，应为日期的列表（注意数据类型为date而非int）
    :param y: 纵轴数据，任意绘制数据的列表
    :param path: 图像保存路径
    """
    # plt.style.use('seaborn')
    plt.figure(figsize=(30, 10))
    plt.rc('font', size=18)
    plt.margins(x=0.02)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    interval = np.clip(len(x) // 12, 1, 200)  # 调整横轴日期间距，避免过挤
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.plot(x, y, linewidth=3, color='blue')
    plt.gcf().autofmt_xdate()
    plt.ticklabel_format(style='sci', axis='y', useMathText=True)
    if path is None:
        print(y)
        plt.show()
    else:
        plt.savefig(path)
    plt.close()


def plot_daily_compare(x: list, y1: list, y2: list, y1_label: str = 'y1',
                       y2_label: str = 'y2', diff_y_scale: bool = False, save_path: str = None):
    """
    绘制两组每日数据比较的辅助函数。第一组为蓝线，第二组为红线。

    :param x: 横轴数据，应为日期的列表（注意数据类型为date而非int）
    :param y1: 第一组数据y值列表
    :param y2: 第二组数据y值列表
    :param diff_y_scale: 两组数据是否采取不同的纵轴数据范围
    :param save_path: 图像保存路径
    :param y1_label: 第一组数据标签
    :param y2_label: 第二组数据标签
    """

    plt.rc('font', size=20)
    plt.figure(figsize=(30, 10))
    plt.rc('legend', fontsize=20, handlelength=3, edgecolor='black')
    plt.margins(x=0.02)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    interval = np.clip(len(x) // 10, 1, 200)  # 调整横轴日期间距，避免过挤
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    plt.xlabel('Date')
    plt.ylabel('TotalAsset')

    if diff_y_scale:
        ax1 = plt.gca()
        ax1.plot(x, y1, linewidth=3, color='blue', label=y1_label)

        ax2 = ax1.twinx()
        ax2.plot(x, y2, linewidth=3, color='red', label=y2_label)
    else:
        plt.plot(x, y1, linewidth=3, color='blue', label=y1_label)
        plt.plot(x, y2, linewidth=3, color='red', label=y2_label)
        plt.legend(loc='upper left')

    plt.gcf().autofmt_xdate()
    plt.ticklabel_format(style='sci', axis='y', useMathText=True)
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()


def plot_daily_multi_y(x: list, ys: list, ys_label: List[str], save_path: str = None):
    """
    绘制多组每日数据比较的辅助函数。线条颜色为蓝、红、绿、青、紫、黑循环

    :param x: 横轴数据，应为日期的列表（注意数据类型为date而非int）
    :param ys: n组y值列表的列表
    :param ys_label: n组y值的标签
    :param save_path: 图像保存路径
    """

    plt.rc('font', size=20)
    plt.figure(figsize=(30, 10))
    plt.rc('legend', fontsize=20, handlelength=3, edgecolor='black')
    plt.margins(x=0.02)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    interval = np.clip(len(x) // 10, 1, 200)  # 调整横轴日期间距，避免过挤
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    plt.xlabel('Date')
    plt.ylabel('TotalAsset')

    from itertools import cycle
    color_cycle = cycle('brgcmk')

    for y, y_label in zip(ys, ys_label):
        plt.plot(x, y, linewidth=3, color=next(color_cycle), label=y_label)

    plt.legend(loc='upper left')

    plt.gcf().autofmt_xdate()
    plt.ticklabel_format(style='sci', axis='y', useMathText=True)
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()


def read_data():
    """仅测试用"""
    import configparser
    conf = configparser.ConfigParser()
    conf.read('./config/config.ini', encoding='utf-8')
    stock_data_path = conf.get('path', 'preprocessed_stock_data')
    stock_data = pd.read_csv(stock_data_path, index_col=0)
    stock_data = remove_anomaly(stock_data)
    return stock_data


def draw_avg_stock_price(stock_data: pd.DataFrame, eval_period_only: bool = False):
    plt.style.use('seaborn')
    if eval_period_only:
        stock_data = subdata_by_range(stock_data, 20180101, 20211231)
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
    stock_data_dict = to_daily_data(stock_data)

    for k, v in stock_data_dict.items():
        y.append(v['close'].mean())
    plt.figure(figsize=(30, 10))
    plt.rc('font', size=18)
    plt.margins(x=0.02)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=120))
    plt.plot(x, y, linewidth=3, color='blue')
    plt.gcf().autofmt_xdate()
    plt.savefig('./figs/stock_price/avg_2010-2021.png')
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


def draw_mean_std(x: list, mean: list, std: list, log_scale: bool, path: str):
    plt.style.use('seaborn')
    plt.figure(figsize=(30, 10))
    plt.rc('font', size=24)
    eb = plt.errorbar(x, mean, std, color='blue', linewidth=3, capsize=5)
    eb[-1][0].set_linestyle('--')
    plt.title('PPO')
    plt.xlabel('Timesteps')
    plt.ylabel('Average Return')
    # plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    if log_scale:
        plt.xscale('log')
    else:
        plt.ticklabel_format(style='sci', useMathText=True)

    plt.savefig(path)
    plt.close()


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


# 绘制Agent学习曲线
def plot_learning_curve(in_log_path, out_path: str, title='Learning Curve'):
    """
    plot the results

    :param out_path:
    :param in_log_path: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(in_log_path), 'timesteps')
    # y = moving_average(y, window=5)
    # Truncate x
    x = x[len(x) - len(y):]

    plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.savefig(out_path)
    plt.close()


if __name__ == '__main__':
    pass

    # plot_results('./models/A2C_test', './models/A2C_test/lerning_curve_smoothed.png')
    # test_sb()
    # data = read_data()
    # data = calc_daily_mean(data)
    # print(get_year_diff(20100101, 20211231))
    # print(get_trade_dates(data))
    # # plot_daily(get_trade_dates(data), data['close'].tolist())
    # plot_daily(get_trade_dates(data), calc_daily_mean(data))

    # data = read_data()
    # data = subdata_by_range(data, 20180101, 20211231)
    # data = calc_daily_mean(data)
    # print(f'start: {data[0]}, end: {data[-1]}, rate: {(data[-1] - data[0])/data[0]*100}')
    # plot_daily(dates, data, './figs/stock_price/cyb10/new_avg10.png')
    # data = to_per_stock_data(data)

    # draw_avg_stock_price(data)
    # for k, v in data.items():
    #     # print('DATESLEN:', len(dates))
    #     # print('VALUELEN:', len(v['close'].tolist()))
    #
    #     plot_daily(get_trade_dates(v), v['close'].tolist(), f'./figs/stock_price/cyb10/{k}.png')
        # plt.xlabel('Date')
        # plt.ylabel('TotalAsset')

    # l = [2146769.06, 2309249.52, 1529266.00, 1107423.98, 1069376.13, 1628285.55, 2308651.76, 2706041.63, 3214475.75]
    # print('mean', np.mean(l), 'std', np.std(l))