from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy

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

def plot_daily_compare(x: list, y1: list, y2: list, diff_y_scale: bool, path:str, label_y1: str, label_y2: str = 'Hold(baseline)'):
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


def draw_avg_stock_price(stock_data: pd.DataFrame, eval_period_only: bool = True):
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
    plt.savefig('./figs/stock_price/avg20_2018-2021_noanomaly.png')
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


# Agent学习曲线
def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, save_path: str, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    # y = moving_average(y, window=5)
    # Truncate x
    x = x[len(x) - len(y):]

    plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.savefig(save_path)
    plt.close()


def test_sb():
    import gym
    from stable_baselines3 import TD3
    from stable_baselines3.common.env_util import make_vec_env
    env = gym.make("Pendulum-v0")
    model = TD3("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    plot_results('./models/A2C_test/tuned', './models/A2C_test/tuned/learning_curve.png')

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        print('A:', action)
        obs, rewards, dones, info = env.step(action)

        # env.render()

if __name__ == '__main__':
    # ppo_perf = {'timesteps':    [0,         10e3,       25e3,       50e3,       10e4,       25e4,       50e4,       1e6,        2e6],
    #             'mean':         [126418.34, 217380.32,  298220.62,  313085.22,  360852.02,  409095.19,  429844.81,  497041.50,  530048.06],
    #             'std':          [16782.47,  52277.96,   151125.63,  155977.04,  157869.34,  69408.79,   138805.86,  143035.92,  124618.02]}
    # draw_mean_std(ppo_perf['timesteps'], ppo_perf['mean'], ppo_perf['std'], True, './figs/simulation/PPO_2M_Eval/timesteps_log.png')

    # plot_results('./models/A2C_test', './models/A2C_test/lerning_curve_smoothed.png')
    test_sb()