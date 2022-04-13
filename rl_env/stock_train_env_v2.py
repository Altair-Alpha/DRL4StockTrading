from typing import List
import bisect
import numpy as np
import pandas as pd
import gym
from gym import spaces

import global_var
import preprocessor as pp


class StockTrainEnvV2(gym.Env):
    """
    符合OpenAI Gym接口的模拟股市环境。
    V2将aciton视为对每只股票资金量的操作而非持股数的操作。
    """

    def __init__(self, data: pd.DataFrame, verbose: int = 0):
        super(StockTrainEnvV2, self).__init__()

        # verbose参数为0不输出，为1输出大部分日志，大于1额外输出每笔交易的记录
        self.verbose = (verbose > 0)
        self.log_trade = (verbose > 1)

        self.stock_codes = pp.get_stock_codes(data)
        self.stock_dim = len(self.stock_codes)
        self.data = pp.to_daily_data(data)
        self.full_dates = self.dates = sorted(list(self.data.keys()))

        if self.verbose:
            print('StockTrainEnvV2:', 'init():', f'trade date from {self.full_dates[0]} to {self.full_dates[-1]}.'
                                               f'trade target: {self.stock_dim} stocks')

        # More attricutes defined in this function
        self.reset()

        # 动作空间，共 stock_dim 维，代表对每只股票的买入/卖出行为，每维定义域为 [-1, 1] 区间
        # 环境处理时会将输入动作乘以放大系数
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.stock_dim,))
        # 状态空间，共 1（可用资金）+ 6（股价+持股+MACD+KDJK+KDJD+RSI）* STOCK_DIM（股票个数）维
        # 状态空间不进行标准化
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1 + 6 * self.stock_dim,))

    def reset(self):
        self.done = False
        self.reward = 0
        self.cur_date = self.dates[0]
        self.last_date = self.dates[-1]
        self.day_count = 0
        self.trade_count = 0

        # 初始状态为：[初始资金, 持股数（全为0），第一天的股价数据，第一天的股价指标]
        self.state = [global_var.INITIAL_BALANCE] + [0] * self.stock_dim + self.data[self.cur_date]['close'].tolist() \
                     + self.data[self.cur_date]['macd'].tolist() \
                     + self.data[self.cur_date]['kdjk'].tolist() \
                     + self.data[self.cur_date]['kdjd'].tolist() \
                     + self.data[self.cur_date]['rsi'].tolist()
        if self.verbose:
            print('StockTrainEnvV2:', f'reset(): init state {self.state}')

        return np.array(self.state)

    def step(self, action: np.ndarray):
        self.done = (self.cur_date == self.last_date)
        self.reward = 0  # 清零，否则最后一个step会重复返回上一个step的reward

        if not self.done:
            # 总资产为：可用资金 + 每只股票持股 * 当日股价
            begin_total_asset = self.state[0] + sum(
                [shares * price for shares, price in zip(self.state[1: self.stock_dim + 1],
                                                         self.state[self.stock_dim + 1: 2 * self.stock_dim + 1])])

            # 对每只股票的单日交易资金量范围为正负当前总资金量 * global_var.MAX_PERCENTAGE_PER_TRADE
            action = action * (begin_total_asset * global_var.TRAIN_MAX_PERCENTAGE_PER_TRADE)

            if self.verbose:
                print(global_var.SEP_LINE1)
                print('StockTrainEnvV2:', f'step(): step begin, today: {self.cur_date}.'
                                          f' Before trade, total asset: {begin_total_asset}')

            # 进行交易
            for index, one_stock_action in enumerate(action):
                if one_stock_action > 0:
                    self._buy_stock(index, abs(one_stock_action))
                elif one_stock_action < 0:
                    self._sell_stock(index, abs(one_stock_action))

            # 更新日期
            self.day_count += 1
            self.cur_date = self.dates[self.day_count]

            # 取新一日的股票数据，更新state
            self.state = self.state[0: 1 + self.stock_dim] + self.data[self.cur_date]['close'].tolist() \
                         + self.data[self.cur_date]['macd'].tolist() \
                         + self.data[self.cur_date]['kdjk'].tolist() \
                         + self.data[self.cur_date]['kdjd'].tolist() \
                         + self.data[self.cur_date]['rsi'].tolist()

            # 结束时的资产计算方式相同，此时使用的state已更新
            end_total_asset = self.state[0] + sum(
                [shares * price for shares, price in zip(self.state[1: self.stock_dim + 1],
                                                         self.state[self.stock_dim + 1: 2 * self.stock_dim + 1])])

            # 当日收益为结束总资产减开始总资产
            self.reward = end_total_asset - begin_total_asset

            if self.verbose:
                print(global_var.SEP_LINE1)
                print('StockTrainEnvV2:', f'step(): after trade, state: {self.state}')
                print('StockTrainEnvV2:', f'step(): total asset: {end_total_asset}. Reward: {self.reward}')

            # 反馈给Agent的Reward需进行缩放
            self.reward *= global_var.REWARD_SCALING

        return np.array(self.state), self.reward, self.done, {}

    def _buy_stock(self, index, amount):
        # 确保入参amount不超过可用资金，避免交易后账户可用资金为负，计算对应股数（向下取整）
        vol = min(amount, self.state[0]) // self.state[1 + self.stock_dim + index]
        # 真正买入资金量（A股印花税为卖方单向收费，其余小额双向收费项暂且不计，买入函数不计费）
        real_amount = vol * self.state[1 + self.stock_dim + index]
        # 更新持股
        self.state[1 + index] += vol
        # 更新可用资金，减少：股价*购买股数
        prev_balance = self.state[0]
        self.state[0] -= real_amount
        self.trade_count += 1

        if self.log_trade:
            print(global_var.SEP_LINE2)
            print('StockTrainEnvV2:', f'trade no.{self.trade_count}, stock: {self.stock_codes[index]}\n'
                                      f'type: buy, vol: {vol}, amount: {real_amount}, '
                                      f'balance drop from {prev_balance} to {self.state[0]}')

    def _sell_stock(self, index, amount):
        # 之前状态持股>0时，才能进行卖出操作
        if self.state[1 + index] > 0:
            vol = amount // self.state[1 + self.stock_dim + index]
            # 确保入参amount对应的股数不超过持股数，避免交易后持股为负
            real_volume = min(vol, self.state[1 + index])
            # 更新持股
            self.state[1 + index] -= real_volume
            # 更新资金，增加：股价*购买股数*(1-交易费率)
            real_amount = self.state[1 + self.stock_dim + index] * real_volume\
                          * (1 - global_var.TRANSACTION_FEE_PERCENTAGE)
            prev_balance = self.state[0]
            self.state[0] += real_amount
            self.trade_count += 1

            if self.log_trade:
                print(global_var.SEP_LINE2)
                print('StockTrainEnvV2:', f'trade no.{self.trade_count}, stock: {self.stock_codes[index]}\n'
                                          f'type: sell, vol: {real_volume}, amount: {real_amount}, '
                                          f'balance increase from {prev_balance} to {self.state[0]}')

    def check_interval_valid(self, start, end) -> bool:
        """
        检查给定日期区间内是否存在股市数据

        :param start: 8位数字开始日期
        :param end: 8位数字结束日期
        """
        dates = [x for x in self.full_dates if (x>=start and x<=end)]
        # print(f'interval {start} to {end} checked, {len(dates)} trade days.')
        return len(dates) > 0

    def reset_date(self, new_start: int, new_end: int, is_last_section: bool = False) -> List[int]:
        """
        调整环境中可以进行交易的日期区间，用于实现分段交易的追踪训练。请先调用check_interval_valid检查该区间内是否存在数据。

        :param new_start: 8位数字开始日期
        :param new_end: 8位数字结束日期
        :param is_last_section: 如果输入区间是最后一个分段请设置为True
        :return: 调整后区间内的交易日期列表
        """

        exact_start = bisect.bisect_left(self.full_dates, new_start)
        exact_end = bisect.bisect_left(self.full_dates, new_end)
        self.dates = self.full_dates[exact_start: exact_end]

        # 由于step中对最后一天last_date不做交易，所以分段时对非最后一段要多补一天，否则会漏掉边界
        if not is_last_section:
            # print(f'sec {new_start} to {new_end}, add one day {self.full_dates[exact_end+1]}')
            self.dates.append(self.full_dates[exact_end + 1])

        self.cur_date = self.dates[0]
        self.last_date = self.dates[-1]
        self.day_count = 0
        return self.dates

    def render(self, mode="human"):
        pass


if __name__ == '__main__':
    pass
    # from stable_baselines3.common.env_checker import check_env
    # from preprocessor import *
    #
    # data = pd.read_csv('../data/szstock_20_preprocessed.csv')
    # data = subdata_by_ndays(data, 10)
    # stock_codes = get_stock_codes(data)
    # data = to_daily_data(data)
    # env = StockTrainEnvV2(data, stock_codes)
    # check_env(env, warn=True)
