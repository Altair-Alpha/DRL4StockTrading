import math
from typing import List, Dict
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gym
from gym import spaces

import global_var
from util import *

class StockEvalEnvV2(gym.Env):
    """符合OpenAI Gym接口的模拟股市环境"""
    metadata = {'render.modes': ['human']}

    def __init__(self, daily_data: Dict[int, pd.DataFrame], stock_codes: List[str], verbose: bool = True):
        super(StockEvalEnvV2, self).__init__()

        self.verbose = verbose
        self.log_trade = verbose

        # stock_codes 必须与 daily_data 中股票的顺序对应
        self.stock_codes = stock_codes
        self.stock_dim = len(stock_codes)
        self.data = daily_data
        self.dates = sorted(list(self.data.keys()))

        if self.verbose:
            print('StockEvalEnvV2:', 'init:', 'trade date from', self.dates[0], 'to', self.dates[-1])
            print('StockEvalEnvV2:', 'init:', f'trade target: {self.stock_dim} stocks')

        self.reset()

        # 动作空间，共 STOCK_DIM 维，代表对每只股票的买入/卖出行为
        # 标准化到 [-1, 1] 区间
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.stock_dim,))
        # 状态空间，共 1（资金）+ 6（股价+持股+MACD+KDJK+KDJD+RSI）* STOCK_DIM（股票个数）维
        # 此问题中无法对状态空间进行标准化
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1 + 6 * self.stock_dim,))

    def step(self, action: np.ndarray):
        self.done = (self.cur_date == self.last_date)
        self.reward = 0  # 清零，否则最后一个step会重复返回上一个step的reward
        if self.done:
            # print('StockTrainEnvV2:', 'episode ended, asset:', self.state[0] + sum(
            #     [shares*price for shares, price in zip(self.state[1 : self.stock_dim+1],
            #                                            self.state[self.stock_dim+1 : 2*self.stock_dim+1])]))
            #save_result(self.dates, self.asset_memory, self.reward_memory, self.verbose)
            pass

        else:
            # 总资产为：余额 + 每只股票持股 * 当前日期股价
            begin_total_asset = self.state[0] + sum(
                [shares * price for shares, price in zip(self.state[1: self.stock_dim + 1],
                                                         self.state[self.stock_dim + 1: 2 * self.stock_dim + 1])])

            self.action_memory.append(action)
            # 对每只股票的单日交易资金量不能超过当前总资金量 * global_var.MAX_PERCENTAGE_PER_TRADE
            action = action * (begin_total_asset * global_var.MAX_PERCENTAGE_PER_TRADE)


            self.asset_memory.append(
                self.state[0: self.stock_dim + 1] + [begin_total_asset])  # state中 0至STOCK_DIM 项为（余额，持股）

            if self.verbose:
                print(global_var.SEP_LINE1)
                print('StockTrainEnvV2:', 'step begin, today:', self.cur_date)
                print('StockTrainEnvV2:', 'before trade, total asset:', begin_total_asset)
                print('StockTrainEnvV2', 'recv action', action)

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
            self.reward = end_total_asset - begin_total_asset
            self.reward_memory.append(self.reward)

            if self.verbose:
                print(global_var.SEP_LINE1)
                print('StockTrainEnvV2:', 'after trade, state:', self.state)
                print('StockTrainEnvV2:', 'after trade, total asset:', end_total_asset)
                print('StockTrainEnvV2:', 'reward:', self.reward)

            self.reward *= global_var.REWARD_SCALING

        return np.array(self.state), self.reward, self.done, {}

    def reset(self):
        self.done = False
        self.reward = 0
        self.cur_date = self.dates[0]
        self.last_date = self.dates[-1]
        self.day_count = 0

        # 初始状态为：[初始资金, 持股数（全为0）, 第一天的股价数据，第一天的股价指标]
        self.state = [global_var.INITIAL_BALANCE] + [0] * self.stock_dim + self.data[self.cur_date]['close'].tolist() \
                     + self.data[self.cur_date]['macd'].tolist() \
                     + self.data[self.cur_date]['kdjk'].tolist() \
                     + self.data[self.cur_date]['kdjd'].tolist() \
                     + self.data[self.cur_date]['rsi'].tolist()
        if self.verbose:
            print('StockEvalEnvV2:', 'reset:', 'init state', self.state)
        self.asset_memory = []      # 记录历史每步资产状态：资金+持股+总资产（前两项合起来）
        self.action_memory = []     # 记录历史每步动作（外部传入）
        self.reward_memory = []     # 记录历史每步收益

        self.trade_count = 0
        return np.array(self.state)

    def _buy_stock(self, index, amount):
        # 确保入参amount不超过余额，避免交易后账户余额为负，计算对应股数（向下取整）
        vol = min(amount, self.state[0]) // self.state[1 + self.stock_dim + index]
        # 真正买入资金量
        # A股印花税为卖方单向收费，其余小额双向收费项暂且不计
        real_amount = vol * self.state[1 + self.stock_dim + index]

        # 更新持股
        self.state[1 + index] += vol
        # 更新资金，减少：股价*购买股数
        prev_balance = self.state[0]
        self.state[0] -= real_amount
        self.trade_count += 1

        if self.log_trade:
            print(global_var.SEP_LINE2)
            print('StockTrainEnvV2:', f'trade no.{self.trade_count}', f'stock: {self.stock_codes[index]}',
                  'type: buy', f'vol: {vol}', f'amount: {real_amount}')
            print('StockTrainEnvV2:', f'balance drop from {prev_balance} to {self.state[0]}')

    def _sell_stock(self, index, amount):
        # 之前状态持股>0时，才能进行卖出操作
        if self.state[1 + index] > 0:
            vol = amount // self.state[1 + self.stock_dim + index]
            real_volume = min(vol, self.state[1 + index])  # 确保入参amount对应的股数不超过持股数，避免交易后持股为负

            # 更新持股
            self.state[1 + index] -= real_volume
            # 更新资金，增加：股价*购买股数*(1-交易费率)
            # A股印花税为卖方单向收费，其余小额双向收费项暂且不计
            real_amount = self.state[1 + self.stock_dim + index] * real_volume \
                     * (1 - global_var.TRANSACTION_FEE_PERCENTAGE)
            prev_balance = self.state[0]
            self.state[0] += real_amount
            self.trade_count += 1

            if self.log_trade:
                print(global_var.SEP_LINE2)
                print('StockTrainEnvV2:', f'trade no.{self.trade_count}', f'stock: {self.stock_codes[index]}',
                      'type: sell', f'vol: {real_volume}', f'amount: {amount}')
                print('StockTrainEnvV2:', f'balance increase from {prev_balance} to {self.state[0]}')

    def reset_date(self, new_start, new_end):
        self.dates = [x for x in self.dates if (x>=new_start and x<=new_end)]
        self.cur_date = self.dates[0]
        self.last_date = self.dates[-1]
        self.day_count = 0

    def save_result(self, asset_graph_path: str, reward_graph_path: str):
        if self.verbose:
            print('StockEvalEnvV2:', 'saving results.')

        x = [datetime.strptime(str(d), '%Y%m%d').date() for d in self.dates]
        del x[-1]
        assets = [a[-1] for a in self.asset_memory] # asset每项中最后一个数是余额+持股价值的总资产
        rewards = self.reward_memory

        plot_daily(x, assets, asset_graph_path)     # 绘制每日资产变化图
        plot_daily(x, rewards, reward_graph_path)    # 绘制每日收益图

    def dump_memory(self, path: str):
        stock_hold_df = pd.DataFrame([x[1:self.stock_dim+1] for x in self.asset_memory], columns=self.stock_codes, index=self.dates[:-1])
        stock_hold_df.T.to_csv(path + 'stock_hold_memory.csv') # 为适配图表制作，进行转置

        asset_dist = []
        for i, d in enumerate(self.dates[:-1]):
            # 持股与股价列表相乘
            # 资产第0项是余额，股价列表前补1；资产最后一项是总金额，不需要，去除
            asset_dist.append([hold*price for hold, price in zip(self.asset_memory[i][:-1], [1] + self.data[d]['close'].tolist())])
            # 验证
            if not math.isclose(sum(asset_dist[i]), self.asset_memory[i][-1]):
                print('NOT EQUAL!!', sum(asset_dist[i]), self.asset_memory[i][-1])
                return
        asset_dist_df = pd.DataFrame(asset_dist, columns=['balance'] + self.stock_codes, index=self.dates[:-1])
        asset_dist_df.T.to_csv(path + 'asset_distribution_memory.csv')

        reward_memory_df = pd.DataFrame(self.reward_memory, columns=['reward'], index=self.dates[:-1])
        reward_memory_df.to_csv(path + 'reward_memory.csv')
        action_meomory_df = pd.DataFrame(self.action_memory, columns=self.stock_codes, index=self.dates[:-1])
        action_meomory_df.T.to_csv(path + 'action_memory.csv')

if __name__ == '__main__':
    pass
    # from stable_baselines3.common.env_checker import check_env
    # from preprocessor import *
    #
    # data = pd.read_csv('../data/szstock_20_preprocessed.csv')
    # data = subdata_by_ndays(data, 10)
    # stock_codes = get_stock_codes(data)
    # data = to_daily_data(data)
    # env = StockEvalEnvV2(data, stock_codes)
    # check_env(env, warn=True)