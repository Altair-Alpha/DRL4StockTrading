import time
import numpy as np
import pandas as pd
import gym
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env

import global_var
import util
from preprocessor import *
from stock_train_env import StockTrainEnv
from stock_eval_env import StockEvalEnv


def run_agent(data: pd.DataFrame = None, model: str = 'DDPG', episode: int = 20):
    stock_codes = get_stock_codes(data)
    data_train = subdata_by_range(data, 20100101, 20171231)
    data_train = to_daily_data(data_train)
    # env_train = StockTrainEnvV1(data_train, stock_codes, verbose=False)

    # env_train = make_vec_env(StockTrainEnv, n_envs=4,
    #                          env_kwargs={'daily_data': data_train, 'stock_codes': stock_codes, 'verbose': False})
    # DDPG不支持多环境
    env_train = StockTrainEnv(data_train, stock_codes, False)

    agent = agent_factory(model, env_train)
    if global_var.VERBOSE:
        print('Agent:', f'using {model} agent')

    train_start_time = time.time()
    agent.learn(timesteps=25000)
    # agent.save('./models/PPO_250K_2018')
    # agent.load('./models/PPO_1M')
    train_end_time = time.time()

    data_eval = subdata_by_range(data, 20180101, 20211231)
    data_eval = to_daily_data(data_eval)
    env_eval = StockEvalEnv(data_eval, stock_codes, verbose=False)

    returns = []
    for i in range(episode):
        total_rewards = 0
        # print(global_var.SEP_LINE1)
        # print('Agent:', f'episode {i+1}/{episode} begins.')
        state = env_eval.reset()
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env_eval.step(action)
            state = next_state
            total_rewards += reward
            if done:
                ret = total_rewards / global_var.REWARD_SCALING
                print('Agent:', 'episode {:0>2d}/{}, return(total reward) {:.2f}'.format(i + 1, episode, ret))
                returns.append(ret)
                break
    return_mean, return_std = np.mean(returns), np.std(returns)
    print('Agent:',
          'total {} episodes, average return {:.2f}, std {:.2f}, return rate {:.2f}%'.format(episode, return_mean,
                                                                                             return_std,
                                                                                             100 * return_mean / global_var.INITIAL_BALANCE))
    print('Agent:', 'model training time {:.2f} minutes'.format((train_end_time - train_start_time) / 60))


def run_agent_test(data: pd.DataFrame = None, model: str = 'DDPG', episode: int = 1):
    if global_var.VERBOSE:
        print('Agent:', f'test version. using {model} agent.')
        print('reward scale', global_var.REWARD_SCALING)
    stock_codes = get_stock_codes(data)
    data_train = subdata_by_range(data, 20100101, 20171231)
    data_train = to_daily_data(data_train)
    data_eval = subdata_by_range(data, 20180101, 20211231)
    data_eval = to_daily_data(data_eval)
    returns = []
    train_times = []
    for e in range(10):
        env_train = StockTrainEnv(data_train, stock_codes, verbose=False)
        # env_train = make_vec_env(StockTrainEnv, n_envs=4,
        #                          env_kwargs={'daily_data': data_train, 'stock_codes': stock_codes, 'verbose': False})
        # data_full = subdata_by_range(data, 20190101, 20211231)
        # data_full = to_daily_data(data_full)
        # env_full = StockTrainEnvV1(data_full, stock_codes, verbose=False)

        agent = agent_factory(model, env_train)
        train_start_time = time.time()
        agent.learn(timesteps=100000)
        train_end_time = time.time()
        # agent.save(f'./models/0303_PPO_1M_RS1e-3_10_Train/{e}')
        train_times.append((train_end_time - train_start_time) / 60)
        env_eval = StockEvalEnv(data_eval, stock_codes, verbose=False)

        ret = 0
        for i in range(episode):
            total_rewards = 0
            # print(global_var.SEP_LINE1)
            # print('Agent:', f'episode {i+1}/{episode} begins.')
            state = env_eval.reset()
            while True:
                action = agent.act(state)
                next_state, reward, done, _ = env_eval.step(action)
                state = next_state
                total_rewards += reward
                if done:
                    ret += total_rewards / global_var.REWARD_SCALING
                    break
        ret /= episode
        print('Agent:', 'episode {:0>2d}/{}, avg return {:.2f}'.format(e + 1, 10, ret))
        returns.append(ret)
    return_mean, return_std = np.mean(returns), np.std(returns)
    print('Agent:',
          'total {} training, average return {:.2f}, std {:.2f}, return rate {:.2f}%'.format(10, return_mean,
                                                                                             return_std,
                                                                                             100 * return_mean / global_var.INITIAL_BALANCE))
    print('Agent:', 'average model training time: {:.2f} minutes'.format(np.mean(train_times)))


def hold_agent_test(data: pd.DataFrame = None, model: str = 'Hold', episode: int = 20):
    if global_var.VERBOSE:
        print('Agent:', f'test version. using {model} agent.')
    stock_codes = get_stock_codes(data)
    data_train = subdata_by_range(data, 20100101, 20171231)
    data_train = to_daily_data(data_train)
    data_eval = subdata_by_range(data, 20180101, 20211231)
    data_eval = to_daily_data(data_eval)
    returns = []
    train_times = []
    for e in range(10):
        env_train = StockTrainEnv(data_train, stock_codes, verbose=False)
        # data_full = subdata_by_range(data, 20190101, 20211231)
        # data_full = to_daily_data(data_full)
        # env_full = StockTrainEnvV1(data_full, stock_codes, verbose=False)

        agent = agent_factory(model, env_train)
        train_start_time = time.time()
        agent.learn(timesteps=1000000)
        train_end_time = time.time()
        train_times.append((train_end_time - train_start_time) / 60)
        env_eval = StockTrainEnv(data_eval, stock_codes, verbose=False)

        ret = 0
        for i in range(episode):
            total_rewards = 0
            agent.initial = True
            # print(global_var.SEP_LINE1)
            # print('Agent:', f'episode {i+1}/{episode} begins.')
            state = env_eval.reset()
            while True:
                action = agent.act(state)
                next_state, reward, done, _ = env_eval.step(action)
                state = next_state
                total_rewards += reward
                if done:
                    ret += total_rewards / global_var.REWARD_SCALING
                    break
        ret /= episode
        print('Agent:', 'episode {:0>2d}/{}, avg return {:.2f}'.format(e + 1, 10, ret))
        returns.append(ret)
    return_mean, return_std = np.mean(returns), np.std(returns)
    print('Agent:',
          'total {} training, average return {:.2f}, std {:.2f}, return rate {:.2f}%'.format(10, return_mean,
                                                                                             return_std,
                                                                                             100 * return_mean / global_var.INITIAL_BALANCE))
    print('Agent:', 'average model training time: {:.2f} minutes'.format(np.mean(train_times)))


def run_agent_keep_train(data: pd.DataFrame = None, model: str = 'Hold', episode: int = 20):
    if global_var.VERBOSE:
        print('Agent:', f'keep train version. using {model} agent.')
    stock_codes = get_stock_codes(data)
    daily_data = to_daily_data(subdata_by_range(data, 20180101, 20211231))
    trade_dates = list(daily_data.keys())
    retrain_dates = [date for index, date in enumerate(trade_dates) if index % 63 == 0]  # 一年250个交易日，除4=62.5
    # 16个 [20180102, 20180424, 20180725, 20181030, 20190219, 20190523, 20190821, 20191203, 20200310, 20200611, 20200910, 20201216, 20210323, 20210625, 20210924, 20211229]
    # print(len(retrain_dates))
    # print('picked dates', retrain_dates)

    returns = []
    for e in range(10):
        for i in range(len(retrain_dates) - 1):
            data_train = to_daily_data(subdata_by_range(data, 20100101, retrain_dates[i]))
            env_train = StockTrainEnv(data_train, stock_codes, verbose=False)
            # data_full = subdata_by_range(data, 20190101, 20211231)
            # data_full = to_daily_data(data_full)
            # env_full = StockTrainEnvV1(data_full, stock_codes, verbose=False)

            agent = agent_factory(model, env_train)
            agent.learn(timesteps=25000)

            data_eval = to_daily_data(subdata_by_range(data, retrain_dates[i], retrain_dates[i + 1]))
            env_eval = StockTrainEnv(data_eval, stock_codes, verbose=False)

            ret = 0
            for i in range(episode):
                total_rewards = 0
                # print(global_var.SEP_LINE1)
                # print('Agent:', f'episode {i+1}/{episode} begins.')
                state = env_eval.reset()
                while True:
                    action = agent.act(state)
                    next_state, reward, done, _ = env_eval.step(action)
                    state = next_state
                    total_rewards += reward
                    if done:
                        ret += total_rewards / global_var.REWARD_SCALING
                        break
            ret /= episode
        print('Agent:', 'episode {:0>2d}/{}, avg return {:.2f}'.format(e + 1, 10, ret))
        returns.append(ret)
    return_mean, return_std = np.mean(returns), np.std(returns)
    print('Agent:',
          'total {} training, average return {:.2f}, std {:.2f}, return rate {:.2f}%'.format(10, return_mean,
                                                                                             return_std,
                                                                                             100 * return_mean / global_var.INITIAL_BALANCE))


def eval_agent(data: pd.DataFrame = None, episode: int = 1):
    stock_codes = get_stock_codes(data)
    data_eval = subdata_by_range(data, 20180101, 20211231)
    data_eval = to_daily_data(data_eval)
    env_eval = StockEvalEnv(data_eval, stock_codes, False)
    path = './models/0223_PPO_2M_10_Train/best.zip'
    agent = agent_factory('PPO', env_eval)
    agent.load(path)

    returns_agent = []
    reward_memory_agent = []
    return_memory_agent = []
    asset_memory_agent = []
    for i in range(episode):
        total_rewards = 0
        # print(global_var.SEP_LINE1)
        # print('Agent:', f'episode {i+1}/{episode} begins.')
        state = env_eval.reset()
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env_eval.step(action)
            state = next_state
            total_rewards += reward
            if done:
                ret = total_rewards / global_var.REWARD_SCALING
                print('Agent:', 'episode {:0>2d}/{}, return(total reward) {:.2f}'.format(i + 1, episode, ret))
                returns_agent.append(ret)
                reward_memory_agent = env_eval.reward_memory
                return_memory_agent = np.cumsum(reward_memory_agent)
                asset_memory_agent = [a[-1] for a in env_eval.asset_memory]
                # env_eval.save_result('./figs/simulation/PPO_2M_Eval/total_assets1.png', './figs/simulation/PPO_2M_Eval/rewards1.png')
                env_eval.dump_memory('./figs/simulation/PPO_2M_Eval/env_memory')
                break
    print('Reward:', f'max:{max(reward_memory_agent)}', f'min:{min(reward_memory_agent)}')
    return_mean, return_std = np.mean(returns_agent), np.std(returns_agent)
    print('Agent:',
          'total {} episodes, average return {:.2f}, std {:.2f}, return rate {:.2f}%'.format(episode, return_mean,
                                                                                             return_std,
                                                                                             100 * return_mean / global_var.INITIAL_BALANCE))

    return

    baseline_agent = agent_factory('Hold', env_eval)
    returns_baseline = []
    reward_memory_baseline = []
    return_memory_baseline = []
    asset_memory_baseline = []
    for i in range(episode):
        total_rewards = 0
        # print(global_var.SEP_LINE1)
        # print('Agent:', f'episode {i+1}/{episode} begins.')
        state = env_eval.reset()
        while True:
            action = baseline_agent.act(state)
            next_state, reward, done, _ = env_eval.step(action)
            state = next_state
            total_rewards += reward
            if done:
                ret = total_rewards / global_var.REWARD_SCALING
                print('Agent:', 'episode {:0>2d}/{}, return(total reward) {:.2f}'.format(i + 1, episode, ret))
                returns_baseline.append(ret)
                reward_memory_baseline = env_eval.reward_memory
                return_memory_baseline = np.cumsum(reward_memory_baseline)
                asset_memory_baseline = [a[-1] for a in env_eval.asset_memory]
                # env_eval.save_result('./figs/simulation/Hold_Eval/total_assets.png', './figs/simulation/Hold_Eval/rewards.png')
                break
    return_mean, return_std = np.mean(returns_baseline), np.std(returns_baseline)
    print('Agent:',
          'total {} episodes, average return {:.2f}, std {:.2f}, return rate {:.2f}%'.format(episode, return_mean,
                                                                                             return_std,
                                                                                             100 * return_mean / global_var.INITIAL_BALANCE))
    from datetime import datetime
    x = [datetime.strptime(str(d), '%Y%m%d').date() for d in env_eval.dates][:-1]

    # util.plot_daily_compare(x, return_memory_agent, return_memory_baseline, diff_y_scale=False, path='./figs/simulation/PPO_2M_Eval/total_assets3.png', label_y1='PPO')
    # util.plot_daily_compare(x, asset_memory_agent, asset_memory_baseline, diff_y_scale=False, path='./figs/simulation/PPO_2M_Eval/total_assets_compare.png', label_y1='PPO')
    # util.plot_daily_compare(x, reward_memory_agent, reward_memory_baseline, diff_y_scale=False, path='./figs/simulation/PPO_2M_Eval/daily_reward_compare.png', label_y1='PPO')

    # 绘制波动期（2020.05-2021.12）每日回报
    util.plot_daily_compare(x[540:900], reward_memory_agent[540:900], reward_memory_baseline[540:900],
                            diff_y_scale=False,
                            path='./figs/simulation/PPO_2M_Eval/daily_reward_compare_2020.png', label_y1='PPO')


class Agent():
    """Agent基类。"""

    def act(self, state: np.ndarray) -> np.ndarray:
        pass

    def learn(self, timesteps: int):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass


class DumbAgent(Agent):
    """每天随机采取行动的Agent。测试和对比用。"""

    def __init__(self, env: gym.Env):
        self.env = env

    def learn(self, timesteps: int):
        pass

    def act(self, state: np.ndarray) -> np.ndarray:
        action = self.env.action_space.sample()
        # print('DumbAgent:', 'random action', action)
        return action


class HoldAgent(Agent):
    """仅在第一天平均买入，之后一直持有的Agent。测试和对比用。"""

    def __init__(self, env: gym.Env):
        self.env = env
        self.initial = True

    def learn(self, timesteps: int):
        pass

    def act(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros(self.env.action_space.shape[0])
        if self.initial:
            balance = state[0]
            stock_dim = self.env.stock_dim
            single_stock_budget = balance / stock_dim
            stock_prices = state[1 + stock_dim: 1 + 2 * stock_dim]
            buy_vol = [single_stock_budget / x / global_var.SHARES_PER_TRADE for x in stock_prices]
            action = np.array(buy_vol)
            # print('HoldAgent:', f'day 0, initial balance {balance}, {stock_dim} stocks, single stock budget {single_stock_budget}')
            # print('HoldAgent:', 'stock prices', stock_prices)
            # print('HoldAgent:', 'buy volume', action)
            self.initial = False
        return action


class PPOAgent(Agent):
    """采用PPO算法的Agent。"""

    def __init__(self, env: gym.Env):
        self.model = PPO('MlpPolicy', env, verbose=0)

    def learn(self, timesteps: int):
        self.model.learn(total_timesteps=timesteps)

    def act(self, state: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(state)
        # print('PPOAgent:', action)
        return action

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = PPO.load(path)


class DDPGAgent(Agent):
    """采用DDPG算法的Agent。"""

    def __init__(self, env: gym.Env):
        self.model = DDPG('MlpPolicy', env, verbose=0)

    def learn(self, timesteps: int):
        self.model.learn(total_timesteps=timesteps)

    def act(self, state: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(state)
        # print('DDPGAgent:', action)
        return action

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = DDPG.load(path)


# class NNAgent(Agent):
#     """DL的神经网络Agent，作为RL算法的对比。"""
#
#     def __init__(self, env: gym.Env):
#         self.env = env
#         self.model = NeuralNetwork(env.observation_space.shape[0], env.action_space.shape[0])
#         self.loss_fn = nn.CrossEntropyLoss()
#         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
#
#     def learn(self, timesteps: int = 10000):
#         self.model.train()
#         epochs = timesteps
#         for e in range(1, epochs+1):
#             state = self.env.reset()
#             while True:
#                 action = self.model.act(state)
#                 next_state, reward, done, _ = self.env.step(action)
#                 state = next_state
#                 if done:
#                     break
#
#
#     def act(self, state: np.ndarray) -> np.ndarray:
#         self.model.eval()
#         return self.model(state)
#
#
# class NeuralNetwork(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(NeuralNetwork, self).__init__()
#         half = input_dim // 2
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, half),
#             nn.ReLU(),
#             nn.Linear(half, half),
#             nn.ReLU(),
#             nn.Linear(half, output_dim)
#         )


def agent_factory(agent_name: str, env) -> Agent:
    if agent_name == 'Dumb':
        return DumbAgent(env)
    elif agent_name == 'Hold':
        return HoldAgent(env)
    elif agent_name == 'PPO':
        return PPOAgent(env)
    elif agent_name == 'DDPG':
        return DDPGAgent(env)
    raise ValueError('所需Agent未定义')
