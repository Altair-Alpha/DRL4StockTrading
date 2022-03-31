import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gym
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

import global_var
import util
from preprocessor import *
# from stock_train_env import StockTrainEnv
# from stock_eval_env import StockEvalEnv
from stock_train_env_v2 import StockTrainEnvV2
from stock_eval_env_v2 import StockEvalEnvV2


def run_agent(data: pd.DataFrame = None, model: str = 'Hold', test_episode: int = 1):
    # 训练一个模型，并测试其表现

    stock_codes = get_stock_codes(data)
    data_train = subdata_by_range(data, 20100101, 20171231)
    # print(data_train['close'].head())

    # print(data_train['close'].head())
    # return
    data_train = to_daily_data(data_train)

    # env_train = make_vec_env(StockTrainEnv, n_envs=4,
    #                          env_kwargs={'daily_data': data_train, 'stock_codes': stock_codes, 'verbose': False})
    # model_path = f'./models/0305_A2C_500K_10_Train/'
    # os.makedirs(model_path, exist_ok=True)

    # env_train = VecMonitor(VecNormalize(DummyVecEnv([lambda: StockTrainEnvV2(data_train, stock_codes, False)])))#, './models/TD3_log')
    # env_train = VecNormalize(DummyVecEnv([lambda: StockTrainEnvV2(data_train, stock_codes, False)]))
    # env_train = Monitor(StockTrainEnvV2(data_train, stock_codes, False), './models/TD3_log')
    env_train = StockTrainEnvV2(data_train, stock_codes, False)

    # env_train = make_vec_env(StockTrainEnvV2, n_envs=4, env_kwargs={'daily_data': data_train, 'stock_codes': stock_codes, 'verbose': False})#,
    #                          monitor_dir='./models/TD3_log')
    # DDPG不支持多环境
    #

    agent = agent_factory(model, env_train)

    if global_var.VERBOSE:
        print('Agent:', f'using {model} agent, timestep {5000}')
        print('reward scale', global_var.REWARD_SCALING)
    train_start_time = time.time()
    agent.learn(timesteps=5000)
    # agent.save('./models/TD3_10K')
    # # agent.load('./models/PPO_1M')
    train_end_time = time.time()
    # util.plot_results('./models/TD3_log', './models/TD3_log/lerning_curve.png')

    data_eval = subdata_by_range(data, 20180101, 20180430)
    data_eval = to_daily_data(data_eval)
    env_eval = StockEvalEnvV2(data_eval, stock_codes, verbose=False)

    returns = []
    for i in range(test_episode):
        # agent.initial = True
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
                print('Agent:', 'episode {:0>2d}/{:0>2d}, return(total reward) {:.2f}'.format(i + 1, test_episode, ret))
                returns.append(ret)
                break
    return_mean, return_std = np.mean(returns), np.std(returns)
    print('Agent:',
          'total {} episodes, average return {:.2f}, std {:.2f}, return rate {:.2f}%'.format(test_episode, return_mean,
                                                                                             return_std,
                                                                                             100 * return_mean / global_var.INITIAL_BALANCE))
    print('Agent:', 'model training time {:.2f} minutes'.format((train_end_time - train_start_time) / 60))


def run_agent_test(data: pd.DataFrame = None, model: str = 'PPO', n_train: int = 10, episode: int = 10):
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

    # model_path = f'./models/EnvV2/A2C/0323_A2C_2M_10_Train/8.zip'
    # model_path = './models/EnvV2/PPO/0308_PPO_2M_10_Train/7.zip'
    model_path = './models/EnvV2/PPO/0330_PPO_2M_10_Train_New/6.zip'
    # os.makedirs(model_path, exist_ok=True)
    for e in range(n_train):
        # env_train = StockTrainEnvV2(data_train, stock_codes, verbose=False)

        # log_path = model_path + f'logs/{e+1}/'
        # os.makedirs(log_path, exist_ok=True)
        # env_train = Monitor(env_train, log_path)

        env_train = make_vec_env(StockTrainEnvV2, n_envs=4,
                                 env_kwargs={'daily_data': data_train, 'stock_codes': stock_codes, 'verbose': False})
                                 # ,monitor_dir=log_path)
        # data_full = subdata_by_range(data, 20190101, 20211231)
        # data_full = to_daily_data(data_full)
        # env_full = StockTrainEnvV1(data_full, stock_codes, verbose=False)

        agent = agent_factory(model, env_train)
        train_start_time = time.time()
        # agent.learn(timesteps=10000)
        train_end_time = time.time()
        # agent.save(model_path + f'{e+1}')
        # util.plot_results(log_path, log_path + 'learning_curve.png')
        train_times.append((train_end_time - train_start_time) / 60)
        agent.load(model_path)
        env_eval = StockEvalEnvV2(data_eval, stock_codes, verbose=False)

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
        # if 100 * ret / global_var.INITIAL_BALANCE < 5:
        #     print('Agent', 'episode {:0>2d}/{} train failed.'.format(e + 1, 10))
        #     e -= 1
        # else:
        returns.append(ret)
    return_mean, return_std = np.mean(returns), np.std(returns)
    print('Agent:',
          'total {} training, average return {:.2f}, std {:.2f}, return rate {:.2f}%'.format(10, return_mean,
                                                                                             return_std,
                                                                                             100 * return_mean / global_var.INITIAL_BALANCE))
    print('Agent:', 'average model training time: {:.2f} minutes'.format(np.mean(train_times)))


def eval_agent_train(data: pd.DataFrame = None, model: str = 'PPO', episode: int = 1, x=0):
    # if global_var.VERBOSE:
    #     print('Agent:', f'evaluating {model} agent on training period(20100101-20171231).')
    stock_codes = get_stock_codes(data)

    data_eval = subdata_by_range(data, 20100101, 20171231)
    data_eval = to_daily_data(data_eval)
    returns = []

    model_path = './models/EnvV2/PPO/0308_PPO_2M_10_Train/'

    for e in range(10):
        # data_full = subdata_by_range(data, 20190101, 20211231)
        # data_full = to_daily_data(data_full)
        # env_full = StockTrainEnvV1(data_full, stock_codes, verbose=False)

        env_eval = StockEvalEnvV2(data_eval, stock_codes, verbose=False)
        agent = agent_factory(model, env_eval)
        agent.load(model_path + f'/{x}.zip')

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
        # print('Agent:', 'episode {:0>2d}/{}, avg return {:.2f}'.format(e + 1, 10, ret))
        returns.append(ret)
    return_mean, return_std = np.mean(returns), np.std(returns)
    print('Agent:',
          'total {} training, average return {:.2f}, std {:.2f}, return rate {:.2f}%'.format(10, return_mean,
                                                                                             return_std,
                                                                                             100 * return_mean / global_var.INITIAL_BALANCE))


def run_agent_keep_train(data: pd.DataFrame = None, model: str = 'PPO', episode: int = 10):
    if global_var.VERBOSE:
        print('Agent:', f'track version. using {model} agent.')
        print('reward scale', global_var.REWARD_SCALING)
    stock_codes = get_stock_codes(data)
    data_train = subdata_by_range(data, 20100101, 20211231)
    data_train = to_daily_data(data_train)
    data_eval = subdata_by_range(data, 20180101, 20211231)
    data_eval = to_daily_data(data_eval)
    returns = []
    train_times = []

    # model_path = f'./models/EnvV2/A2C/0323_A2C_2M_10_Train/'
    # a2c_model_path = './models/EnvV2/A2C/0323_A2C_2M_10_Train/8.zip'
    ppo_model_path = './models/EnvV2/PPO/0308_PPO_2M_10_Train/7.zip'
    # os.makedirs(model_path, exist_ok=True)
    retrain_dates = util.get_quarter_dates(20180101, 20211231)
    for e in range(episode):
        env_train = StockTrainEnvV2(data_train, stock_codes, verbose=False)

        # log_path = model_path + f'logs/{e+1}/'
        # os.makedirs(log_path, exist_ok=True)
        # env_train = Monitor(env_train, log_path)

        # env_train = make_vec_env(StockTrainEnvV2, n_envs=4,
        #                          env_kwargs={'daily_data': data_train, 'stock_codes': stock_codes, 'verbose': False})  #
        # ,monitor_dir=log_path)
        # data_full = subdata_by_range(data, 20190101, 20211231)
        # data_full = to_daily_data(data_full)
        # env_full = StockTrainEnvV1(data_full, stock_codes, verbose=False)

        # load a pre-trained model
        agent = agent_factory(model, env_train)
        agent.model.set_parameters(ppo_model_path)
        # train_start_time = time.time()
        # agent.learn(timesteps=2000000)
        # train_end_time = time.time()
        # agent.save(model_path + f'{e+1}')
        # util.plot_results(log_path, log_path + 'learning_curve.png')
        # train_times.append((train_end_time - train_start_time) / 60)
        env_eval = StockEvalEnvV2(data_eval, stock_codes, verbose=False)
        state = env_eval.reset()
        # agent.initial = True
        ret = 0
        for i in range(len(retrain_dates)-1):
            # print('Agent:', '=================================new quarter=====================================')
            # perform trading on the next quarter
            total_rewards = 0
            env_eval.reset_date(retrain_dates[i], retrain_dates[i+1])
            while True:
                action = agent.act(state)
                next_state, reward, done, _ = env_eval.step(action)
                state = next_state
                total_rewards += reward
                if done:
                    ret += total_rewards / global_var.REWARD_SCALING
                    break
            # print('Agent:', f'quarter {retrain_dates[i]} to {retrain_dates[i+1]} ended, current return {ret}. Training on its data.')
            # continue training on the quarter's data after trading
            if i != len(retrain_dates)-2:
                env_train.reset_date(retrain_dates[i], retrain_dates[i+1])
                agent.learn(25000)
        # ret = 0
        # for i in range(episode):
        #     total_rewards = 0
        #     agent.initial = True
        #     # print(global_var.SEP_LINE1)
        #     # print('Agent:', f'episode {i+1}/{episode} begins.')
        #     state = env_eval.reset()
        #     while True:
        #         action = agent.act(state)
        #         next_state, reward, done, _ = env_eval.step(action)
        #         state = next_state
        #         total_rewards += reward
        #         if done:
        #             ret += total_rewards / global_var.REWARD_SCALING
        #             break
        print('Agent:', 'episode {:0>2d}/{}, avg return {:.2f}'.format(e + 1, 10, ret))
        # if 100 * ret / global_var.INITIAL_BALANCE < 5:
        #     print('Agent', 'episode {:0>2d}/{} train failed.'.format(e + 1, 10))
        #     e -= 1
        # else:
        returns.append(ret)
    return_mean, return_std = np.mean(returns), np.std(returns)
    print('Agent:',
          'total {} training, average return {:.2f}, std {:.2f}, return rate {:.2f}%'.format(10, return_mean,
                                                                                             return_std,
                                                                                             100 * return_mean / global_var.INITIAL_BALANCE))
    # print('Agent:', 'average model training time: {:.2f} minutes'.format(np.mean(train_times)))


def eval_agent(data: pd.DataFrame = None, model: str = 'PPO', episode: int = 1):
    stock_codes = get_stock_codes(data)
    data_eval = subdata_by_range(data, 20180101, 20211231)
    data_eval = to_daily_data(data_eval)
    env_eval = StockEvalEnvV2(data_eval, stock_codes, False)

    model_path = './models/EnvV2/PPO/0308_PPO_2M_10_Train/7.zip'
    output_path = './figs/simulation/EnvV2_PPO_2M_Eval1/'
    os.makedirs(output_path, exist_ok=True)

    agent = agent_factory(model, env_eval)
    agent.load(model_path)

    returns_agent = []
    reward_memory_agent = []
    return_memory_agent = []
    asset_memory_agent = []
    for i in range(episode):
        total_rewards = 0
        # print(global_var.SEP_LINE1)
        # print('Agent:', f'episode {i+1}/{episode} begins.')
        agent.initial = True

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
                # return_memory_agent = np.cumsum(reward_memory_agent)
                asset_memory_agent = [a[-1] for a in env_eval.asset_memory]
                env_eval.save_result(output_path + 'total_assets.png', output_path + 'rewards.png')
                os.makedirs(output_path + 'env_memory', exist_ok=True)
                env_eval.dump_memory(output_path + 'env_memory/')
                break
    print('Reward:', f'max:{max(reward_memory_agent)}', f'min:{min(reward_memory_agent)}')
    return_mean, return_std = np.mean(returns_agent), np.std(returns_agent)
    print('Agent:',
          'total {} episodes, average return {:.2f}, std {:.2f}, return rate {:.2f}%'.format(episode, return_mean,
                                                                                             return_std,
                                                                                             100 * return_mean / global_var.INITIAL_BALANCE))

    # return

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
                # return_memory_baseline = np.cumsum(reward_memory_baseline)
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

    # util.plot_daily_compare(x, return_memory_agent, return_memory_baseline, diff_y_scale=False, path='./figs/simulation/DDPG_500K_Eval/total_assets3.png', label_y1='DDPG')
    util.plot_daily_compare(x, asset_memory_baseline, asset_memory_agent, y1_label='Hold(baseline)', y2_label=model, diff_y_scale=False, save_path=output_path + 'total_assets_compare.png')
    util.plot_daily_compare(x, reward_memory_baseline, reward_memory_agent, y1_label='Hold(baseline)', y2_label=model, diff_y_scale=False, save_path=output_path + 'daily_reward_compare.png')

    # # 绘制波动期（2020.05-2021.12）每日回报
    # util.plot_daily_compare(x[540:900], reward_memory_agent[540:900], reward_memory_baseline[540:900],
    #                         diff_y_scale=False,
    #                         path='./figs/simulation/PPO_2M_Eval/daily_reward_compare_2020.png', label_y1='PPO')


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
    """仅在第一天平均买入，之后一直持有的Agent。基线。"""

    def __init__(self, env: gym.Env):
        self.env = env
        global_var.MAX_PERCENTAGE_PER_TRADE = 0.0588
        self.initial = True

    def learn(self, timesteps: int):
        pass

    def act(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros(self.env.action_space.shape[0])
        if self.initial:
            balance = state[0]
            stock_dim = self.env.stock_dim
            single_stock_budget = balance / stock_dim

            action = np.array([1 for _ in range(stock_dim)])
            # stock_prices = state[1 + stock_dim: 1 + 2 * stock_dim]
            # buy_vol = [single_stock_budget / x / global_var.SHARES_PER_TRADE for x in stock_prices]
            # action = np.array(buy_vol)

            # print('HoldAgent:', f'day 0, initial balance {balance}, {stock_dim} stocks, single stock budget {single_stock_budget}')
            # print('HoldAgent:', 'stock prices', stock_prices)
            # print('HoldAgent:', 'buy amount', action)
            self.initial = False
        return action


class PPOAgent(Agent):
    """采用PPO算法的Agent。"""

    def __init__(self, env: gym.Env):
        self.model = PPO('MlpPolicy', env, verbose=0)

    def learn(self, timesteps: int):
        self.model.learn(total_timesteps=timesteps)

    def act(self, state: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(state) # eval时使用确定策略
        # print('PPOAgent:', action)
        return action

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = PPO.load(path)


class A2CAgent(Agent):
    """采用A2C算法的Agent。"""

    def __init__(self, env: gym.Env):
        self.model = A2C('MlpPolicy', env, verbose=0)

    def learn(self, timesteps: int):
        self.model.learn(total_timesteps=timesteps)

    def act(self, state: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(state)
        # print('A2CAgent:', action)
        return action

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = A2C.load(path)


class TD3Agent(Agent):
    """采用TD3算法的Agent。"""

    def __init__(self, env: gym.Env):
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        self.model = TD3('MlpPolicy', env, verbose=1, action_noise=action_noise, learning_starts=1000, learning_rate=0.0002)

    def learn(self, timesteps: int):
        self.model.learn(total_timesteps=timesteps)

    def act(self, state: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(state)
        print('TD3Agent:', action.tolist())
        return action

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = TD3.load(path)


class SACAgent(Agent):
    """采用SAC算法的Agent。"""

    def __init__(self, env: gym.Env):
        self.model = SAC('MlpPolicy', env, verbose=1, gamma=0.99, learning_rate=0.0001, learning_starts=100, batch_size=512)

    def learn(self, timesteps: int):
        self.model.learn(total_timesteps=timesteps)

    def act(self, state: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(state)
        print('SACAgent:', action.tolist())
        return action

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = SAC.load(path)

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
    elif agent_name == 'A2C':
        return A2CAgent(env)
    elif agent_name == 'TD3':
        return TD3Agent(env)
    elif agent_name == 'SAC':
        return SACAgent(env)
    raise ValueError('所需Agent未定义')
