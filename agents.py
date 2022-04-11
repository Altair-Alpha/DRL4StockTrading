import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import gym
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

import global_var
import util
import preprocessor as pp
from stock_train_env_v2 import StockTrainEnvV2
from stock_eval_env_v2 import StockEvalEnvV2


def train_test_single_agent(data: pd.DataFrame, model: str = 'A2C',
                            train_timesteps: int = 10000, test_episode: int = 10,
                            model_save_path: str = None, log_path: str = None):
    """
    训练一个模型，并测试其表现

    :param data: 预处理后的完整股票数据
    :param model: 使用的模型名称，目前可用的有'Dumb', 'Hold', 'A2C', 'PPO'
    :param train_timesteps: 模型训练交互步数
    :param test_episode: 模型测试重复次数
    :param model_save_path: 模型保存路径，如为None（默认）则不保存
    :param log_path: 模型训练日志保存路径，如为None（默认）则不保存
    """

    stock_codes = pp.get_stock_codes(data)
    data_train = pp.subdata_by_range(data, global_var.TRAIN_START_DATE, global_var.TRAIN_END_DATE)
    data_train = pp.to_daily_data(data_train)

    # A2C and PPO support multi-environment training
    if model == 'A2C' or model == 'PPO':
        if log_path is not None:
            env_train = make_vec_env(StockTrainEnvV2, n_envs=4,
                                     env_kwargs={'daily_data': data_train,
                                                 'stock_codes': stock_codes, 'verbose': False},
                                     monitor_dir=log_path)
        else:
            env_train = make_vec_env(StockTrainEnvV2, n_envs=4,
                                     env_kwargs={'daily_data': data_train,
                                                 'stock_codes': stock_codes, 'verbose': False})
    else:
        if log_path is not None:
            env_train = Monitor(StockTrainEnvV2(data_train, stock_codes, False), log_path)
        else:
            env_train = StockTrainEnvV2(data_train, stock_codes, False)

    agent = agent_factory(model, env_train)

    if global_var.VERBOSE:
        print('Agent:', f'training {model} agent, timesteps {train_timesteps}')

    train_start_time = time.time()  # record training time
    agent.learn(timesteps=train_timesteps)
    train_end_time = time.time()

    if model_save_path is not None:
        os.makedirs(model_save_path, exist_ok=True)
        agent.save(model_save_path)
    if log_path is not None:
        util.plot_results(log_path, log_path + 'lerning_curve.png')

    # agent.load('./models/EnvV2/CYB_Data/0407_A2C_1M_10_Train/1.zip')

    data_eval = pp.subdata_by_range(data, global_var.TEST_START_DATE, global_var.TEST_END_DATE)
    data_eval = pp.to_daily_data(data_eval)
    env_eval = StockEvalEnvV2(data_eval, stock_codes, verbose=False)

    returns = []
    for i in range(test_episode):
        if model == 'Hold':
            agent.is_first_day = True

        total_rewards = 0
        state = env_eval.reset()
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env_eval.step(action)
            state = next_state
            total_rewards += reward
            if done:
                ret = total_rewards / global_var.REWARD_SCALING
                print('Agent:', 'episode {:0>2d}/{:0>2d}, return {:.2f}'.format(i + 1, test_episode, ret))
                returns.append(ret)
                break

    return_mean, return_std = np.mean(returns), np.std(returns)
    # 因为测试区间定为xxxx0101-xxxx1231，故它们求year_diff再+1才是正确的总年数
    yearly_return = 100 * return_mean / global_var.INITIAL_BALANCE\
                    / (util.get_year_diff(global_var.TEST_START_DATE, global_var.TEST_END_DATE) + 1)

    print('Agent:', 'total {} episodes, '
                    'average return {:.2f}, std {:.2f}, '
                    'yearly return rate {:.2f}%'.format(test_episode, return_mean, return_std, yearly_return))
    print('Agent:', '{} model average training time: {:.2f} minutes'.format(model, (train_end_time - train_start_time) / 60))


def run_agent_test(data: pd.DataFrame = None, model: str = 'PPO', n_train: int = 10, episode: int = 10):
    if global_var.VERBOSE:
        print('Agent:', f'test version. using {model} agent.')
        print('reward scale', global_var.REWARD_SCALING)
    stock_codes = pp.get_stock_codes(data)
    data_train = pp.subdata_by_range(data, 20120101, 20171231)
    data_train = pp.to_daily_data(data_train)
    data_eval = pp.subdata_by_range(data, 20180101, 20211231)
    data_eval = pp.to_daily_data(data_eval)
    returns = []
    train_times = []

    # model_path = f'./models/EnvV2/A2C/0323_A2C_2M_10_Train/8.zip'
    # model_path = './models/EnvV2/PPO/0308_PPO_2M_10_Train/7.zip'
    model_path = f'./models/EnvV2/CYB_Data/0409_PPO_2M_10_Train/'
    os.makedirs(model_path, exist_ok=True)
    for e in range(n_train):
        # env_train = StockTrainEnvV2(data_train, stock_codes, verbose=False)

        log_path = model_path + f'logs/{e+1}/'
        os.makedirs(log_path, exist_ok=True)
        # env_train = Monitor(env_train, log_path)

        env_train = make_vec_env(StockTrainEnvV2, n_envs=4,
                                 env_kwargs={'daily_data': data_train, 'stock_codes': stock_codes, 'verbose': False}#)
                                 ,monitor_dir=log_path)
        # data_full = subdata_by_range(data, 20190101, 20211231)
        # data_full = to_daily_data(data_full)
        # env_full = StockTrainEnvV1(data_full, stock_codes, verbose=False)

        agent = agent_factory(model, env_train)
        train_start_time = time.time()
        agent.learn(timesteps=2000000)
        train_end_time = time.time()
        agent.save(model_path + f'{e+1}')
        util.plot_results(log_path, log_path + 'learning_curve.png')
        train_times.append((train_end_time - train_start_time) / 60)
        # agent.load(model_path)
        env_eval = StockEvalEnvV2(data_eval, stock_codes, verbose=False)

        ret = 0
        for i in range(episode):
            total_rewards = 0
            # agent.initial = True
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
    stock_codes = pp.get_stock_codes(data)

    data_eval = pp.subdata_by_range(data, 20100101, 20171231)
    data_eval = pp.to_daily_data(data_eval)
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


def run_agent_keep_train(data: pd.DataFrame = None, model: str = 'A2C', episode: int = 10):
    if global_var.VERBOSE:
        print('Agent:', f'track version. using {model} agent.')
        print('reward scale', global_var.REWARD_SCALING)
    stock_codes = pp.get_stock_codes(data)
    data_train = pp.subdata_by_range(data, 20120101, 20211231)
    data_train = pp.to_daily_data(data_train)
    data_eval = pp.subdata_by_range(data, 20180101, 20211231)
    data_eval = pp.to_daily_data(data_eval)
    returns = []
    train_times = []

    # model_path = f'./models/EnvV2/A2C/0323_A2C_2M_10_Train/'
    a2c_model_path = './models/EnvV2/CYB_Data/0407_A2C_2M_10_Train/10.zip'#'./models/EnvV2/A2C/0323_A2C_2M_10_Train/8.zip'
    # ppo_model_path = './models/EnvV2/PPO/0308_PPO_2M_10_Train/7.zip'
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

        # load a pre-trained model
        agent = agent_factory(model, env_train)
        agent.model.set_parameters(a2c_model_path)
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
            if not env_train.check_interval_valid(retrain_dates[i], retrain_dates[i + 1]):
                # print(f'Quarter {retrain_dates[i]} to {retrain_dates[i + 1]} has no data, therefore skipped.')
                continue
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
                """TO FIX"""
                env_train.reset_date(retrain_dates[i], retrain_dates[i + 1])
                agent.learn(25000)

        print('Agent:', 'episode {:0>2d}/{}, avg return {:.2f}'.format(e + 1, 10, ret))
        returns.append(ret)
    return_mean, return_std = np.mean(returns), np.std(returns)
    print('Agent:',
          'total {} training, average return {:.2f}, std {:.2f}, return rate {:.2f}%'.format(10, return_mean,
                                                                                             return_std,
                                                                                             100 * return_mean / global_var.INITIAL_BALANCE))
    # print('Agent:', 'average model training time: {:.2f} minutes'.format(np.mean(train_times)))


def eval_agent(data: pd.DataFrame = None, model: str = 'A2C', episode: int = 1):
    stock_codes = pp.get_stock_codes(data)
    data_eval = pp.subdata_by_range(data, 20180101, 20211231)
    data_eval = pp.to_daily_data(data_eval)
    env_eval = StockEvalEnvV2(data_eval, stock_codes, False)

    # model_path = './models/EnvV2/PPO/0308_PPO_2M_10_Train/7.zip'
    # output_path = './figs/simulation/EnvV2_PPO_2M_Eval1/'

    # model_path = './models/EnvV2/CYB_Data/0407_A2C_1M_10_Train/1.zip'
    model_path = './models/EnvV2/CYB_Data/0409_PPO_2M_10_Train/6.zip'
    output_path = './figs/simulation/CYB_PPO_2M_Eval/'
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
    x = [datetime.strptime(str(d), '%Y%m%d').date() for d in env_eval.dates][:-1]

    # util.plot_daily_compare(x, return_memory_agent, return_memory_baseline, diff_y_scale=False, path='./figs/simulation/DDPG_500K_Eval/total_assets3.png', label_y1='DDPG')
    util.plot_daily_compare(x, asset_memory_baseline, asset_memory_agent, y1_label='Hold(baseline)', y2_label=model, diff_y_scale=False, save_path=output_path + 'total_assets_compare.png')
    util.plot_daily_compare(x, reward_memory_baseline, reward_memory_agent, y1_label='Hold(baseline)', y2_label=model, diff_y_scale=False, save_path=output_path + 'daily_reward_compare.png')

    # # 绘制波动期（2020.05-2021.12）每日回报
    # util.plot_daily_compare(x[540:900], reward_memory_agent[540:900], reward_memory_baseline[540:900],
    #                         diff_y_scale=False,
    #                         path='./figs/simulation/PPO_2M_Eval/daily_reward_compare_2020.png', label_y1='PPO')


def eval_track_train_agent(data: pd.DataFrame = None, model: str = 'A2C', track_train_step: int = 10000):
    if global_var.VERBOSE:
        print('Agent:', f'evaluating track trained {model} agent.')
    stock_codes = pp.get_stock_codes(data)
    data_train = pp.subdata_by_range(data, 20100101, 20211231)
    data_train = pp.to_daily_data(data_train)
    data_eval = pp.subdata_by_range(data, 20180101, 20211231)
    data_eval = pp.to_daily_data(data_eval)
    returns = []
    train_times = []

    # model_path = f'./models/EnvV2/A2C/0323_A2C_2M_10_Train/'
    a2c_model_path = './models/EnvV2/A2C/0323_A2C_2M_10_Train/8.zip'
    # './models/EnvV2/CYB_Data/0407_A2C_2M_10_Train/10.zip'
    ppo_model_path = './models/EnvV2/PPO/0308_PPO_2M_10_Train/7.zip'

    model_path = a2c_model_path
    output_path = './figs/simulation/EnvV2_A2C_2M_Track_10K_Eval/'
    os.makedirs(output_path, exist_ok=True)
    if global_var.VERBOSE:
        print('Agent:', f'original model file at {model_path}. outputing eval result at {output_path}')


    ###################### Track Trained Model Perf ######################
    retrain_dates = util.get_quarter_dates(20180101, 20220101)

    env_train = StockTrainEnvV2(data_train, stock_codes, verbose=False)
    env_eval = StockEvalEnvV2(data_eval, stock_codes, verbose=False)

    # load a pre-trained model
    agent_track = agent_factory(model, env_train)

    # agent.load() will clear out the env so that the agent can't learn anymore
    # therefore use set_parameters() to load the model
    agent_track.model.set_parameters(model_path)

    state = env_eval.reset()
    ret_agent_track = 0

    for i in range(len(retrain_dates)-1):
        # print('Agent:', '=================================new quarter=====================================')
        # perform trading on the next quarter
        if not env_train.check_interval_valid(retrain_dates[i], retrain_dates[i + 1]):
            # the quarter has no data, so we shouldn't test or train the model on it.
            # print(f'Quarter {retrain_dates[i]} to {retrain_dates[i + 1]} has no data, therefore skipped.')
            continue
        total_rewards = 0
        reseted_dates = env_eval.reset_date(retrain_dates[i], retrain_dates[i+1], is_last_section=(i==len(retrain_dates)-2))
        while True:
            action = agent_track.act(state)
            next_state, reward, done, _ = env_eval.step(action)
            state = next_state
            total_rewards += reward
            if done:
                ret_agent_track += total_rewards / global_var.REWARD_SCALING
                # print(f'Quarter {retrain_dates[i]} to {retrain_dates[i + 1]} done. traded dates: {reseted_dates}')
                break
        # print('Agent:', f'quarter {retrain_dates[i]} to {retrain_dates[i+1]} ended, current return {ret_agent}. Training on its data.')
        # continue training on the quarter's data after trading
        if i != len(retrain_dates)-2:
            env_train.reset_date(retrain_dates[i], retrain_dates[i + 1])
            agent_track.learn(track_train_step)
    reward_memory_agent_track = env_eval.reward_memory
    asset_memory_agent_track = [a[-1] for a in env_eval.asset_memory]
    os.makedirs(output_path + 'env_memory', exist_ok=True)
    env_eval.dump_memory(output_path + 'env_memory/')

    print('Agent:',
          'Track trained agent average return {:.2f}, yearly return rate {:.2f}%'.format(ret_agent_track, 100 * ret_agent_track / 4 / global_var.INITIAL_BALANCE))


    ###################### Original Model Perf (No Track Train) ######################
    while True:
        env_eval = StockEvalEnvV2(data_eval, stock_codes, verbose=False)
        original_agent = agent_factory(model, env_eval)
        original_agent.load(model_path)

        total_rewards = 0
        reward_memory_original_agent = []
        asset_memory_original_agent = []
        state = env_eval.reset()
        while True:
            action = original_agent.act(state)
            next_state, reward, done, _ = env_eval.step(action)
            state = next_state
            total_rewards += reward
            if done:
                ret_original_agent = total_rewards / global_var.REWARD_SCALING
                reward_memory_original_agent = env_eval.reward_memory
                asset_memory_original_agent = [a[-1] for a in env_eval.asset_memory]
                break
        print('Agent:', 'Original agent average return {:.2f}, yearly return rate {:.2f}%'.format(ret_original_agent,
                                                                                            100 * ret_original_agent / 4 / global_var.INITIAL_BALANCE))
        if ret_original_agent > 900000: break

    ###################### Baseline Perf (Hold Agent) ######################
    env_eval = StockEvalEnvV2(data_eval, stock_codes, verbose=False)
    baseline_agent = agent_factory('Hold', env_eval)
    total_rewards = 0

    state = env_eval.reset()
    while True:
        action = baseline_agent.act(state)
        next_state, reward, done, _ = env_eval.step(action)
        state = next_state
        total_rewards += reward
        if done:
            ret_baseline = total_rewards / global_var.REWARD_SCALING
            reward_memory_baseline = env_eval.reward_memory
            asset_memory_baseline = [a[-1] for a in env_eval.asset_memory]
            break
    print('Agent:','Baseline average return {:.2f}, yearly return rate {:.2f}%'.format(ret_baseline,100 * ret_baseline / 4 / global_var.INITIAL_BALANCE))

    dates = [datetime.strptime(str(d), '%Y%m%d').date() for d in env_eval.full_dates][:-1]

    util.plot_daily_multi_y(x=dates, ys=[asset_memory_agent_track, asset_memory_original_agent, asset_memory_baseline],
                            ys_label=[model+f'(Track Step={track_train_step})', model+'(No Track Train)', 'Hold(Baseline)'],
                            save_path=output_path + 'total_assets_compare.png')
    util.plot_daily_multi_y(x=dates, ys=[reward_memory_agent_track, reward_memory_original_agent, reward_memory_baseline],
                            ys_label=[model + f'(Track Step={track_train_step})', model + '(No Track Train)', 'Hold(Baseline)'],
                            save_path=output_path + 'daily_reward_compare.png')


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
        self.is_first_day = True

    def learn(self, timesteps: int):
        pass

    def act(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros(self.env.action_space.shape[0])
        if self.is_first_day:
            balance = state[0]
            stock_dim = self.env.stock_dim

            # HoldAgent在测试时不应该受TEST_MAX_PERCENTAGE_PER_TRADE参数影响，而是永远保证在第一个交易日将所有资金平均分配到各股票中
            action = np.array([1 / stock_dim / global_var.TEST_MAX_PERCENTAGE_PER_TRADE for _ in range(stock_dim)])

            # print('HoldAgent:', f'day 0, initial balance {balance}, {stock_dim} stocks, single stock budget {single_stock_budget}')
            # print('HoldAgent:', 'stock prices', stock_prices)
            # print('HoldAgent:', 'buy amount', action)
            self.is_first_day = False
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
