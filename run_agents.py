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
import agents
from stock_train_env_v2 import StockTrainEnvV2
from stock_eval_env_v2 import StockEvalEnvV2


def get_train_env(data: pd.DataFrame, stock_codes: list,
                  agent_name: str, log_path: str = None):
    """
    根据Agent类型和是否输出日生成合适的训练环境

    :param data: 每日股票数据
    :param stock_codes: 股票代码列表
    :param agent_name: 模型名称，目前可选：'Dumb', 'Hold', 'A2C', 'PPO'
    :param log_path: 日志保存路径，如为None（默认）则不保存
    """
    # A2C and PPO support multi-environment training
    if agent_name == 'A2C' or agent_name == 'PPO':
        if log_path is not None:
            env = make_vec_env(StockTrainEnvV2, n_envs=4,
                               env_kwargs={'daily_data': data,
                                           'stock_codes': stock_codes, 'verbose': False},
                               monitor_dir=log_path)
        else:
            env = make_vec_env(StockTrainEnvV2, n_envs=4,
                               env_kwargs={'daily_data': data,
                                           'stock_codes': stock_codes, 'verbose': False})
    else:
        if log_path is not None:
            env = Monitor(StockTrainEnvV2(data, stock_codes, False), log_path)
        else:
            env = StockTrainEnvV2(data, stock_codes, False)

    return env


def eval_agent(agent: agents.Agent, env_eval: gym.Env) -> float:
    """
    测试模型在模拟环境中完成一轮交易的表现

    :param agent: 模型
    :param env_eval: 环境
    :return: 模型在环境中完成一轮交易的收益金额
    """
    if isinstance(agent, agents.HoldAgent):
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
            break
    return ret


def train_eval_agent(data: pd.DataFrame, agent_name: str = 'A2C',
                     train_timesteps: int = 10000, eval_episode: int = 10,
                     model_save_path: str = None, log_path: str = None):
    """
    训练一个模型，并测试其表现

    :param data: 预处理后的完整股票数据
    :param agent_name: Agent名称，目前可选：'Dumb', 'Hold', 'A2C', 'PPO'
    :param train_timesteps: 模型训练交互步数
    :param eval_episode: 模型重复测试次数
    :param model_save_path: 模型保存路径，如为None（默认）则不保存
    :param log_path: 模型训练日志保存路径，如为None（默认）则不保存
    """

    stock_codes = pp.get_stock_codes(data)
    data_train = pp.subdata_by_range(data, global_var.TRAIN_START_DATE, global_var.TRAIN_END_DATE)
    data_train = pp.to_daily_data(data_train)
    env_train = get_train_env(data_train, stock_codes, agent_name, log_path)

    # Create and train the agent
    if global_var.VERBOSE:
        print('RunAgent:', f'training {agent_name} agent, timesteps {train_timesteps}')

    agent = agents.agent_factory(agent_name, env_train)
    train_start_time = time.time()  # record training time
    agent.learn(timesteps=train_timesteps)
    train_end_time = time.time()

    # Save the agent's model and plot learning curve(optional)
    if model_save_path is not None:
        os.makedirs(model_save_path, exist_ok=True)
        agent.save(model_save_path)
    if log_path is not None:
        util.plot_results(log_path, log_path + 'learning_curve.png')

    # agent.load('./models/EnvV2/CYB_Data/0407_A2C_1M_10_Train/1.zip')

    data_eval = pp.subdata_by_range(data, global_var.TEST_START_DATE, global_var.TEST_END_DATE)
    data_eval = pp.to_daily_data(data_eval)
    env_eval = StockEvalEnvV2(data_eval, stock_codes, verbose=False)

    # Evaluate the agent
    if global_var.VERBOSE:
        print('RunAgent:', f'evaluating trained {agent_name} agent for {eval_episode} episodes:')

    returns = []
    for i in range(eval_episode):
        ret = eval_agent(agent, env_eval)
        print('RunAgent:', 'episode {:0>2d}/{:0>2d}, return {:.2f}'.format(i + 1, eval_episode, ret))
        returns.append(ret)

    return_mean, return_std = np.mean(returns), np.std(returns)
    # 因为实验中一般将测试区间定为某年1月1日至某年12月31日，故对这两个日期求year_diff再+1才是总年数
    yearly_return_rate = 100 * return_mean / global_var.INITIAL_BALANCE\
                    / (util.get_year_diff(global_var.TEST_START_DATE, global_var.TEST_END_DATE) + 1)

    print('RunAgent:',
          'total {} episodes, average return {:.2f}, std {:.2f}, '
          'yearly return rate {:.2f}%'.format(eval_episode, return_mean, return_std, yearly_return_rate))
    print('RunAgent:',
          '{} agent average training time: {:.2f} minutes'.format(agent_name,
                                                                  (train_end_time - train_start_time) / 60))


def train_test_agent_ntimes(data: pd.DataFrame = None, agent_name: str = 'PPO', train_timesteps: int = 10000,
                            n_train: int = 10, eval_episode: int = 10,
                            model_save_path: str = None, log_path: str = None):
    """
    训练一种模型n_train次，测试其平均表现

    :param data: 预处理后的完整股票数据
    :param agent_name: Agent名称，目前可选：'Dumb', 'Hold', 'A2C', 'PPO'
    :param train_timesteps: 模型每次训练的交互步数
    :param n_train: 重复训练次数
    :param eval_episode: 模型每次训练的重复测试次数
    :param model_save_path: 模型保存路径，如为None（默认）则不保存，否则n_train个模型分别保存至该目录下的1.zip-n.zip文件
    :param log_path: 模型训练日志保存路径，如为None（默认）则不保存，否则n_train次训练的日志分别保存至该目录下的1-n目录中
    """

    stock_codes = pp.get_stock_codes(data)
    data_train = pp.subdata_by_range(data, global_var.TRAIN_START_DATE, global_var.TRAIN_END_DATE)
    data_train = pp.to_daily_data(data_train)
    data_eval = pp.subdata_by_range(data, global_var.TEST_START_DATE, global_var.TEST_END_DATE)
    data_eval = pp.to_daily_data(data_eval)

    # model_path = f'./models/EnvV2/A2C/0323_A2C_2M_10_Train/8.zip'
    # model_path = './models/EnvV2/PPO/0308_PPO_2M_10_Train/7.zip'
    model_path = f'./models/EnvV2/CYB_Data/0409_PPO_2M_10_Train/'
    # model_save_path = model_path

    if model_save_path is not None:
        os.makedirs(model_save_path, exist_ok=True)

    if global_var.VERBOSE:
        print('RunAgent:', f'training and evaluating {agent_name} agent for {n_train} times')

    returns = []
    train_elapsed_times = []

    for i in range(n_train):
        train_i_log_path = None
        if log_path is not None:
            train_i_log_path = log_path + f'{i+1}/'
            os.makedirs(train_i_log_path, exist_ok=True)

        # Train
        env_train = get_train_env(data_train, stock_codes, agent_name, train_i_log_path)
        agent = agents.agent_factory(agent_name, env_train)
        train_start_time = time.time()
        agent.learn(timesteps=train_timesteps)
        train_end_time = time.time()
        train_elapsed_times.append((train_end_time - train_start_time) / 60)

        if model_save_path is not None:
            agent.save(model_path + f'{i+1}')
        if log_path is not None:
            util.plot_results(train_i_log_path, train_i_log_path + 'learning_curve.png')

        # agent.load(model_path)

        # Eval
        env_eval = StockEvalEnvV2(data_eval, stock_codes, verbose=False)
        ret = 0
        for _ in range(eval_episode):
            ret += eval_agent(agent, env_eval)
        ret /= eval_episode
        print('RunAgent:', 'episode {:0>2d}/{}, average return {:.2f}'.format(i+1, eval_episode, ret))
        returns.append(ret)

    return_mean, return_std = np.mean(returns), np.std(returns)
    yearly_return_rate = 100 * return_mean / global_var.INITIAL_BALANCE \
                         / (util.get_year_diff(global_var.TEST_START_DATE, global_var.TEST_END_DATE) + 1)

    print('RunAgent:',
          'total {} training, average return {:.2f}, std {:.2f}, '
          'yearly return rate {:.2f}%'.format(n_train, return_mean, return_std, yearly_return_rate))
    print('RunAgent:',
          '{} agent average training time: {:.2f} minutes'.format(agent_name, np.mean(train_elapsed_times)))


# TO BE FIXED
def eval_agent_train(data: pd.DataFrame = None, model: str = 'PPO', episode: int = 1, x=0):
    # if global_var.VERBOSE:
    #     print('RunAgent:', f'evaluating {model} agent on training period(20100101-20171231).')
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
        agent = agents.agent_factory(model, env_eval)
        agent.load(model_path + f'/{x}.zip')

        ret = 0
        for i in range(episode):
            total_rewards = 0
            agent.initial = True
            # print(global_var.SEP_LINE1)
            # print('RunAgent:', f'episode {i+1}/{episode} begins.')
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
        # print('RunAgent:', 'episode {:0>2d}/{}, avg return {:.2f}'.format(e + 1, 10, ret))
        returns.append(ret)
    return_mean, return_std = np.mean(returns), np.std(returns)
    print('RunAgent:',
          'total {} training, average return {:.2f}, std {:.2f}, return rate {:.2f}%'.format(10, return_mean,
                                                                                             return_std,
                                                                                             100 * return_mean / global_var.INITIAL_BALANCE))


def run_agent_keep_train(data: pd.DataFrame = None, model: str = 'A2C', episode: int = 10):
    if global_var.VERBOSE:
        print('RunAgent:', f'track version. using {model} agent.')
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
            # print('RunAgent:', '=================================new quarter=====================================')
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
            # print('RunAgent:', f'quarter {retrain_dates[i]} to {retrain_dates[i+1]} ended, current return {ret}. Training on its data.')
            # continue training on the quarter's data after trading
            if i != len(retrain_dates)-2:
                """TO FIX"""
                env_train.reset_date(retrain_dates[i], retrain_dates[i + 1])
                agent.learn(25000)

        print('RunAgent:', 'episode {:0>2d}/{}, avg return {:.2f}'.format(e + 1, 10, ret))
        returns.append(ret)
    return_mean, return_std = np.mean(returns), np.std(returns)
    print('RunAgent:',
          'total {} training, average return {:.2f}, std {:.2f}, return rate {:.2f}%'.format(10, return_mean,
                                                                                             return_std,
                                                                                             100 * return_mean / global_var.INITIAL_BALANCE))
    # print('RunAgent:', 'average model training time: {:.2f} minutes'.format(np.mean(train_times)))


def eval_agent_with_log(data: pd.DataFrame = None, model: str = 'A2C', episode: int = 1):
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
        # print('RunAgent:', f'episode {i+1}/{episode} begins.')
        agent.initial = True

        state = env_eval.reset()
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env_eval.step(action)
            state = next_state
            total_rewards += reward
            if done:
                ret = total_rewards / global_var.REWARD_SCALING
                print('RunAgent:', 'episode {:0>2d}/{}, return(total reward) {:.2f}'.format(i + 1, episode, ret))
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
    print('RunAgent:',
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
        # print('RunAgent:', f'episode {i+1}/{episode} begins.')
        state = env_eval.reset()
        while True:
            action = baseline_agent.act(state)
            next_state, reward, done, _ = env_eval.step(action)
            state = next_state
            total_rewards += reward
            if done:
                ret = total_rewards / global_var.REWARD_SCALING
                print('RunAgent:', 'episode {:0>2d}/{}, return(total reward) {:.2f}'.format(i + 1, episode, ret))
                returns_baseline.append(ret)
                reward_memory_baseline = env_eval.reward_memory
                # return_memory_baseline = np.cumsum(reward_memory_baseline)
                asset_memory_baseline = [a[-1] for a in env_eval.asset_memory]
                # env_eval.save_result('./figs/simulation/Hold_Eval/total_assets.png', './figs/simulation/Hold_Eval/rewards.png')
                break
    return_mean, return_std = np.mean(returns_baseline), np.std(returns_baseline)
    print('RunAgent:',
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
        print('RunAgent:', f'evaluating track trained {model} agent.')
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
        print('RunAgent:', f'original model file at {model_path}. outputing eval result at {output_path}')


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
        # print('RunAgent:', '=================================new quarter=====================================')
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
        # print('RunAgent:', f'quarter {retrain_dates[i]} to {retrain_dates[i+1]} ended, current return {ret_agent}. Training on its data.')
        # continue training on the quarter's data after trading
        if i != len(retrain_dates)-2:
            env_train.reset_date(retrain_dates[i], retrain_dates[i + 1])
            agent_track.learn(track_train_step)
    reward_memory_agent_track = env_eval.reward_memory
    asset_memory_agent_track = [a[-1] for a in env_eval.asset_memory]
    os.makedirs(output_path + 'env_memory', exist_ok=True)
    env_eval.dump_memory(output_path + 'env_memory/')

    print('RunAgent:',
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
        print('RunAgent:', 'Original agent average return {:.2f}, yearly return rate {:.2f}%'.format(ret_original_agent,
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
    print('RunAgent:','Baseline average return {:.2f}, yearly return rate {:.2f}%'.format(ret_baseline,100 * ret_baseline / 4 / global_var.INITIAL_BALANCE))

    dates = [datetime.strptime(str(d), '%Y%m%d').date() for d in env_eval.full_dates][:-1]

    util.plot_daily_multi_y(x=dates, ys=[asset_memory_agent_track, asset_memory_original_agent, asset_memory_baseline],
                            ys_label=[model+f'(Track Step={track_train_step})', model+'(No Track Train)', 'Hold(Baseline)'],
                            save_path=output_path + 'total_assets_compare.png')
    util.plot_daily_multi_y(x=dates, ys=[reward_memory_agent_track, reward_memory_original_agent, reward_memory_baseline],
                            ys_label=[model + f'(Track Step={track_train_step})', model + '(No Track Train)', 'Hold(Baseline)'],
                            save_path=output_path + 'daily_reward_compare.png')

