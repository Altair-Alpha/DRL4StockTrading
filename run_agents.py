import os
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

import global_var
import util
import preprocessor as pp
from agents import agent_factory
from agents import Agent, DumbAgent, HoldAgent, A2CAgent, PPOAgent
from stock_train_env_v2 import StockTrainEnvV2
from stock_eval_env_v2 import StockEvalEnvV2


def get_train_env(data: pd.DataFrame, agent_name: str, log_path: str = None):
    """
    根据Agent类型和是否输出日生成合适的训练环境

    :param data: 每日股票数据（请先调用to_daily_data处理成字典形式）
    :param agent_name: 模型名称，目前可选：'Dumb', 'Hold', 'A2C', 'PPO'
    :param log_path: 日志保存路径，如为None（默认）则不保存
    """
    # A2C and PPO support multi-environment training

    if agent_name == 'A2C' or agent_name == 'PPO':
        if log_path is not None:
            env = make_vec_env(StockTrainEnvV2, n_envs=4,
                               env_kwargs={'data': data, 'verbose': 0},
                               monitor_dir=log_path)
        else:
            env = make_vec_env(StockTrainEnvV2, n_envs=4,
                               env_kwargs={'data': data, 'verbose': 0})
    else:
        if log_path is not None:
            env = Monitor(StockTrainEnvV2(data, verbose=0), log_path)
        else:
            env = StockTrainEnvV2(data, verbose=0)

    return env


def eval_agent_simple(agent: Agent, env_eval: gym.Env) -> float:
    """
    测试模型在模拟环境中完成一轮交易的表现，仅返回其收益值

    :param agent: Agent对象
    :param env_eval: 环境
    :return: 模型在环境中完成一轮交易的收益金额
    """

    if isinstance(agent, HoldAgent):
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


def eval_agent(agent: Agent, env_eval: StockEvalEnvV2, output_path: str) -> float:
    """
    测试模型在模拟环境中完成一轮交易的表现，将模型的历史输出记录到csv文件，绘制与基线的每日收益和资产曲线对比图，同时返回收益值

    :param agent: Agent对象
    :param env_eval: 环境
    :param output_path: 图表输出路径
    :return: 模型在环境中完成一轮交易的收益金额
    """

    # stock_codes = pp.get_stock_codes(data)
    # data_eval = pp.subdata_by_range(data, global_var.EVAL_START_DATE, global_var.EVAL_END_DATE)
    # data_eval = pp.to_daily_data(data_eval)
    # env_eval = StockEvalEnvV2(data_eval, stock_codes, False)

    # model_path = './models/EnvV2/PPO/0308_PPO_2M_10_Train/7.zip'
    # output_path = './figs/simulation/EnvV2_PPO_2M_Eval1/'

    # model_path = './models/EnvV2/CYB_Data/0407_A2C_1M_10_Train/1.zip'
    # model_path = './models/EnvV2/CYB_Data/0409_PPO_2M_10_Train/6.zip'
    # output_path = './figs/simulation/CYB_PPO_2M_Eval/'
    # agent = agent_factory(agent_name, env_eval)
    # agent.load(model_path)

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path + 'env_memory', exist_ok=True)

    # Eval input agent
    total_rewards = 0
    state = env_eval.reset()
    while True:
        action = agent.act(state)
        next_state, reward, done, _ = env_eval.step(action)
        state = next_state
        total_rewards += reward
        if done:
            ret_agent = total_rewards / global_var.REWARD_SCALING
            reward_memory_agent = env_eval.reward_memory
            asset_memory_agent = [a[-1] for a in env_eval.asset_memory]
            env_eval.plot_memory(output_path)
            env_eval.dump_memory(output_path + 'env_memory/')
            break

    yearly_return_rate = 100 * ret_agent / global_var.INITIAL_BALANCE \
                         / (util.get_year_diff(global_var.EVAL_START_DATE, global_var.EVAL_END_DATE) + 1)
    print('RunAgent:',
          'eval {} agent, average return {:.2f}, yearly return rate {:.2f}%'.format(agent.name,
                                                                                    ret_agent,
                                                                                    yearly_return_rate))

    # Eval baseline agent
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

    yearly_return_rate = 100 * ret_baseline / global_var.INITIAL_BALANCE \
                         / (util.get_year_diff(global_var.EVAL_START_DATE, global_var.EVAL_END_DATE) + 1)
    print('RunAgent:',
          'baseline {} agent, average return {:.2f}, yearly return rate {:.2f}%'.format(baseline_agent.name,
                                                                                        ret_baseline,
                                                                                        yearly_return_rate))

    # Plot compare graph
    trade_dates = [datetime.strptime(str(d), '%Y%m%d').date() for d in env_eval.full_dates][:-1]
    util.plot_daily_compare(x=trade_dates, y1=asset_memory_baseline, y2=asset_memory_agent,
                            y1_label=f'{baseline_agent.name}(baseline)', y2_label=agent.name,
                            diff_y_scale=False, save_path=output_path + 'total_assets_compare.png')
    util.plot_daily_compare(x=trade_dates, y1=reward_memory_baseline, y2=reward_memory_agent,
                            y1_label=f'{baseline_agent.name}(baseline)', y2_label=agent.name,
                            diff_y_scale=False, save_path=output_path + 'daily_reward_compare.png')

    return ret_agent


def train_agent(data: pd.DataFrame, agent_name: str, train_timesteps: int = 10000,
                eval_episode: int = 10, model_save_path: str = None, log_path: str = None):
    """
    训练一个模型，并测试其表现

    :param data: 预处理后的完整股票数据
    :param agent_name: Agent名称，目前可选：'Dumb', 'Hold', 'A2C', 'PPO'
    :param train_timesteps: 模型训练交互步数
    :param eval_episode: 模型重复测试次数
    :param model_save_path: 模型保存路径，如为None（默认）则不保存
    :param log_path: 模型训练日志保存路径，如为None（默认）则不保存
    """

    data_train = pp.subdata_by_range(data, global_var.TRAIN_START_DATE, global_var.TRAIN_END_DATE)
    data_eval = pp.subdata_by_range(data, global_var.EVAL_START_DATE, global_var.EVAL_END_DATE)

    # Create and train the agent
    if global_var.VERBOSE:
        print('RunAgent:', f'training {agent_name} agent, timesteps {train_timesteps}')

    env_train = get_train_env(data_train, agent_name, log_path)
    agent = agent_factory(agent_name, env_train)
    train_start_time = time.time()  # record training time
    agent.learn(timesteps=train_timesteps)
    train_end_time = time.time()

    # Save the agent's model and plot learning curve(optional)
    if model_save_path is not None:
        os.makedirs(model_save_path, exist_ok=True)
        agent.save(model_save_path)
    if log_path is not None:
        util.plot_learning_curve(log_path, log_path + 'learning_curve.png')

    # Evaluate the agent
    agent.eval_mode()
    env_eval = StockEvalEnvV2(data_eval, verbose=0)
    if global_var.VERBOSE:
        print('RunAgent:', f'evaluating trained {agent_name} agent for {eval_episode} episodes:')

    returns = []
    for i in range(eval_episode):
        ret = eval_agent_simple(agent, env_eval)
        print('RunAgent:', 'episode {:0>2d}/{:0>2d}, return {:.2f}'.format(i + 1, eval_episode, ret))
        returns.append(ret)

    return_mean, return_std = np.mean(returns), np.std(returns)
    # 因为实验中一般将测试区间定为某年1月1日至某年12月31日，故对这两个日期求year_diff再+1才是总年数
    yearly_return_rate = 100 * return_mean / global_var.INITIAL_BALANCE\
                         / (util.get_year_diff(global_var.EVAL_START_DATE, global_var.EVAL_END_DATE) + 1)

    print('RunAgent:',
          'total {} episodes, average return {:.2f}, std {:.2f}, yearly return rate {:.2f}%'.format(eval_episode,
                                                                                                    return_mean,
                                                                                                    return_std,
                                                                                                    yearly_return_rate))
    print('RunAgent:',
          '{} agent average training time: {:.2f} minutes'.format(agent_name, (train_end_time - train_start_time) / 60))


def train_agent_ntimes(data: pd.DataFrame, agent_name: str, train_timesteps: int = 10000,
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

    data_train = pp.subdata_by_range(data, global_var.TRAIN_START_DATE, global_var.TRAIN_END_DATE)
    data_eval = pp.subdata_by_range(data, global_var.EVAL_START_DATE, global_var.EVAL_END_DATE)

    if model_save_path is not None:
        os.makedirs(model_save_path, exist_ok=True)
        print('RunAgent:', f'models will be saved to {model_save_path}')

    if global_var.VERBOSE:
        print('RunAgent:', f'training and evaluating {agent_name} agent for {n_train} times, timesteps {train_timesteps}')

    returns = []
    train_elapsed_times = []

    for i in range(n_train):
        train_i_log_path = None
        if log_path is not None:
            train_i_log_path = log_path + f'{i+1}/'
            os.makedirs(train_i_log_path, exist_ok=True)

        # Train
        env_train = get_train_env(data_train, agent_name, train_i_log_path)
        agent = agent_factory(agent_name, env_train)
        train_start_time = time.time()
        agent.learn(timesteps=train_timesteps)
        train_end_time = time.time()
        train_elapsed_times.append((train_end_time - train_start_time) / 60)

        if model_save_path is not None:
            agent.save(model_save_path + f'{i+1}')
        if log_path is not None:
            util.plot_learning_curve(train_i_log_path, train_i_log_path + 'learning_curve.png')

        # Eval
        agent.eval_mode()
        env_eval = StockEvalEnvV2(data_eval, verbose=0)
        ret = 0
        for _ in range(eval_episode):
            ret += eval_agent_simple(agent, env_eval)
        ret /= eval_episode
        print('RunAgent:', 'episode {:0>2d}/{}, average return {:.2f}'.format(i+1, n_train, ret))
        returns.append(ret)

    return_mean, return_std = np.mean(returns), np.std(returns)
    yearly_return_rate = 100 * return_mean / global_var.INITIAL_BALANCE \
                         / (util.get_year_diff(global_var.EVAL_START_DATE, global_var.EVAL_END_DATE) + 1)

    print('RunAgent:',
          'total {} training, average return {:.2f}, std {:.2f}, yearly return rate {:.2f}%'.format(n_train,
                                                                                                    return_mean,
                                                                                                    return_std,
                                                                                                    yearly_return_rate))
    print('RunAgent:',
          '{} agent average training time: {:.2f} minutes'.format(agent_name, np.mean(train_elapsed_times)))


def track_train_agent_ntimes(data: pd.DataFrame, agent_name: str, track_train_timesteps: int,
                             base_model_path: str, n_train: int = 10):
    """
    在一个基础预训练模型的基础上，在测试集上每完成一个季度的交易后使用该段时间上的数据继续训练该模型，使模型能追踪近期趋势。
    共训练n_train次，测试其平均表现（仅输出收益率数字）。

    :param data: 预处理后的完整股票数据
    :param agent_name: Agent名称，目前可选：'Dumb', 'Hold', 'A2C', 'PPO'
    :param track_train_timesteps: 模型在每个时间窗口上继续训练的交互步数
    :param base_model_path: 基础模型文件路径
    :param n_train: 重复训练次数
    """

    if global_var.VERBOSE:
        print('RunAgent:', f'track train {agent_name} agent on eval period for {n_train} times, timesteps {track_train_timesteps}. '
                           f'Load base model from {base_model_path}')

    if not os.path.isfile(base_model_path):
        raise ValueError('Specified base model file does not exist')

    # 由于是在测试期上继续训练，因此data_train和data_eval的范围相同
    data_train = pp.subdata_by_range(data, global_var.EVAL_START_DATE, global_var.EVAL_END_DATE)
    data_eval = pp.subdata_by_range(data, global_var.EVAL_START_DATE, global_var.EVAL_END_DATE)

    retrain_dates = util.get_quarter_dates(global_var.EVAL_START_DATE,
                                           int(datetime.strftime(
                                               datetime.strptime(str(global_var.EVAL_END_DATE), '%Y%m%d') + relativedelta(days=1),
                                           '%Y%m%d')))

    returns = []
    for e in range(n_train):
        # 追踪训练分多个时间段，不能用多环境，也无法记录训练日志
        env_train = StockTrainEnvV2(data_train, verbose=0)

        # 加载预训练模型
        agent = agent_factory(agent_name, env_train)
        if isinstance(agent, A2CAgent) or isinstance(agent, PPOAgent):
            agent.model.set_parameters(base_model_path)

        # 在每个季度上测试和继续训练
        env_eval = StockEvalEnvV2(data_eval, verbose=0)
        state = env_eval.reset()
        ret = 0
        for i in range(len(retrain_dates) - 1):
            # 如果该季度区间内没有任何数据，则跳过
            if not env_train.check_interval_valid(retrain_dates[i], retrain_dates[i + 1]):
                # print(f'Quarter {retrain_dates[i]} to {retrain_dates[i+1]} has no data, therefore skipped.')
                continue

            # perform trading on the next quarter
            agent.eval_mode()
            env_eval.reset_date(retrain_dates[i], retrain_dates[i+1], is_last_section=(i==len(retrain_dates)-2))
            # 多个分段间状态连续，不能用eval_agent_simple否则会在分段处重置回初始状态
            total_rewards = 0
            while True:
                action = agent.act(state)
                next_state, reward, done, _ = env_eval.step(action)
                state = next_state
                total_rewards += reward
                if done:
                    ret += total_rewards / global_var.REWARD_SCALING
                    break

            # print('RunAgent:', f'quarter {retrain_dates[i]} to {retrain_dates[i+1]} ended, current return {ret:.2f}. Training on its data.')

            # training on the quarter's data after trading
            if i != len(retrain_dates) - 2:  # 最后一个季度上不用训练
                agent.train_mode()
                env_train.reset_date(retrain_dates[i], retrain_dates[i + 1])
                agent.learn(track_train_timesteps)

        print('RunAgent:', 'episode {:0>2d}/{}, avg return {:.2f}'.format(e + 1, n_train, ret))
        returns.append(ret)

    return_mean, return_std = np.mean(returns), np.std(returns)
    yearly_return_rate = 100 * return_mean / global_var.INITIAL_BALANCE \
                         / (util.get_year_diff(global_var.EVAL_START_DATE, global_var.EVAL_END_DATE) + 1)

    print('RunAgent:',
          'total {} training, average return {:.2f}, std {:.2f},'
          ' yearly return rate {:.2f}%'.format(n_train, return_mean, return_std, yearly_return_rate))


def track_train_agent(data: pd.DataFrame, agent_name: str, track_train_timesteps: int,
                      base_model_path: str, output_path: str) -> float:
    """
    在一个基础预训练模型的基础上，在测试集上每完成一个季度的交易后使用该段时间上的数据继续训练该模型，使模型能追踪近期趋势。
    训练+测试一次，输出历史状态以及追踪训练的Agent和不进行追踪训练以及基线的资产曲线对比图

    :param data: 预处理后的完整股票数据
    :param agent_name: Agent名称，目前可选：'Dumb', 'Hold', 'A2C', 'PPO'
    :param track_train_timesteps: 模型在每个时间窗口上继续训练的交互步数
    :param base_model_path: 基础模型文件路径
    :param output_path: 图表输出路径
    :return: 模型在环境中完成一轮交易的收益金额
    """

    if global_var.VERBOSE:
        print('RunAgent:', f'track train {agent_name} agent on eval period, timesteps {track_train_timesteps}. '
                           f'Load base model from {base_model_path}, outputing eval result at {output_path}')

    if not os.path.isfile(base_model_path):
        raise ValueError('Specified base model file does not exist')

    data_train = pp.subdata_by_range(data, global_var.TRAIN_START_DATE, global_var.TRAIN_END_DATE)
    data_eval = pp.subdata_by_range(data, global_var.EVAL_START_DATE, global_var.EVAL_END_DATE)

    os.makedirs(output_path, exist_ok=True)

    ###################### Track Trained Model Perf ######################
    retrain_dates = util.get_quarter_dates(global_var.EVAL_START_DATE,
                                           int(datetime.strftime(global_var.EVAL_END_DATE + relativedelta(days=1), '%Y%m%d')))

    env_train = get
    env_eval = StockEvalEnvV2(data_eval, verbose=0)

    # load a pre-trained model
    agent_track = agent_factory(agent_name, env_train)

    # agent.load() will clear out the env so that the agent can't learn anymore
    # therefore use set_parameters() to load the model
    agent_track.model.set_parameters(model_path)

    state = env_eval.reset()
    ret_agent_track = 0

    for i in range(len(retrain_dates) - 1):
        # print('RunAgent:', '=================================new quarter=====================================')
        # perform trading on the next quarter
        if not env_train.check_interval_valid(retrain_dates[i], retrain_dates[i + 1]):
            # the quarter has no data, so we shouldn't test or train the model on it.
            # print(f'Quarter {retrain_dates[i]} to {retrain_dates[i + 1]} has no data, therefore skipped.')
            continue
        total_rewards = 0
        reseted_dates = env_eval.reset_date(retrain_dates[i], retrain_dates[i + 1],
                                            is_last_section=(i == len(retrain_dates) - 2))
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
        if i != len(retrain_dates) - 2:
            env_train.reset_date(retrain_dates[i], retrain_dates[i + 1])
            agent_track.learn(track_train_timesteps)
    reward_memory_agent_track = env_eval.reward_memory
    asset_memory_agent_track = [a[-1] for a in env_eval.asset_memory]
    os.makedirs(output_path + 'env_memory', exist_ok=True)
    env_eval.dump_memory(output_path + 'env_memory/')

    print('RunAgent:',
          'Track trained agent average return {:.2f}, yearly return rate {:.2f}%'.format(ret_agent_track,
                                                                                         100 * ret_agent_track / 4 / global_var.INITIAL_BALANCE))

    ###################### Original Model Perf (No Track Train) ######################
    while True:
        env_eval = StockEvalEnvV2(data_eval, stock_codes, verbose=False)
        original_agent = agent_factory(agent_name, env_eval)
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
    print('RunAgent:', 'Baseline average return {:.2f}, yearly return rate {:.2f}%'.format(ret_baseline,
                                                                                           100 * ret_baseline / 4 / global_var.INITIAL_BALANCE))

    dates = [datetime.strptime(str(d), '%Y%m%d').date() for d in env_eval.full_dates][:-1]

    util.plot_daily_multi_y(x=dates, ys=[asset_memory_agent_track, asset_memory_original_agent, asset_memory_baseline],
                            ys_label=[agent_name + f'(Track Step={track_train_timesteps})', agent_name + '(No Track Train)',
                                      'Hold(Baseline)'],
                            save_path=output_path + 'total_assets_compare.png')
    util.plot_daily_multi_y(x=dates,
                            ys=[reward_memory_agent_track, reward_memory_original_agent, reward_memory_baseline],
                            ys_label=[agent_name + f'(Track Step={track_train_timesteps})', agent_name + '(No Track Train)',
                                      'Hold(Baseline)'],
                            save_path=output_path + 'daily_reward_compare.png')

    return ret_agent_track
