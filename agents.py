import numpy as np
import pandas as pd
import gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import global_var
from preprocessor import *
from stock_train_env import *


def run_agent(data: pd.DataFrame = None, model: str = 'PPO', episode: int = 100):
    stock_codes = get_stock_codes(data)
    data_train = subdata_by_range(data, 20100101, 20181231)
    data_train = to_daily_data(data_train)
    env_train = StockTrainEnvV1(data_train, stock_codes, verbose=False)

    # data_full = subdata_by_range(data, 20190101, 20211231)
    # data_full = to_daily_data(data_full)
    # env_full = StockTrainEnvV1(data_full, stock_codes, verbose=False)

    agent = agent_factory(model, env_train)
    if global_var.VERBOSE:
        print('Agent:', f'using {model} agent')

    agent.learn(timesteps=25000)
    #agent.load('./models/PPO_1M')

    data_eval = subdata_by_range(data, 20190101, 20211231)
    data_eval = to_daily_data(data_eval)
    env_eval = StockTrainEnvV1(data_eval, stock_codes, verbose=False)

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
                print('Agent:', 'episode {:0>2d}/{}, return(total reward) {:.2f}'.format(i+1, episode, ret))
                returns.append(ret)
                break
    return_mean, return_std = np.mean(returns), np.std(returns)
    print('Agent:', 'total {} episodes, average return {:.2f}, std {:2f}, return rate {:.2f}%'.format(episode, return_mean, return_std, 100 * return_mean / global_var.INITIAL_BALANCE))

class Agent():
    """Agent基类。"""

    def act(self, state: np.ndarray) -> np.ndarray:
        pass

    def learn(self, timesteps: int = 10000):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass


class DumbAgent(Agent):
    """随机采取行动的Agent，仅供测试用。"""

    def __init__(self, env: gym.Env):
        self.env = env

    def learn(self, timesteps: int = 10000):
        pass

    def act(self, state: np.ndarray) -> np.ndarray:
        action = self.env.action_space.sample()
        # print('DumbAgent:', 'random action', action)
        return action


class PPOAgent(Agent):
    """采用PPO算法的Agent。"""

    def __init__(self, env: gym.Env):
        self.model = PPO('MlpPolicy', env, verbose=1)

    def learn(self, timesteps: int = 10000):
        self.model.learn(total_timesteps=timesteps)

    def act(self, state: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(state)
        # print('PPOAgent:', action)
        return action

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = PPO.load(path)

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


def agent_factory(agent_name: str, env: gym.Env) -> Agent:
    if agent_name == 'Dumb':
        return DumbAgent(env)
    elif agent_name == 'PPO':
        return PPOAgent(env)
    raise ValueError('所需Agent未定义')


if __name__ == '__main__':
    global_var.init()
    data = pd.read_csv('./data/szstock_20_preprocessed.csv')
    stock_codes = get_stock_codes(data)
    data = to_daily_data(data)
    # a = NNAgent(StockTrainEnvV1(data, stock_codes, verbose=False))
    # run_agent()
