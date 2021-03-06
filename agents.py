import numpy as np
import gym

from stable_baselines3 import A2C, PPO

import global_var

"""RLAgents"""


class Agent:
    """Agent基类。"""

    def __init__(self):
        self.name = None

    def act(self, state: np.ndarray) -> np.ndarray:
        pass

    def learn(self, timesteps: int):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass

    def train_mode(self):
        pass

    def eval_mode(self):
        pass


class DumbAgent(Agent):
    """每天随机采取行动的Agent。测试和基础对比用。"""

    def __init__(self, env: gym.Env):
        super().__init__()
        self.name = 'Dumb'
        self.env = env

    def learn(self, timesteps: int):
        pass  # DumbAgent doesn't  learn

    def act(self, state: np.ndarray) -> np.ndarray:
        action = self.env.action_space.sample()  # sample an action from action space
        # print('DumbAgent:', 'random action:', action)
        return action

    def train_mode(self):
        pass

    def eval_mode(self):
        pass


class HoldAgent(Agent):
    """仅在第一天将余额平均分配买入各只股票，之后不再采取买卖动作的Agent。主要基线。"""

    def __init__(self, env: gym.Env):
        super().__init__()
        self.name = 'Hold'
        self.env = env
        self.is_first_day = True

    def learn(self, timesteps: int):
        pass  # HoldAgent doesn't learn

    def act(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros(self.env.action_space.shape[0])
        if self.is_first_day:  # only perform at the first day
            stock_dim = self.env.stock_dim
            # 由于action被传入环境后会被乘以EVAL_MAX_PERCENTAGE_PER_TRADE参数放大，而HoldAgent应该永远保证在第一个交易日
            # 将所有资金平均分配到各股票中，因此要做如下调整：
            action_adjusted = 1 / stock_dim / global_var.EVAL_MAX_PERCENTAGE_PER_TRADE
            action = np.array([action_adjusted for _ in range(stock_dim)])

            # single_stock_budget = global_var.INITIAL_BALANCE * global_var.EVAL_MAX_PERCENTAGE_PER_TRADE * action_adjusted
            # print('HoldAgent:', f'day 0, {stock_dim} stocks,'
            #                     f' single stock budget {single_stock_budget}')
            self.is_first_day = False

        return action

    def train_mode(self):
        pass

    def eval_mode(self):
        pass


class A2CAgent(Agent):
    """采用A2C算法的Agent。"""

    def __init__(self, env: gym.Env):
        super().__init__()
        self.name = 'A2C'
        self.model = A2C('MlpPolicy', env, verbose=0)
        self.determ_policy = False

    def learn(self, timesteps: int):
        self.model.learn(total_timesteps=timesteps)

    def act(self, state: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(state, deterministic=self.determ_policy)
        # print('A2CAgent:', 'action:', action)
        return action

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = A2C.load(path)

    def train_mode(self):
        self.determ_policy = False

    def eval_mode(self):
        self.determ_policy = True


class PPOAgent(Agent):
    """采用PPO算法的Agent。"""

    def __init__(self, env: gym.Env):
        super().__init__()
        self.name = 'PPO'
        self.model = PPO('MlpPolicy', env, verbose=0)
        self.determ_policy = False

    def learn(self, timesteps: int):
        self.model.learn(total_timesteps=timesteps)

    def act(self, state: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(state, deterministic=self.determ_policy)
        # print('PPOAgent:', 'action:', action)
        return action

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = PPO.load(path)

    def train_mode(self):
        self.determ_policy = False

    def eval_mode(self):
        self.determ_policy = True


# class TD3Agent(Agent):
#     """采用TD3算法的Agent。"""
#
#     def __init__(self, env: gym.Env):
#         n_actions = env.action_space.shape[-1]
#         action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
#          self.model = TD3('MlpPolicy', env, verbose=1,
#                           action_noise=action_noise, learning_starts=1000, learning_rate=0.0002)
#
#     def learn(self, timesteps: int):
#         self.model.learn(total_timesteps=timesteps)
#
#     def act(self, state: np.ndarray) -> np.ndarray:
#         action, _ = self.model.predict(state)
#         print('TD3Agent:', action.tolist())
#         return action
#
#     def save(self, path: str):
#         self.model.save(path)
#
#     def load(self, path: str):
#         self.model = TD3.load(path)
#
#
# class SACAgent(Agent):
#     """采用SAC算法的Agent。"""
#
#     def __init__(self, env: gym.Env):
#         self.model = SAC('MlpPolicy', env, verbose=1,
#                          gamma=0.99, learning_rate=0.0001, learning_starts=100, batch_size=512)
#
#     def learn(self, timesteps: int):
#         self.model.learn(total_timesteps=timesteps)
#
#     def act(self, state: np.ndarray) -> np.ndarray:
#         action, _ = self.model.predict(state)
#         print('SACAgent:', action.tolist())
#         return action
#
#     def save(self, path: str):
#         self.model.save(path)
#
#     def load(self, path: str):
#         self.model = SAC.load(path)
#
#
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
    raise ValueError('所需Agent未定义')
