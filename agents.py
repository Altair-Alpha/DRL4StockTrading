import pandas as pd
import gym
from stable_baselines3 import PPO
import global_var
from preprocessor import *
from stock_train_env import *


def run_agent(data: pd.DataFrame=None, model: str='PPO', episode: int=100):
    #data = subdata_by_ndays(data, 5, 20100104)
    stock_codes = get_stock_codes(data)
    data = to_daily_data(data)
    env = StockTrainEnvV1(data, stock_codes, verbose=False)
    agent = agent_factory('Dumb', env)
    if global_var.VERBOSE:
        print('Agent:', f'using {model} agent')
    agent.learn(25000)

    avg_reward = 0
    for i in range(episode):
        total_rewards = 0
        # print(global_var.SEP_LINE1)
        # print('Agent:', f'episode {i+1}/{episode} begins.')
        state = env.reset()
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_rewards += reward
            if done:
                print('Agent:', f'episode {i + 1}/{episode}, total reward {total_rewards}')
                avg_reward += total_rewards
                break
    avg_reward /= episode
    print('Agent:', f'total {episode} episodes, average reward {avg_reward}, return rate {100 * avg_reward / global_var.INITIAL_BALANCE} %')


class Agent():
    """Agent基类。"""
    def act(self, state: np.ndarray) -> np.ndarray:
        pass

    def learn(self, timesteps: int=10000):
        pass


class DumbAgent(Agent):
    """随机采取行动的Agent，仅供测试用。"""

    def __init__(self, env: gym.Env):
        self.env = env

    def learn(self, timesteps: int=10000):
        pass

    def act(self, state: np.ndarray) -> np.ndarray:
        action = self.env.action_space.sample()
        # print('DumbAgent:', 'random action', action)
        return action


class PPOAgent(Agent):
    """采用PPO算法的Agent。"""

    def __init__(self, env: gym.Env):
        self.model = PPO('MlpPolicy', env, verbose=1)

    def learn(self, timesteps: int=10000):
        self.model.learn(total_timesteps=timesteps)

    def act(self, state: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(state)
        #print('PPOAgent:', action)
        return action



def agent_factory(agent_name: str, env: gym.Env) -> Agent:
    if agent_name == 'Dumb':
        return DumbAgent(env)
    elif agent_name == 'PPO':
        return PPOAgent(env)
    raise ValueError('所需Agent未定义')

if __name__ == '__main__':
    global_var.init()
    run_agent()
