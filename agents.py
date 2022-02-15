import pandas as pd
import gym

import global_var
#from global_var import *
from stock_train_env import *


def run_agent(data: pd.DataFrame = None, episode: int = 100):
    import preprocessor
    data = preprocessor.pd.read_csv('./data/szstock_20_preprocessed.csv')
    data = preprocessor.subdata_by_ndays(data, 20, 20150105)
    stock_codes = preprocessor.get_stock_codes(data)
    data = preprocessor.to_daily_data(data)
    env = StockTrainEnvV1(data, stock_codes, verbose=False)
    agent = DumbAgent(env)
    avg_reward = 0
    for i in range(episode):
        total_rewards = 0
        # print(sep_line1)
        # print('Agent:', f'episode {i+1}/{episode} begins.')
        state = env.reset()
        while True:
            action = agent.act()
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_rewards += reward
            if done:
                #print('Agent:', f'episode {i + 1}/{episode}, total reward {total_rewards}')
                avg_reward += total_rewards
                break
    avg_reward /= episode
    print('Agent:', f'total {episode} episodes, average reward {avg_reward}')


class DumbAgent():
    """随机采取行动的Agent，仅供测试用。"""

    def __init__(self, env: gym.Env):
        self.env = env

    def act(self):
        action = self.env.action_space.sample()
        # print('DumbAgent:', 'random action', action)
        return action


class PPOAgent():
    """"""


if __name__ == '__main__':
    global_var.init()
    run_agent()
