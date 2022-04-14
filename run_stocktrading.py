import time
import global_var
import util
from preprocessor import *
from run_agents import *

if __name__ == '__main__':
    start_time = time.time()
    global_var.init()

    if global_var.VERBOSE:
        print('Main:', 'logging mode set to verbose.')

    if os.path.exists(global_var.PREPROCESSED_DATA_PATH):
        if global_var.VERBOSE:
            print('Main:', 'preprocessed stock data found. using it.')
        preprocessed_stock_data = pd.read_csv(global_var.PREPROCESSED_DATA_PATH, index_col=0)
    else:
        if global_var.VERBOSE:
            print('Main:', 'start loading and preprocessing stock data from path:', global_var.STOCK_DATA_PATH)
        preprocessed_stock_data = load_and_preprocess(global_var.STOCK_DATA_PATH)
        preprocessed_stock_data.to_csv(global_var.PREPROCESSED_DATA_PATH)
        if global_var.VERBOSE:
            print('Main:', 'stock data preprocessing complete and saved to:', global_var.PREPROCESSED_DATA_PATH)

    # model_path = f'./models/EnvV2/A2C/0323_A2C_2M_10_Train/8.zip'
    # model_path = './models/EnvV2/PPO/0308_PPO_2M_10_Train/7.zip'
    # model_path = f'./models/EnvV2/CYB_Data/0409_PPO_2M_10_Train/'

    # model_path = f'./models/EnvV2/A2C/0323_A2C_2M_10_Train/'
    # a2c_model_path = './models/EnvV2/A2C/0323_A2C_2M_10_Train/8.zip'
    # './models/EnvV2/CYB_Data/0407_A2C_2M_10_Train/10.zip'
    # ppo_model_path = './models/EnvV2/PPO/0308_PPO_2M_10_Train/7.zip'
    # output_path = './figs/simulation/EnvV2_A2C_2M_Track_10K_Eval/'

    # train_agent_ntimes(preprocessed_stock_data, 'A2C', 100000, 10, 1)
    track_train_agent_ntimes(preprocessed_stock_data, 'A2C', 25000, './models/EnvV2/CYB_Data/0407_A2C_2M_10_Train/9.zip', n_train=10)
    # track_train_agent(preprocessed_stock_data, 'PPO', 25000, '')

    # data_eval = pp.subdata_by_range(preprocessed_stock_data, global_var.EVAL_START_DATE, global_var.EVAL_END_DATE)
    # env = StockEvalEnvV2(data_eval, 0)
    # agent = PPOAgent(env)
    # agent.load('./models/EnvV2/CYB_Data/0409_PPO_2M_10_Train/6.zip')
    # agent.eval_mode()
    # # eval_agent(agent, env, './figs/simulation/CYB_PPO_2M_Eval/')
    # print(eval_agent_simple(agent, env))
    # data_eval = pp.subdata_by_range(preprocessed_stock_data, global_var.EVAL_START_DATE, global_var.EVAL_END_DATE)
    # env = StockEvalEnvV2(data_eval, verbose=0)
    # # agent = PPOAgent(env)
    # rets = []
    # agent.load(f'./models/EnvV2/CYB_Data/0409_PPO_2M_10_Train/6.zip')
    # for _ in range(10):
    #     ret = eval_agent_simple(agent, env)
    #     print(ret)
    #     rets.append(ret)
    # print('avg', np.mean(rets), 'std', np.std(rets), 'yearly', np.mean(rets) / global_var.INITIAL_BALANCE * 100 / 4)
    # agent = A2CAgent(env)
    # for i in range(10):
    #     agent.load(f'./models/EnvV2/CYB_Data/0407_A2C_2M_10_Train/{i+1}.zip')
    #     agent.eval_mode()
    #     ret = eval_agent_simple(agent, env)
    #     print(f'Model {i+1}:', ret)
    #     rets.append(ret)
    # print('avg', np.mean(rets), 'std', np.std(rets), 'yearly', np.mean(rets) / global_var.INITIAL_BALANCE * 100 / 4)


    # track_train_agent_ntimes(data, 'A2C', 5000, './models/EnvV2/CYB_Data/0407_A2C_2M_10_Train/9.zip', n_train=1)
    # run_agent_test(remove_anomaly(preprocessed_stock_data))
    # for i in range(10):
    #     print(f'Model {i+1}')
    #     eval_agent_train(remove_anomaly(preprocessed_stock_data), x=i+1)
    # run_agent_keep_train(remove_anomaly(preprocessed_stock_data))

    # eval_agent(remove_anomaly(preprocessed_stock_data))
    # eval_track_train_agent(remove_anomaly(preprocessed_stock_data))
    # util.draw_avg_stock_price(remove_anomaly(preprocessed_stock_data))

    end_time = time.time()
    if global_var.VERBOSE:
        print('Main:', 'total time elapsed: {:.2f} minutes'.format((end_time - start_time) / 60))
