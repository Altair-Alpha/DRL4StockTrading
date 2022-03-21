import configparser

"""全局变量，通过读取config获取"""

'''日志'''
VERBOSE = 0
# print辅助
SEP_LINE1 = '============================================================================='
SEP_LINE2 = '-----------------------------------------------------------------------------'

'''路径'''
STOCK_DATA_PATH = ''
PREPROCESSED_DATA_PATH = ''

'''股市环境参数'''
INITIAL_BALANCE = 0             # 初始资金
SHARES_PER_TRADE = 0            # 每笔交易股数单位（因数）
TRANSACTION_FEE_PERCENTAGE = 0  # 交易费率（百分比）
MAX_PERCENTAGE_PER_TRADE = 0    # 每步交易最大占当前总资金量比例

REWARD_SCALING = 0              # 每步reward较大，乘以该值进行缩放(?)

def read_config():
    conf = configparser.ConfigParser()
    conf.read('./config/config.ini', encoding='utf-8')

    # 展示所有可用配置项
    # sections = conf.sections()
    # print(sections)

    global VERBOSE
    VERBOSE = int(conf.get('log', 'verbose'))

    global STOCK_DATA_PATH, PREPROCESSED_DATA_PATH
    STOCK_DATA_PATH = conf.get('path', 'stock_data')
    PREPROCESSED_DATA_PATH = conf.get('path', 'preprocessed_stock_data')

    global INITIAL_BALANCE, TRANSACTION_FEE_PERCENTAGE, SHARES_PER_TRADE, MAX_PERCENTAGE_PER_TRADE
    INITIAL_BALANCE = int(conf.get('stock_env', 'initial_balance'))
    SHARES_PER_TRADE = int(conf.get('stock_env', 'shares_per_trade'))
    TRANSACTION_FEE_PERCENTAGE = float(conf.get('stock_env', 'transaction_fee_percentage'))
    MAX_PERCENTAGE_PER_TRADE = float(conf.get('stock_env', 'max_percentage_per_trade'))

    global REWARD_SCALING
    REWARD_SCALING = float(conf.get('hyper_param', 'reward_scaling'))

def init():
    read_config()