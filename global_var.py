import configparser

"""全局变量，通过读取config.ini文件获取"""

'''日志'''
VERBOSE = 0  # 是否输出详细运行信息，0不输出，1输出
# print辅助
SEP_LINE1 = '============================================================================='
SEP_LINE2 = '-----------------------------------------------------------------------------'

'''路径'''
STOCK_DATA_PATH = ''
PREPROCESSED_DATA_PATH = ''

'''股市环境参数'''
INITIAL_BALANCE                 = 0  # 初始资金
TRANSACTION_FEE_PERCENTAGE      = 0  # 交易费率（百分比）
TRAIN_MAX_PERCENTAGE_PER_TRADE  = 0  # 每步交易最大金额占当前总资金量比例（训练）
TEST_MAX_PERCENTAGE_PER_TRADE   = 0  # 每步交易最大金额占当前总资金量比例（测试）
TRAIN_START_DATE                = 0  # 训练开始日期
TRAIN_END_DATE                  = 0  # 训练结束日期
TEST_START_DATE                 = 0  # 测试开始日期
TEST_END_DATE                   = 0  # 测试结束日期

REWARD_SCALING = 0  # 每步reward较大，乘以该值进行缩放


def read_config():
    conf = configparser.ConfigParser()
    conf.read('./config/config.ini', encoding='utf-8')

    # 展示所有可用配置项
    # sections = conf.sections()
    # print(sections)

    global VERBOSE
    VERBOSE = int(conf.get('log', 'verbose'))

    global STOCK_DATA_PATH, PREPROCESSED_DATA_PATH
    STOCK_DATA_PATH         = conf.get('path', 'stock_data')
    PREPROCESSED_DATA_PATH  = conf.get('path', 'preprocessed_stock_data')

    global INITIAL_BALANCE, TRANSACTION_FEE_PERCENTAGE, TRAIN_MAX_PERCENTAGE_PER_TRADE, TEST_MAX_PERCENTAGE_PER_TRADE
    global TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE
    INITIAL_BALANCE = int(conf.get('stock_env', 'initial_balance'))

    TRANSACTION_FEE_PERCENTAGE     = float(conf.get('stock_env', 'transaction_fee_percentage'))
    TRAIN_MAX_PERCENTAGE_PER_TRADE = float(conf.get('stock_env', 'train_max_percentage_per_trade'))
    TEST_MAX_PERCENTAGE_PER_TRADE  = float(conf.get('stock_env', 'test_max_percentage_per_trade'))

    TRAIN_START_DATE = int(conf.get('stock_env', 'train_start_date'))
    TRAIN_END_DATE   = int(conf.get('stock_env', 'train_end_date'))
    TEST_START_DATE  = int(conf.get('stock_env', 'test_start_date'))
    TEST_END_DATE    = int(conf.get('stock_env', 'test_end_date'))

    global REWARD_SCALING
    REWARD_SCALING = float(conf.get('hyper_param', 'reward_scaling'))


def init():
    read_config()
