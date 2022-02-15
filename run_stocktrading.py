import global_var
from preprocessor import *


if __name__ == '__main__':

    if global_var.VERBOSE:
        print('Main:', 'logging mode set to verbose.')

    if os.path.exists(global_var.PREPROCESSED_DATA_PATH):
        if global_var.VERBOSE:
            print('Main:', 'cached preprocessed stock data found. using it.')
        preprocessed_stock_data = pd.read_csv(global_var.PREPROCESSED_DATA_PATH, index_col=0)
    else:
        if global_var.VERBOSE:
            print('Main:', 'start loading and preprocessing stock data from path:', global_var.STOCK_DATA_PATH)
        preprocessed_stock_data = load_and_preprocess(global_var.STOCK_DATA_PATH)
        preprocessed_stock_data.to_csv(global_var.PREPROCESSED_DATA_PATH)
        if global_var.VERBOSE:
            print('Main:', 'stock data preprocessing complete and saved.')

    print(preprocessed_stock_data.head())
    print('===========================================')
    # print(subdata_by_ndays(preprocessed_stock_data, 10))
    # print('===========================================')
    # print(subdata_by_ndays(preprocessed_stock_data, 5, 20201105))
    # print('===========================================')
    print(subdata_by_range(preprocessed_stock_data, 20150101, 20150110))
    print('===========================================')
    print(to_daily_data(data=subdata_by_ndays(preprocessed_stock_data, 10)))