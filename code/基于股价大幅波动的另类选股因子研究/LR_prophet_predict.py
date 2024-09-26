import uuid

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
#from neuralprophet import NeuralProphet, set_log_level

# Disable logging messages unless there is an error
#set_log_level("ERROR")
import statsmodels.tsa.stattools as tsa
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
#import seaborn as sn
from datetime import datetime
import datetime as dt
from timeit import default_timer as timer
from dateutil.relativedelta import relativedelta


from bond_LR_predict import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'
import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

#start = '2022-01-4 00:00:00'
cutoff = '2023-09-1 00:00:00'
#bond_codes = ["210203.IB", "190215.IB", "210205.IB", "210210.IB", "220205.IB",
 #             "220210.IB", "220215.IB", "220220.IB", "230205.IB", "210220.IB"]
src_files = ['diff_gk.xlsx','diff_gz.xlsx','diff_nf.xlsx','diff_jc.xlsx']
#src_files = ['diff_gk.xlsx']

#spread_file = 'E://meridian//债券//信号统计//NSS信号.xlsx'
#spread_df = pd.read_excel(spread_file, sheet_name='bid-ask-spread')

def get_half_life2(z_array: pd.Series):
    z_lag = np.roll(z_array, 1)
    z_lag[0] = 0
    z_ret = z_array - z_lag
    z_ret[0] = 0
    # adds intercept terms to X variable for regression
    z_lag2 = sm.add_constant(z_lag)
    model = sm.OLS(z_ret, z_lag2)
    res = model.fit()
    halflife = -np.log(2) / res.params[1]
    #print('Halflife = ', halflife)
    return halflife
def get_half_life(data: pd.Series):
    if (len(data) < 2):
        return -1
    data_lag = data.shift(1)
    Y = data[1:] - data_lag[1:]
    X = sm.add_constant(data_lag[1:])  # Adds a column of ones to an array
    model = sm.OLS(Y, X)
    res = model.fit()
    np_log = -np.log(2)
    if(len(res.params) < 2):
        return -1
    res_params = res.params.iloc[1]
    param_1 = np_log / res_params
    halflife = round(param_1, 0)
    return halflife

def stationary_test(data2: pd.Series):
    # compute ADF test statistic
    adf_ = tsa.adfuller(data2,maxlag=1)
    #print('adf', adf_[0])
    #print('p-value', adf_[1])
    #print('T values', adf_[4])
    return adf_[1] < 0.1

def decode(x,i):
    if x >= i:
        return 'D'
    elif x < 0:
        return 'U'
    else:
        return 'NaN'
def get_prophet_signal(src_file: str):
    print("starting reading ")
    #request_file = 'D://git//strategy-repos-master//butterfly//nss-data//' + src_file
    request_file = 'C://git//ficc-code//nss-data//' + src_file
    src_df = pd.read_excel(request_file)
    src_df = src_df.drop('Yld', axis=1)
    src_df = src_df.drop('TradeTime', axis=1)
    src_df = src_df.drop('Volume', axis=1)
    src_df = src_df.drop('Coupon', axis=1)
    src_df = src_df.drop('StartDate', axis=1)
    #src_df = src_df.drop('MaturityDate', axis=1)
    src_df = src_df.drop('BondName', axis=1)
    src_df = src_df.drop('Frequency', axis=1)
    src_df = src_df.drop('ActiveUnit', axis=1)
    src_df = src_df.drop('ValuationYield', axis=1)
    src_df = src_df.drop('NssYield', axis=1)
    src_df = src_df.drop('DiffYield', axis=1)
    src_df = src_df.drop('DiffZero', axis=1)
    #src_df = src_df.drop('Yield', axis=1)
    src_df = src_df.drop('ValuationDirtyPrice', axis=1)
    #df = df.drop('ValuationDirtyPrice', axis=1)
    src_df = src_df.drop('IsCurve', axis=1)
    codes = src_df.Code.unique()

    filename= str(uuid.uuid4().hex) + "_prophet_"+ src_file
    #writer = pd.ExcelWriter("B://git//Quantitative-Trading//bond-data//"+filename)
    writer = pd.ExcelWriter("C://git//logistics-Quantitative-Trading//bond-data//" + filename)
    print("codes length ", len(codes))
    for code in codes:
        if(len(code) <= 9 and code.startswith('2')):
            newdf = src_df[(src_df.Code == code)]
            #newdf['Date'] = pd.to_datetime(newdf['Date'])
            newdf.set_index('Date', inplace=True)
            skip_df = newdf.loc[cutoff:]
            if(len(skip_df) <= 2):
               continue
            print("code -》 ", code)
            if code == '230302.IB':
                print("code is 230302.IB ")
            newdf = newdf.tail(400)
            if (len(newdf) <= 100):
                continue
            #
            data_df = newdf.iloc[:-1, :]
            #data_df = newdf
            halflife_Zero = get_half_life(data_df["Zero"])
            halflife_NssZero = get_half_life(data_df["NssZero"])
            if(halflife_Zero > 0 and halflife_NssZero > 0 ):
                window_size = int(np.rint(halflife_Zero + halflife_NssZero)/2)
                if (len(data_df)/1.1 <= window_size):
                    continue
                data_df['dayRet'] = data_df['Zero'].diff()*10000
                #rolling skew to predict month cum
                data_df['skew'] = data_df['dayRet'].rolling(window=int(window_size), center=False).skew()

                data_df["dates_2"] = data_df.index
                data_df["month_cum"] = data_df["dates_2"].apply(
                    lambda row: sum_from_months_prior(row, data_df, 'dayRet'))
                #cum yield < 0, bond is up
                data_df["signal"] = data_df["month_cum"].apply(lambda x: signal(x, 0))
                data_df = data_df.dropna()
                min = data_df["dates_2"].min()
                data_df['month_diff'] = data_df["dates_2"].apply(lambda row: get_month_diff(row, min))
                monthes = data_df.month_diff.unique()
                result_df = pd.DataFrame()
                for m in monthes:
                    if (m + 11 >= monthes.max()):
                        predict(m, monthes.max(), data_df, 40)
                        break
                    if code == '220406.IB':
                        print("m is  ", m)
                    train_df, test_df = make_prophet(m, m + 11, m + 11, m + 12, data_df)

                    test_df["prophet-signal-decode"] = test_df["prophet-signal"].apply(lambda x: decode(x, 0))
                    prophet_signal_desc = test_df['prophet-signal-decode'].describe()
                    #within predict period, if 55% is UP(month_cum), then buy at begining and sell at end
                    if(prophet_signal_desc[2] == 'U' and prophet_signal_desc[3]/prophet_signal_desc[0] >= 0.45):
                        test_df.at[-1, 'PnLYield'] = (test_df.iloc[0]['Yield'] - test_df.iloc[-1]['Yield'])*10000
                    # 亏损加大了？？？
                    #elif (prophet_signal_desc[2] == 'D' and prophet_signal_desc[3] / prophet_signal_desc[0] >= 0.55):
                    #    test_df.at[-1, 'PnLYield'] = (test_df.iloc[-1]['Yield'] - test_df.iloc[0]['Yield'] ) * 10000
                    else:
                        test_df.at[-1, 'PnLYield'] = 0
                    test_df.fillna(0, axis=1, inplace=True)
                    if len(result_df) == 0:
                        #result_df = train_df.append(test_df)
                        result_df = pd.concat([train_df, test_df], ignore_index=True)
                    else:
                        #result_df = result_df.append(test_df)
                        result_df = pd.concat([result_df, test_df], ignore_index=True)
                if len(result_df) > 0:
                    result_df['signal_check-signal'] = result_df[['signal', 'prophet-signal']].apply(
                        lambda x: "Y" if x['signal'] == x['prophet-signal'] else 'N', axis=1)
                    result_df.to_excel(writer, sheet_name=code)
                    forcast_df=result_df.dropna()
                    signal_desc = forcast_df['signal_check-signal'].describe()
                    print("%%%%%%%%%%%% code is  {0}, total forcast_df {1}, top value {2}, freq of top value  {3}, win rate % {4}, PnL {5}"
                          .format(code,
                                  signal_desc[0],
                                  signal_desc[2],
                                  signal_desc[3],
                                  signal_desc[3]/signal_desc[0]*100, result_df['PnLYield'].sum()))
    writer.close()
    print("ALl done ", src_file)

for src_file in src_files:
    get_prophet_signal(src_file)
