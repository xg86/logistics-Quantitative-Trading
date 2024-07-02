#from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
from sklearn import preprocessing
from sklearn import utils
from dateutil.relativedelta import relativedelta
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
#from neuralprophet import NeuralProphet
from webclient import *
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

column_names=['dates', 'Zero', 'NssZero', 'DiffZero','Yield','NssYield','DiffYield','dayRet','stdev','skew','month_cum_2','month_diff']
#column_x=['Volume', 'Zero', 'DiffZero','Yield','dayRet','stdev','skew']
#it is for Logic regresssion. need to predict
#column_x=['NssZero','month_cum']
column_x=['NssZero','skew']
def sum_from_months_prior(row, df: DataFrame, col_name: str):
    '''returns sum of values in row month,
    from all dates in df prior to row date'''
    month = pd.to_datetime(row).month
    year = pd.to_datetime(row).year
    all_dates_prior = df[df.index <= row]
    same_month = all_dates_prior[np.logical_and(all_dates_prior.index.month == month, all_dates_prior.index.year == year)]
    return same_month[col_name].sum()
def cross_validation(data):
    X = np.array(data[column_x])
    Y = np.array(data['signal'])
    return X, Y

def mse(X, Y, data):
    lm = LogisticRegression()
    #lab = preprocessing.LabelEncoder()
    #y_transformed = lab.fit_transform(Y)
    #print(y_transformed)
    scores = model_selection.cross_val_score(lm, X, Y, cv=10, scoring='neg_mean_squared_error')
    print(scores)
    # scores都是负数,这里的mean_squared_error是一种损失函数，优化的目标的使其最小化，而分类准确率是一种奖励函数，优化的目标是使其最大化。
    mse_scores = -scores
    print(mse_scores)
    rmse_scores = np.sqrt(mse_scores)
    print(rmse_scores)
    print(rmse_scores.mean())
    feature_cols = column_x
    X = data[feature_cols]

    # convert from MSE to RMSE # calculate the average RMSE
    print(np.sqrt(-model_selection.cross_val_score(lm, X, Y, cv=10, scoring='neg_mean_squared_error')).mean())

def proc_data(sheet_name: str, excel_file: str):
    print("starting {0} ".format(sheet_name))
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    df.set_index('dates', inplace=True)
    df.index = pd.to_datetime(df.index)
    df["dates_2"] = df.index
    df.sort_index(inplace=True)
    df["month_cum"] = df["dates_2"].apply(lambda row: sum_from_months_prior(row, df, 'dayRet'))
    df["month_pct"] = df["dates_2"].apply(lambda row: sum_from_months_prior(row, df, 'pct_chg'))
    df = df.dropna()
    return df
#数据标准化
def Standard(X_train, X_test):
    scaler =MinMaxScaler()
    X_train_scaled =scaler.fit_transform(X_train)
    X_test_scaled =scaler.fit_transform(X_test)
    return  X_train_scaled,X_test_scaled

#预测准确率
#NOT using
def LR(X_train,Y_train,X_test,Y_test):
    logreg =LogisticRegression(C=1).fit(X_train,Y_train)
    train_score=logreg.score(X_train,Y_train)
    test_score=logreg.score(X_test,Y_test)
#     print('Accuracy of LR classifier on train set:{:.3f}'.format(logreg.score(X_train,Y_train)))
#     print('Accuracy of LR classifier on test set:{:.3f}'.format(logreg.score(X_test,Y_test)))
    return  train_score,test_score

#预测概率值（没有Y_test来检验）
clf = LogisticRegression()
def LR_probo(x_train, y_train, x_test, y_test, columns1, columns2):
    clf.fit(x_train, y_train)
    #   返回预测标签
#     print(clf.predict(X_test))
    probo=clf.predict_proba(x_test)
    # 返回预测属于某标签的概率
#    probo=pd.DataFrame(probo, columns=[columns1,columns2])
#    probo['STOCK']=X1_test[:,1]
#    probo['date']=X1_test[:,0]
#     probo['PCT_CHG']=Y_test
    return probo
    pass

def get_month_delta(start_date: datetime, end_date: datetime):
    r= relativedelta(start_date, end_date)
    return abs(r.months + (12*r.years))
    pass

def get_month_diff(row, start_date: datetime):
    '''returns sum of values in row month,
    from all dates in df prior to row date'''
    month = pd.to_datetime(row).month
    year = pd.to_datetime(row).year
    min_month =start_date.month
    min_year =start_date.year
    return abs(month-min_month + 12*(year - min_year))
def signal(x,i):
    if x >= i:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def Splitdata(data_df: DataFrame, train_start: int, train_end: int, test_start: int, test_end: int):
    train_df = data_df[np.logical_and(data_df.month_diff >= train_start, data_df.month_diff <=train_end)]
    x_train = np.array(train_df[column_x])
    y_train = np.array(train_df['signal'])

    if test_start == 0 or test_end == 0:
        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        return x_train_scaled, y_train, None, None, train_df, None
    # 第x个月，测试集
    test_df = data_df[np.logical_and(data_df.month_diff > test_start, data_df.month_diff <= test_end)]
    i = 1
    while len(test_df) == 0:
        test_df = data_df[np.logical_and(data_df.month_diff > test_start, data_df.month_diff <= test_end + i)]
        i = i + 1
    x_test = np.array(test_df[column_x])
    y_test = np.array(test_df['signal'])
    x_train_scaled, x_test_scaled = Standard(x_train, x_test)
    return x_train_scaled, y_train, x_test_scaled, y_test, train_df, test_df

sheets=['210013.IB']

src_file="bond_data.xlsx"
request_file = 'D://git//Quantitative-Trading//bond-data//' + src_file

#for sheet in sheets:
#    df=proc_data(sheet, request_file)
#    df = df.drop('dates_2', axis=1)
#    df.to_csv(sheet+"_df_bond.csv")
'''
data_file="210013.IB_df_bond.csv"
data_file_full = 'D://git//Quantitative-Trading//code//基于股价大幅波动的另类选股因子研究//' + data_file
data_df = pd.read_csv(data_file_full)
data_df.set_index('dates', inplace=True)
data_df.index = pd.to_datetime(data_df.index)
data_df["dates_2"] = data_df.index
#data_df['month_cum_2']=[data_df.loc[x- pd.tseries.offsets.DateOffset(months=1):x, 'dayRet'].sum() for x in data_df.index]
data_df["month_cum_2"] = data_df["dates_2"].apply(lambda row: sum_from_months_prior(row, data_df, 'dayRet'))
data_df["signal"] = data_df["month_cum_2"].apply(lambda x: signal(x,0))
min = data_df["dates_2"].min()
data_df['month_delta'] = data_df["dates_2"].apply(lambda curr_date: get_month_delta(min, curr_date))
data_df['month_diff'] = data_df["dates_2"].apply(lambda row: get_month_diff(row, min))
#data_df.to_csv("m_delta_"+data_file)
X, Y = cross_validation(data_df)
mse(X, Y, data_df)
'''
data_file_2="m_delta_210013.IB_df_bond.csv"
data_file_full_2 = 'D://git//Quantitative-Trading//code//基于股价大幅波动的另类选股因子研究//' + data_file_2
#data_df_2 = pd.read_csv(data_file_full_2)

def predict(train_start: int, train_end: int, data_df: DataFrame, predict_days: int):
    x_train_scaled, y_train, x_test_scaled, y_test, train_df, test_df = Splitdata(data_df, train_start, train_end, 0, 0)

    prophet_df=pd.DataFrame()
    prophet_df['ds'] = train_df['dates_2']
    # prophet_df['y'] = train_df['month_cum_2']
    # prophet_df['y'] = train_df['dayRet']
    prophet_df['y'] = train_df['skew']
    # prophet_df['NssZero'] = train_df['NssZero']
    m = Prophet()
    # m.add_regressor('NssZero', standardize=False)
    # m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    # m.add_country_holidays(country_name='CN')
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=predict_days)
    # future = m.make_future_dataframe(periods=len(result.values)-len(prophet_df))
    # future['NssZero'] = result.values[:len(future)]
    forecast = m.predict(future)

    ref_date = train_df.dates_2.max()
    mat_dates = train_df.MaturityDate.unique()
    codes = train_df.Code.unique()

    forecast.set_index('ds', inplace=True)
    predict_df = forecast.loc[ref_date:]
    predict_df["dates_2"] = predict_df.index
    predict_df['NssZero_curve'] = get_ylds(ref_date, mat_dates[0], codes[0], predict_df['dates_2'])

    prophet_test = np.array(predict_df[['NssZero_curve', 'yhat']])
    # NssZero is from curve constructed in real
    # prophet_test = np.array(test_df[['NssZero','yhat']])
    scaler = MinMaxScaler()
    prophet_test_scaled = scaler.fit_transform(prophet_test)
    probo_prophet = LR_probo(x_train_scaled, y_train, prophet_test_scaled, None, None, None)
    # probo_prophet=LR_probo(x_train_scaled, y_train, x_test_scaled, None, None, None)
    # print(probo_prophet)
    # give column name, probability up or down
    probo_prophet_df = pd.DataFrame(probo_prophet, columns=['up', 'down'])
    # decode probability to -1, yeild is -1(down), price is up
    probo_prophet_df['prophet-signal'] = probo_prophet_df[['up', 'down']].apply(
        lambda x: 'U' if x['up'] > x['down'] else 'D', axis=1)
    predict_desc = probo_prophet_df['prophet-signal'].describe()
    print(
        "@@@@@@@@@@@ code is  {0}, total predict_df {1}, top value {2}, freq of top value  {3}, PCT % {4},"
        .format(codes[0],
                predict_desc[0],
                predict_desc[2],
                predict_desc[3],
                predict_desc[3] / predict_desc[0] * 100))
    pass

def make_prophet(train_start: int, train_end: int, test_start: int, test_end: int, data_df: DataFrame):
    x_train_scaled, y_train, x_test_scaled, y_test, train_df, test_df = Splitdata(data_df, train_start, train_end, test_start, test_end)
    #probo_test=LR_probo(x_train_scaled, y_train, x_test_scaled, None, 'Crashp1m5', 'Jackpotp1m5')
    #print(probo_test)

    prophet_df=pd.DataFrame()
    prophet_df['ds'] = train_df['dates_2']
    #prophet_df['y'] = train_df['month_cum_2']
    #prophet_df['y'] = train_df['dayRet']
    prophet_df['y'] = train_df['skew']
    #prophet_df['NssZero'] = train_df['NssZero']
    m=Prophet()
    #m.add_regressor('NssZero', standardize=False)
    #m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    #m.add_country_holidays(country_name='CN')
    m.fit(prophet_df)




    ref_date = train_df.dates_2.max()
    mat_dates = train_df.MaturityDate.unique()
    codes = train_df.Code.unique()
    test_df['NssZero_curve'] = get_ylds(ref_date, mat_dates[0], codes[0], test_df['dates_2'])
    #result = pd.concat([train_df['NssZero'], test_df['NssZero_curve']])

    future = m.make_future_dataframe(periods=len(test_df))
    #future = m.make_future_dataframe(periods=len(result.values)-len(prophet_df))
    #future['NssZero'] = result.values[:len(future)]
    forecast = m.predict(future)
    #forecast.to_csv("210013.IB_forecast.csv")0
    '''
    ValueError: Detected multiple frequencies in the timeseries please pre-process data.
    '''
    #m = NeuralProphet()
    #metrics = m.fit(prophet_df)
    #df_future = m.make_future_dataframe(prophet_df, n_historic_predictions=True, periods=len(test_df)+1)
    #forecast = m.predict(df_future)

    test_df['yhat'] = np.array(forecast.tail(len(test_df))['yhat'])
    performance_baseline_MAPE = mean_absolute_percentage_error(test_df['skew'], test_df['yhat'])
    print(f'The MAPE for the baseline model is {performance_baseline_MAPE}')

    prophet_test = np.array(test_df[['NssZero_curve','yhat']])
    # NssZero is from curve constructed in real
    #prophet_test = np.array(test_df[['NssZero','yhat']])
    scaler =MinMaxScaler()
    prophet_test_scaled =scaler.fit_transform(prophet_test)
    probo_prophet=LR_probo(x_train_scaled, y_train, prophet_test_scaled, None, None, None)
    #probo_prophet=LR_probo(x_train_scaled, y_train, x_test_scaled, None, None, None)
    #print(probo_prophet)
    # give column name, probability up or down
    probo_prophet_df = pd.DataFrame(probo_prophet, columns =['up', 'down'])
    # decode probability to -1, yeild is -1(down), price is up
    probo_prophet_df['prophet-signal'] = probo_prophet_df[['up', 'down']].apply(lambda x : -1 if x['up'] > x['down'] else 1, axis=1)
    test_df['prophet-signal'] = np.array(probo_prophet_df['prophet-signal'])

    return train_df, test_df


'''
monthes = data_df_2.month_diff.unique()

result_df=pd.DataFrame()

for m in monthes:
    if(m+11 >= monthes.max()):
        break
    print("m is  ", m)
    train_df, test_df =make_prophet(m, m+11, m+11, m+12)
    if len(result_df) == 0:
        result_df = train_df.append(test_df)
    else:
        result_df = result_df.append(test_df)

result_df.to_csv("result_df.csv")
'''
#print(forecast)
    #if len(bond_df) == 0:
    #    bond_df= df
    #else:
    #bond_df = bond_df.append(df)


#bond_df.to_csv("df_bond.csv")

