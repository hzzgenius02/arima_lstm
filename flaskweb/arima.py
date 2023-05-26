import tushare as ts
pro = ts.pro_api('647f6840944a4425d46c97c08cf20af6b656bb79673bd1635ebdf0ce')
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import itertools
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.stattools as st
from sklearn.metrics import *
from scipy import  stats
from statsmodels.graphics.api import qqplot


def cal(daima):
    begin = '20220101'
    from datetime import datetime, date
    dayofWeek = datetime.today().weekday()
    import datetime
    today = datetime.date.today()
    if dayofWeek == 0:
        end = (datetime.date.today() + datetime.timedelta(days=-3)).strftime('%Y%m%d')
        end_ = (datetime.date.today() + datetime.timedelta(days=-3)).strftime('%Y-%m-%d')
    elif dayofWeek == 6:
        end = (datetime.date.today() + datetime.timedelta(days=-2)).strftime('%Y%m%d')
        end_ = (datetime.date.today() + datetime.timedelta(days=-2)).strftime('%Y-%m-%d')
    else:
        end = (datetime.date.today() + datetime.timedelta(days=-1)).strftime('%Y%m%d')
        end_ = (datetime.date.today() + datetime.timedelta(days=-1)).strftime('%Y-%m-%d')

    data=pro.query('daily', ts_code=daima+'.SZ', start_date=begin, end_date=end) #放假期间股票停止交易

    df = pd.DataFrame(data)
    df.to_csv(daima+'.csv')

    Stock_XRHJ = pd.read_csv(daima+'.csv',index_col = 'trade_date',parse_dates=['trade_date'])
    df = pd.DataFrame(Stock_XRHJ)
    df=df.iloc[::-1]

    #1.数据准备
    df.index = pd.to_datetime(df.index)
    sub    = df['2022-01-01':end_]['close']
    train  = df['2022-01-01':'2022-12-31']['close']
    val    = df['2023-01-01':'2023-02-28']['close']
    test   = df['2023-03-01':end_]['close']

    train_results = st.arma_order_select_ic(train, max_ar=8, max_ma=8, ic=['aic', 'bic'])
    bic = train_results.bic_min_order
    print('BIC:',bic)
    model = sm.tsa.ARIMA(sub, order=(bic[0], 0, bic[1]))  # ARIMA(1,0,0)模型
    # model = sm.tsa.ARIMA(sub, order=(1, 0, 0))  # ARIMA(1,0,0)模型
    results = model.fit()
    predict_sunspots = results.predict(start=str('2023-03-01'), end=end_, dynamic=False)
    mae = mean_absolute_error(test, predict_sunspots)
    print('MAE:', mae)
    mse = mean_squared_error(test, predict_sunspots)
    print('MSE:', mse)
    r2 = r2_score(test, predict_sunspots)
    print('R2:', r2)

    fore_sunspots = results.forecast(20)
    fore_sunspots.index = pd.date_range(start=end_, periods=20, freq='D')
    fore_sunspots.index = pd.to_datetime(fore_sunspots.index)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax = sub.plot(ax=ax, label='Raw data')
    predict_sunspots.plot(ax=ax, label='Validation data')
    fore_sunspots.plot(ax=ax, label='Forecast data')
    plt.legend()
    plt.title('ARIMA'+' '+daima, fontsize=10)
    plt.savefig('./static/predict.png')
    # plt.show()
    ans = [round(x, 2) for x in fore_sunspots.tolist()]
    print(ans)
    return ans

# cal('300813')