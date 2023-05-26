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
from torch.nn import LSTM, Module, Linear


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

    data = pro.query('daily', ts_code=daima + '.SZ', start_date=begin, end_date=end)  # 放假期间股票停止交易
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv(daima + '.csv')

    Stock = pd.read_csv(daima + '.csv', index_col='trade_date', parse_dates=['trade_date'])
    df = pd.DataFrame(Stock)
    df = df.iloc[::-1]

    dataX = []  # 属性
    dataY = []  # 标签
    k = 0
    tempX = []  # 储存某个历史200天数据
    tempY = []  # 储存某个未来10天数据
    for index, rows in df.iterrows():
        if k < 200:
            k += 1
            tempX.append([rows['close']])
            continue
        if k < 220:
            k += 1
            tempY.append([rows['close']])
            continue
        dataX.append(tempX[:])
        dataY.append(tempY[:])

        tempX = tempX[1:] + tempY[:1]

        tempY = tempY[1:]
        tempY.append([rows['close']])
    dataX.append(tempX[:])  # 加上最后一项
    dataY.append(tempY[:])  # 加上最后一项

    import torch
    import torch.utils.data as Data

    dataX = torch.tensor(dataX)  # 列表转Tensor
    dataY = torch.tensor(dataY)  # 列表转Tensor

    dataset = Data.TensorDataset(dataX, dataY)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])  # 以8:2比例划分训练集和测试集

    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True
    )
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=True
    )

    class MyModel(Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.lstm = LSTM(input_size=1, hidden_size=100, num_layers=2, batch_first=True)
            self.linear = Linear(20000, 20)  # 将结果映射到10天的数据

        def forward(self, x):
            return self.linear(self.lstm(x)[0].reshape(-1, 20000))

    import torch.nn.functional as F
    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lossList = []  # 记录训练loss
    lossListTest = []  # 记录测试loss
    for epoch in range(100):
        loss_nowEpoch = []
        model.train()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            out = model(batch_x)  # 模型输入
            Loss = F.mse_loss(out, batch_y.view(-1, 20))  # loss计算，将batch_y从(64,10,1)变形为(64,10)
            optimizer.zero_grad()  # 当前batch的梯度不会再用到，所以清除梯度
            Loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新参数
            loss_nowEpoch.append(Loss.item())

        lossList.append(sum(loss_nowEpoch) / len(loss_nowEpoch))

        loss_nowEpochTest = []
        model.eval()
        for step, (batch_x, batch_y) in enumerate(test_loader):
            out = model(batch_x)
            Loss = F.mse_loss(out, batch_y.view(-1, 20))  # 将batch_y从(64,10,1)变形为(64,10)
            loss_nowEpochTest.append(Loss.item())
            break
        lossListTest.append(sum(loss_nowEpochTest) / len(loss_nowEpochTest))

        print(">>> EPOCH{} averTrainLoss:{:.3f} averTestLoss:{:.3f}".format(epoch + 1, lossList[-1], lossListTest[-1]))

    import pandas as pd
    X = torch.tensor(df['close'][-200:].to_numpy().copy()).reshape(-1, 1)

    Y = model(X.view(1, 200, 1).float()).reshape(-1, 1)

    x1 = list(range(1, 201))
    x2 = [i for i in range(200, 221)]
    Xp = X.detach().flatten().tolist()
    Yp = Y.detach().flatten().tolist()
    Yp.insert(0, round(Xp[-1], 2))
    # 绘制折线图
    plt.plot(x1, Xp, label='Raw data')
    plt.plot(x2, Yp, label='Forecast data')
    plt.legend()
    plt.title('LSTM'+' '+daima, fontsize=10)
    plt.savefig('./static/predict.png')

    # plt.show()
    ans = [round(x, 2) for x in Y.detach().flatten().tolist()]
    print(ans)
    return ans


# cal('000681')