import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class indicators():
    def __init__(self, data):
        self.__data = data

    def rsv(self):
        # rsv = (今日收盤價 - 最近九天的最低價)/(最近九天的最高價 - 最近九天最低價)
        data = self.__data
        rsv = (
                      data['收盤價'] - data['收盤價'].rolling(window=9).min()
              ) / (
                      data['收盤價'].rolling(window=9).max() - data['收盤價'].rolling(window=9).min()
              ) * 100
        rsv = np.nan_to_num(rsv)
        self.__data['RSV'] = rsv

    def kv(self):
        data = self.__data
        if 'RSV' not in data:
            self.rsv()
        rsv = self.__data['RSV'].tolist()
        # 當日K值=前一日K值 * 2/3 + 當日RSV * 1/3
        kv = [20 for x in range(8)]
        ktemp = kv[0]
        for i in range(len(rsv) - 8):
            ktemp = ktemp * (2 / 3) + rsv[i + 8] * (1 / 3)
            kv.append(round(ktemp, 2))
        self.__data['K'] = kv

    def dv(self):
        data = self.__data
        if 'K' not in data:
            self.kv()
        kv = self.__data['K'].tolist()

        # 當日D值=前一日D值 * 2/3 + 當日K值 * 1/3
        dv = [50 for x in range(8)]
        dtemp = dv[0]
        for i in range(len(kv) - 8):
            dtemp = dtemp * (2 / 3) + kv[i + 8] * (1 / 3)
            dv.append(round(dtemp, 2))
        self.__data['D'] = dv

    def macd(self):
        data = self.__data
        data['12_EMA'] = data['收盤價'].ewm(span=12).mean()
        data['26_EMA'] = data['收盤價'].ewm(span=26).mean()
        data['DIF'] = data['12_EMA'] - data['26_EMA']
        data['MACD'] = data['DIF'].ewm(span=9).mean()

    def kd_line(self):
        data = self.__data

        if 'K' not in data:
            self.kv()
        if 'D' not in data:
            self.dv()

        data['K'].plot()
        data['D'].plot()
        plt.show()

    def __str__(self):
        return self.__data.__str__()


if __name__ == "__main__":
    df = pd.read_csv('data/20130101~20220715.csv')

    tsmc = indicators(df[df['證券代號'] == 2330])

    tsmc.kv()
    tsmc.dv()
    tsmc.macd()

    print(tsmc)

    # tsmc.kd_line()
