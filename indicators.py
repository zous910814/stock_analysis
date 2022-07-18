import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf


class Indicators:
    def __init__(self, data):
        self.__data = data

    def rsv(self):
        '''
        rsv = (今日收盤價 - 最近九天的最低價)/(最近九天的最高價 - 最近九天最低價)
        '''
        data = self.__data
        rsv = (
                      data['收盤價'] - data['收盤價'].rolling(window=9).min()
              ) / (
                      data['收盤價'].rolling(window=9).max() - data['收盤價'].rolling(window=9).min()
              ) * 100
        rsv = np.nan_to_num(rsv)
        self.__data['RSV'] = rsv

    def kv(self):
        '''
        當日K值=前一日K值 * 2/3 + 當日RSV * 1/3
        '''
        data = self.__data
        if 'RSV' not in data:
            self.rsv()
        rsv = self.__data['RSV'].tolist()

        kv = [20 for _ in range(8)]
        ktemp = kv[0]
        for i in range(len(rsv) - 8):
            ktemp = ktemp * (2 / 3) + rsv[i + 8] * (1 / 3)
            kv.append(round(ktemp, 2))
        self.__data['K'] = kv

    def dv(self):
        '''
        當日D值=前一日D值 * 2/3 + 當日K值 * 1/3
        '''
        data = self.__data
        if 'K' not in data:
            self.kv()
        kv = self.__data['K'].tolist()

        dv = [50 for _ in range(8)]
        dtemp = dv[0]
        for i in range(len(kv) - 8):
            dtemp = dtemp * (2 / 3) + kv[i + 8] * (1 / 3)
            dv.append(round(dtemp, 2))
        self.__data['D'] = dv

    def macd(self):
        '''
        EMA(指數移動平均)：計算MACD時，會先計算長、短天期的指數移動平均線(EMA)
        ，一般來說短期常使用12日(n=12)、長期為26日(m=26)
        DIF = 12日EMA – 26日EMA
        MACD =快線取9日EMA
        柱狀圖(直方圖) = 快線–慢線，EMA快慢線相減後，得出來的差額就是在MACD圖形看到的柱狀圖。
        :return:
        '''
        data = self.__data
        data['12_EMA'] = data['收盤價'].ewm(span=12).mean()
        data['26_EMA'] = data['收盤價'].ewm(span=26).mean()
        data['DIF'] = data['12_EMA'] - data['26_EMA']
        data['MACD'] = data['DIF'].ewm(span=9).mean()

    def kd_line(self):
        '''
        Make KD indicator's picture
        '''
        data = self.__data

        if 'K' not in data:
            self.kv()
        if 'D' not in data:
            self.dv()

        data.index = pd.DatetimeIndex(data['日期'])
        data['K'].plot()
        data['D'].plot()
        plt.show()

    def macd_line(self):
        '''
        Make MACD indicator's picture
        '''
        data = self.__data

        if 'DIF' not in data or 'MACD' not in data:
            self.macd()

        data.index = pd.DatetimeIndex(data['日期'])
        data['MACD'].plot()
        data['DIF'].plot()

        plt.show()

    def candlestick_chart(self):
        data = self.__data

        data.index = pd.DatetimeIndex(data['日期'])
        data = data.rename(columns={'收盤價': 'Close', '開盤價': 'Open',
                                    '最高價': 'High', '最低價': 'Low',
                                    '成交股數': 'Volume'})

        mc = mpf.make_marketcolors(up='r', down='g', inherit=True)
        s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)

        mpf.plot(data, style=s, type='candle', volume=True)

    def __str__(self):
        return self.__data.__str__()


if __name__ == "__main__":
    df = pd.read_csv('data/20130101~20220715.csv')

    tsmc = Indicators(df[df['證券代號'] == 2330])

    # tsmc.kv()
    # tsmc.dv()
    # tsmc.macd()
    print(tsmc)

    # tsmc.kd_line()
    # tsmc.macd_line()
    tsmc.candlestick_chart()