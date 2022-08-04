import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf


class Indicators:
    '''
    input a company's dataframe to create indicators series
    :return series
    '''

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
        data['MACD_histogram'] = data['DIF'] - data['MACD']

    def ma(self, day=20):
        data = self.__data
        data['{}_MA'.format(day)] = data['收盤價'].rolling(day).mean()
        data['{}_MA'.format(day)] = np.nan_to_num(data['{}_MA'.format(day)])

    def bias(self, day=6):
        data = self.__data
        if '{}_MA'.format(day) not in data:
            self.ma(day)

        data['Bias'] = 100 * (data['收盤價'] - data['{}_MA'.format(day)]) / data['收盤價'].rolling(day).mean()

    def kd_line(self, date: str, savefig: bool = True):
        '''
        Make KD indicator's picture
        '''
        data = self.__data

        if 'K' not in data:
            self.kv()
        if 'D' not in data:
            self.dv()

        data.index = pd.DatetimeIndex(data['日期'])
        data = data[data.index > date]
        data['K'].plot()
        data['D'].plot()
        plt.legend()
        plt.title('KD')

        if savefig == True:
            plt.savefig('picture/' + str(round(data['證券代號'].values[0])) + 'KD.png')
        else:
            plt.show()

    def macd_line(self, date: str, savefig=True):
        '''
        Make MACD indicator's picture
        '''
        data = self.__data

        if 'DIF' not in data or 'MACD' not in data:
            self.macd()

        data.index = pd.DatetimeIndex(data['日期'])
        data = data[data.index > date]

        data['MACD'].plot(kind='line')
        data['DIF'].plot(kind='line')
        for index, row in data.iterrows():
            if (row['MACD_histogram'] > 0):
                plt.bar(row['日期'], row['MACD_histogram'], width=0.5, color='red')
            else:
                plt.bar(row['日期'], row['MACD_histogram'], width=0.5, color='green')

        # major_index = data.index[data.index % 10 == 0]
        # major_xtics = data['日期'][data.index % 10 == 0]
        # plt.xticks(major_index, major_xtics)
        # plt.setp(plt.gca().get_xticklabels(), rotation=30)

        plt.legend()
        plt.title('MACD')

        if savefig == True:
            plt.savefig('picture/' + str(round(data['證券代號'].values[0])) + 'MACD.png')
        else:
            plt.show()

    def bias_line(self, date: str, savefig=True):
        data = self.__data

        if 'Bias' not in data:
            self.bias()

        data.index = pd.DatetimeIndex(data['日期'])
        data = data[data.index > date]
        data['Bias'].plot(color='red')
        plt.legend()
        plt.title('Bias')

        if savefig == True:
            plt.savefig('picture/' + str(round(data['證券代號'].values[0])) + 'Bias.png')
        else:
            plt.show()

    def candlestick_chart(self, date, savefig=True):
        data = self.__data

        data.index = pd.DatetimeIndex(data['日期'])
        data = data[data.index > date]
        data = data.rename(columns={'收盤價': 'Close', '開盤價': 'Open',
                                    '最高價': 'High', '最低價': 'Low',
                                    '成交股數': 'Volume'})

        mc = mpf.make_marketcolors(up='r', down='g', inherit=True)
        s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)

        if savefig == True:

            # 5b,20o,60g,120r,240p
            mpf.plot(data, style=s, type='candle', volume=True, mav=(5, 20, 60, 120, 240),
                     savefig='picture/' + str(round(data['證券代號'].values[0])) + '.png')
        else:
            mpf.plot(data, style=s, type='candle', volume=True, mav=(5, 20, 60, 120, 240))

    def __str__(self):
        return self.__data.__str__()


class Selections:
    '''
    input all company's dataframe to get stock selections
    :return list
    '''

    def __init__(self, data):
        self.__data = data

    def kd_goldencross(self) -> list:
        data = self.__data
        symbol = data['證券代號'].unique()
        li = []
        for i in symbol:
            df = data[data['證券代號'] == i]
            ind_df = Indicators(df)
            ind_df.kv()
            ind_df.dv()

            k = df['K'].tolist()
            d = df['D'].tolist()

            if k[-1] < 20 and d[-1] < 20:
                li.append(i)

        return li

    def kd_deathcross(self) -> list:
        data = self.__data
        symbol = data['證券代號'].unique()
        li = []
        for i in symbol:
            df = data[data['證券代號'] == i]
            ind_df = Indicators(df)
            ind_df.kv()
            ind_df.dv()

            k = df['K'].tolist()
            d = df['D'].tolist()

            if k[-1] > 80 and d[-1] > 80:
                li.append(i)

        return li


if __name__ == "__main__":
    df = pd.read_csv('data/20130101~20220803.csv')

    tsmc_df = df[df['證券代號'] == 2330]
    ind_df = Indicators(tsmc_df)
    ind_df.bias()
    ind_df.bias_line(date='2022-01')
    print(ind_df)

    # ind_df.macd_line(date='2022-05', savefig=False)
    # select_df = Selections(df)
    # print(select_df.kd_goldencross())

    '''
    KD<20
    [1234.0, 1235.0, 1304.0, 1315.0, 1324.0, 1325.0, 1402.0, 1435.0, 1436.0, 1469.0, 1472.0, 1525.0, 1538.0, 1590.0,
     1701.0, 1729.0, 2049.0, 2109.0, 2201.0, 2204.0, 2340.0, 2361.0, 2397.0, 2414.0, 2425.0, 2427.0, 2429.0, 2436.0,
     2486.0, 2499.0, 2610.0, 2722.0, 2833.0, 2841.0, 2852.0, 2905.0, 2911.0, 2923.0, 3014.0, 3026.0, 3312.0, 3406.0,
     3519.0, 3536.0, 3573.0, 3584.0, 3588.0, 3593.0, 4133.0, 4414.0, 4935.0, 4976.0, 5269.0, 5471.0, 6117.0, 6145.0,
     6172.0, 6213.0, 6281.0, 910069.0, 910322.0, 910708.0, 910948.0, 911201.0, 911602.0, 911612.0, 911616.0, 911626.0,
     911868.0, 9910.0, 9919.0, 9926.0, 9931.0, 9940.0, 9941.0, 3682.0, 2712.0, 5259.0, 6415.0, 2929.0, 6449.0, 6452.0,
     4927.0, 8467.0, 1256.0, 6525.0, 8466.0, 4943.0, 8488.0, 4807.0, 8499.0, 8497.0, 8482.0, 8367.0, 6655.0, 6674.0,
     4564.0, 8462.0, 6672.0, 4439.0, 6743.0, 6781.0, 4440.0, 2945.0, 6770.0, 6719.0, 5306.0, 4583.0, 6799.0]
    '''
    '''
    KD>80
    [1301.0, 1303.0, 1321.0, 1437.0, 1439.0, 1451.0, 1456.0, 1464.0, 1516.0, 1530.0, 1536.0, 1537.0, 1604.0, 1616.0, 1618.0,
     1712.0, 1726.0, 1737.0, 2059.0, 2062.0, 2106.0, 2107.0, 2206.0, 2301.0, 2305.0, 2308.0, 2317.0, 2324.0, 2356.0, 2364.0,
     2380.0, 2388.0, 2395.0, 2419.0, 2420.0, 2421.0, 2438.0, 2459.0, 2467.0, 2468.0, 2505.0, 2538.0, 2543.0, 2608.0, 2611.0,
     2809.0, 2880.0, 2885.0, 2886.0, 2890.0, 2892.0, 3017.0, 3022.0, 3024.0, 3040.0, 3229.0, 3231.0, 3474.0, 3494.0, 3518.0,
     3557.0, 3579.0, 3596.0, 3599.0, 3617.0, 3638.0, 3703.0, 4733.0, 4915.0, 4938.0, 4984.0, 5434.0, 5880.0, 6112.0, 6128.0,
     6133.0, 6152.0, 6176.0, 6201.0, 6216.0, 6225.0, 6269.0, 8105.0, 8201.0, 8271.0, 8926.0, 911610.0, 9907.0, 9924.0,
     9929.0, 9934.0, 9942.0, 1817.0, 5519.0, 8996.0, 6442.0, 4557.0, 3321.0, 2243.0, 6581.0, 6625.0, 5876.0, 4571.0, 6754.0,
     6515.0, 6776.0, 2211.0, 6796.0]
    '''
