import numpy as np
import pandas as pd


def rsv(data):
    # rsv = (今日收盤價 - 最近九天的最低價)/(最近九天的最高價 - 最近九天最低價)
    rsv = (
                  data['收盤價'] - data['收盤價'].rolling(window=9).min()
          ) / (
                  data['收盤價'].rolling(window=9).max() - data['收盤價'].rolling(window=9).min()
          ) * 100
    rsv = np.nan_to_num(rsv)
    return rsv


def kv(rsv):
    # 當日K值=前一日K值 * 2/3 + 當日RSV * 1/3
    kv = [20 for x in range(8)]
    ktemp = kv[0]
    for i in range(len(rsv) - 8):
        ktemp = ktemp * (2 / 3) + rsv[i + 8] * (1 / 3)
        kv.append(round(ktemp, 2))
    return kv


def dv(kv):
    # 當日D值=前一日D值 * 2/3 + 當日K值 * 1/3
    dv = [50 for x in range(8)]
    dtemp = dv[0]
    for i in range(len(kv) - 8):
        dtemp = dtemp * (2 / 3) + kv[i + 8] * (1 / 3)
        dv.append(round(dtemp, 2))
    return dv


if __name__ == "__main__":
    df = pd.read_csv('data/20130101~20220715.csv')
    tsmc = df[df['證券代號'] == 2330]

    rsv = rsv(tsmc)
    kv = kv(rsv)
    dv = dv(kv)

    tsmc['RSV'] = rsv
    tsmc['K'] = kv
    tsmc['D'] = dv
    print(tsmc)
