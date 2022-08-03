import pandas as pd
import datetime

def str2datetime(path):
    df = pd.read_csv(path)
    date = df['日期'].tolist()
    nd = []
    for i in date:
        a = str(i)
        a = datetime.datetime.strptime(a, "%Y%m%d")
        nd.append(str(a))
    df['日期'] = nd
    print(df)
    df.to_csv(path, encoding="utf-8_sig", index=False)


def concat(path1: str, path2: str):
    '''
    input two path and concat two file
    :param path1: '2022-07-11'
    :param path2: '2022-07-13'
    :return: csv
    '''
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df3 = pd.concat([df1, df2])
    df3.to_csv('data/' + path1[5:13] + '~' + path2[5:13] + '.csv', encoding="utf-8_sig", index=False)
    print(df3)

