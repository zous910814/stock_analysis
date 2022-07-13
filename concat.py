import pandas as pd


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


if __name__ == "__main__":
    a = 'data/20180102~20220712.csv'
    b = 'data/20220713.csv'
    concat(a, b)
