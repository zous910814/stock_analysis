import random
import requests
from io import StringIO
import pandas as pd
import datetime
import time


def get_date(days: int) -> list:
    begin_date = (datetime.datetime.now() - datetime.timedelta(days=days - 1)).strftime("%Y-%m-%d")
    date_list = []
    begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(time.strftime('%Y-%m-%d', time.localtime(time.time())), "%Y-%m-%d")
    while begin_date <= end_date:
        date_str = begin_date.strftime("%Y%m%d")
        date_list.append(date_str)
        begin_date += datetime.timedelta(days=1)
    return date_list


def download_stock_csv(headers: dict):
    '''
    Download today's stock info:
    column:[證券代號,成交股數,成交筆數,成交金額,開盤價,
            最高價,最低價,收盤價,漲跌(+/-),漲跌價差,
            最後揭示買價,最後揭示買量,最後揭示賣價,最後揭示賣量,本益比,日期]

    :return: csv
    '''
    time.sleep(random.randint(5, 10))
    datestr = time.strftime('%Y%m%d', time.localtime())

    # download stock
    r = requests.post('https://www.twse.com.tw/exchangeReport/MI_INDEX?response=csv&date=' + datestr + '&type=ALL',
                      headers=headers)

    # Organize data into tables
    df = pd.read_csv(StringIO(r.text.replace("=", "")),
                     header=["證券代號" in l for l in r.text.split("\n")].index(True) - 1)
    # Arrange strings:
    df = df.apply(
        lambda s: pd.to_numeric(s.astype(str).str.replace(",", "").replace("+", "1").replace("-", "-1"),
                                errors='coerce'))
    TCC_index_list = df.index[df['證券代號'] == 1101].tolist()

    df = df[df.index >= TCC_index_list[0]]
    df = df.dropna(subset=['證券代號'])
    df = df.drop(['證券名稱', 'Unnamed: 16'], axis=1)
    df['日期'] = datestr
    df.to_csv('data/' + datestr + ".csv", encoding="utf-8_sig", index=False)


# 顯示出來
# print(df)
if __name__ == "__main__":
    # li = get_date(2)
    with open("data/headers.txt", 'r') as f:
        headers = eval(f.read())
    download_stock_csv(headers)
