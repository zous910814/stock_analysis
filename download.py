import requests
from io import StringIO
import pandas as pd
import datetime
import time
import os
import random


def get_date(begin_date: str, end_date: str) -> list:
    '''
    Give a string of two dates and return the consecutive time of the two dates
    :param begin_date: '2022-07-11'
    :param end_date: '2022-07-13'
    :return: list
    '''
    begin_date = begin_date
    end_date = end_date
    date_list = []
    begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    while begin_date <= end_date:
        date_str = begin_date.strftime("%Y%m%d")
        date_list.append(date_str)
        begin_date += datetime.timedelta(days=1)
    return date_list


def download_stock_csv(headers: dict, datestr: str, datestrlist: list):
    '''
    Download today's stock info:
    column:[證券代號,成交股數,成交筆數,成交金額,開盤價,
            最高價,最低價,收盤價,漲跌(+/-),漲跌價差,
            最後揭示買價,最後揭示買量,最後揭示賣價,最後揭示賣量,本益比,日期]

    :return: csv
    '''
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

    if os.path.exists(
            'data/' + datestrlist[0] + '~' + datestrlist[-1] + '.csv'
    ) == False and os.path.exists(
            'data/' + datestrlist[0] + '.csv'
    ) == False:
        if datestrlist[0] == datestrlist[-1]:
            df.to_csv('data/' + datestrlist[0] + '.csv', encoding="utf-8_sig", index=False)
        else:
            df.to_csv('data/' + datestrlist[0] + '~' + datestrlist[-1] + '.csv', encoding="utf-8_sig", index=False)
    else:
        df1 = pd.read_csv('data/' + datestrlist[0] + '~' + datestrlist[-1] + ".csv")
        df1 = pd.concat([df1, df])
        df1.to_csv('data/' + datestrlist[0] + '~' + datestrlist[-1] + '.csv', encoding="utf-8_sig", index=False)
        return df1
    time.sleep(random.randint(5, 10))


if __name__ == "__main__":
    date_liist = get_date('2022-07-13', '2022-07-13')

    with open("data/headers.txt", 'r') as f:
        headers = eval(f.read())

    for date in date_liist:
        try:
            a = download_stock_csv(headers, date, date_liist)
            print('Success', date)
        except ValueError:
            print('Fail', date)
            time.sleep(random.randint(2, 5))
        except requests.exceptions.ConnectionError as e:
            time.sleep(2)
            print("ERROR",e)
            continue
